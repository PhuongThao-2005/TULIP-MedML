"""
File: train.py

Description:
    Main training entrypoint for GCN-based CheXpert experiments.
    This script loads YAML config, builds dataset/model/loss/optimizer,
    runs training, and reports final validation metrics.

Main Components:
    - load_cfg: Read YAML configuration file.
    - build_optimizer: Build SGD optimizer with backbone/GCN parameter groups.
    - build_criterion: Build BCE or uncertainty-aware loss.
    - build_model: Build GCN model by selected backbone.
    - main: End-to-end training and final evaluation pipeline.

Inputs:
    - YAML config path (`--config`)
    - Optional subset size (`--subset`)
    - Optional data root override (`--data_root`)

Outputs:
    - Trained checkpoints in `output.save_dir`
    - Training logs in `output.log_dir`
    - Printed metrics on official and uncertain validation splits

Notes:
    - Uses `GCNMultiLabelMAPEngine` from `src.engine`.
    - Targets follow {-1, 0, 1}; handling depends on selected loss.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml
import glob

# add project root to PYTHONPATH so `src.*` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chexpert import CheXpert, NUM_CLASSES
from src.engine import GCNMultiLabelMAPEngine
from src.models.gcn import gcn_resnet101, gcn_swin_t, gcn_swin_b
from src.evaluate import evaluate, print_metrics
from src.loss.ua_asl import UncertaintyAwareASL


def load_cfg(path: str) -> dict:
    """
    Load experiment config from YAML file.

    Args:
        path (str): Absolute or relative path to YAML config.

    Returns:
        dict: Parsed configuration dictionary.

    Shape:
        - None (configuration loader).

    Notes:
        - Keep config as dictionary for easy key-based access.
    """
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    """
    Build SGD optimizer for GCN model.

    Args:
        model (nn.Module): Model exposing `get_config_optim`.
        cfg (dict): Parsed YAML config.

    Returns:
        torch.optim.Optimizer: Configured SGD optimizer.

    Shape:
        - None (optimizer builder).

    Notes:
        - Uses `lr` and `lrp` from config to separate LR groups.
    """
    return torch.optim.SGD(
        model.get_config_optim(cfg['train']['lr'], cfg['train']['lrp']),
        lr=cfg['train']['lr'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay'],
    )


def build_criterion(cfg: dict) -> nn.Module:
    """
    Build training criterion from config.

    Args:
        cfg (dict): Parsed YAML config.

    Returns:
        nn.Module: Loss module (`BCEWithLogitsLoss` or `UncertaintyAwareASL`).

    Shape:
        logits: (B, C)
        targets: (B, C)

    Notes:
        - `BCEWithLogitsLoss` expects targets in {0, 1}.
        - `UncertaintyAwareASL` supports targets in {-1, 0, 1}.
    """
    loss_cfg = cfg.get('loss', {})
    loss_type = loss_cfg.get('type', 'bce').lower()

    if loss_type == 'bce':
        # standard multi-label classification loss
        # requires targets in {0, 1}
        return nn.BCEWithLogitsLoss()

    if loss_type == 'ua_asl':
        # uncertainty-aware asymmetric loss
        # supports targets in {-1, 0, 1}
        return UncertaintyAwareASL(
            gamma_pos=loss_cfg.get('gamma_pos', 0.0),
            gamma_neg=loss_cfg.get('gamma_neg', 4.0),
            margin=loss_cfg.get('margin', 0.05),
            lambda_unc=loss_cfg.get('lambda_unc', 0.5),
            alpha=loss_cfg.get('alpha', 0.5),
            reduction=loss_cfg.get('reduction', 'mean'),
        )

    raise ValueError(f"Unsupported loss type: {loss_type}")


def find_latest_checkpoint(save_dir):
    """
    Find latest epoch checkpoint in a save directory.

    Args:
        save_dir (str): Directory containing checkpoint files.

    Returns:
        str | None: Latest checkpoint path or None if no checkpoint found.

    Shape:
        - None (filesystem utility).

    Notes:
        - Expects checkpoint format: `checkpoint_epoch_{epoch}.pth.tar`.
    """
    if not os.path.exists(save_dir):
        return None

    # Search only inside this experiment's save directory.
    pattern = os.path.join(save_dir, 'checkpoint_epoch_*.pth.tar')
    files = glob.glob(pattern)

    if not files:
        return None

    # sort by epoch index extracted from filename
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    return files[-1]


def build_model(cfg: dict):
    """
    Build GCN model according to selected backbone.

    Args:
        cfg (dict): Parsed YAML config.

    Returns:
        nn.Module: Instantiated GCN model.

    Shape:
        image_features: (B, D)
        label_embeddings: (C, D)
        logits: (B, C), where C = NUM_CLASSES

    Notes:
        - Terminology:
            - image_features: visual representation from backbone
            - label_embeddings: class/node embeddings propagated by GCN
            - adjacency_matrix: loaded from `cfg['data']['adj']`
    """
    backbone = cfg['model'].get('backbone', 'resnet101').lower()

    # shared arguments for all backbones
    common_kwargs = {
        'num_classes': NUM_CLASSES,
        't': cfg['model']['t'],                  # threshold for adjacency
        'pretrained': cfg['model']['pretrained'],
        'adj_file': cfg['data']['adj'],          # adjacency_matrix (C, C)
        'in_channel': cfg['model']['gcn_in'],    # embedding dimension
        'inp_file': cfg['data']['word_vec'],     # label_embeddings init
    }

    # choose backbone architecture
    if backbone == 'resnet101':
        return gcn_resnet101(**common_kwargs)
    if backbone == 'swin_t':
        return gcn_swin_t(**common_kwargs)
    if backbone == 'swin_b':
        return gcn_swin_b(**common_kwargs)

    raise ValueError(f"Unsupported backbone: {backbone}")


def main():
    """
    Run end-to-end training and final evaluation.

    Args:
        None: Parsed from CLI.

    Returns:
        None

    Shape:
        image_features: (B, 3, H, W)
        targets: (B, C)
        logits: (B, C)
        probs: (B, C) after sigmoid in evaluation

    Notes:
        - Uses `uncertain='keep'` so evaluation can compute uncertain AUC.
        - Engine handles target remapping depending on loss type.
    """
    parser = argparse.ArgumentParser(description='Train GCN on CheXpert')
    parser.add_argument('--config',    required=True,
                        help='Path to YAML config file')
    parser.add_argument('--subset',    type=int, default=None,
                        help='Use only N images (quick smoke-test)')
    parser.add_argument('--data_root', default=None,
                        help='Override data.root from config '
                             '(e.g. /kaggle/input/chexpert-v10-small)')
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # Load config
    # ─────────────────────────────────────────────────────────────────────────
    cfg = load_cfg(args.config)
    print(f"Config: {cfg['name']}")

    # allow overriding dataset root (useful for Kaggle / cluster)
    root = args.data_root or cfg['data']['root']

    # fix random seed for reproducibility
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    # Keep uncertain targets as -1 so uncertain-aware metrics can be computed.
    uncertain_policy = cfg['data'].get('uncertain', 'keep')

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────────────────────────────────────────
    # train_ds sample format:
    #   image_features: (3, H, W), targets: (C,)
    train_ds = CheXpert(
        root=root,
        csv_file=cfg['data']['train_csv'],
        inp_name=cfg['data']['word_vec'],
        uncertain=uncertain_policy,
    )

    # val_ds sample format:
    #   image_features: (3, H, W), targets: (C,)
    val_ds = CheXpert(
        root=root,
        csv_file=cfg['data']['val_csv'],
        inp_name=cfg['data']['word_vec'],
        uncertain=uncertain_policy,
    )
    # val_uncertain_ds sample format:
    #   image_features: (3, H, W), targets: (C,)
    val_uncertain_ds = CheXpert(
        root=root,
        csv_file=cfg['data']['val_uncertain_csv'],
        inp_name=cfg['data']['word_vec'],
        uncertain='keep',
    )

    # subset mode: use only N images for quick test
    if args.subset:
        n_val = max(50, args.subset // 9)
        train_ds.df = train_ds.df.head(args.subset).reset_index(drop=True)
        val_ds.df   = val_ds.df.head(n_val).reset_index(drop=True)
        print(f'Subset mode: {len(train_ds)} train / {len(val_ds)} val')


    # ─────────────────────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────────────────────
    model = build_model(cfg) # build GCN model

    # ─────────────────────────────────────────────────────────────────────────
    # Loss & Optimizer
    # ─────────────────────────────────────────────────────────────────────────
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # Engine state
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(cfg['output']['save_dir'], exist_ok=True)
    os.makedirs(cfg['output']['log_dir'], exist_ok=True)
    
    # auto-resume latest checkpoint
    resume_path = find_latest_checkpoint(cfg['output']['save_dir'])
    if resume_path:
        print(f"Auto-resume from: {resume_path}")
    else:
        print("Train from scratch")
    state = {
        # Data pipeline
        'batch_size'        : cfg['train']['batch_size'],
        'image_size'        : cfg['data']['img_size'],

        # Training schedule
        'max_epochs'        : cfg['train']['epochs'],
        'workers'           : cfg['train']['workers'],
        'epoch_step'        : cfg['train']['epoch_step'],

        # Logging/checkpoint
        'save_model_path'   : cfg['output']['save_dir'],
        'log_dir'           : cfg['output']['log_dir'],

        # Runtime behavior
        'print_freq'        : 100,
        'use_pb'            : True,
        'difficult_examples': False,
        'resume'            : resume_path,

        # Loss behavior switch in engine
        'loss_type'        : cfg['loss']['type'],
    }


    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────
    engine     = GCNMultiLabelMAPEngine(state)
    
    best_score = engine.learning(model, criterion, train_ds, val_ds, val_uncertain_dataset=val_uncertain_ds, optimizer = optimizer)
    print(f'\n[{cfg["name"]}] Training complete.  Best val score = {best_score:.4f}')

    # ─────────────────────────────────────────────────────────────────────────
    # Final evaluation (load best checkpoint)
    # ─────────────────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # unwrap DataParallel for evaluation
    raw_model = model.module if hasattr(model, 'module') else model

    best_path = os.path.join(cfg['output']['save_dir'], 'model_best.pth.tar')
    print(f'Loading best model from: {best_path}')

    checkpoint = torch.load(best_path, map_location=device)

    # load best weights
    raw_model.load_state_dict(checkpoint['state_dict'])

    # Build validation loader for final report.
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=state['batch_size'],
        shuffle=False,
        num_workers=state['workers'],
        pin_memory=(device == 'cuda'),
    )

    print('\n=== Final validation metrics ===')
    
    # compute metrics: mAP, AUC, uncertain AUC
    results = evaluate(raw_model, val_loader, device=device)
    print_metrics(results)
    
    # Evaluate uncertain split separately.
    val_unc_loader = torch.utils.data.DataLoader(
        val_uncertain_ds,
        batch_size=state['batch_size'],
        shuffle=False,
        num_workers=state['workers'],
        pin_memory=(device == 'cuda'),
    )

    print('\n=== Validation (Uncertain split) ===')

    results_unc = evaluate(raw_model, val_unc_loader, device=device)
    print_metrics(results_unc)


if __name__ == '__main__':
    main()
