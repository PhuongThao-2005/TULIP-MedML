import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml
import glob


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chexpert import CheXpert, NUM_CLASSES
from src.engine import GCNMultiLabelMAPEngine
from src.models.gcn import gcn_resnet101, gcn_swin_t, gcn_swin_b
from src.evaluate import evaluate, print_metrics
from src.loss.ua_asl import UncertaintyAwareASL
from src.loss.asl import AsymmetricLoss

def load_cfg(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.get_config_optim(cfg['train']['lr'], cfg['train']['lrp']),
        lr=cfg['train']['lr'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay'],
    )


def build_criterion(cfg: dict) -> nn.Module:
    loss_cfg = cfg.get('loss', {})
    loss_type = loss_cfg.get('type', 'bce').lower()

    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()

    if loss_type == 'ua_asl':
        return UncertaintyAwareASL(
            gamma_pos=loss_cfg.get('gamma_pos', 0.0),
            gamma_neg=loss_cfg.get('gamma_neg', 4.0),
            margin=loss_cfg.get('margin', 0.05),
            lambda_unc=loss_cfg.get('lambda_unc', 0.5),
            alpha=loss_cfg.get('alpha', 0.5),
            reduction=loss_cfg.get('reduction', 'mean'),
        )

    if loss_type == 'asl':
        return AsymmetricLoss(
            gamma_pos=loss_cfg.get('gamma_pos', 0.0),
            gamma_neg=loss_cfg.get('gamma_neg', 4.0),
            margin=loss_cfg.get('margin', 0.05),
            reduction=loss_cfg.get('reduction', 'mean'),
            disable_torch_grad_focal_loss=loss_cfg.get(
                'disable_torch_grad_focal_loss', True
            ),
        )

    raise ValueError(f"Unsupported loss type: {loss_type}")

def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None

    # Only search in the specific save_dir for this config
    pattern = os.path.join(save_dir, 'checkpoint_epoch_*.pth.tar')
    files = glob.glob(pattern)

    if not files:
        return None

    files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    return files[-1]

def build_model(cfg: dict):
    backbone = cfg['model'].get('backbone', 'resnet101').lower()
    common_kwargs = {
        'num_classes': NUM_CLASSES,
        't': cfg['model']['t'],
        'pretrained': cfg['model']['pretrained'],
        'adj_file': cfg['data']['adj'],
        'in_channel': cfg['model']['gcn_in'],
        'inp_file': cfg['data']['word_vec'],
    }

    if backbone == 'resnet101':
        return gcn_resnet101(**common_kwargs)
    if backbone == 'swin_t':
        return gcn_swin_t(**common_kwargs)
    if backbone == 'swin_b':
        return gcn_swin_b(**common_kwargs)

    raise ValueError(f"Unsupported backbone: {backbone}")

def main():
    parser = argparse.ArgumentParser(description='Train GCN on CheXpert')
    parser.add_argument('--config',    required=True,
                        help='Path to YAML config file')
    parser.add_argument('--subset',    type=int, default=None,
                        help='Use only N images (quick smoke-test)')
    parser.add_argument('--data_root', default=None,
                        help='Override data.root from config '
                             '(e.g. /kaggle/input/chexpert-v10-small)')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    print(f"Config: {cfg['name']}")

    root = args.data_root or cfg['data']['root']

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    # ── Datasets ─────────────────────────────────────────────────────────────
    # uncertain='keep' preserves -1 labels so evaluate.py can compute
    # AUC-uncertain.  The engine's on_start_batch remaps -1 → 0 for BCE loss
    # but keeps target_gt intact for evaluation.  Override via config if needed.
    uncertain_policy = cfg['data'].get('uncertain', 'keep')

    train_ds = CheXpert(
        root=root,
        csv_file=cfg['data']['train_csv'],
        inp_name=cfg['data']['word_vec'],
        uncertain=uncertain_policy,
    )
    val_ds = CheXpert(
        root=root,
        csv_file=cfg['data']['val_csv'],
        inp_name=cfg['data']['word_vec'],
        uncertain=uncertain_policy,
    )
    
    val_uncertain_ds = CheXpert(
        root=root,
        csv_file=cfg['data']['val_uncertain_csv'],  # thêm trong config
        inp_name=cfg['data']['word_vec'],
        uncertain='keep',  # giữ -1 để compute unc_auc
    )

    if args.subset:
        n_val = max(50, args.subset // 9)
        train_ds.df = train_ds.df.head(args.subset).reset_index(drop=True)
        val_ds.df   = val_ds.df.head(n_val).reset_index(drop=True)
        print(f'Subset mode: {len(train_ds)} train / {len(val_ds)} val')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)

    # ── Engine state ──────────────────────────────────────────────────────────
    os.makedirs(cfg['output']['save_dir'], exist_ok=True)
    os.makedirs(cfg['output']['log_dir'], exist_ok=True)
    resume_path = find_latest_checkpoint(cfg['output']['save_dir'])
    if resume_path:
        print(f"Auto-resume from: {resume_path}")
    else:
        print("Train from scratch")
    state = {
        'batch_size'        : cfg['train']['batch_size'],
        'image_size'        : cfg['data']['img_size'],
        'max_epochs'        : cfg['train']['epochs'],
        'workers'           : cfg['train']['workers'],
        'epoch_step'        : cfg['train']['epoch_step'],
        'save_model_path'   : cfg['output']['save_dir'],
        'log_dir'           : cfg['output']['log_dir'],
        'print_freq'        : 100,
        'use_pb'            : True,
        'difficult_examples': False,
        'resume'            : resume_path,
        'loss_type'        : cfg['loss']['type'],
    }

    # ── Training ──────────────────────────────────────────────────────────────
    engine     = GCNMultiLabelMAPEngine(state)
    best_score = engine.learning(model, criterion, train_ds, val_ds, val_uncertain_dataset=val_uncertain_ds, optimizer = optimizer)
    print(f'\n[{cfg["name"]}] Training complete.  Best val score = {best_score:.4f}')

    # ── Final evaluation on validation set ───────────────────────────────────
    # Run a clean, full-dataset evaluation using evaluate.py for a definitive
    # summary table.  Uses the model as stored (DataParallel-wrapped if GPU).
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Unwrap DataParallel if needed for evaluate()
    raw_model = model.module if hasattr(model, 'module') else model

    best_path = os.path.join(cfg['output']['save_dir'], 'model_best.pth.tar')
    print(f'Loading best model from: {best_path}')

    checkpoint = torch.load(best_path, map_location=device)
    raw_model.load_state_dict(checkpoint['state_dict'])

    # Build a fresh val loader with the val transform the engine set
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=state['batch_size'],
        shuffle=False,
        num_workers=state['workers'],
        pin_memory=(device == 'cuda'),
    )

    print('\n=== Final validation metrics ===')
    results = evaluate(raw_model, val_loader, device=device)
    print_metrics(results)
    
    # --- Val uncertain ---
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
