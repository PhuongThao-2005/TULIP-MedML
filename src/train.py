import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chexpert import CheXpert, NUM_CLASSES
from src.engine import GCNMultiLabelMAPEngine
from src.models.gcn import gcn_resnet101
from src.evaluate import evaluate, print_metrics


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.get_config_optim(cfg['train']['lr'], cfg['train']['lrp']),
        lr=cfg['train']['lr'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay'],
    )


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

    if args.subset:
        n_val = max(50, args.subset // 9)
        train_ds.df = train_ds.df.head(args.subset).reset_index(drop=True)
        val_ds.df   = val_ds.df.head(n_val).reset_index(drop=True)
        print(f'Subset mode: {len(train_ds)} train / {len(val_ds)} val')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = gcn_resnet101(
        num_classes=NUM_CLASSES,
        t=cfg['model']['t'],
        pretrained=cfg['model']['pretrained'],
        adj_file=cfg['data']['adj'],
        in_channel=cfg['model']['gcn_in'],
        inp_file=cfg['data']['word_vec'],
    )

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    # BCEWithLogitsLoss works on {0, 1} labels produced by on_start_batch
    # (uncertain -1 is remapped to 0 = treat as negative for loss).
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, cfg)

    # ── Engine state ──────────────────────────────────────────────────────────
    os.makedirs(cfg['output']['save_dir'], exist_ok=True)
    state = {
        'batch_size'        : cfg['train']['batch_size'],
        'image_size'        : cfg['data']['img_size'],
        'max_epochs'        : cfg['train']['epochs'],
        'workers'           : cfg['train']['workers'],
        'epoch_step'        : cfg['train']['epoch_step'],
        'save_model_path'   : cfg['output']['save_dir'],
        'print_freq'        : 100,
        'use_pb'            : True,
        'difficult_examples': False,
    }

    # ── Training ──────────────────────────────────────────────────────────────
    engine     = GCNMultiLabelMAPEngine(state)
    best_score = engine.learning(model, criterion, train_ds, val_ds, optimizer)
    print(f'\n[{cfg["name"]}] Training complete.  Best val score = {best_score:.4f}')

    # ── Final evaluation on validation set ───────────────────────────────────
    # Run a clean, full-dataset evaluation using evaluate.py for a definitive
    # summary table.  Uses the model as stored (DataParallel-wrapped if GPU).
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Unwrap DataParallel if needed for evaluate()
    raw_model = model.module if hasattr(model, 'module') else model

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


if __name__ == '__main__':
    main()
