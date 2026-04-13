# train_chexnet.py
"""
Training script cho CheXNet baseline trên CheXpert-small.

Tái dùng tối đa từ repo:
  - src/data/chexpert.py    → CheXpert dataset (giữ nguyên)
  - src/engine.py           → MultiLabelMAPEngine (không phải GCN version)
  - src/evaluate.py         → evaluate(), print_metrics() (giữ nguyên)

KHÔNG dùng:
  - GCNMultiLabelMAPEngine  (CheXNet không có GCN)
  - word_vec / adj_file     (không cần)

Cách chạy trên Kaggle:
  !python train_chexnet.py --config configs/chexnet.yaml
  !python train_chexnet.py --config configs/chexnet.yaml --subset 1000  # smoke test
"""

import argparse
import os
import sys
import glob

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chexpert import CheXpert
from src.evaluate import evaluate, print_metrics
from src.models.chexnet import build_chexnet, NUM_CLASSES
from src.engine import MultiLabelMAPEngine

# ─────────────────────────────────────────────────────────────────────────────
#  CheXNet Engine — extend MultiLabelMAPEngine
# ─────────────────────────────────────────────────────────────────────────────

class CheXNetEngine(MultiLabelMAPEngine):
    """
    Extend MultiLabelMAPEngine cho CheXNet (không có GCN).

    Thay đổi so với GCNMultiLabelMAPEngine:
      1. on_start_batch: unpack ((img, path), label) — giống GCN version
         nhưng KHÔNG cần inp (word_vec) vì CheXNet là CNN thuần
      2. on_forward: model(img) — 1 argument, không có word_vec
      3. Xử lý uncertain: -1 → 0 cho BCE (giống C1 baseline)
      4. Thu thập val scores để tính AUC — giống GCN version
    """

    def on_start_epoch(self, training, model, criterion, data_loader,
                       optimizer=None, display=True):
        MultiLabelMAPEngine.on_start_epoch(
            self, training, model, criterion, data_loader, optimizer)
        if not training:
            self.state['_val_scores']  = []
            self.state['_val_targets'] = []

    def on_start_batch(self, training, model, criterion, data_loader,
                       optimizer=None, display=True):
        # Giữ label gốc (có -1) cho evaluate
        self.state['target_gt'] = self.state['target'].clone()

        # Remap uncertain -1 → 0 cho BCE loss (giống C1)
        target = self.state['target'].clone().float()
        target[target < 0] = 0.0
        self.state['target'] = target

        # Unpack ((img, path), label) — giống chexpert.py __getitem__
        inp = self.state['input']
        self.state['feature'] = inp[0]   # img tensor (B, 3, H, W)
        self.state['name']    = inp[1]   # list of paths (không dùng)

    def on_end_batch(self, training, model, criterion, data_loader,
                     optimizer=None, display=True):
        MultiLabelMAPEngine.on_end_batch(
            self, training, model, criterion, data_loader, display)

        if not training:
            probs = torch.sigmoid(self.state['output']).detach().cpu().numpy()
            gt    = self.state['target_gt'].detach().cpu().numpy()
            self.state['_val_scores'].append(probs)
            self.state['_val_targets'].append(gt)

    def on_end_epoch(self, training, model, criterion, data_loader,
                     optimizer=None, display=True):
        import numpy as np
        from src.evaluate import compute_mAP, compute_mean_AUC, compute_AUC_uncertain

        loss = self.state['meter_loss'].avg

        if training:
            map_val = 100.0 * self.state['ap_meter'].value().mean()
            if display:
                print(f'Epoch: [{self.state["epoch"]}]\t'
                      f'Loss {loss:.4f}\tmAP {map_val:.3f}')
            return map_val

        # ── Validation ──
        val_scores  = self.state.get('_val_scores', [])
        val_targets = self.state.get('_val_targets', [])

        if not val_scores:
            print(f'Val:\tLoss {loss:.4f}  (no predictions)')
            return 0.0

        scores  = np.concatenate(val_scores,  axis=0)
        targets = np.concatenate(val_targets, axis=0)

        mAP,     per_ap  = compute_mAP(scores, targets)
        mean_auc, per_auc = compute_mean_AUC(scores, targets)
        unc_auc, per_unc  = compute_AUC_uncertain(scores, targets)

        results = {
            'map'              : round(mAP,      4) if not np.isnan(mAP)      else None,
            'mean_auc'         : round(mean_auc, 4) if not np.isnan(mean_auc) else None,
            'unc_auc'          : round(unc_auc,  4) if not np.isnan(unc_auc)  else None,
            'per_class_auc'    : per_auc,
            'per_class_ap'     : per_ap,
            'per_class_unc_auc': per_unc,
        }

        if display:
            print(f'\nVal:\tLoss {loss:.4f}')
            print_metrics(results)

        scheduler = self.state.get('scheduler')
        if scheduler is not None:
            scheduler.step(loss)
            if display:
                lrs = [pg['lr'] for pg in scheduler.optimizer.param_groups]
                print(f'  ReduceLROnPlateau: val_loss={loss:.4f}  lr={lrs}')

        score = mAP if not np.isnan(mAP) else (mean_auc if not np.isnan(mean_auc) else 0.0)
        return float(score)

    def on_forward(self, training, model, criterion, data_loader,
                   optimizer=None, display=True):
        """
        CheXNet forward: model nhận 1 argument (img tensor).
        Khác GCNEngine vốn truyền model(img, word_vec).
        """
        feature_var = self.state['feature'].float()
        target_var  = self.state['target'].float()

        if self.state['use_gpu']:
            feature_var = feature_var.cuda()
            target_var  = target_var.cuda()

        self.state['output'] = model(feature_var)          # logits (B, 14)
        self.state['loss']   = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (tái dùng pattern từ train.py gốc)
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg['train']
    return torch.optim.Adam(
        model.parameters(),          # toàn bộ model, không chia lr
        lr=train_cfg['lr'],          # 0.001
        betas=(train_cfg['beta1'], train_cfg['beta2']),  # (0.9, 0.999)
    )

def build_scheduler(optimizer, cfg: dict):
    """
    ReduceLROnPlateau — step sau mỗi val epoch với val loss (không phải mỗi train epoch).
    Decay × factor khi val loss không giảm sau `patience` epoch.
    """
    train_cfg = cfg['train']
    kind = train_cfg.get('scheduler', 'plateau')
    if kind != 'plateau':
        return None
    kwargs = dict(
        mode='min',
        factor=train_cfg['lr_factor'],
        patience=train_cfg['lr_patience'],
        min_lr=train_cfg['lr_min'],
    )
    try:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **kwargs, verbose=True
        )
    except TypeError:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

def find_latest_checkpoint(save_dir: str):
    """Giống hàm trong train.py gốc - auto-resume."""
    if not os.path.exists(save_dir):
        return None
    pattern = os.path.join(save_dir, 'checkpoint_epoch_*.pth.tar')
    files   = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    return files[-1]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train CheXNet on CheXpert')
    parser.add_argument('--config',    required=True,
                        help='Path to YAML config (configs/chexnet.yaml)')
    parser.add_argument('--subset',    type=int, default=None,
                        help='Dùng N ảnh đầu — quick smoke-test')
    parser.add_argument('--data_root', default=None,
                        help='Override data.root trong config')
    args = parser.parse_args()

    cfg  = load_cfg(args.config)
    root = args.data_root or cfg['data']['root']
    print(f"Config: {cfg['name']}")

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    # ── Datasets ─────────────────────────────────────────────────────────────
    # uncertain='keep' → giữ -1 để engine remap khi tính loss
    # và evaluate.py compute unc_auc đúng
    uncertain_policy = cfg['data'].get('uncertain', 'keep')

    train_ds = CheXpert(
        root=root,
        csv_file=cfg['data']['train_csv'],
        inp_name=cfg['data'].get('word_vec'),   # None cho CheXNet
        uncertain=uncertain_policy,
    )
    val_ds = CheXpert(
        root=root,
        csv_file=cfg['data']['val_csv'],
        inp_name=cfg['data'].get('word_vec'),
        uncertain=uncertain_policy,
    )

    if args.subset:
        n_val = max(50, args.subset // 9)
        train_ds.df = train_ds.df.head(args.subset).reset_index(drop=True)
        val_ds.df   = val_ds.df.head(n_val).reset_index(drop=True)
        print(f'Subset mode: {len(train_ds)} train / {len(val_ds)} val')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_chexnet(
        ckpt_path=cfg['model']['ckpt_path'],
        num_classes=cfg['model']['num_classes'],
    )

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()   # model output là logits
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── Engine ────────────────────────────────────────────────────────────────
    os.makedirs(cfg['output']['save_dir'], exist_ok=True)
    os.makedirs(cfg['output']['log_dir'],  exist_ok=True)

    resume_path = find_latest_checkpoint(cfg['output']['save_dir'])
    print(f"Auto-resume: {resume_path}" if resume_path else "Train from scratch")

    state = {
        'batch_size'        : cfg['train']['batch_size'],
        'image_size'        : cfg['data']['img_size'],
        'max_epochs'        : cfg['train']['epochs'],
        'workers'           : cfg['train']['workers'],
        'epoch_step'        : cfg['train'].get('epoch_step', []),
        'save_model_path'   : cfg['output']['save_dir'],
        'log_dir'           : cfg['output']['log_dir'],
        'print_freq'        : 100,
        'use_pb'            : True,
        'difficult_examples': False,
        'resume'            : resume_path,
        'loss_type'         : cfg['loss']['type'],   # 'bce'
        'scheduler'         : scheduler,
        'skip_adjust_learning_rate': scheduler is not None,
    }

    engine     = CheXNetEngine(state)
    best_score = engine.learning(model, criterion, train_ds, val_ds,
                                 optimizer=optimizer)
    print(f'\n[{cfg["name"]}] Training complete. Best val score = {best_score:.4f}')

    # ── Final evaluation ──────────────────────────────────────────────────────
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_model = model.module if hasattr(model, 'module') else model

    best_path = os.path.join(cfg['output']['save_dir'], 'model_best.pth.tar')
    print(f'\nLoading best model from: {best_path}')
    checkpoint = torch.load(best_path, map_location=device)
    raw_model.load_state_dict(checkpoint['state_dict'])

    # Dùng đúng evaluate() từ evaluate.py — giống C1/C5
    # evaluate() nhận model(imgs) → tự gọi sigmoid bên trong
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=state['batch_size'],
        shuffle=False,
        num_workers=state['workers'],
        pin_memory=(device == 'cuda'),
    )

    print('\n=== Final validation metrics (CheXNet) ===')
    results = evaluate(raw_model, val_loader, device=device)
    print_metrics(results)


if __name__ == '__main__':
    main()
