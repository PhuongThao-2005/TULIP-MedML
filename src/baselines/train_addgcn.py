import argparse
import glob
import os
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.chexpert import CheXpert, NUM_CLASSES, get_transform
from src.evaluate import compute_AUC_uncertain, compute_mAP, compute_mean_AUC, print_metrics
from src.models.addgcn import addgcn_resnet101


def load_cfg(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(save_dir):
    pattern = os.path.join(save_dir, 'checkpoint_epoch_*.pth.tar')
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
    return files[-1]


def build_loader(dataset, batch_size, workers, shuffle):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def to_train_targets(targets):
    targets = targets.clone()
    targets[targets == -1] = 0
    return targets


def evaluate_addgcn(model, loader, criterion, device):
    model.eval()
    losses = []
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            inputs, labels = batch
            imgs = inputs[0]

            imgs = imgs.to(device)
            labels = labels.to(device)
            train_targets = to_train_targets(labels)

            out1, out2 = model(imgs)
            logits = (out1 + out2) / 2.0
            loss = criterion(logits, train_targets)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)
            all_scores.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    map_score, per_class_ap = compute_mAP(scores, targets)
    mean_auc, per_class_auc = compute_mean_AUC(scores, targets)
    unc_auc, per_class_unc = compute_AUC_uncertain(scores, targets)

    results = {
        'map': round(map_score, 4) if not np.isnan(map_score) else None,
        'mean_auc': round(mean_auc, 4) if not np.isnan(mean_auc) else None,
        'unc_auc': round(unc_auc, 4) if not np.isnan(unc_auc) else None,
        'per_class_auc': per_class_auc,
        'per_class_ap': per_class_ap,
        'per_class_unc_auc': per_class_unc,
        'loss': float(np.mean(losses)) if losses else None,
    }
    return results


def save_checkpoint(path, model, optimizer, epoch, best_score):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_score': best_score,
    }
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description='ADD-GCN baseline for CheXpert')
    parser.add_argument('--config', required=True, help='Path to baseline config YAML')
    parser.add_argument('--evaluate', action='store_true', help='Only run evaluation')
    parser.add_argument('--resume', default='', help='Checkpoint path for resume/eval')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    train_ds = CheXpert(
        root=cfg['data']['root'],
        csv_file=cfg['data']['train_csv'],
        inp_name=cfg['data']['word_vec'],
        uncertain=cfg['data'].get('uncertain', 'keep'),
        split='train',
        transform=get_transform('train', size=cfg['data']['img_size']),
    )
    val_ds = CheXpert(
        root=cfg['data']['root'],
        csv_file=cfg['data']['val_csv'],
        inp_name=cfg['data']['word_vec'],
        uncertain='keep',
        split='val',
        transform=get_transform('val', size=cfg['data']['img_size']),
    )

    val_unc_ds = None
    if cfg['data'].get('val_uncertain_csv'):
        val_unc_ds = CheXpert(
            root=cfg['data']['root'],
            csv_file=cfg['data']['val_uncertain_csv'],
            inp_name=cfg['data']['word_vec'],
            uncertain='keep',
            split='val',
            transform=get_transform('val', size=cfg['data']['img_size']),
        )

    train_loader = build_loader(train_ds, cfg['train']['batch_size'], cfg['train']['workers'], True)
    val_loader = build_loader(val_ds, cfg['train']['batch_size'], cfg['train']['workers'], False)
    val_unc_loader = None
    if val_unc_ds is not None:
        val_unc_loader = build_loader(val_unc_ds, cfg['train']['batch_size'], cfg['train']['workers'], False)

    model = addgcn_resnet101(
        num_classes=NUM_CLASSES,
        pretrained=cfg['model'].get('pretrained', True),
    ).to(device)

    criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    optimizer = torch.optim.SGD(
        model.get_config_optim(cfg['train']['lr'], cfg['train']['lrp']),
        lr=cfg['train']['lr'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay'],
    )

    save_dir = cfg['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    best_score = -1.0

    resume_path = args.resume or find_latest_checkpoint(save_dir)
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if not args.evaluate and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_score = checkpoint.get('best_score', -1.0)
        print(f'Resumed from {resume_path} at epoch {start_epoch}')

    if args.evaluate:
        print('\n=== Validation ===')
        results = evaluate_addgcn(model, val_loader, criterion, device)
        print_metrics(results)

        if val_unc_loader is not None:
            print('\n=== Validation (Uncertain split) ===')
            results_unc = evaluate_addgcn(model, val_unc_loader, criterion, device)
            print_metrics(results_unc)
        return

    milestones = set(cfg['train'].get('epoch_step', []))
    for epoch in range(start_epoch, cfg['train']['epochs']):
        if epoch in milestones:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1

        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg["train"]["epochs"]}')
        for batch in pbar:
            inputs, labels = batch
            imgs = inputs[0].to(device)
            labels = labels.to(device)

            train_targets = to_train_targets(labels)
            out1, out2 = model(imgs)
            logits = (out1 + out2) / 2.0
            loss = criterion(logits, train_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['train'].get('max_clip_grad_norm', 10.0))
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f'{np.mean(losses):.4f}')

        print(f'Train loss: {np.mean(losses):.4f}')

        print('\n=== Validation ===')
        results = evaluate_addgcn(model, val_loader, criterion, device)
        print_metrics(results)

        score = results['mean_auc'] if results['mean_auc'] is not None else (results['map'] or -1.0)
        running_best = max(best_score, score)
        ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth.tar')
        save_checkpoint(ckpt_path, model, optimizer, epoch, running_best)

        if score > best_score:
            best_score = score
            save_checkpoint(os.path.join(save_dir, 'model_best.pth.tar'), model, optimizer, epoch, best_score)
            print(f'New best score: {best_score:.4f}')

        if val_unc_loader is not None:
            print('\n=== Validation (Uncertain split) ===')
            results_unc = evaluate_addgcn(model, val_unc_loader, criterion, device)
            print_metrics(results_unc)

    print(f'Finished training. Best score = {best_score:.4f}')


if __name__ == '__main__':
    main()
