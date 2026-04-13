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
    if not save_dir or not os.path.isdir(save_dir):
        return None

    # Prefer explicit epoch checkpoints if present.
    epoch_files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth.tar'))
    if epoch_files:
        epoch_files.sort(key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
        return epoch_files[-1]

    # Otherwise, search recursively for any common checkpoint format.
    candidates = []
    for pattern in ('**/model_best.pth.tar', '**/*.pth.tar', '**/*.pth', '**/*.pt'):
        candidates.extend(glob.glob(os.path.join(save_dir, pattern), recursive=True))

    if not candidates:
        return None

    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def find_latest_checkpoint_multi(search_dirs):
    for d in search_dirs:
        if not d:
            continue
        ckpt = find_latest_checkpoint(d)
        if ckpt:
            return ckpt
    return None


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


def parse_gpu_ids(cfg, args_gpu_ids):
    if args_gpu_ids:
        return [int(x.strip()) for x in args_gpu_ids.split(',') if x.strip()]
    gpu_ids = cfg.get('train', {}).get('gpu_ids', [0, 1])
    return [int(x) for x in gpu_ids]


def build_checkpoint_search_dirs(cfg, save_dir):
    search_dirs = [save_dir]
    search_dirs.extend(cfg.get('output', {}).get('resume_dirs', []))
    search_dirs.extend([
        '/kaggle/working/checkpoints/add_gcn',
        '/kaggle/working/checkpoints/addgcn_baseline',
        '/kaggle/working/checkpoints/addgcn',
    ])
    unique_dirs = []
    seen = set()
    for d in search_dirs:
        if d and d not in seen:
            unique_dirs.append(d)
            seen.add(d)
    return unique_dirs


def add_module_prefix(state_dict):
    return {k if k.startswith('module.') else f'module.{k}': v for k, v in state_dict.items()}


def strip_module_prefix(state_dict):
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def load_model_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    try:
        model.load_state_dict(strip_module_prefix(state_dict))
        return
    except RuntimeError:
        pass

    model.load_state_dict(add_module_prefix(state_dict))


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError('Unsupported checkpoint format: cannot find model state dict.')


def load_checkpoint(resume_path, model, optimizer, device, evaluate_only=False):
    checkpoint = torch.load(resume_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    load_model_state_dict(model, state_dict)

    start_epoch = 0
    best_score = -1.0

    if isinstance(checkpoint, dict):
        if not evaluate_only and 'optimizer' in checkpoint and isinstance(checkpoint['optimizer'], dict):
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_score = checkpoint.get('best_score', -1.0)

    return start_epoch, best_score


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
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    state = {
        'epoch': epoch,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_score': best_score,
    }
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description='ADD-GCN baseline for CheXpert')
    parser.add_argument('--config', required=True, help='Path to baseline config YAML')
    parser.add_argument('--evaluate', action='store_true', help='Only run evaluation')
    parser.add_argument('--resume', default='', help='Checkpoint path for resume/eval')
    parser.add_argument('--checkpoint', default='', help='Alias of --resume for checkpoint path')
    parser.add_argument('--subset', type=int, default=None, help='Use only N images for quick smoke-test')
    parser.add_argument('--gpu-ids', default='', help='Comma-separated GPU ids, e.g. "0,1"')
    parser.add_argument('--no-dp-fallback', action='store_true', help='Do not fallback to single GPU if DataParallel fails')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_ids = parse_gpu_ids(cfg, args.gpu_ids)
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

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

    if args.subset:
        n_val = max(50, args.subset // 9)
        train_ds.df = train_ds.df.head(args.subset).reset_index(drop=True)
        val_ds.df = val_ds.df.head(n_val).reset_index(drop=True)
        if val_unc_ds is not None:
            val_unc_ds.df = val_unc_ds.df.head(n_val).reset_index(drop=True)
        print(f'Subset mode: {len(train_ds)} train / {len(val_ds)} val / '
              f'{len(val_unc_ds) if val_unc_ds is not None else 0} val_unc')

    train_loader = build_loader(train_ds, cfg['train']['batch_size'], cfg['train']['workers'], True)
    val_loader = build_loader(val_ds, cfg['train']['batch_size'], cfg['train']['workers'], False)
    val_unc_loader = None
    if val_unc_ds is not None:
        val_unc_loader = build_loader(val_unc_ds, cfg['train']['batch_size'], cfg['train']['workers'], False)

    model = addgcn_resnet101(
        num_classes=NUM_CLASSES,
        pretrained=cfg['model'].get('pretrained', True),
    ).to(device)

    if device.type == 'cuda':
        valid_gpu_ids = [i for i in gpu_ids if 0 <= i < available_gpus]
        if len(valid_gpu_ids) >= 2:
            model = torch.nn.DataParallel(model, device_ids=valid_gpu_ids)
            print(f'Using DataParallel on GPUs: {valid_gpu_ids}')
        else:
            current_gpu = torch.cuda.current_device()
            print(f'Using single GPU: {current_gpu}')
    else:
        print('CUDA not available. Running on CPU.')

    criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    model_for_optim = model.module if isinstance(model, torch.nn.DataParallel) else model
    optimizer = torch.optim.SGD(
        model_for_optim.get_config_optim(cfg['train']['lr'], cfg['train']['lrp']),
        lr=cfg['train']['lr'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay'],
    )

    save_dir = cfg['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_search_dirs = build_checkpoint_search_dirs(cfg, save_dir)

    start_epoch = 0
    best_score = -1.0

    resume_path = args.checkpoint or args.resume or find_latest_checkpoint_multi(checkpoint_search_dirs)
    if resume_path and os.path.isfile(resume_path):
        start_epoch, best_score = load_checkpoint(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            evaluate_only=args.evaluate,
        )
        print(f'Resumed from {resume_path} at epoch {start_epoch}')
    elif not (args.checkpoint or args.resume):
        print(f'No checkpoint found in: {checkpoint_search_dirs}')
    elif args.checkpoint or args.resume:
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint or args.resume}')

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
            try:
                out1, out2 = model(imgs)
            except torch.AcceleratorError as e:
                if isinstance(model, torch.nn.DataParallel) and not args.no_dp_fallback:
                    print('DataParallel failed with AcceleratorError, falling back to single GPU (device 0).')
                    print(f'AcceleratorError: {e}')
                    model = model.module.to(device)
                    model.train()
                    out1, out2 = model(imgs)
                else:
                    raise
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

        score = results['map'] if results['map'] is not None else -1.0
        running_best = max(best_score, score)
        ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth.tar')
        save_checkpoint(ckpt_path, model, optimizer, epoch, running_best)

        if score > best_score:
            best_score = score
            save_checkpoint(os.path.join(save_dir, 'model_best.pth.tar'), model, optimizer, epoch, best_score)
            print(f'New best mAP: {best_score:.4f}')

        if val_unc_loader is not None:
            print('\n=== Validation (Uncertain split) ===')
            results_unc = evaluate_addgcn(model, val_unc_loader, criterion, device)
            print_metrics(results_unc)

    print(f'Finished training. Best mAP = {best_score:.4f}')


if __name__ == '__main__':
    main()
