"""
File: train_addgcn.py
Description:
    Training and evaluation script for the ADD-GCN baseline on CheXpert.
    Supports multi-GPU training via DataParallel, automatic checkpoint
    resumption, and optional evaluation-only mode.
Main Components:
    - main: Full training loop with validation, checkpointing, and final eval.
    - evaluate_addgcn: Evaluation loop returning mAP, mean AUC, and uncertain AUC.
    - load_checkpoint / save_checkpoint: Checkpoint utilities.
    - find_latest_checkpoint / find_latest_checkpoint_multi: Auto-resume helpers.
Inputs:
    CheXpert images (B, 3, H, W) from CheXpertDataset.
Outputs:
    Per-epoch checkpoints and model_best.pth.tar in the configured save_dir.
Notes:
    - Final logits = (out1 + out2) / 2.0 (ensemble of global and GCN branches).
    - Uncertain labels (-1) are remapped to 0 for MultiLabelSoftMarginLoss.
    - cuDNN is disabled by default to avoid misaligned address errors on Kaggle.
"""

import argparse
import glob
import os
import sys
import torch.nn as nn

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.chexpert import CheXpert, NUM_CLASSES, get_transform
from src.evaluate import compute_AUC_uncertain, compute_mAP, compute_mean_AUC, print_metrics
from src.models.addgcn import addgcn_resnet101

# Config and Checkpoint Utilities

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(save_dir: str) -> str | None:
    """
    Find the most recent checkpoint file in a directory.

    Prefers epoch-numbered checkpoint files (checkpoint_epoch_N.pth.tar).
    Falls back to any .pth.tar, .pth, or .pt file sorted by modification time.

    Args:
        save_dir (str): Directory to search for checkpoint files.

    Returns:
        str or None: Path to the latest checkpoint, or None if not found.
    """
    if not save_dir or not os.path.isdir(save_dir):
        return None

    # Prefer epoch-numbered checkpoints for deterministic ordering
    epoch_files = glob.glob(
        os.path.join(save_dir, "checkpoint_epoch_*.pth.tar")
    )
    if epoch_files:
        epoch_files.sort(
            key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0])
        )
        return epoch_files[-1]

    # Fall back to any checkpoint format, sorted by modification time
    candidates = []
    for pattern in ("**/model_best.pth.tar", "**/*.pth.tar", "**/*.pth", "**/*.pt"):
        candidates.extend(
            glob.glob(os.path.join(save_dir, pattern), recursive=True)
        )

    candidates = [p for p in candidates if os.path.isfile(p)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def find_latest_checkpoint_multi(search_dirs: list[str]) -> str | None:
    """
    Search multiple directories for the latest checkpoint, in priority order.

    Args:
        search_dirs (list[str]): Directories to search, tried in order.

    Returns:
        str or None: Path to the first valid checkpoint found.
    """
    for directory in search_dirs:
        if not directory:
            continue
        checkpoint = find_latest_checkpoint(directory)
        if checkpoint:
            return checkpoint
    return None


def build_checkpoint_search_dirs(cfg: dict, save_dir: str) -> list[str]:
    """
    Build the ordered list of directories to search for auto-resume checkpoints.

    Kaggle notebook output and working directories are checked first, followed
    by the configured save directory and any extra directories from config.

    Args:
        cfg (dict): Full configuration dictionary.
        save_dir (str): Primary output checkpoint directory.

    Returns:
        list[str]: Deduplicated list of directories to search.
    """
    candidates = [
        "/kaggle/input/notebooks/myvnthdim/notebook0de352a96d/checkpoints/addgcn_baseline",
        "/kaggle/working/checkpoints/addgcn_baseline",
        save_dir,
    ] + cfg.get("output", {}).get("resume_dirs", [])

    # Preserve insertion order while deduplicating
    seen = set()
    unique_dirs = []
    for directory in candidates:
        if directory and directory not in seen:
            unique_dirs.append(directory)
            seen.add(directory)
    return unique_dirs


def _add_module_prefix(state_dict: dict) -> dict:
    """Prepend 'module.' to all keys for DataParallel compatibility."""
    return {
        k if k.startswith("module.") else f"module.{k}": v
        for k, v in state_dict.items()
    }


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by DataParallel from state dict keys."""
    return {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def _load_model_state_dict(model: nn.Module, state_dict: dict) -> None:
    """
    Load a state dict into a model, handling DataParallel prefix mismatches.

    Tries direct loading first, then strips or adds 'module.' prefix as needed.

    Args:
        model (nn.Module): Target model.
        state_dict (dict): State dict to load.
    """
    for attempt in (
        state_dict,
        _strip_module_prefix(state_dict),
        _add_module_prefix(state_dict),
    ):
        try:
            model.load_state_dict(attempt)
            return
        except RuntimeError:
            continue
    raise RuntimeError("Failed to load state dict: incompatible keys.")


def _extract_state_dict(checkpoint) -> dict:
    """
    Extract the model state dict from a checkpoint, handling various formats.

    Args:
        checkpoint: Loaded checkpoint (dict or raw state dict).

    Returns:
        dict: Model state dict.

    Raises:
        ValueError: If the checkpoint format is unrecognized.
    """
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        # Assume the entire dict is the state dict
        return checkpoint
    raise ValueError("Unsupported checkpoint format: cannot find model state dict.")


def load_checkpoint(
    resume_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    evaluate_only: bool = False,
) -> tuple[int, float]:
    """
    Load model (and optionally optimizer) state from a checkpoint file.

    Args:
        resume_path (str): Path to the checkpoint file.
        model (nn.Module): Model to load weights into.
        optimizer: Optimizer to restore state into (skipped when evaluate_only).
        device (torch.device): Device to map the checkpoint tensors to.
        evaluate_only (bool): If True, skip optimizer state loading.

    Returns:
        tuple:
            - start_epoch (int): Next epoch index to resume training from.
            - best_score (float): Best validation score recorded in checkpoint.
    """
    checkpoint = torch.load(resume_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    _load_model_state_dict(model, state_dict)

    start_epoch = 0
    best_score = -1.0

    if isinstance(checkpoint, dict):
        if (
            not evaluate_only
            and "optimizer" in checkpoint
            and isinstance(checkpoint["optimizer"], dict)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_score = checkpoint.get("best_score", -1.0)

    return start_epoch, best_score


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
) -> None:
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        path (str): Output file path.
        model (nn.Module): Model to save (DataParallel wrapper is unwrapped).
        optimizer: Optimizer whose state dict to save.
        epoch (int): Current epoch index.
        best_score (float): Best validation score achieved so far.
    """
    # Unwrap DataParallel to save only the underlying model state
    model_to_save = (
        model.module if isinstance(model, torch.nn.DataParallel) else model
    )
    state = {
        "epoch": epoch,
        "state_dict": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
    }
    torch.save(state, path)

# Data Loading

def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """
    Construct a DataLoader with pinned memory when CUDA is available.

    Args:
        dataset: A PyTorch Dataset instance.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        shuffle (bool): Whether to shuffle at each epoch.

    Returns:
        DataLoader: Configured data loader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _remap_uncertain_to_negative(targets: torch.Tensor) -> torch.Tensor:
    """
    Remap uncertain labels (-1) to negative (0) for MultiLabelSoftMarginLoss.

    MultiLabelSoftMarginLoss expects targets in {0, 1}.
    The original targets with -1 are preserved for AUC metric computation.

    Args:
        targets (Tensor): Labels with values in {-1, 0, 1}.

    Returns:
        Tensor: Labels with -1 replaced by 0.
    """
    remapped = targets.clone()
    remapped[remapped == -1] = 0
    return remapped


def _parse_gpu_ids(cfg: dict, args_gpu_ids: str) -> list[int]:
    """
    Parse GPU IDs from CLI argument or config fallback.

    Args:
        cfg (dict): Full configuration dictionary.
        args_gpu_ids (str): Comma-separated GPU IDs from CLI (may be empty).

    Returns:
        list[int]: List of GPU device indices.
    """
    if args_gpu_ids:
        return [int(x.strip()) for x in args_gpu_ids.split(",") if x.strip()]
    return [int(x) for x in cfg.get("train", {}).get("gpu_ids", [0, 1])]

# Evaluation

def evaluate_addgcn(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
) -> dict:
    """
    Run evaluation loop and compute mAP, mean AUC, and uncertain-aware AUC.

    Args:
        model (nn.Module): Model in eval mode.
        loader (DataLoader): Validation data loader.
        criterion: Loss function compatible with (logits, targets) signature.
        device (torch.device): Device to run inference on.

    Returns:
        dict: Evaluation metrics with keys:
            "map", "mean_auc", "unc_auc",
            "per_class_auc", "per_class_ap", "per_class_unc_auc", "loss"

    Notes:
        - Final logits are the average of the two ADD-GCN output branches.
        - Targets with -1 are kept for AUC computation; only remapped for loss.
    """
    model.eval()
    losses = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            (images, _), targets = batch
            images = images.to(device)
            targets = targets.to(device)

            # Remap uncertain to negative for loss computation only
            train_targets = _remap_uncertain_to_negative(targets)

            # ADD-GCN produces two outputs; ensemble by averaging
            out1, out2 = model(images)             # (B, C), (B, C)
            logits = (out1 + out2) / 2.0           # (B, C)
            loss = criterion(logits, train_targets)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)          # (B, C)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Stack predictions and targets across all batches
    probs_np = np.concatenate(all_probs, axis=0)      # (N, C)
    targets_np = np.concatenate(all_targets, axis=0)  # (N, C)

    map_score, per_class_ap = compute_mAP(probs_np, targets_np)
    mean_auc, per_class_auc = compute_mean_AUC(probs_np, targets_np)
    unc_auc, per_class_unc = compute_AUC_uncertain(probs_np, targets_np)

    return {
        "map": round(map_score, 4) if not np.isnan(map_score) else None,
        "mean_auc": round(mean_auc, 4) if not np.isnan(mean_auc) else None,
        "unc_auc": round(unc_auc, 4) if not np.isnan(unc_auc) else None,
        "per_class_auc": per_class_auc,
        "per_class_ap": per_class_ap,
        "per_class_unc_auc": per_class_unc,
        "loss": float(np.mean(losses)) if losses else None,
    }

# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADD-GCN training and evaluation on CheXpert."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation only (no training)"
    )
    parser.add_argument("--resume", default="", help="Checkpoint path to resume from")
    parser.add_argument(
        "--checkpoint", default="", help="Alias for --resume"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Use only N images for a quick smoke-test"
    )
    parser.add_argument(
        "--gpu-ids", default="", help='Comma-separated GPU IDs, e.g. "0,1"'
    )
    parser.add_argument(
        "--no-dp-fallback", action="store_true",
        help="Disable single-GPU fallback when DataParallel fails"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch_size from config"
    )
    parser.add_argument(
        "--force-single-gpu", action="store_true",
        help="Disable DataParallel and train on a single GPU"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = _parse_gpu_ids(cfg, args.gpu_ids)
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Disable cuDNN to avoid misaligned address errors on Kaggle / some multi-GPU setups
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("(cuDNN disabled to avoid misaligned address errors)")

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
        print(f"Batch size overridden to {args.batch_size}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Training uses the configured uncertainty policy; validation always keeps -1.
    train_dataset = CheXpert(
        root=cfg["data"]["root"],
        csv_file=cfg["data"]["train_csv"],
        inp_name=cfg["data"]["word_vec"],
        uncertain=cfg["data"].get("uncertain", "keep"),
        split="train",
        transform=get_transform("train", size=cfg["data"]["img_size"]),
    )
    val_dataset = CheXpert(
        root=cfg["data"]["root"],
        csv_file=cfg["data"]["val_csv"],
        inp_name=cfg["data"]["word_vec"],
        uncertain="keep",
        split="val",
        transform=get_transform("val", size=cfg["data"]["img_size"]),
    )

    # Optional second validation split with uncertain labels
    val_uncertain_dataset = None
    if cfg["data"].get("val_uncertain_csv"):
        val_uncertain_dataset = CheXpert(
            root=cfg["data"]["root"],
            csv_file=cfg["data"]["val_uncertain_csv"],
            inp_name=cfg["data"]["word_vec"],
            uncertain="keep",
            split="val",
            transform=get_transform("val", size=cfg["data"]["img_size"]),
        )

    if args.subset:
        n_val = max(50, args.subset // 9)
        train_dataset.df = train_dataset.df.head(args.subset).reset_index(drop=True)
        val_dataset.df = val_dataset.df.head(n_val).reset_index(drop=True)
        if val_uncertain_dataset is not None:
            val_uncertain_dataset.df = (
                val_uncertain_dataset.df.head(n_val).reset_index(drop=True)
            )
        print(
            f"Subset mode: {len(train_dataset)} train / {len(val_dataset)} val / "
            f"{len(val_uncertain_dataset) if val_uncertain_dataset else 0} val_unc"
        )

    train_loader = build_loader(
        train_dataset, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=True
    )
    val_loader = build_loader(
        val_dataset, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=False
    )
    val_uncertain_loader = (
        build_loader(
            val_uncertain_dataset,
            cfg["train"]["batch_size"],
            cfg["train"]["workers"],
            shuffle=False,
        )
        if val_uncertain_dataset is not None
        else None
    )

    # Model 
    model = addgcn_resnet101(
        num_classes=NUM_CLASSES,
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    if device.type == "cuda":
        if args.force_single_gpu:
            print("Forced single GPU mode (--force-single-gpu)")
        else:
            valid_gpu_ids = [i for i in gpu_ids if 0 <= i < available_gpus]
            if len(valid_gpu_ids) >= 2:
                model = torch.nn.DataParallel(model, device_ids=valid_gpu_ids)
                print(f"Using DataParallel on GPUs: {valid_gpu_ids}")
            else:
                print(f"Using single GPU: {torch.cuda.current_device()}")
    else:
        print("CUDA not available. Running on CPU.")

    # Loss and Optimizer
    criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    model_for_optim = (
        model.module if isinstance(model, torch.nn.DataParallel) else model
    )
    optimizer = torch.optim.SGD(
        model_for_optim.get_config_optim(
            cfg["train"]["lr"], cfg["train"]["lrp"]
        ),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # Checkpoint Resume 
    save_dir = cfg["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_search_dirs = build_checkpoint_search_dirs(cfg, save_dir)

    start_epoch = 0
    best_score = -1.0

    resume_path = (
        args.checkpoint or args.resume
        or find_latest_checkpoint_multi(checkpoint_search_dirs)
    )
    if resume_path and os.path.isfile(resume_path):
        start_epoch, best_score = load_checkpoint(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            evaluate_only=args.evaluate,
        )
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
    elif not (args.checkpoint or args.resume):
        print(f"No checkpoint found in: {checkpoint_search_dirs}")
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint or args.resume}"
        )

    # Evaluation
    if args.evaluate:
        print("\n=== Validation ===")
        results = evaluate_addgcn(model, val_loader, criterion, device)
        print_metrics(results)

        if val_uncertain_loader is not None:
            print("\n=== Validation (Uncertain split) ===")
            results_unc = evaluate_addgcn(
                model, val_uncertain_loader, criterion, device
            )
            print_metrics(results_unc)
        return

    # Training Loop 
    lr_decay_epochs = set(cfg["train"].get("epoch_step", []))

    for epoch in range(start_epoch, cfg["train"]["epochs"]):

        # Step-based learning rate decay at configured milestone epochs
        if epoch in lr_decay_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.1

        model.train()
        epoch_losses = []

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{cfg["train"]["epochs"]}',
        )
        for (images, _), targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Remap uncertain to negative before loss computation
            train_targets = _remap_uncertain_to_negative(targets)

            try:
                out1, out2 = model(images)
                # Ensure layout is contiguous after DataParallel scatter/gather
                out1 = out1.contiguous()
                out2 = out2.contiguous()
                logits = (out1 + out2) / 2.0  # (B, C)
                loss = criterion(logits, train_targets)

            except torch.AcceleratorError as exc:
                if isinstance(model, torch.nn.DataParallel) and not args.no_dp_fallback:
                    # Fall back to single GPU if DataParallel causes a CUDA error
                    print(
                        "\n" + "=" * 80 + "\n"
                        "DataParallel AcceleratorError: falling back to single GPU.\n"
                        f"Error: {exc}\n" + "=" * 80 + "\n"
                    )
                    torch.cuda.empty_cache()
                    model = model.module.to(device)
                    model.train()
                    out1, out2 = model(images)
                    out1 = out1.contiguous()
                    out2 = out2.contiguous()
                    logits = (out1 + out2) / 2.0
                    loss = criterion(logits, train_targets)
                else:
                    raise

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg["train"].get("max_clip_grad_norm", 10.0),
            )
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")

        print(f"Train loss: {np.mean(epoch_losses):.4f}")

        # Validation
        print("\n=== Validation ===")
        results = evaluate_addgcn(model, val_loader, criterion, device)
        print_metrics(results)

        score = results["map"] if results["map"] is not None else -1.0
        running_best = max(best_score, score)

        # Save per-epoch checkpoint
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth.tar")
        save_checkpoint(ckpt_path, model, optimizer, epoch, running_best)

        # Save best model checkpoint if validation score improved
        if score > best_score:
            best_score = score
            save_checkpoint(
                os.path.join(save_dir, "model_best.pth.tar"),
                model, optimizer, epoch, best_score,
            )
            print(f"New best mAP: {best_score:.4f}")

        if val_uncertain_loader is not None:
            print("\n=== Validation (Uncertain split) ===")
            results_unc = evaluate_addgcn(
                model, val_uncertain_loader, criterion, device
            )
            print_metrics(results_unc)

    # Final Evaluation with Best Model
    print("\n" + "=" * 80)
    print("FINAL RESULTS — Best Model Evaluation")
    print("=" * 80)

    best_model_path = os.path.join(save_dir, "model_best.pth.tar")
    if not os.path.isfile(best_model_path):
        print(f"Best model not found at: {best_model_path}")
        print(f"Final Best mAP = {best_score:.4f}")
        return

    load_checkpoint(
        resume_path=best_model_path,
        model=model,
        optimizer=optimizer,
        device=device,
        evaluate_only=True,
    )
    print(f"Loaded best model from {best_model_path}")

    print(f"\n{'─' * 80}\nOfficial Validation Set:\n{'─' * 80}")
    final_results = evaluate_addgcn(model, val_loader, criterion, device)
    print_metrics(final_results)

    if val_uncertain_loader is not None:
        print(f"\n{'─' * 80}\nUncertain Validation Set:\n{'─' * 80}")
        final_results_unc = evaluate_addgcn(
            model, val_uncertain_loader, criterion, device
        )
        print_metrics(final_results_unc)

    def _fmt(key: str) -> str:
        val = final_results.get(key)
        return f"{val:.4f}" if val is not None else "N/A"

    print(f"\n{'=' * 80}\nFINAL SUMMARY\n{'=' * 80}")
    print(f"Best mAP:            {_fmt('map')}")
    print(f"Best Mean AUC:       {_fmt('mean_auc')}")
    print(f"Best Uncertain AUC:  {_fmt('unc_auc')}")
    print(f"Validation Loss:     {_fmt('loss')}")
    print(f"Best model saved to: {best_model_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()