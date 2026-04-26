"""
File: train_chexnet.py
Description:
    Training script for the CheXNet DenseNet-121 baseline on CheXpert.
    Reuses the shared engine and evaluation utilities from the GCN codebase.
Main Components:
    - CheXNetEngine: Extends MultiLabelMAPEngine with CheXNet-specific forward
      pass (no word vectors or GCN) and per-epoch AUC / mAP computation.
    - main: Dataset construction, model instantiation, engine setup, and training.
Inputs:
    CheXpert images (B, 3, H, W); no word vectors or adjacency files needed.
Outputs:
    Per-epoch checkpoints and model_best.pth.tar in the configured save_dir.
Notes:
    - Model outputs logits; BCEWithLogitsLoss is used as the training criterion.
    - Uncertain labels (-1) are remapped to 0 for BCE; kept as -1 for AUC eval.
    - ReduceLROnPlateau scheduler steps on validation loss (not epoch count).
    - Unlike GCNMultiLabelMAPEngine, this engine calls model(images) with
      a single argument — no label embedding input.
Usage:
    python train_chexnet.py --config configs/chexnet.yaml
    python train_chexnet.py --config configs/chexnet.yaml --subset 1000  # smoke test
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

# Resolve repo root (parent of src/) so that src.* imports work from any CWD
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _REPO_ROOT)

from src.data.chexpert import CheXpert
from src.engine import MultiLabelMAPEngine
from src.evaluate import (
    compute_AUC_uncertain,
    compute_mAP,
    compute_mean_AUC,
    evaluate,
    print_metrics,
)
from src.models.chexnet import NUM_CLASSES, build_chexnet


# Engine

class CheXNetEngine(MultiLabelMAPEngine):
    """
    Training engine for the CheXNet DenseNet-121 baseline.

    Differences from GCNMultiLabelMAPEngine:
      - on_start_batch: unpacks (images, paths) from the dataset's __getitem__
        tuple, but does NOT extract label embeddings (CheXNet has no GCN branch).
      - on_forward: calls model(images) with a single argument.
      - on_end_batch: accumulates sigmoid probabilities and original targets
        (including -1) for post-epoch AUC computation.
      - on_end_epoch: computes mAP, mean AUC, and uncertain-aware AUC;
        steps ReduceLROnPlateau on the official validation loss.
    """

    def on_start_epoch(
        self,
        training: bool,
        model: nn.Module,
        criterion,
        data_loader,
        optimizer=None,
        display: bool = True,
    ) -> None:
        MultiLabelMAPEngine.on_start_epoch(
            self, training, model, criterion, data_loader, optimizer
        )
        # Initialize per-epoch accumulators for validation AUC computation
        if not training:
            self.state["_val_probs"] = []
            self.state["_val_targets"] = []

    def on_start_batch(
        self,
        training: bool,
        model: nn.Module,
        criterion,
        data_loader,
        optimizer=None,
        display: bool = True,
    ) -> None:
        # Preserve original targets (with -1) for AUC evaluation
        self.state["target_gt"] = self.state["target"].clone()

        # Remap uncertain labels (-1) to 0 for BCEWithLogitsLoss
        # (Giải thích: BCEWithLogitsLoss chỉ chấp nhận target ∈ {0, 1})
        targets = self.state["target"].clone().float()
        targets[targets < 0] = 0.0
        self.state["target"] = targets

        # Unpack the (images, paths) tuple returned by CheXpertDataset.__getitem__
        batch_input = self.state["input"]
        self.state["feature"] = batch_input[0]  # images (B, 3, H, W)
        self.state["name"] = batch_input[1]      # image paths (not used for training)

    def on_end_batch(
        self,
        training: bool,
        model: nn.Module,
        criterion,
        data_loader,
        optimizer=None,
        display: bool = True,
    ) -> None:
        MultiLabelMAPEngine.on_end_batch(
            self, training, model, criterion, data_loader, display
        )
        # Accumulate predictions and original targets for validation AUC
        if not training:
            probs = torch.sigmoid(self.state["output"]).detach().cpu().numpy()
            targets_gt = self.state["target_gt"].detach().cpu().numpy()
            self.state["_val_probs"].append(probs)
            self.state["_val_targets"].append(targets_gt)

    def on_end_epoch(
        self,
        training: bool,
        model: nn.Module,
        criterion,
        data_loader,
        optimizer=None,
        display: bool = True,
    ) -> float:
        """
        Compute and log epoch-level metrics.

        For training: logs average loss and mAP from ap_meter.
        For validation: computes mAP, mean AUC, and uncertain-aware AUC
        from accumulated predictions; steps ReduceLROnPlateau scheduler.

        Returns:
            float: Epoch score used for best-model tracking
                (mAP, or mean AUC as fallback).
        """
        loss = self.state["meter_loss"].avg

        if training:
            map_val = 100.0 * self.state["ap_meter"].value().mean()
            if display:
                print(
                    f'Epoch: [{self.state["epoch"]}]\t'
                    f"Loss {loss:.4f}\tmAP {map_val:.3f}"
                )
            self.logger.info(
                f'Train Epoch {self.state["epoch"]} '
                f"- Loss: {loss:.4f}, mAP: {map_val:.3f}"
            )
            return float(map_val)

        # ── Validation Metrics ────────────────────────────────────────────────
        val_probs = self.state.get("_val_probs", [])
        val_targets = self.state.get("_val_targets", [])

        if not val_probs:
            if display:
                print(f"Val:\tLoss {loss:.4f}  (no predictions)")
            self.logger.info(
                f'Validation ({self.state.get("val_split", "official")}) '
                f"- Loss: {loss:.4f}, no predictions"
            )
            return 0.0

        # Stack accumulated predictions across batches
        probs_np = np.concatenate(val_probs, axis=0)      # (N, C)
        targets_np = np.concatenate(val_targets, axis=0)  # (N, C)

        map_score, per_class_ap = compute_mAP(probs_np, targets_np)
        mean_auc, per_class_auc = compute_mean_AUC(probs_np, targets_np)
        unc_auc, per_class_unc = compute_AUC_uncertain(probs_np, targets_np)

        results = {
            "map": round(map_score, 4) if not np.isnan(map_score) else None,
            "mean_auc": round(mean_auc, 4) if not np.isnan(mean_auc) else None,
            "unc_auc": round(unc_auc, 4) if not np.isnan(unc_auc) else None,
            "per_class_auc": per_class_auc,
            "per_class_ap": per_class_ap,
            "per_class_unc_auc": per_class_unc,
        }

        if display:
            print(f"\nVal:\tLoss {loss:.4f}")
            print_metrics(results)

        # Step ReduceLROnPlateau on the official validation split only
        # (Giải thích: tránh step scheduler 2 lần khi có val_uncertain_split)
        scheduler = self.state.get("scheduler")
        val_split = self.state.get("val_split", "official")
        if scheduler is not None and val_split == "official":
            scheduler.step(loss)
            if display:
                lrs = [pg["lr"] for pg in scheduler.optimizer.param_groups]
                print(f"  ReduceLROnPlateau: val_loss={loss:.4f}  lr={lrs}")

        self.logger.info(
            "Validation (%s) - Loss: %.4f, mAP: %s, Mean AUC: %s, Unc AUC: %s",
            val_split,
            loss,
            results["map"],
            results["mean_auc"],
            results["unc_auc"],
        )

        # Use mAP as the primary score; fall back to mean AUC if NaN
        score = (
            map_score if not np.isnan(map_score)
            else mean_auc if not np.isnan(mean_auc)
            else 0.0
        )
        return float(score)

    def on_forward(
        self,
        training: bool,
        model: nn.Module,
        criterion,
        data_loader,
        optimizer=None,
        display: bool = True,
    ) -> None:
        """
        CheXNet forward pass: single-argument model call (no label embeddings).

        Args:
            training (bool): Whether the engine is in training mode.
            model (nn.Module): The CheXNet model.
            criterion: Loss function (BCEWithLogitsLoss).
            data_loader: Active data loader (unused here).
            optimizer: Optimizer (used for gradient step during training).
            display (bool): Whether to print progress.

        Notes:
            - Differs from GCNEngine which calls model(images, label_embeddings).
            - Gradient clipping is applied before optimizer step.
        """
        images = self.state["feature"].float()
        targets = self.state["target"].float()

        if self.state["use_gpu"]:
            images = images.cuda()
            targets = targets.cuda()

        logits = model(images)                          # (B, num_classes)
        self.state["output"] = logits
        self.state["loss"] = criterion(logits, targets)

        if training:
            optimizer.zero_grad()
            self.state["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()


# Utilities

def _load_config(path: str) -> dict:
    """Load a YAML configuration file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Construct an Adam optimizer for all model parameters.

    Args:
        model (nn.Module): The model whose parameters to optimize.
        cfg (dict): Full configuration dictionary.

    Returns:
        torch.optim.Adam: Configured optimizer.
    """
    train_cfg = cfg["train"]
    return torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """
    Construct a ReduceLROnPlateau scheduler if configured.

    The scheduler steps on validation loss after each epoch. Only the
    official validation split triggers a step to avoid double-stepping
    when an uncertain split is also evaluated.

    Args:
        optimizer: The optimizer to attach the scheduler to.
        cfg (dict): Full configuration dictionary.

    Returns:
        ReduceLROnPlateau or None if scheduler type is not "plateau".
    """
    train_cfg = cfg["train"]
    if train_cfg.get("scheduler", "plateau") != "plateau":
        return None

    kwargs = dict(
        mode="min",
        factor=train_cfg["lr_factor"],
        patience=train_cfg["lr_patience"],
        min_lr=train_cfg["lr_min"],
    )
    try:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **kwargs, verbose=True
        )
    except TypeError:
        # Older PyTorch versions do not accept verbose= in ReduceLROnPlateau
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)


def _find_latest_checkpoint(save_dir: str) -> str | None:
    """
    Find the most recent epoch checkpoint in save_dir for auto-resume.

    Args:
        save_dir (str): Directory containing checkpoint files.

    Returns:
        str or None: Path to the latest checkpoint, or None if none found.
    """
    if not os.path.exists(save_dir):
        return None
    pattern = os.path.join(save_dir, "checkpoint_epoch_*.pth.tar")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(
        key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
    )
    return files[-1]


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CheXNet DenseNet-121 baseline on CheXpert."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config (e.g. configs/chexnet.yaml)"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Use only N images — quick smoke-test"
    )
    parser.add_argument(
        "--data_root", default=None,
        help="Override data.root from config"
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data_root = args.data_root or cfg["data"]["root"]
    print(f"Config: {cfg['name']}")

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    # ── Datasets ──────────────────────────────────────────────────────────────
    # uncertain="keep" preserves -1 labels so the engine can remap them
    # to 0 for BCE loss while keeping them intact for AUC evaluation.
    uncertain_policy = cfg["data"].get("uncertain", "keep")

    train_dataset = CheXpert(
        root=data_root,
        csv_file=cfg["data"]["train_csv"],
        inp_name=cfg["data"].get("word_vec"),  # None for CheXNet (no GCN)
        uncertain=uncertain_policy,
    )
    val_dataset = CheXpert(
        root=data_root,
        csv_file=cfg["data"]["val_csv"],
        inp_name=cfg["data"].get("word_vec"),
        uncertain=uncertain_policy,
    )
    # Second validation split retaining uncertain labels for unc_auc metrics
    val_uncertain_dataset = CheXpert(
        root=data_root,
        csv_file=cfg["data"]["val_uncertain_csv"],
        inp_name=cfg["data"].get("word_vec"),
        uncertain="keep",
    )

    if args.subset:
        n_val = max(50, args.subset // 9)
        train_dataset.df = train_dataset.df.head(args.subset).reset_index(drop=True)
        val_dataset.df = val_dataset.df.head(n_val).reset_index(drop=True)
        print(f"Subset mode: {len(train_dataset)} train / {len(val_dataset)} val")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = build_chexnet(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        ckpt_path=model_cfg.get("ckpt_path"),  # accepted but intentionally ignored
    )

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()  # expects logits; sigmoid applied internally
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    # ── Engine Setup ──────────────────────────────────────────────────────────
    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["log_dir"], exist_ok=True)

    resume_path = _find_latest_checkpoint(cfg["output"]["save_dir"])
    print(f"Auto-resume: {resume_path}" if resume_path else "Training from scratch")

    engine_state = {
        "batch_size": cfg["train"]["batch_size"],
        "image_size": cfg["data"]["img_size"],
        "max_epochs": cfg["train"]["epochs"],
        "workers": cfg["train"]["workers"],
        "epoch_step": cfg["train"].get("epoch_step", []),
        "save_model_path": cfg["output"]["save_dir"],
        "log_dir": cfg["output"]["log_dir"],
        "print_freq": 100,
        "use_pb": True,
        "difficult_examples": False,
        "resume": resume_path,
        "loss_type": cfg["loss"]["type"],           # "bce"
        "scheduler": scheduler,
        # Skip engine's built-in LR step when ReduceLROnPlateau is active
        "skip_adjust_learning_rate": scheduler is not None,
    }

    engine = CheXNetEngine(engine_state)
    best_score = engine.learning(
        model,
        criterion,
        train_dataset,
        val_dataset,
        val_uncertain_dataset=val_uncertain_dataset,
        optimizer=optimizer,
    )
    print(f"\n[{cfg['name']}] Training complete. Best val score = {best_score:.4f}")

    # ── Final Evaluation ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_model = model.module if hasattr(model, "module") else model

    best_path = os.path.join(cfg["output"]["save_dir"], "model_best.pth.tar")
    print(f"\nLoading best model from: {best_path}")
    checkpoint = torch.load(best_path, map_location=device)
    raw_model.load_state_dict(checkpoint["state_dict"])

    batch_size = engine_state["batch_size"]
    num_workers = engine_state["workers"]
    pin_memory = device == "cuda"

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print("\n=== Final validation metrics (official val split) ===")
    final_results = evaluate(raw_model, val_loader, device=device)
    print_metrics(final_results)

    val_uncertain_loader = torch.utils.data.DataLoader(
        val_uncertain_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print("\n=== Final validation metrics (uncertain split) ===")
    final_results_unc = evaluate(raw_model, val_uncertain_loader, device=device)
    print_metrics(final_results_unc)


if __name__ == "__main__":
    main()