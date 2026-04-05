"""
grid_search_c5.py
=================
Grid search over UA-ASL hyperparameters for C5 (Swin-T + BioMedCLIP + UA-ASL).

Grid:
    γ+  = 0  (fixed — paper recommendation)
    γ-  ∈ {2, 4, 6}
    λu  ∈ {0.3, 0.5, 0.7}
    α   ∈ {0.3, 0.5}
    → 18 combinations

Each run:
    - Train 5 epochs on train_small_v3.csv
    - Evaluate on val_uncertain split
    - Track: mAP, mean_AUC, unc_AUC

Outputs (all written to log_dir):
    grid_results.json           — full results table
    best_params.json            — best hyperparams (by unc_AUC)
    grid_heatmap_<metric>.png   — γ- (y) × λu (x), one subplot per α
    grid_scatter.png            — scatter: mAP vs unc_AUC, colour = mean_AUC
    grid_parallel.png           — parallel coordinates plot

Usage (terminal):
    python grid_search_c5.py --config configs/c5_tulip.yaml

Usage (notebook):
    from src.grid_search_c5 import run_grid_search
    results, best = run_grid_search("configs/c5_tulip.yaml")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.chexpert import CheXpert, NUM_CLASSES
from src.evaluate import compute_AUC_uncertain, compute_mAP, compute_mean_AUC
from src.util import Warp
from src.models.gcn import gcn_swin_t
from src.loss.ua_asl import UncertaintyAwareASL


# ──────────────────────────────────────────────────────────────────────────────
# Config / seed helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_val_transform(image_size: int = 224) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        Warp(image_size),
        transforms.ToTensor(),
        normalize,
    ])


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    from src.util import MultiScaleCrop
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        MultiScaleCrop(image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Single-run trainer
# ──────────────────────────────────────────────────────────────────────────────

def _run_one(
    cfg: dict,
    gamma_neg: float,
    lambda_unc: float,
    alpha: float,
    train_ds: CheXpert,
    val_ds: CheXpert,
    device: str,
    n_epochs: int,
    run_id: int,
    total_runs: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one grid-search run and return metrics dict."""

    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"[{run_id:02d}/{total_runs}]  γ+=0 (fixed)  γ-={gamma_neg}  λu={lambda_unc}  α={alpha}")
    print(f"{'─'*60}")

    # ── dry_run: skip training, return dummy metrics ──────────────────────────
    if dry_run:
        print("  [dry_run] skipping training")
        return {
            "run_id":      run_id,
            "gamma_pos":   0,
            "gamma_neg":   gamma_neg,
            "lambda_unc":  lambda_unc,
            "alpha":       alpha,
            "mAP":         0.0,
            "mean_auc":    0.0,
            "unc_auc":     0.0,
            "elapsed_min": 0.0,
        }

    # ── Model ─────────────────────────────────────────────────────────────────
    model = gcn_swin_t(
        num_classes=NUM_CLASSES,
        t=cfg["model"]["t"],
        pretrained=cfg["model"]["pretrained"],
        adj_file=cfg["data"]["adj"],
        in_channel=cfg["model"]["gcn_in"],
        inp_file=cfg["data"]["word_vec"],
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = UncertaintyAwareASL(
        gamma_pos=0,            # fixed per paper
        gamma_neg=gamma_neg,
        margin=cfg["loss"].get("margin", 0.05),
        lambda_unc=lambda_unc,
        alpha=alpha,
        reduction="mean",
        disable_torch_grad_focal_loss=True,
    )

    # ── Optimiser — SGD ───────────────────────────────────────────────────────
    lr  = cfg["train"]["lr"]
    lrp = cfg["train"]["lrp"]
    optimizer = torch.optim.SGD(
        model.get_config_optim(lr, lrp),
        lr=lr,
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # ── Scheduler — MultiStepLR at 60% and 85% of n_epochs ───────────────────
    step1 = max(1, round(n_epochs * 0.60))
    step2 = max(step1 + 1, round(n_epochs * 0.85))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[step1, step2], gamma=lrp,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    img_size = cfg["data"].get("img_size", 224)
    bs       = cfg["train"]["batch_size"]
    workers  = cfg["train"]["workers"]

    train_ds.transform = get_train_transform(img_size)
    val_ds.transform   = get_val_transform(img_size)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=(device == "cuda"),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        model.train()
        running_loss, n_batch = 0.0, 0

        for batch in train_loader:
            (imgs, _paths), targets = batch[0], batch[1]
            imgs    = imgs.to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            running_loss += loss.item()
            n_batch      += 1

        avg_loss   = running_loss / max(n_batch, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch+1:02d}/{n_epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}")
        scheduler.step()

    # ── Evaluation on val_uncertain ───────────────────────────────────────────
    model.eval()
    all_scores, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            (imgs, _paths), targets = batch[0], batch[1]
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
            all_targets.append(targets.numpy())

    scores  = np.concatenate(all_scores,  axis=0)   # (N, C)
    targets = np.concatenate(all_targets, axis=0)   # (N, C) — may have -1

    mAP,      _ = compute_mAP(scores, targets)
    mean_auc, _ = compute_mean_AUC(scores, targets)
    unc_auc,  _ = compute_AUC_uncertain(scores, targets)

    elapsed = (time.time() - t0) / 60.0

    result = {
        "run_id":      run_id,
        "gamma_pos":   0,
        "gamma_neg":   gamma_neg,
        "lambda_unc":  lambda_unc,
        "alpha":       alpha,
        "mAP":         round(float(mAP),      4) if not np.isnan(mAP)      else 0.0,
        "mean_auc":    round(float(mean_auc), 4) if not np.isnan(mean_auc) else 0.0,
        "unc_auc":     round(float(unc_auc),  4) if not np.isnan(unc_auc)  else 0.0,
        "elapsed_min": round(elapsed, 2),
    }
    print(
        f"  → mAP={result['mAP']:.4f}  "
        f"AUC={result['mean_auc']:.4f}  "
        f"unc_AUC={result['unc_auc']:.4f}  "
        f"({elapsed:.1f} min)"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _make_visualisations(results: list[dict], out_dir: str):
    """
    Create 3 plots and save to out_dir:
        grid_heatmap_<metric>.png  — γ- (y) × λu (x), one subplot per α
        grid_scatter.png           — scatter mAP vs unc_AUC, colour = mean_AUC
        grid_parallel.png          — parallel coordinates
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except ImportError:
        print("[visualise] matplotlib not available — skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)

    gn_vals  = sorted(set(r["gamma_neg"]  for r in results))
    lu_vals  = sorted(set(r["lambda_unc"] for r in results))
    al_vals  = sorted(set(r["alpha"]      for r in results))
    metrics  = ["unc_auc", "mAP", "mean_auc"]
    m_labels = {"unc_auc": "unc_AUC", "mAP": "mAP", "mean_auc": "mean AUC"}

    # ── 1. Heatmaps: γ- (y) × λu (x), subplot per α ─────────────────────────
    for metric in metrics:
        fig, axes = plt.subplots(
            1, len(al_vals),
            figsize=(5 * len(al_vals), 4.2),
            constrained_layout=True,
        )
        if len(al_vals) == 1:
            axes = [axes]

        all_vals = [r[metric] for r in results]
        vmin, vmax = min(all_vals), max(all_vals)
        if vmin == vmax:
            vmin, vmax = vmin - 0.001, vmax + 0.001

        cmap = "Blues" if metric == "unc_auc" else "Greens" if metric == "mAP" else "Oranges"

        for ax, al in zip(axes, al_vals):
            grid_data = np.zeros((len(gn_vals), len(lu_vals)))
            for r in results:
                if r["alpha"] == al:
                    ri = gn_vals.index(r["gamma_neg"])
                    ci = lu_vals.index(r["lambda_unc"])
                    grid_data[ri, ci] = r[metric]

            im = ax.imshow(grid_data, vmin=vmin, vmax=vmax,
                           cmap=cmap, aspect="auto", origin="lower")
            ax.set_xticks(range(len(lu_vals)))
            ax.set_xticklabels([f"λu={v}" for v in lu_vals], fontsize=10)
            ax.set_yticks(range(len(gn_vals)))
            ax.set_yticklabels([f"γ-={v}" for v in gn_vals], fontsize=10)
            ax.set_title(f"α = {al}", fontsize=11, fontweight="bold")

            for ri in range(len(gn_vals)):
                for ci in range(len(lu_vals)):
                    v = grid_data[ri, ci]
                    text_color = "white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55 else "#222"
                    ax.text(ci, ri, f"{v:.4f}", ha="center", va="center",
                            fontsize=9, color=text_color)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"{m_labels[metric]}  —  γ- (y) × λu (x) × α (subplot)  |  γ+=0 fixed",
            fontsize=13, fontweight="bold",
        )
        path = os.path.join(out_dir, f"grid_heatmap_{metric}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    # ── 2. Scatter: mAP vs unc_AUC, colour = mean_AUC ────────────────────────
    best = max(results, key=lambda r: r["unc_auc"])

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r["mAP"]      for r in results]
    ys = [r["unc_auc"]  for r in results]
    cs = [r["mean_auc"] for r in results]

    sc   = ax.scatter(xs, ys, c=cs, cmap="plasma", s=80, alpha=0.85,
                      edgecolors="#555", linewidths=0.4)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("mean AUC", fontsize=10)

    ax.scatter(best["mAP"], best["unc_auc"], marker="*",
               s=260, color="#E24B4A", zorder=5, label="best unc_AUC")
    ax.annotate(
        f"γ-={best['gamma_neg']}, λu={best['lambda_unc']}, α={best['alpha']}",
        xy=(best["mAP"], best["unc_auc"]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=8.5, color="#E24B4A",
    )
    ax.set_xlabel("mAP", fontsize=11)
    ax.set_ylabel("unc_AUC", fontsize=11)
    ax.set_title(
        "mAP vs unc_AUC across 18 runs\n(colour = mean AUC, γ+=0 fixed)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    path = os.path.join(out_dir, "grid_scatter.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── 3. Parallel coordinates ───────────────────────────────────────────────
    param_cols  = ["gamma_neg", "lambda_unc", "alpha"]
    metric_cols = ["mAP", "mean_auc", "unc_auc"]
    col_labels  = ["γ-", "λu", "α", "mAP", "mean AUC", "unc_AUC"]
    all_cols    = param_cols + metric_cols

    fig, axes = plt.subplots(1, len(all_cols) - 1, figsize=(13, 5), sharey=False)

    col_vals = {}
    for col in all_cols:
        vals = [r[col] for r in results]
        lo, hi = min(vals), max(vals)
        col_vals[col] = (vals, lo, hi)

    def norm(val, lo, hi):
        return (val - lo) / (hi - lo + 1e-9)

    unc_vals   = [r["unc_auc"] for r in results]
    cmap_fn    = cm.get_cmap("cool")
    cmin, cmax = min(unc_vals), max(unc_vals)

    for r in results:
        ys_line = [norm(r[c], *col_vals[c][1:]) for c in all_cols]
        color   = cmap_fn((r["unc_auc"] - cmin) / (cmax - cmin + 1e-9))
        is_best = r["run_id"] == best["run_id"]
        lw      = 2.2 if is_best else 0.7
        al      = 0.95 if is_best else 0.35

        for i in range(len(axes)):
            axes[i].plot([0, 1], [ys_line[i], ys_line[i + 1]],
                         color=color, lw=lw, alpha=al)

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([col_labels[i], col_labels[i + 1]], fontsize=10)
        ax.yaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)

        col = all_cols[i]
        vals_list, lo, hi = col_vals[col]
        for v in sorted(set(vals_list)):
            y = norm(v, lo, hi)
            ax.plot([0, 0.05], [y, y], color="#888", lw=0.8)
            ax.text(-0.08, y, str(round(v, 3)), ha="right", va="center", fontsize=8)

    sm = cm.ScalarMappable(cmap="cool",
                           norm=mcolors.Normalize(vmin=cmin, vmax=cmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("unc_AUC", fontsize=10)

    fig.suptitle(
        "Parallel coordinates — each line = one run (colour = unc_AUC)  |  γ+=0 fixed",
        fontsize=12, fontweight="bold",
    )
    path = os.path.join(out_dir, "grid_parallel.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Public API  (called from notebook)
# ──────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    config_path: str,
    data_root: str | None = None,
    n_epochs: int | None = None,
    dry_run: bool = False,
) -> tuple[list[dict], dict]:
    """
    Main entry point — usable from both CLI and notebooks.

    Args:
        config_path : path to YAML config (e.g. 'configs/c5_tulip.yaml')
        data_root   : override cfg['data']['root'] if provided
        n_epochs    : override cfg['grid_search']['epochs_per_run']
        dry_run     : skip actual training, write fake results (pipeline test)

    Returns:
        results : list of dicts — one per run
        best    : dict with best hyperparams by unc_AUC
    """
    cfg      = load_cfg(config_path)
    set_seed(cfg["seed"])

    root     = data_root or cfg["data"]["root"]
    gs_cfg   = cfg.get("grid_search", {})
    n_epochs = n_epochs or gs_cfg.get("epochs_per_run", 5)
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    # Grid: γ+ fixed = 0, search γ-, λu, α
    gn_list = gs_cfg.get("gamma_neg",  [2, 4, 6])
    lu_list = gs_cfg.get("lambda_unc", [0.3, 0.5, 0.7])
    al_list = gs_cfg.get("alpha",      [0.3, 0.5])
    grid    = list(product(gn_list, lu_list, al_list))

    print(f"Device        : {device}")
    print(f"Grid size     : {len(grid)} runs")
    print(f"  γ+ = 0 (fixed)")
    print(f"  γ- ∈ {gn_list}")
    print(f"  λu ∈ {lu_list}")
    print(f"  α  ∈ {al_list}")
    print(f"Epochs / run  : {n_epochs}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = CheXpert(
        root=root,
        csv_file=cfg["data"]["train_csv"],
        inp_name=cfg["data"]["word_vec"],
        uncertain="keep",
    )
    val_ds = CheXpert(
        root=root,
        csv_file=cfg["data"]["val_uncertain_csv"],
        inp_name=cfg["data"]["word_vec"],
        uncertain="keep",
    )

    print(f"Train         : {len(train_ds):,} samples")
    print(f"Val uncertain : {len(val_ds):,} samples\n")

    # ── Resume support ────────────────────────────────────────────────────────
    log_dir      = cfg["output"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(log_dir, "grid_results.json")

    completed_ids = set()
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
        completed_ids = {r["run_id"] for r in results}
        print(f"Resuming — {len(completed_ids)} runs already done: {sorted(completed_ids)}")
    else:
        results: list[dict] = []

    # ── Run grid ──────────────────────────────────────────────────────────────
    for run_id, (gn, lu, al) in enumerate(grid, start=1):
        if run_id in completed_ids:
            print(f"[{run_id:02d}/{len(grid)}] SKIP (already done)")
            continue

        res = _run_one(
            cfg=cfg,
            gamma_neg=gn, lambda_unc=lu, alpha=al,
            train_ds=train_ds, val_ds=val_ds,
            device=device, n_epochs=n_epochs,
            run_id=run_id, total_runs=len(grid),
            dry_run=dry_run,
        )
        results.append(res)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [checkpoint] {len(results)} runs saved → {results_path}")

    # ── Best params ───────────────────────────────────────────────────────────
    best      = max(results, key=lambda r: r["unc_auc"])
    best_path = os.path.join(log_dir, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    print("\n" + "═" * 60)
    print("BEST  (by unc_AUC)")
    print("═" * 60)
    print(f"  γ+  = 0 (fixed)")
    print(f"  γ-  = {best['gamma_neg']}")
    print(f"  λu  = {best['lambda_unc']}")
    print(f"  α   = {best['alpha']}")
    print(f"  mAP      = {best['mAP']:.4f}")
    print(f"  mean_AUC = {best['mean_auc']:.4f}")
    print(f"  unc_AUC  = {best['unc_auc']:.4f}")
    print(f"\nBest params saved → {best_path}")

    print("\nTop-5 runs (by unc_AUC):")
    print(f"{'γ-':>4} {'λu':>5} {'α':>5}  {'mAP':>7} {'AUC':>7} {'unc_AUC':>9}")
    print("─" * 48)
    for r in sorted(results, key=lambda x: x["unc_auc"], reverse=True)[:5]:
        marker = " ←" if r["run_id"] == best["run_id"] else ""
        print(
            f"{r['gamma_neg']:>4} {r['lambda_unc']:>5} {r['alpha']:>5}  "
            f"{r['mAP']:>7.4f} {r['mean_auc']:>7.4f} {r['unc_auc']:>9.4f}{marker}"
        )

    print("\nGenerating plots...")
    _make_visualisations(results, log_dir)

    return results, best


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UA-ASL grid search for C5")
    parser.add_argument("--config",    required=True,
                        help="Path to YAML config (e.g. configs/c5_tulip.yaml)")
    parser.add_argument("--data_root", default=None,
                        help="Override cfg.data.root")
    parser.add_argument("--epochs",    type=int, default=None,
                        help="Override epochs_per_run (default: from config)")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Skip training — smoke-test pipeline only")
    args = parser.parse_args()

    run_grid_search(
        config_path=args.config,
        data_root=args.data_root,
        n_epochs=args.epochs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()