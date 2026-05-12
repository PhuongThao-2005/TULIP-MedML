"""
File: evaluate.py

Description:
    Evaluation utilities for multi-label CheXpert experiments.
    This module computes mAP, mean AUC, and uncertain AUC.

Main Components:
    - _unpack_batch: Parse dataloader batch into tensors.
    - compute_mAP: Compute mean AP over classes.
    - compute_mean_AUC: Compute mean ROC-AUC over classes.
    - compute_AUC_uncertain: Compute AUC on uncertain subset.
    - evaluate: Run full inference and aggregate metrics.
    - print_metrics: Print per-class and mean metrics.

Inputs:
    - logits from model forward pass
    - targets in {-1, 0, 1}

Outputs:
    - Metric dictionary with scalar metrics and per-class tables

Notes:
    - Uses `logits` for raw model outputs and `probs` after sigmoid.
    - Uses `targets` terminology for ground-truth labels.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.chexpert import CHEXPERT_CLASSES


def _unpack_batch(batch):
    """
    Unpack a dataloader batch into image tensor, label embeddings, and targets.

    Args:
        batch (tuple | list): Batch from dataloader.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            image_features, label_embeddings, targets

    Shape:
        image_features: (B, 3, H, W)
        label_embeddings: (C, D) or (B, C, D), optional
        targets: (B, C)

    Notes:
        - Accepts both plain tensor input and tuple/list input.
    """
    
    # Ensure batch has the expected structure: (inputs, targets)
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError('Expected batch format (input, target).')

    inputs, targets = batch
    image_features = None
    label_embeddings = None

    # Case 1: inputs is directly a tensor → treat as image tensor
    if torch.is_tensor(inputs):
        image_features = inputs

    # Case 2: inputs is a tuple/list → may contain additional info
    elif isinstance(inputs, (tuple, list)):
        if len(inputs) == 0:
            raise ValueError('Input tuple is empty.')

        # First element is always the image tensor
        image_features = inputs[0]

        # Try to extract label embeddings if present
        if len(inputs) >= 3 and torch.is_tensor(inputs[2]):
            label_embeddings = inputs[2]
        elif len(inputs) >= 2 and torch.is_tensor(inputs[1]):
            label_embeddings = inputs[1]

    else:
        raise ValueError(f'Unsupported input type: {type(inputs)}')

    # Ensure targets are tensors
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)

    return image_features, label_embeddings, targets


def compute_mAP(
    probs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, dict]:
    """
    Compute mean Average Precision across classes.

    Args:
        probs (np.ndarray): Sigmoid probabilities.
        targets (np.ndarray): Ground-truth labels in {-1, 0, 1}.

    Returns:
        tuple[float, dict]: (mean_ap, per_class_ap)

    Shape:
        probs: (N, C)
        targets: (N, C)

    Notes:
        - Ignores uncertain targets (`-1`) for AP computation.
        - Requires at least one positive target per class.
    """
    per_class = {}
    aps = []

    for class_idx, class_name in enumerate(CHEXPERT_CLASSES):
        targets_class = targets[:, class_idx]
        probs_class = probs[:, class_idx]

        # Ignore uncertain targets for standard AP.
        mask = targets_class != -1
        if mask.sum() == 0:
            continue

        # Convert to binary labels {0,1}
        targets_binary = (targets_class[mask] == 1).astype(int)

        # Skip if no positive samples exist
        if targets_binary.sum() == 0:
            continue

        try:
            # Compute Average Precision for this class
            ap = average_precision_score(targets_binary, probs_class[mask])
            per_class[class_name] = round(float(ap), 4)
            aps.append(ap)
        except ValueError:
            pass

    # Mean AP across valid classes
    mean_ap = float(np.mean(aps)) if aps else float('nan')
    return mean_ap, per_class


def compute_mean_AUC(
    probs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, dict]:
    """
    Compute mean ROC-AUC across valid classes.

    Args:
        probs (np.ndarray): Sigmoid probabilities.
        targets (np.ndarray): Ground-truth labels in {-1, 0, 1}.

    Returns:
        tuple[float, dict]: (mean_auc, per_class_auc)

    Shape:
        probs: (N, C)
        targets: (N, C)

    Notes:
        - Ignores uncertain targets (`-1`) for ROC-AUC.
        - Requires both positive and negative examples per class.
    """
    per_class = {}
    aucs = []

    for class_idx, class_name in enumerate(CHEXPERT_CLASSES):
        targets_class = targets[:, class_idx]
        probs_class = probs[:, class_idx]

        # Ignore uncertain targets for standard ROC-AUC.
        mask = targets_class != -1
        if mask.sum() == 0:
            continue

        targets_binary = (targets_class[mask] == 1).astype(int)
        
        # Require both positive and negative samples
        if targets_binary.sum() == 0 or (1 - targets_binary).sum() == 0:
            continue

        try:
            # Compute ROC-AUC
            auc = roc_auc_score(targets_binary, probs_class[mask])
            per_class[class_name] = round(float(auc), 4)
            aucs.append(auc)
        except ValueError:
            pass

    # Mean AUC across valid classes
    mean_auc = float(np.mean(aucs)) if aucs else float('nan')
    return mean_auc, per_class


def compute_AUC_uncertain(
    probs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, dict]:
    """
    Compute AUC on samples that contain uncertain targets.

    Args:
        probs (np.ndarray): Sigmoid probabilities.
        targets (np.ndarray): Ground-truth labels in {-1, 0, 1}.

    Returns:
        tuple[float, dict]: (mean_uncertain_auc, per_class_uncertain_auc)

    Shape:
        probs: (N, C)
        targets: (N, C)

    Notes:
        - Selects rows containing at least one `-1`.
        - Remaps `-1 -> 1` for this metric.
        - Useful for uncertainty-aware training diagnostics.
    """
    # Select samples that contain at least one uncertain label (-1)
    has_uncertain = (targets == -1).any(axis=1)
    if has_uncertain.sum() == 0:
        return float('nan'), {}

    # Filter only uncertain samples
    probs_uncertain = probs[has_uncertain]                 # (M, C)
    targets_uncertain = targets[has_uncertain].copy()      # (M, C)

    # Convert uncertain labels (-1) → positive (1)
    targets_uncertain[targets_uncertain == -1] = 1

    aucs = []
    per_class = {}

    for class_idx, class_name in enumerate(CHEXPERT_CLASSES):
        targets_class = targets_uncertain[:, class_idx]
        probs_class = probs_uncertain[:, class_idx]

        # Require both positive and negative samples
        if targets_class.sum() == 0 or (1 - targets_class).sum() == 0:
            continue
        try:
            # Compute ROC-AUC
            auc = roc_auc_score(targets_class, probs_class)
            aucs.append(auc)
            per_class[class_name] = round(float(auc), 4)
        except ValueError:
            pass

    # Mean uncertain AUC
    mean_auc = float(np.mean(aucs)) if aucs else float('nan')
    return mean_auc, per_class


def evaluate(model, loader, device='cuda') -> dict:
    """
    Run inference on a loader and compute summary metrics.

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Evaluation dataloader.
        device (str): Compute device, e.g. 'cuda' or 'cpu'.

    Returns:
        dict: Metric dictionary including scalar and per-class metrics.

    Shape:
        image_features: (B, 3, H, W)
        label_embeddings: (C, D) or (B, C, D), optional
        logits: (B, C)
        probs: (B, C)
        targets: (B, C)

    Notes:
        - Uses `logits` for raw model outputs.
        - Uses `probs = sigmoid(logits)` for metric computation.
    """

    # Switch model to evaluation mode
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():  # disable gradient computation for efficiency
        for batch in loader:
            # Unpack batch into image, label embeddings (optional), and targets
            image_features, label_embeddings, targets = _unpack_batch(batch)

            # Move data to device
            image_features = image_features.to(device)
            targets = targets.to(device)

            # Forward pass (support both 1-input and 2-input models)
            if label_embeddings is not None:
                label_embeddings = label_embeddings.to(device)
                # Two-input forward path (image + label embeddings).
                logits = model(image_features, label_embeddings)
            else:
                # One-input forward path (model stores label embeddings internally).
                logits = model(image_features)

            # Convert logits → probabilities using sigmoid
            probs = torch.sigmoid(logits)  # probs: (B, C)

            # Collect outputs for full-dataset evaluation
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Handle empty loader case
    if not all_probs:
        return {
            'map': None,
            'mean_auc': None,
            'unc_auc': None,
            'per_class_auc': {},
            'per_class_ap': {},
            'per_class_unc_auc': {},
        }

    # Concatenate all batches
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute all evaluation metrics
    map_score, per_class_ap = compute_mAP(probs, targets)
    mean_auc, per_class_auc = compute_mean_AUC(probs, targets)
    unc_auc, per_class_unc = compute_AUC_uncertain(probs, targets)

    return {
        'map': round(map_score, 4) if not np.isnan(map_score) else None,
        'mean_auc': round(mean_auc, 4) if not np.isnan(mean_auc) else None,
        'unc_auc': round(unc_auc, 4) if not np.isnan(unc_auc) else None,
        'per_class_auc': per_class_auc,
        'per_class_ap': per_class_ap,
        'per_class_unc_auc': per_class_unc,
    }


def print_metrics(results: dict):
    """
    Print per-class and mean metric table.

    Args:
        results (dict): Output dictionary from `evaluate`.

    Returns:
        None

    Shape:
        - None (logging utility).
    """
    per_auc = results.get('per_class_auc', {})
    per_ap = results.get('per_class_ap', {})
    per_unc = results.get('per_class_unc_auc', {})

    print(f"\n{'Class':35s} {'AP':>8} {'AUC':>8} {'Unc_AUC':>8}")
    print('-' * 67)

    for class_name in CHEXPERT_CLASSES:
        auc = per_auc.get(class_name, float('nan'))
        ap = per_ap.get(class_name, float('nan'))
        unc = per_unc.get(class_name, float('nan'))

        auc_str = f'{auc:.4f}' if not np.isnan(auc) else 'nan'
        ap_str = f'{ap:.4f}' if not np.isnan(ap) else 'nan'
        unc_str = f'{unc:.4f}' if not np.isnan(unc) else 'nan'

        print(f'{class_name:35s} {ap_str:>8} {auc_str:>8} {unc_str:>8}')

    print('-' * 67)

    mean_auc = results.get('mean_auc')
    map_val = results.get('map')
    unc = results.get('unc_auc')

    mean_auc_str = 'nan' if mean_auc is None else f'{mean_auc:.4f}'
    map_str = 'nan' if map_val is None else f'{map_val:.4f}'
    unc_str = 'nan' if unc is None else f'{unc:.4f}'

    print(f"{'Mean':35s} {map_str:>8} {mean_auc_str:>8} {unc_str:>8}")