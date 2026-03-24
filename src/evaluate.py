import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.chexpert import CHEXPERT_CLASSES


def _unpack_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError('Expected batch format (input, target).')

    inputs, labels = batch
    imgs = None
    word_vecs = None

    if torch.is_tensor(inputs):
        imgs = inputs
    elif isinstance(inputs, (tuple, list)):
        if len(inputs) == 0:
            raise ValueError('Input tuple is empty.')

        imgs = inputs[0]

        # Support both:
        #   (img, word_vec)
        #   (img, path, word_vec)
        if len(inputs) >= 3 and torch.is_tensor(inputs[2]):
            word_vecs = inputs[2]
        elif len(inputs) >= 2 and torch.is_tensor(inputs[1]):
            word_vecs = inputs[1]
    else:
        raise ValueError(f'Unsupported input type: {type(inputs)}')

    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)

    return imgs, word_vecs, labels


# ─────────────────────────────────────────────────────────
# 1. mAP — exclude nhãn -1
# ─────────────────────────────────────────────────────────
def compute_mAP(scores: np.ndarray, targets: np.ndarray) -> float:
    """
    scores:  [N, 14] sigmoid output
    targets: [N, 14] {-1, 0, 1}
    Bỏ sample có nhãn -1, chỉ tính trên {0, 1}.
    """
    aps = []
    for i in range(len(CHEXPERT_CLASSES)):
        t = targets[:, i]
        s = scores[:, i]

        mask = t != -1                      # bỏ uncertain
        if mask.sum() == 0:
            continue
        t_bin = (t[mask] == 1).astype(int)
        if t_bin.sum() == 0:               # không có positive
            continue

        try:
            ap = average_precision_score(t_bin, s[mask])
            aps.append(ap)
        except ValueError:
            pass

    return float(np.mean(aps)) if aps else float('nan')


# ─────────────────────────────────────────────────────────
# 2. mean AUC — bỏ nhãn không có dương tính
# ─────────────────────────────────────────────────────────
def compute_mean_AUC(
    scores : np.ndarray,
    targets: np.ndarray,
) -> tuple[float, dict]:
    """
    Trả về (mean_auc, per_class_dict).
    per_class_dict: {label_name: auc} để in bảng.
    """
    per_class = {}
    aucs      = []

    for i, cls in enumerate(CHEXPERT_CLASSES):
        t = targets[:, i]
        s = scores[:, i]

        mask = t != -1                      # bỏ uncertain
        if mask.sum() == 0:
            continue
        t_bin = (t[mask] == 1).astype(int)
        if t_bin.sum() == 0 or (1 - t_bin).sum() == 0:
            continue                        # cần cả pos lẫn neg

        try:
            auc = roc_auc_score(t_bin, s[mask])
            per_class[cls] = round(float(auc), 4)
            aucs.append(auc)
        except ValueError:
            pass

    mean_auc = float(np.mean(aucs)) if aucs else float('nan')
    return mean_auc, per_class


# ─────────────────────────────────────────────────────────
# 3. AUC uncertain — subset sample có ≥1 nhãn -1
# ─────────────────────────────────────────────────────────
def compute_AUC_uncertain(
    scores : np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Chỉ tính trên subset sample có ít nhất 1 nhãn = -1.
    Map uncertain (-1) → positive (1) khi evaluate.

    Lý do: ảnh uncertain thường có dấu hiệu bệnh mờ nhạt
    → model tốt phải cho score cao hơn ảnh âm tính rõ ràng
    → nếu UA-ASL hoạt động đúng, unc_auc sẽ cao hơn BCE.
    """
    # Lấy subset có ít nhất 1 nhãn -1
    has_uncertain = (targets == -1).any(axis=1)
    if has_uncertain.sum() == 0:
        return float('nan')

    s_unc = scores[has_uncertain]           # [M, 14]
    t_unc = targets[has_uncertain].copy()   # [M, 14]

    # Map -1 → 1 (uncertain = positive khi evaluate)
    t_unc[t_unc == -1] = 1

    aucs = []
    for i in range(len(CHEXPERT_CLASSES)):
        t_c = t_unc[:, i]
        s_c = s_unc[:, i]

        if t_c.sum() == 0 or (1 - t_c).sum() == 0:
            continue

        try:
            auc = roc_auc_score(t_c, s_c)
            aucs.append(auc)
        except ValueError:
            pass

    return float(np.mean(aucs)) if aucs else float('nan')


# ─────────────────────────────────────────────────────────
# Interface chính
# ─────────────────────────────────────────────────────────
def evaluate(model, loader, device='cuda') -> dict:
    """
    Chạy inference toàn bộ loader, tính 3 metrics.

    Args:
        model  : PyTorch model, output [B, 14] logits
        loader : DataLoader trả về (img, word_vec), labels
                 labels: [B, 14] tensor {-1, 0, 1}
        device : 'cuda' hoặc 'cpu'

    Returns:
        {
            "map"          : float,
            "mean_auc"     : float,
            "unc_auc"      : float,
            "per_class_auc": dict   # {label: auc} để in bảng
        }
    """
    model.eval()

    all_scores  = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            imgs, word_vecs, labels = _unpack_batch(batch)

            imgs      = imgs.to(device)
            labels    = labels.to(device)

            if word_vecs is not None:
                word_vecs = word_vecs.to(device)
                logits = model(imgs, word_vecs)     # [B, 14]
            else:
                logits = model(imgs)                # [B, 14]

            probs  = torch.sigmoid(logits)          # [B, 14]

            all_scores.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())   # giữ nguyên {-1,0,1}

    if not all_scores:
        return {
            'map': None,
            'mean_auc': None,
            'unc_auc': None,
            'per_class_auc': {},
        }

    scores  = np.concatenate(all_scores,  axis=0)  # [N, 14]
    targets = np.concatenate(all_targets, axis=0)  # [N, 14]

    map_score          = compute_mAP(scores, targets)
    mean_auc, per_class = compute_mean_AUC(scores, targets)
    unc_auc            = compute_AUC_uncertain(scores, targets)

    return {
        'map'          : round(map_score, 4) if not np.isnan(map_score) else None,
        'mean_auc'     : round(mean_auc,  4) if not np.isnan(mean_auc) else None,
        'unc_auc'      : round(unc_auc,   4) if not np.isnan(unc_auc) else None,
        'per_class_auc': per_class,
    }


# ─────────────────────────────────────────────────────────
# Giữ lại hàm print từ code gốc — dùng sau evaluate()
# ─────────────────────────────────────────────────────────
def print_metrics(results: dict):
    per = results.get('per_class_auc', {})
    print(f"\n{'Class':35s} {'AUC':>8}")
    print('-' * 45)
    for cls in CHEXPERT_CLASSES:
        auc = per.get(cls, float('nan'))
        val = f'{auc:.4f}' if not np.isnan(auc) else '   nan'
        print(f'{cls:35s} {val:>8}')
    print('-' * 45)
    mean_auc = results.get('mean_auc')
    map_val = results.get('map')
    print(f"{'mean_auc':35s} {'N/A' if mean_auc is None else f'{mean_auc:.4f}':>8}")
    print(f"{'map':35s} {'N/A' if map_val is None else f'{map_val:.4f}':>8}")
    unc = results['unc_auc']
    print(f"{'unc_auc':35s} {'N/A' if unc is None else f'{unc:.4f}':>8}")