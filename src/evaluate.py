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
def compute_mAP(
    scores : np.ndarray,
    targets: np.ndarray,
) -> tuple[float, dict]:
    """
    scores:  [N, 14] sigmoid output
    targets: [N, 14] {-1, 0, 1}
    Bỏ sample có nhãn -1, chỉ tính trên {0, 1}.

    Trả về (mean_ap, per_class_dict) — khớp với compute_mean_AUC.
    per_class_dict: {label_name: ap} để in bảng.
    """
    per_class = {}
    aps       = []

    for i, cls in enumerate(CHEXPERT_CLASSES):
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
            per_class[cls] = round(float(ap), 4)
            aps.append(ap)
        except ValueError:
            pass

    mean_ap = float(np.mean(aps)) if aps else float('nan')
    return mean_ap, per_class


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
) -> tuple[float, dict]:
    """
    Chỉ tính trên subset sample có ít nhất 1 nhãn = -1.
    Map uncertain (-1) → positive (1) khi evaluate.

    Lý do: ảnh uncertain thường có dấu hiệu bệnh mờ nhạt
    → model tốt phải cho score cao hơn ảnh âm tính rõ ràng
    → nếu UA-ASL hoạt động đúng, unc_auc sẽ cao hơn BCE.

    Lưu ý: CheXpert official val set chỉ có {0, 1} → luôn trả nan.
    Metric này chỉ có ý nghĩa khi eval trên train subset hoặc uncertain split.
    """
    # Lấy subset có ít nhất 1 nhãn -1
    has_uncertain = (targets == -1).any(axis=1)
    if has_uncertain.sum() == 0:
        return float('nan'), {}

    s_unc = scores[has_uncertain]           # [M, 14]
    t_unc = targets[has_uncertain].copy()   # [M, 14]
    t_unc[t_unc == -1] = 1                  # uncertain → positive

    aucs = []
    per_class = {}
    for i, cls in enumerate(CHEXPERT_CLASSES):
        t_c = t_unc[:, i]
        s_c = s_unc[:, i]
        if t_c.sum() == 0 or (1 - t_c).sum() == 0:
            continue
        try:
            auc = roc_auc_score(t_c, s_c)
            aucs.append(auc)
            per_class[cls] = round(float(auc), 4)
        except ValueError:
            pass

    mean_auc = float(np.mean(aucs)) if aucs else float('nan')
    return mean_auc, per_class


# ─────────────────────────────────────────────────────────
# Interface chính
# ─────────────────────────────────────────────────────────
def evaluate(model, loader, device='cuda') -> dict:
    """
    Chạy inference toàn bộ loader, tính 3 metrics.

    Returns:
        {
            "map"          : float,
            "mean_auc"     : float,
            "unc_auc"      : float,
            "per_class_auc": dict,   # {label: auc}
            "per_class_ap" : dict,   # {label: ap}
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
            'map'          : None,
            'mean_auc'     : None,
            'unc_auc'      : None,
            'per_class_auc': {},
            'per_class_ap' : {},
            'per_class_unc_auc': {},
        }

    scores  = np.concatenate(all_scores,  axis=0)  # [N, 14]
    targets = np.concatenate(all_targets, axis=0)  # [N, 14]

    map_score,  per_class_ap  = compute_mAP(scores, targets)
    mean_auc,   per_class_auc = compute_mean_AUC(scores, targets)
    unc_auc,    per_class_unc = compute_AUC_uncertain(scores, targets)

    return {
        'map'          : round(map_score, 4) if not np.isnan(map_score) else None,
        'mean_auc'     : round(mean_auc,  4) if not np.isnan(mean_auc)  else None,
        'unc_auc'      : round(unc_auc,   4) if not np.isnan(unc_auc)   else None,
        'per_class_auc': per_class_auc,
        'per_class_ap' : per_class_ap,
        'per_class_unc_auc': per_class_unc,
    }


# ─────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────
def print_metrics(results: dict, show_unc: bool = False):
    """
    In bảng Class | AUC | AP per-class, rồi các dòng tổng.

    show_unc: chỉ in unc_auc khi dùng UA-ASL (C4/C5).
              Với BCE (C1/C2/C3) val set không có -1 → unc_auc luôn nan,
              không cần in.
    """
    per_auc = results.get('per_class_auc', {})
    per_ap  = results.get('per_class_ap',  {})
    per_unc = results.get('per_class_unc_auc', {}) if show_unc else {}

    # Header
    if show_unc:
        print(f"\n{'Class':35s} {'AP':>8} {'AUC':>8} {'Unc_AUC':>8}")
        print('-' * 67)
    else:
        print(f"\n{'Class':35s} {'AP':>8} {'AUC':>8}")
        print('-' * 55)

    for cls in CHEXPERT_CLASSES:
        auc = per_auc.get(cls, float('nan'))
        ap  = per_ap.get(cls,  float('nan'))
        unc = per_unc.get(cls, float('nan')) if show_unc else None

        auc_str = f'{auc:.4f}' if not np.isnan(auc) else 'nan'
        ap_str  = f'{ap:.4f}'  if not np.isnan(ap)  else 'nan'
        unc_str = f'{unc:.4f}' if show_unc and not np.isnan(unc) else ('nan' if show_unc else '')

        if show_unc:
            print(f'{cls:35s} {ap_str:>8} {auc_str:>8} {unc_str:>8}')
        else:
            print(f'{cls:35s} {ap_str:>8} {auc_str:>8}')

    # Separator
    if show_unc:
        print('-' * 67)
    else:
        print('-' * 55)

    mean_auc = results.get('mean_auc')
    map_val  = results.get('map')
    unc      = results.get('unc_auc')

    mean_auc_str = 'nan' if mean_auc is None else f'{mean_auc:.4f}'
    map_str      = 'nan' if map_val  is None else f'{map_val:.4f}'
    unc_str      = 'nan' if unc is None else f'{unc:.4f}' if show_unc else ''

    # Mean line
    if show_unc:
        print(f"{'Mean':35s} {map_str:>8} {mean_auc_str:>8} {unc_str:>8}")
    else:
        print(f"{'Mean':35s} {map_str:>8} {mean_auc_str:>8}")