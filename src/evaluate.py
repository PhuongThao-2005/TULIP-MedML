import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

def compute_metrics(all_scores, all_targets):
    """
    all_scores  : np.array (N, 14) — sigmoid output
    all_targets : np.array (N, 14) — {-1, 0, 1}
    """
    results = {}
    aucs, aps = [], []

    for i, cls in enumerate(CHEXPERT_CLASSES):
        scores  = all_scores[:, i]
        targets = all_targets[:, i].copy()

        # Chỉ tính trên các sample có label rõ ràng (bỏ uncertain -1)
        known = targets != -1
        if known.sum() == 0 or targets[known].sum() == 0:
            results[cls] = {'auc': float('nan'), 'ap': float('nan')}
            continue

        t_bin = (targets[known] == 1).astype(int)
        s     = scores[known]

        try:
            auc = roc_auc_score(t_bin, s)
            ap  = average_precision_score(t_bin, s)
        except ValueError:
            auc = ap = float('nan')

        results[cls] = {'auc': auc, 'ap': ap}
        if not np.isnan(auc): aucs.append(auc)
        if not np.isnan(ap):  aps.append(ap)

    results['mean_auc'] = np.mean(aucs) if aucs else float('nan')
    results['mean_ap']  = np.mean(aps)  if aps  else float('nan')
    return results


def print_metrics(results):
    print(f"\n{'Class':35s} {'AUC':>8} {'AP':>8}")
    print('-' * 54)
    for cls in CHEXPERT_CLASSES:
        r = results[cls]
        auc = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else '   nan'
        ap  = f"{r['ap']:.4f}"  if not np.isnan(r['ap'])  else '   nan'
        print(f"{cls:35s} {auc:>8} {ap:>8}")
    print('-' * 54)
    print(f"{'Mean':35s} {results['mean_auc']:>8.4f} {results['mean_ap']:>8.4f}")