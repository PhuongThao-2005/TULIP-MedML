"""
gen_chexpert_data.py
Chạy 1 lần để tạo:
  data/chexpert_glove_word2vec.npy   — word embeddings (14, 300)
  data/chexpert_adj.pkl              — co-occurrence adjacency matrix

Cách chạy:
  python gen_chexpert_data.py --csv CheXpert-v1.0-small/train.csv \
      --glove /path/to/glove.840B.300d.txt
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch

from open_clip import create_model_from_pretrained, get_tokenizer
from sklearn.metrics.pairwise import cosine_similarity

CHEXPERT_DESCRIPTIONS = {
    'No Finding':
        'chest X-ray with no significant cardiopulmonary abnormality detected',

    'Enlarged Cardiomediastinum':
        'widening of the mediastinum or enlarged cardiac silhouette '
        'on frontal chest X-ray, may indicate vascular or lymph node pathology',

    'Cardiomegaly':
        'abnormal enlargement of the heart shadow on chest X-ray, '
        'cardiothoracic ratio greater than 0.5, suggesting cardiac disease',

    'Lung Opacity':
        'increased density in the lung parenchyma on chest X-ray, '
        'presenting as consolidation, ground-glass opacity, or airspace disease',

    'Lung Lesion':
        'focal abnormality in the lung including nodule, mass, or cavitary '
        'lesion visible on chest X-ray, may represent neoplasm or infection',

    'Edema':
        'pulmonary edema with bilateral airspace opacities, perihilar haziness, '
        'and vascular congestion on chest X-ray due to fluid in lung interstitium',

    'Consolidation':
        'homogeneous opacification of lung parenchyma replacing air with fluid '
        'or tissue on chest X-ray, consistent with pneumonia or hemorrhage',

    'Pneumonia':
        'infectious or inflammatory consolidation of lung parenchyma on chest '
        'X-ray, often presenting as lobar or segmental airspace opacity with fever',

    'Atelectasis':
        'partial or complete collapse of lung tissue on chest X-ray, '
        'causing volume loss and increased density due to airway obstruction',

    'Pneumothorax':
        'presence of air in the pleural space on chest X-ray, '
        'visible as pleural line with absence of lung markings beyond the margin',

    'Pleural Effusion':
        'abnormal accumulation of fluid in the pleural cavity on chest X-ray, '
        'appearing as blunting of costophrenic angle or meniscus sign',

    'Pleural Other':
        'pleural abnormality other than effusion on chest X-ray, '
        'including pleural thickening, calcification, or fibrothorax',

    'Fracture':
        'cortical disruption or break in bony structures visible on chest X-ray, '
        'commonly involving ribs or clavicle due to trauma',

    'Support Devices':
        'medical support devices visible on chest X-ray including endotracheal '
        'tube, central venous catheter, pacemaker lead, or nasogastric tube',
}

CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]
NUM_CLASSES = 14


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Word Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def load_glove(glove_path):
    """Load GloVe file → dict {word: np.array(300)}"""
    print(f"Loading GloVe from {glove_path} ...")
    glove = {}
    with open(glove_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) != 301:
                continue
            word = parts[0].lower()
            glove[word] = np.array(parts[1:], dtype=np.float32)
    print(f"  Loaded {len(glove)} tokens")
    return glove


def build_word_vectors(glove_path=None, out_path='data/chexpert_glove_word2vec.npy'):
    """
    Tạo word embedding cho 14 CheXpert classes.
    Mỗi class name → split thành words → average GloVe vectors.
    Nếu không có GloVe → random (reproducible seed) để smoke-test.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    if glove_path and os.path.exists(glove_path):
        glove = load_glove(glove_path)
        vecs = np.zeros((NUM_CLASSES, 300), dtype=np.float32)
        for i, cls_name in enumerate(CHEXPERT_CLASSES):
            words = cls_name.lower().split()
            found = [glove[w] for w in words if w in glove]
            if found:
                vecs[i] = np.mean(found, axis=0)
                print(f"  [{i:2d}] {cls_name:35s} → {len(found)}/{len(words)} words found")
            else:
                rng = np.random.RandomState(i)
                vecs[i] = rng.randn(300).astype(np.float32)
                print(f"  [{i:2d}] {cls_name:35s} → NOT FOUND, using random")
    else:
        print("[INFO] GloVe not provided → using random vectors (reproducible seed)")
        print("       Để có kết quả tốt hơn, download GloVe và truyền --glove")
        rng = np.random.RandomState(42)
        vecs = rng.randn(NUM_CLASSES, 300).astype(np.float32)

    np.save(out_path, vecs)
    print(f"\n✓ Word vectors saved: {out_path}  shape={vecs.shape}")
    return vecs

def gen_biomedclip_embeddings(
    save_path: str = 'data/chexpert_biomedclip_vec.npy',
    device   : str = 'cpu',
) -> np.ndarray:
    """
    Encode 14 mô tả bệnh lý bằng BioMedCLIP text encoder.
    Output: [14, 512] float32, L2-normalized.
    Freeze toàn bộ encoder — không train, chỉ extract features.
    """

    print("[BioMedCLIP] Loading model...")
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )

    # Freeze toàn bộ — chỉ dùng để extract, không train
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)

    # ── Encode 14 descriptions ────────────────────────
    texts  = [CHEXPERT_DESCRIPTIONS[c] for c in CHEXPERT_CLASSES]
    tokens = tokenizer(texts, context_length=256)
    tokens = tokens.to(device)

    with torch.no_grad():
        embeddings = model.encode_text(tokens)       # [14, 512]
        embeddings = embeddings / embeddings.norm(   # L2 normalize
            dim=-1, keepdim=True
        )

    Z = embeddings.cpu().numpy().astype(np.float32)
    np.save(save_path, Z)

    print(f"[BioMedCLIP] Saved {save_path}")
    print(f"  shape : {Z.shape}")          # (14, 512)
    print(f"  dtype : {Z.dtype}")          # float32
    print(f"  norm  : {np.linalg.norm(Z, axis=1).mean():.4f}")  # ≈ 1.0

    return Z


def verify_embeddings(path: str = 'data/chexpert_biomedclip_vec.npy'):
    """
    Kiểm tra embeddings phân biệt được các nhãn không.
    In cosine similarity matrix — cặp bệnh liên quan phải có sim cao.
    """

    Z   = np.load(path)                          # [14, 512]
    sim = cosine_similarity(Z)                   # [14, 14]

    print(f"\n[Verify] shape={Z.shape}, norm≈{np.linalg.norm(Z,axis=1).mean():.3f}")

    sim_copy = sim.copy()
    np.fill_diagonal(sim_copy, 0)
    flat_idx = np.argsort(sim_copy.ravel())[::-1][:3]
    rows, cols = np.unravel_index(flat_idx, sim_copy.shape)

    print("\nTop 3 cặp similarity cao (bệnh liên quan):")
    for r, c in zip(rows, cols):
        print(f"  {CHEXPERT_CLASSES[r]:30s} ↔ "
              f"{CHEXPERT_CLASSES[c]:30s}: {sim_copy[r,c]:.4f}")
    # Kỳ vọng:
    # Consolidation ↔ Pneumonia     : cao (cùng airspace disease)
    # Edema ↔ Pleural Effusion      : cao (cùng fluid-related)
    # Lung Opacity ↔ Consolidation  : cao (cùng density increase)

    # Top 3 cặp similarity thấp nhất
    np.fill_diagonal(sim_copy, 1)
    flat_idx2 = np.argsort(sim_copy.ravel())[:3]
    rows2, cols2 = np.unravel_index(flat_idx2, sim_copy.shape)

    print("\nTop 3 cặp similarity thấp (bệnh không liên quan):")
    for r, c in zip(rows2, cols2):
        print(f"  {CHEXPERT_CLASSES[r]:30s} ↔ "
              f"{CHEXPERT_CLASSES[c]:30s}: {sim_copy[r,c]:.4f}")
    # Kỳ vọng:
    # No Finding ↔ Fracture         : thấp
    # Support Devices ↔ Pneumothorax: thấp



# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Adjacency Matrix (Co-occurrence)
# ─────────────────────────────────────────────────────────────────────────────

def build_adj_matrix(csv_path, out_path='data/chexpert_adj.pkl', uncertain='zeros'):
    """
    Tính co-occurrence matrix nhanh bằng vectorization.

    A[i][j] = số ảnh có label_i=1 và label_j=1
    nums[j] = số ảnh có label_j=1
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    label_df = df[CHEXPERT_CLASSES].copy()

    # Xử lý NaN và uncertain (-1)
    label_df = label_df.fillna(0)

    if uncertain == 'ones':
        label_df = label_df.replace(-1, 1)
    else:  # 'zeros'
        label_df = label_df.replace(-1, 0)

    labels = label_df.values.astype(np.float32)   # (N, 14)
    N = len(labels)

    binary = (labels == 1).astype(np.float32)     # (N, 14)

    A    = binary.T @ binary                      # (14, 14)
    nums = binary.sum(axis=0)                     # (14,)

    # ===== save =====
    result = {'adj': A, 'nums': nums}
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    # ===== log =====
    print(f"\nAdjacency matrix saved: {out_path}  shape={A.shape}")
    print("  Class counts (positive labels):")

    for cls, n in zip(CHEXPERT_CLASSES, nums):
        pct = 100 * n / N
        print(f"    {cls:35s}: {int(n):6d} ({pct:.1f}%)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate CheXpert auxiliary data')
    p.add_argument('--csv',   default='CheXpert-v1.0-small/train.csv', help='Path to CheXpert train.csv')
    p.add_argument('--glove', default=None, help='Path to GloVe 300d file (optional)')
    p.add_argument('--uncertain', default='zeros', choices=['zeros', 'ones'], help='How to handle uncertain(-1) labels when building adj')
    p.add_argument('--out_dir', default='data', help='Output directory')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Word Vectors")
    print("=" * 60)
    vec_path = os.path.join(args.out_dir, 'chexpert_glove_word2vec.npy')
    build_word_vectors(glove_path=args.glove, out_path=vec_path)

    print("\n" + "=" * 60)
    print("STEP 2: Adjacency Matrix")
    print("=" * 60)
    adj_path = os.path.join(args.out_dir, 'chexpert_adj.pkl')
    build_adj_matrix(csv_path=args.csv, out_path=adj_path, uncertain=args.uncertain)

    print("\n✓ Done. Files created:")
    print(f"   {vec_path}")
    print(f"   {adj_path}")

if __name__ == '__main__':
    Z = gen_biomedclip_embeddings(
        save_path='data/chexpert_biomedclip_word2vec.npy',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    verify_embeddings('data/chexpert_biomedclip_word2vec.npy')
