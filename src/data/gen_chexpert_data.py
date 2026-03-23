"""
gen_chexpert_data.py
Chạy 1 lần để tạo:
  data/chexpert_glove_word2vec.npy   — word embeddings (14, 300)
  data/chexpert_adj.pkl              — co-occurrence adjacency matrix

Cách chạy:
  # Không có GloVe (dùng random để test nhanh):
  python gen_chexpert_data.py --csv CheXpert-v1.0-small/train.csv

  # Có GloVe 300d (kết quả tốt hơn):
  python gen_chexpert_data.py --csv CheXpert-v1.0-small/train.csv \
      --glove /path/to/glove.840B.300d.txt
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

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


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Adjacency Matrix (Co-occurrence)
# ─────────────────────────────────────────────────────────────────────────────

def build_adj_matrix(csv_path, out_path='data/chexpert_adj.pkl', uncertain='zeros'):
    """
    Đọc train.csv, tính co-occurrence matrix:
        A[i][j] = count(label_i=1 AND label_j=1)
        nums[j]  = count(label_j=1)

    gen_A() trong util.py sẽ tính P(i|j) = A[i][j] / nums[j]
    rồi threshold với t để tạo adjacency 0/1.

    uncertain: 'zeros' (uncertain→0) | 'ones' (uncertain→1)
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

    # Tính co-occurrence counts
    A    = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    nums = np.zeros(NUM_CLASSES, dtype=np.float32)

    for i in range(NUM_CLASSES):
        nums[i] = np.sum(labels[:, i] == 1)
        for j in range(NUM_CLASSES):
            A[i][j] = np.sum((labels[:, i] == 1) & (labels[:, j] == 1))

    result = {'adj': A, 'nums': nums}
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\n✓ Adjacency matrix saved: {out_path}  shape={A.shape}")
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
    p.add_argument('--csv',   default='CheXpert-v1.0-small/train.csv',
                   help='Path to CheXpert train.csv')
    p.add_argument('--glove', default=None,
                   help='Path to GloVe 300d file (optional)')
    p.add_argument('--uncertain', default='zeros', choices=['zeros', 'ones'],
                   help='How to handle uncertain(-1) labels when building adj')
    p.add_argument('--out_dir', default='data',
                   help='Output directory')
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
