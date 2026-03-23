"""
Chạy 1 lần: python src/data/make_splits.py
Tạo ra data/train_split.csv và data/val_split.csv
dùng MultilabelStratifiedKFold để đảm bảo tỷ lệ 14 nhãn đều nhau
"""
import os
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

CHEXPERT_ROOT = '/kaggle/input/datasets/ashery/chexpert'
OUT_DIR       = '/kaggle/working/data'
SEED          = 42

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(CHEXPERT_ROOT, 'train.csv'))
    print(f'Loaded {len(df)} rows')

    # ── EDA: thống kê pos/neg/uncertain ──────────────────────────────
    print('\n── Label statistics ──')
    for col in LABELS:
        pos = (df[col] == 1).sum()
        neg = (df[col] == 0).sum()
        unc = (df[col] == -1).sum()
        nan = df[col].isna().sum()
        print(f'  {col:35s} pos={pos:6d} neg={neg:6d} unc={unc:6d} nan={nan:6d}')

    # ── Stratified split ──────────────────────────────────────────────
    # Binarize: uncertain(-1) và nan → 0 để stratify
    label_mat = df[LABELS].fillna(0).replace(-1, 0).values.astype(int)

    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    # Lấy fold 0 làm val (10%), còn lại làm train (90%)
    for train_idx, val_idx in mskf.split(df, label_mat):
        break   # chỉ cần fold đầu tiên

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_path = os.path.join(OUT_DIR, 'train_split.csv')
    val_path   = os.path.join(OUT_DIR, 'val_split.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)

    print(f'\nTrain: {len(train_df)} rows → {train_path}')
    print(f'Val  : {len(val_df)} rows → {val_path}')

    # Verify phân phối đều
    print('\n── Verify label distribution (pos rate) ──')
    for col in LABELS:
        tr = (train_df[col] == 1).mean()
        vl = (val_df[col]   == 1).mean()
        diff = abs(tr - vl)
        flag = '  ← OK' if diff < 0.02 else '  ← CHECK'
        print(f'  {col:35s} train={tr:.3f}  val={vl:.3f}  diff={diff:.3f}{flag}')

if __name__ == '__main__':
    main()