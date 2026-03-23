import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data

CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]
NUM_CLASSES = 14


class CheXpert(data.Dataset):
    def __init__(self, root, csv_file='train.csv',
                 inp_name=None, transform=None, uncertain='zeros'):
        self.root      = root
        self.transform = transform
        self.uncertain = uncertain
        self.classes   = CHEXPERT_CLASSES
        self.num_classes = NUM_CLASSES

        # ── Load CSV ─────────────────────────────────────────────────
        csv_path = os.path.join(root, csv_file)
        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
        df_raw = pd.read_csv(csv_path)
        print(f'[CheXpert] {csv_file}: {len(df_raw)} rows in CSV')

        # ── FIX: chỉ giữ lại rows có file ảnh thực sự tồn tại ───────
        # Dataset Kaggle có thể thiếu nhiều file so với CSV gốc
        print(f'[CheXpert] Scanning files...')
        exists_mask = df_raw['Path'].apply(
            lambda p: os.path.isfile(self._build_path(root, p))
        )
        self.df = df_raw[exists_mask].reset_index(drop=True)

        n_removed = len(df_raw) - len(self.df)
        print(f'[CheXpert] {len(self.df)} usable  '
              f'({n_removed} skipped - file not found)')

        if len(self.df) == 0:
            sample = self._build_path(root, df_raw['Path'].iloc[0])
            raise FileNotFoundError(
                f"No images found! Kiểm tra CHEXPERT_ROOT.\n"
                f"  root = {root}\n"
                f"  Sample path tried: {sample}"
            )

        # ── Load word embeddings ──────────────────────────────────────
        assert inp_name and os.path.exists(inp_name), \
            f"Word embedding not found: {inp_name}"
        self.inp = torch.from_numpy(
            np.load(inp_name).astype(np.float32))   # (14, 300)
        self.inp_name = inp_name

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std  = [0.229, 0.224, 0.225]

    @staticmethod
    def _build_path(root, row_path):
        # CSV Path: "CheXpert-v1.0-small/train/patientXXX/studyY/viewZ.jpg"
        # root:     ".../CheXpert-v1.0-small"
        # join với '..' để lên 1 cấp, rồi nối lại path từ CSV
        return os.path.normpath(os.path.join(root, '..', row_path))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = self._build_path(self.root, row['Path'])
        img = Image.open(img_path).convert('RGB')   # grayscale -> RGB
        if self.transform:
            img = self.transform(img)

        # ── Labels ────────────────────────────────────────────────────
        target = np.full(NUM_CLASSES, -1, dtype=np.float32)
        for i, cls in enumerate(CHEXPERT_CLASSES):
            val = row.get(cls, np.nan)
            if pd.isna(val) or val == 0.0:
                target[i] = -1          # absent / not mentioned
            elif val == 1.0:
                target[i] = 1           # positive
            elif val == -1.0:           # uncertain
                if   self.uncertain == 'ones':   target[i] = 1
                elif self.uncertain == 'ignore':  target[i] = 0
                else:                             target[i] = -1  # zeros

        return (img, row['Path'], self.inp), torch.from_numpy(target)
