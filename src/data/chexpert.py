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
        self.root      = root.rstrip('/')
        self.transform = transform
        self.uncertain = uncertain
        self.classes   = CHEXPERT_CLASSES
        self.num_classes = NUM_CLASSES

        # ── Load CSV ──────────────────────────────────────────────────
        csv_path = csv_file if os.path.isabs(csv_file) \
                   else os.path.join(self.root, csv_file)
        assert os.path.exists(csv_path), f'CSV not found: {csv_path}'
        df_raw = pd.read_csv(csv_path)
        print(f'[CheXpert] {os.path.basename(csv_path)}: {len(df_raw)} rows in CSV')

        # ── Filter: chỉ giữ rows có file tồn tại trên disk ───────────
        print('[CheXpert] Filtering missing files...')
        mask = df_raw['Path'].apply(lambda p: os.path.isfile(self._build(p)))
        self.df = df_raw[mask].reset_index(drop=True)
        n_skip  = len(df_raw) - len(self.df)
        print(f'[CheXpert] {len(self.df)} usable  ({n_skip} skipped)')

        assert len(self.df) > 0, (
            f'No images found!\n'
            f'  root  = {self.root}\n'
            f'  tried = {self._build(df_raw["Path"].iloc[0])}'
        )

        # ── Load word embeddings ──────────────────────────────────────
        assert inp_name and os.path.exists(inp_name), \
            f'Word embedding not found: {inp_name}'
        self.inp = torch.from_numpy(
            np.load(inp_name).astype(np.float32))
        self.inp_name = inp_name

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std  = [0.229, 0.224, 0.225]

    def _build(self, csv_path):
        clean  = csv_path.lstrip('./')
        parts  = clean.replace('\\', '/').split('/')
        rel    = '/'.join(parts[1:])   # bỏ parts[0] = 'CheXpert-v1.0-small'
        return os.path.join(self.root, rel)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(self._build(row['Path'])).convert('RGB')
        if self.transform:
            img = self.transform(img)

        target = np.full(NUM_CLASSES, -1, dtype=np.float32)
        for i, cls in enumerate(CHEXPERT_CLASSES):
            val = row.get(cls, np.nan)
            if pd.isna(val) or val == 0.0:
                target[i] = -1
            elif val == 1.0:
                target[i] = 1
            elif val == -1.0:
                if   self.uncertain == 'ones':   target[i] = 1
                elif self.uncertain == 'ignore':  target[i] = 0
                else:                             target[i] = -1

        return (img, row['Path'], self.inp), torch.from_numpy(target)