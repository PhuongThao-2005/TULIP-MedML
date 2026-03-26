import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

NUM_CLASSES = len(CHEXPERT_CLASSES)


# ================= TRANSFORM =================
def get_transform(split='train', size=224):
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    if split == 'train':
        return T.Compose([
            T.Resize(size + 32),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize(size + 32),
            T.CenterCrop(size),
            T.ToTensor(),
            normalize,
        ])


# ================= DATASET =================
class CheXpertDataset(Dataset):

    def __init__(
        self,
        root: str,
        csv_file: str,
        inp_name: Optional[str] = None,
        label_policy: str = 'zeros',
        uncertain: Optional[str] = None,
        transform=None,
        target_transform=None,
    ):
        policy = uncertain if uncertain is not None else label_policy
        assert policy in ('zeros', 'ones', 'ignore', 'keep')

        self.root = root
        self.label_policy = policy
        self.transform = transform
        self.target_transform = target_transform

        csv_path = csv_file if os.path.isabs(csv_file) \
            else os.path.join(root, csv_file)
        inp_path = inp_name if inp_name and os.path.isabs(inp_name) \
            else os.path.join(root, inp_name) if inp_name else None

        self.df = pd.read_csv(csv_path)
        print(f'[CheXpert] {len(self.df)} samples | policy={self.label_policy}')

        if inp_path:
            self.inp = torch.from_numpy(np.load(inp_path).astype(np.float32))
        else:
            self.inp = torch.zeros((NUM_CLASSES, 300), dtype=torch.float32)

        if self.transform is None:
            self.transform = get_transform(split='train')

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, path):
        path = path.replace('\\', '/')

        path = path.replace('CheXpert-v1.0-small/', '')
        if '/._' in path:
            return None

        return os.path.join(self.root, path)

    def _prepare_labels(self, row):
        labels = (
            row[CHEXPERT_CLASSES]
            .astype(float)
            .fillna(0)
            .values.astype(np.float32)
        )
        if self.label_policy == 'zeros':
            labels[labels == -1] = 0
        elif self.label_policy == 'ones':
            labels[labels == -1] = 1

        return labels

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['Path']

        img_path = self._resolve_path(path)

        if img_path is None or not os.path.exists(img_path):
            # skip sample lỗi
            return self.__getitem__((idx + 1) % len(self.df))

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        label = torch.from_numpy(self._prepare_labels(row))
        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, path, self.inp), label

    def label_stats(self):
        stats = {}
        label_df = self.df[CHEXPERT_CLASSES].fillna(0).copy()
        for i, name in enumerate(CHEXPERT_CLASSES):
            col = label_df.iloc[:, i].to_numpy()
            stats[name] = {
                'pos': int((col == 1).sum()),
                'neg': int((col == 0).sum()),
                'unc': int((col == -1).sum()),
            }
        return stats


# Alias giữ tương thích import trong train.py
CheXpert = CheXpertDataset


# ================= FACTORY =================
def build_dataset(
    root,
    csv_file,
    inp_name,
    split='train',
    uncertain='zeros',
):
    ds = CheXpertDataset(
        root=root,
        csv_file=csv_file,
        inp_name=inp_name,
        uncertain=uncertain,
        transform=get_transform(split),
    )
    return ds, ds.inp
