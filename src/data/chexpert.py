import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import transforms


CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

NUM_CLASSES = len(CHEXPERT_CLASSES)


# ================= CONFIG =================
EXPERIMENT_CONFIG = {
    'c1': {'inp': '../data/chexpert_glove_word2vec.npy', 'policy': 'zeros', 'dim': 300},
    'c2': {'inp': '../data/chexpert_glove_word2vec.npy', 'policy': 'zeros', 'dim': 300},
    # 'c3': {'inp': '../data/chexpert_biomedclip_word2vec.npy', 'policy': 'zeros', 'dim': 512},
    'c4': {'inp': '../data/chexpert_glove_word2vec.npy', 'policy': 'keep',  'dim': 300},
    # 'c5': {'inp': '../data/chexpert_biomedclip_word2vec.npy', 'policy': 'keep',  'dim': 512},
}


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
        label_policy: str = 'zeros',
        transform=None,
    ):
        assert label_policy in ('zeros', 'keep')

        self.root = root
        self.label_policy = label_policy
        self.transform = transform or get_transform()

        # Load CSV
        csv_path = csv_file if os.path.isabs(csv_file) \
            else os.path.join(root, csv_file)

        df = pd.read_csv(csv_path)
        print(f'[CheXpert] {len(df)} samples | policy={label_policy}')

        # Labels
        label_df = df[CHEXPERT_CLASSES].fillna(0)

        if label_policy == 'zeros':
            label_df = label_df.replace(-1, 0)

        self.labels = label_df.values.astype(np.float32)
        self.paths = df['Path'].tolist()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def _resolve_path(self, path):
        return os.path.join(self.root, path.replace('\\', '/').lstrip('./'))

    def __getitem__(self, idx):
        img = Image.open(
            self._resolve_path(self.paths[idx])
        ).convert('RGB')

        img = self.transform(img) 

        label = torch.from_numpy(self.labels[idx])

        return img, label

    def label_stats(self):
        stats = {}
        for i, name in enumerate(CHEXPERT_CLASSES):
            col = self.labels[:, i]
            stats[name] = {
                'pos': int((col == 1).sum()),
                'neg': int((col == 0).sum()),
                'unc': int((col == -1).sum()),
            }
        return stats


# ================= FACTORY =================
def build_dataset(exp, root, csv_file, split='train'):
    assert exp in EXPERIMENT_CONFIG

    cfg = EXPERIMENT_CONFIG[exp]

    print(f'[build_dataset] {exp.upper()} | {split} | {cfg["policy"]}')

    ds = CheXpertDataset(
        root=root,
        csv_file=csv_file,
        label_policy=cfg['policy'],
        transform=get_transform(split)
    )

    emb = torch.from_numpy(
        np.load(cfg['inp']).astype(np.float32)
    )

    print(f'[embedding] shape={emb.shape}')

    return ds, emb