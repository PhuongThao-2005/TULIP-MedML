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
NUM_CLASSES = len(CHEXPERT_CLASSES)  # 14

# Exp   Backbone    Node Init    Loss     label_policy  inp_name
# C1    ResNet-101  GloVe        BCE      zeros         glove.npy
# C2    Swin-T      GloVe        BCE      zeros         glove.npy
# C3    ResNet-101  BioMedCLIP   BCE      zeros         biomedclip.npy
# C4    ResNet-101  GloVe        UA-ASL   keep          glove.npy
# C5    Swin-T      BioMedCLIP   UA-ASL   keep          biomedclip.npy

EXPERIMENT_CONFIG = {
    'c1': {
        'inp_name'    : 'data/chexpert_glove_word2vec.npy',
        'label_policy': 'zeros',
        'in_channel'  : 300,
    },
    'c2': {
        'inp_name'    : 'data/chexpert_glove_word2vec.npy',
        'label_policy': 'zeros',
        'in_channel'  : 300,
    },
    'c3': {
        'inp_name'    : 'data/chexpert_biomedclip_word2vec.npy',
        'label_policy': 'zeros',
        'in_channel'  : 512,
    },
    'c4': {
        'inp_name'    : 'data/chexpert_glove_word2vec.npy',
        'label_policy': 'keep',
        'in_channel'  : 300,
    },
    'c5': {
        'inp_name'    : 'data/chexpert_biomedclip_word2vec.npy',
        'label_policy': 'keep',
        'in_channel'  : 512,
    },
}


class CheXpert(data.Dataset):
    """
    CheXpert Dataset.

    Output __getitem__:
        (img, word_vec), target
            img      : Tensor [3, 224, 224]
            word_vec : Tensor [14, D]
                       D=300 nếu GloVe       (C1, C2, C4)
                       D=512 nếu BioMedCLIP  (C3, C5)
            target   : Tensor [14] theo label_policy

    label_policy:
        "zeros" — NaN→0, -1→0   dùng cho C1, C2, C3 (BCE)
        "keep"  — NaN→0, -1→-1  dùng cho C4, C5     (UA-ASL)
    """

    def __init__(
        self,
        root        : str,
        csv_file    : str = 'train.csv',
        inp_name    : str = None,
        transform         = None,
        label_policy: str = 'zeros',
    ):
        assert label_policy in ('zeros', 'keep'), \
            f"label_policy phải là 'zeros' hoặc 'keep', nhận: {label_policy}"

        self.root         = root.rstrip('/')
        self.transform    = transform
        self.label_policy = label_policy
        self.classes      = CHEXPERT_CLASSES
        self.num_classes  = NUM_CLASSES

        # engine.py dùng để build Normalize transform
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std  = [0.229, 0.224, 0.225]

        # ── Load CSV ──────────────────────────────────────
        csv_path = csv_file if os.path.isabs(csv_file) \
                   else os.path.join(self.root, csv_file)
        assert os.path.exists(csv_path), f'CSV not found: {csv_path}'

        df = pd.read_csv(csv_path)
        print(f'[CheXpert] {os.path.basename(csv_path)}: '
              f'{len(df)} rows | policy={label_policy}')

        # ── Xử lý labels ──────────────────────────────────
        label_df = df[CHEXPERT_CLASSES].fillna(0)  # NaN → 0

        if label_policy == 'zeros':
            # -1 → 0: uncertain = negative
            # C1: ResNet-101 + GloVe      + BCE
            # C2: Swin-T     + GloVe      + BCE
            # C3: ResNet-101 + BioMedCLIP + BCE
            label_df = label_df.replace(-1, 0)
        else:
            # 'keep': giữ nguyên -1 để UA-ASL xử lý
            # C4: ResNet-101 + GloVe      + UA-ASL
            # C5: Swin-T     + BioMedCLIP + UA-ASL
            pass

        self.paths  = df['Path'].tolist()
        self.labels = label_df.values.astype(np.float32)  # [N, 14]

        # ── Load word embeddings ───────────────────────────
        # [14, 300] GloVe      → C1, C2, C4
        # [14, 512] BioMedCLIP → C3, C5
        assert inp_name and os.path.exists(inp_name), \
            f'Embedding not found: {inp_name}'
        self.inp = torch.from_numpy(
            np.load(inp_name).astype(np.float32)
        )
        print(f'[CheXpert] embedding: {self.inp.shape} '
              f'({"BioMedCLIP" if self.inp.shape[1]==512 else "GloVe"})')

    def _resolve_path(self, csv_path: str) -> str:
        """Chuyển path trong CSV thành absolute path."""
        clean = csv_path.replace('\\', '/').lstrip('./')
        parts = clean.split('/')
        rel   = '/'.join(parts[1:])
        return os.path.join(self.root, rel)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(
            self._resolve_path(self.paths[index])
        ).convert('RGB')

        if self.transform:
            img = self.transform(img)           # [3, 224, 224]

        target = torch.from_numpy(self.labels[index])  # [14]

        return (img, self.inp), target

    def label_stats(self) -> dict:
        stats = {}
        for i, name in enumerate(CHEXPERT_CLASSES):
            col = self.labels[:, i]
            stats[name] = {
                'pos (1)' : int((col ==  1).sum()),
                'neg (0)' : int((col ==  0).sum()),
                'unc (-1)': int((col == -1).sum()),
            }
        return stats


# ── Factory function — dùng trong train.py ────────────
def build_dataset(
    exp     : str,
    root    : str,
    csv_file: str,
    split   : str = 'train',
    transform     = None,
) -> CheXpert:
    """
    Tạo dataset đúng config cho từng experiment.

    Usage:
        ds = build_dataset('c5', root, 'data/train_small.csv')

    Args:
        exp      : 'c1' | 'c2' | 'c3' | 'c4' | 'c5'
        root     : thư mục gốc chứa ảnh
        csv_file : path tới CSV
        split    : 'train' hoặc 'val' (để log)
        transform: nếu None thì engine.py set sau
    """
    assert exp in EXPERIMENT_CONFIG, \
        f"exp phải là một trong {list(EXPERIMENT_CONFIG.keys())}"

    cfg = EXPERIMENT_CONFIG[exp]

    print(f'[build_dataset] exp={exp.upper()} | split={split} | '
          f'policy={cfg["label_policy"]} | '
          f'emb={"BioMedCLIP" if cfg["in_channel"]==512 else "GloVe"}')

    return CheXpert(
        root         = root,
        csv_file     = csv_file,
        inp_name     = cfg['inp_name'],
        transform    = transform,
        label_policy = cfg['label_policy'],
    )