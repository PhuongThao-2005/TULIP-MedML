import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data

# 14 nhãn bệnh của CheXpert
CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]
NUM_CLASSES = 14


class CheXpert(data.Dataset):
    """
    Dataset class cho CheXpert multi-label chest X-ray.

    Mỗi __getitem__ trả về:
        (img, path, word_vec), target
           state['feature'] = img        (B, 3, H, W)
           state['out']     = path       string
           state['input']   = word_vec   (14, 300)
        target: Tensor (14,) với giá trị {-1, 0, 1}
    """
    def __init__(self, root, csv_file='train.csv',
                 inp_name=None, transform=None, uncertain='zeros'):
        """
        root      : thư mục chứa ảnh, VD '/kaggle/input/.../chexpert'
        csv_file  : path đầy đủ hoặc tên file CSV
        inp_name  : path tới file .npy chứa word embeddings (14, 300)
        transform : torchvision transform, được engine.py set sau khi khởi tạo
        uncertain : cách xử lý label -1 trong CSV
                    'zeros'  → coi là negative (mặc định, đơn giản nhất)
                    'ones'   → coi là positive (U-Ones strategy)
                    'ignore' → đánh dấu 0 để engine mask khỏi loss
        """
        self.root      = root.rstrip('/')
        self.transform = transform
        self.uncertain = uncertain
        self.classes   = CHEXPERT_CLASSES
        self.num_classes = NUM_CLASSES

        # Load CSV
        csv_path = csv_file if os.path.isabs(csv_file) \
                   else os.path.join(self.root, csv_file)
        assert os.path.exists(csv_path), f'CSV not found: {csv_path}'
        self.df = pd.read_csv(csv_path)
        print(f'[CheXpert] {os.path.basename(csv_path)}: {len(self.df)} rows')

        # ── Load word embeddings ──────────────────────────────────────
        # inp shape (14, 300): mỗi row là GloVe vector của 1 class
        # Load 1 lần ở đây, dùng chung cho mọi sample → không load lại trong __getitem__
        assert inp_name and os.path.exists(inp_name), \
            f'Word embedding not found: {inp_name}'
        self.inp = torch.from_numpy(
            np.load(inp_name).astype(np.float32))
        self.inp_name = inp_name

        # Dùng để engine.py tự động tạo transform chuẩn hóa ảnh
        self.image_normalization_mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self.image_normalization_std  = [0.229, 0.224, 0.225]  # ImageNet std

    def _build(self, csv_path):
        """
        Chuyển path trong CSV thành absolute path trên disk.
        """
        clean = csv_path.lstrip('./')
        parts = clean.replace('\\', '/').split('/')
        rel   = '/'.join(parts[1:])
        return os.path.join(self.root, rel)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # ── Load ảnh ─────────────────────────────────────────────────
        # X-ray gốc là grayscale (1 channel)
        # .convert('RGB') nhân ra 3 channels để ResNet-101 pretrained nhận được
        img = Image.open(self._build(row['Path'])).convert('RGB')
        
        # Transform được engine.py set sau khi khởi tạo dataset:
        #   train → MultiScaleCrop + RandomHorizontalFlip + Normalize
        #   val   → Warp (resize) + Normalize
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