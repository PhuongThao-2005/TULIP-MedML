"""
File: chexpert.py
Description:
    Dataset class and data transforms for the CheXpert chest X-ray dataset.
    Supports label uncertainty handling via multiple policies.
Main Components:
    - get_transform: Build train/val image augmentation pipeline.
    - CheXpertDataset: PyTorch Dataset for CheXpert with uncertainty handling.
    - build_dataset: Factory function for constructing a CheXpertDataset.
Inputs:
    CSV files with image paths and multi-label annotations (14 classes).
Outputs:
    (image_tensor, label_tensor) pairs; optional label embedding tensor (inp).
Notes:
    - Labels can be {0, 1, -1} where -1 indicates uncertain.
    - Uncertainty policy must be declared at dataset construction time.
    - CheXpert is an alias for CheXpertDataset for backward compatibility.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


CHEXPERT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

NUM_CLASSES = len(CHEXPERT_CLASSES)


# Transforms

def get_transform(split: str = "train", size: int = 224) -> T.Compose:
    """
    Build an image transform pipeline for train or validation splits.

    Args:
        split (str): One of "train" or "val"/"test". Train applies random
            crop and horizontal flip; val applies center crop only.
        size (int): Target crop size in pixels.

    Returns:
        T.Compose: Composed torchvision transform pipeline.

    Notes:
        - Images are resized to (size + 32) before cropping to preserve context.
        - Normalization uses ImageNet statistics.
    """
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return T.Compose([
            T.Resize(size + 32),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

    return T.Compose([
        T.Resize(size + 32),
        T.CenterCrop(size),
        T.ToTensor(),
        normalize,
    ])


# Dataset

class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for the CheXpert chest X-ray multi-label classification task.

    Args:
        root (str): Root directory of the CheXpert dataset.
        csv_file (str): Path to a split CSV (absolute or relative to root).
        inp_name (str, optional): Path to a .npy file of label embeddings
            (num_classes, embed_dim). Falls back to zeros if not provided.
        split (str): "train" or "val". Used to select the default transform.
        label_policy (str): Legacy alias for uncertain. One of
            "zeros", "ones", "ignore", "keep".
        uncertain (str, optional): How to handle -1 labels.
            "zeros" → treat uncertain as negative.
            "ones"  → treat uncertain as positive.
            "ignore"/"keep" → pass -1 through unchanged.
        transform: Image transform. Defaults to get_transform(split).
        target_transform: Optional transform applied to the label tensor.

    Notes:
        - When an image file is missing, the loader skips forward to the
          next valid sample rather than raising immediately.
        - The 'uncertain' argument takes priority over 'label_policy'.
    """

    def __init__(
        self,
        root: str,
        csv_file: str,
        inp_name: Optional[str] = None,
        split: str = "train",
        label_policy: str = "zeros",
        uncertain: Optional[str] = None,
        transform=None,
        target_transform=None,
    ):
        policy = uncertain if uncertain is not None else label_policy
        assert policy in ("zeros", "ones", "ignore", "keep"), (
            f"Unknown uncertainty policy: {policy!r}"
        )

        self.root = root
        self.label_policy = policy
        self.transform = transform
        self.target_transform = target_transform

        # Resolve CSV and embedding paths relative to root when not absolute
        csv_path = (
            csv_file
            if os.path.isabs(csv_file)
            else os.path.join(root, csv_file)
        )
        inp_path = (
            inp_name
            if (inp_name and os.path.isabs(inp_name))
            else os.path.join(root, inp_name) if inp_name
            else None
        )

        self.df = pd.read_csv(csv_path)
        print(f"[CheXpert] {len(self.df)} samples | policy={self.label_policy}")

        # Load label embeddings if provided; otherwise use zero placeholders
        if inp_path:
            self.inp = torch.from_numpy(
                np.load(inp_path).astype(np.float32)
            )  # (NUM_CLASSES, embed_dim)
        else:
            self.inp = torch.zeros((NUM_CLASSES, 300), dtype=torch.float32)

        if self.transform is None:
            self.transform = get_transform(split=split)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, raw_path: str) -> Optional[str]:
        """
        Normalize a raw CSV path to an absolute file path.

        Args:
            raw_path (str): Path string as stored in the CSV.

        Returns:
            str or None: Absolute path if resolvable, None if path is
                a macOS resource fork (._) artifact.
        """
        path = raw_path.replace("\\", "/")
        path = path.replace("CheXpert-v1.0-small/", "")

        if "/._" in path:
            return None

        return os.path.join(self.root, path)

    def _prepare_targets(self, row: pd.Series) -> np.ndarray:
        """
        Extract and remap multi-label targets from a DataFrame row.

        Args:
            row (pd.Series): A single row from the dataset CSV.

        Returns:
            np.ndarray: Float32 array of shape (NUM_CLASSES,) with values
                in {0.0, 1.0} or {-1.0, 0.0, 1.0} depending on policy.

        Notes:
            - NaN values (unlabeled) are treated as negative (0).
            - "zeros" maps uncertain (-1) to 0 (conservative baseline).
            - "ones" maps uncertain (-1) to 1 (optimistic baseline).
            - "keep"/"ignore" leaves -1 unchanged for downstream handling.
        """
        targets = (
            row[CHEXPERT_CLASSES]
            .astype(float)
            .fillna(0)
            .values.astype(np.float32)
        )

        if self.label_policy == "zeros":
            targets[targets == -1] = 0
        elif self.label_policy == "ones":
            targets[targets == -1] = 1

        return targets

    def __getitem__(self, idx: int):
        """
        Load and return one (image, targets) sample.

        Returns:
            tuple:
                - (image_tensor, image_path): Transformed image and its path.
                - targets (Tensor): Multi-label target of shape (NUM_CLASSES,).

        Notes:
            - If the image at idx is missing, the loader advances to the next
              valid sample cyclically. Raises RuntimeError if none are valid.
        """
        for _ in range(len(self.df)):
            row = self.df.iloc[idx]
            img_path = self._resolve_path(row["Path"])

            if img_path and os.path.exists(img_path):
                break

            idx = (idx + 1) % len(self.df)
        else:
            raise RuntimeError("No valid image found in the dataset.")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # (3, H, W)

        targets = torch.from_numpy(self._prepare_targets(row))  # (NUM_CLASSES,)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return (img, img_path), targets

    def label_stats(self) -> dict:
        """
        Compute per-class label statistics across the dataset.

        Returns:
            dict: Maps each class name to {"pos": int, "neg": int, "unc": int},
                counting positive, negative, and uncertain samples respectively.
        """
        stats = {}
        label_df = self.df[CHEXPERT_CLASSES].fillna(0).copy()
        for name in CHEXPERT_CLASSES:
            col = label_df[name].to_numpy()
            stats[name] = {
                "pos": int((col == 1).sum()),
                "neg": int((col == 0).sum()),
                "unc": int((col == -1).sum()),
            }
        return stats


# Alias for backward compatibility with train scripts importing CheXpert
CheXpert = CheXpertDataset

# Factory

def build_dataset(
    root: str,
    csv_file: str,
    inp_name: str,
    split: str = "train",
    uncertain: str = "zeros",
) -> tuple:
    """
    Construct a CheXpertDataset and return it with its label embedding tensor.

    Args:
        root (str): Root directory of the dataset.
        csv_file (str): Path to the split CSV.
        inp_name (str): Path to the label embedding .npy file.
        split (str): "train" or "val".
        uncertain (str): Uncertainty label policy ("zeros", "ones", "keep").

    Returns:
        tuple:
            - CheXpertDataset: The constructed dataset.
            - Tensor: Label embeddings of shape (NUM_CLASSES, embed_dim).
    """
    dataset = CheXpertDataset(
        root=root,
        csv_file=csv_file,
        inp_name=inp_name,
        uncertain=uncertain,
        transform=get_transform(split),
    )
    return dataset, dataset.inp