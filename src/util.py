"""
File: util.py

Description:
    Utility components used across training and evaluation.
    Includes image transforms, AP meter, download helper, and graph utilities.

Main Components:
    - Warp: Resize image to a fixed square size.
    - MultiScaleCrop: Data augmentation with multi-scale random/fixed crop.
    - AveragePrecisionMeter: Running AP meter for multi-label tasks.
    - gen_A: Build thresholded adjacency matrix from co-occurrence stats.
    - gen_adj: Normalize adjacency matrix for GCN propagation.

Inputs:
    - PIL images, numpy arrays, torch tensors, and adjacency pickle files.

Outputs:
    - Transformed images, metric values, and normalized graph matrices.

Notes:
    - Used by `src.engine`, `src.models.gcn`, and training scripts.
"""

import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F


class Warp(object):
    """
    Resize image to a fixed square size.

    Args:
        size (int): Output width/height.
        interpolation (int): PIL interpolation mode.

    Returns:
        PIL.Image: Resized image.

    Shape:
        input image: (H, W, 3)
        output image: (size, size, 3)

    Notes:
        - This transform is mainly used for validation.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Apply fixed-size warp to input image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Resized image.

        Shape:
            img: (H, W, 3)
            output: (size, size, 3)
        """
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(
            size=self.size, interpolation=self.interpolation)


class MultiScaleCrop(object):
    """
    Multi-scale crop augmentation with optional fixed crop positions.

    Args:
        input_size (int | list[int, int]): Final output size.
        scales (list[float] | None): Relative crop scales.
        max_distort (int): Maximum aspect mismatch index.
        fix_crop (bool): Whether to use fixed crop offsets.
        more_fix_crop (bool): Whether to use additional fixed offsets.

    Returns:
        PIL.Image: Cropped and resized image.

    Shape:
        input image: (H, W, 3)
        output image: (input_size[1], input_size[0], 3)
    """
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        """
        Apply multi-scale crop and resize.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Cropped and resized output image.
        """
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        """
        Sample crop size and offsets based on configured scales.

        Args:
            im_size (tuple[int, int]): Original image size as (W, H).

        Returns:
            tuple[int, int, int, int]: (crop_w, crop_h, offset_w, offset_h)
        """
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        """
        Randomly choose one fixed offset from precomputed candidates.
        """
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        """
        Generate candidate fixed crop offsets.

        Args:
            more_fix_crop (bool): Enable extra offsets.
            image_w (int): Image width.
            image_h (int): Image height.
            crop_w (int): Crop width.
            crop_h (int): Crop height.

        Returns:
            list[tuple[int, int]]: Candidate (offset_w, offset_h) pairs.
        """
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4
        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))
        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__


def download_url(url, destination=None, progress_bar=True):
    """
    Download file from URL with optional progress bar.

    Args:
        url (str): Source URL.
        destination (str | None): Destination file path.
        progress_bar (bool): Whether to show tqdm progress.

    Returns:
        None
    """
    def my_hook(t):
        last_b = [0]
        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner
    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
    """
    Store predictions/targets and compute multi-label AP metrics.

    Args:
        difficult_examples (bool): Skip negatives when computing AP if True.

    Notes:
        - Accumulates over batches, then computes per-class metrics.
        - Uses `targets` terminology for ground truth.
    """
    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """
        Reset internal buffers for scores and targets.
        """
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Add one batch of logits/probs and targets to internal storage.

        Args:
            output (torch.Tensor | np.ndarray): Model outputs.
            target (torch.Tensor | np.ndarray): Ground-truth targets.

        Returns:
            None

        Shape:
            output: (B, C) or (B,)
            target: (B, C) or (B,)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1)
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            # Grow storage buffer to reduce frequent reallocations.
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """
        Compute per-class AP values from accumulated buffers.

        Returns:
            torch.Tensor | int: AP tensor of shape (C,) or 0 if empty.
        """
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        for k in range(self.scores.size(1)):
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):
        """
        Compute AP for one class.

        Args:
            output (torch.Tensor): Class scores.
            target (torch.Tensor): Class targets.
            difficult_examples (bool): Ignore negatives when True.

        Returns:
            float: AP value.

        Shape:
            output: (N,)
            target: (N,)
        """
        sorted, indices = torch.sort(output, dim=0, descending=True)
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count == 0:
            return 0.
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        """
        Compute overall multi-label metrics from all classes.
        """
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        """
        Compute overall metrics after keeping only top-k predictions per sample.
        """
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        """
        Compute OP/OR/OF1/CP/CR/CF1 metrics.

        Args:
            scores_ (np.ndarray): Predicted sign matrix.
            targets_ (np.ndarray): Binary targets matrix.

        Returns:
            tuple[float, float, float, float, float, float]: Multi-label metrics.

        Shape:
            scores_: (N, C)
            targets_: (N, C)
        """
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)
        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1


def gen_A(num_classes, t, adj_file):
    """
    Generate thresholded adjacency matrix from co-occurrence statistics.

    Args:
        num_classes (int): Number of classes.
        t (float): Threshold for binarizing co-occurrence.
        adj_file (str): Pickle file containing `adj` and `nums`.

    Returns:
        np.ndarray: Normalized adjacency matrix with self-connections.

    Shape:
        adjacency_matrix: (C, C)

    Notes:
        - Adds identity to ensure self-loop for each class node.
    """
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    adjacency_matrix = result['adj']
    class_counts = result['nums']  # class_counts[c] = number of positive samples for class c
    class_counts = class_counts[:, np.newaxis]
    # Normalize co-occurrence by class frequency.
    adjacency_matrix = adjacency_matrix / (class_counts + 1e-6)
    # Threshold weak edges.
    adjacency_matrix[adjacency_matrix < t] = 0
    adjacency_matrix[adjacency_matrix >= t] = 1
    # Scale columns to keep graph magnitude stable.
    adjacency_matrix = adjacency_matrix * 0.25 / (adjacency_matrix.sum(0, keepdims=True) + 1e-6)
    # Add identity (self-loop) so each node keeps its own information.
    adjacency_matrix = adjacency_matrix + np.identity(num_classes, np.int32)
    return adjacency_matrix


def gen_adj(A):
    """
    Symmetrically normalize adjacency matrix for GCN propagation.

    Args:
        A (torch.Tensor): Raw adjacency matrix.

    Returns:
        torch.Tensor: Normalized adjacency matrix.

    Shape:
        adjacency_matrix: (C, C)
        output: (C, C)

    Notes:
        - Equivalent to D^{-1/2} A D^{-1/2} form.
    """
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    # adjacency_matrix_norm = D^{-1/2} * A * D^{-1/2}
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
