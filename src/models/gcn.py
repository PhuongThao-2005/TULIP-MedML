"""
File: gcn.py
Description:
    Graph Convolutional Network (GCN) models for multi-label image classification.
    Labels are treated as graph nodes; a co-occurrence adjacency matrix encodes
    semantic relationships between classes.
Main Components:
    - GraphConvolution: Single GCN layer (Kipf & Welling, 2017).
    - GCNResnet: ResNet-101 backbone + two-layer GCN for label embedding propagation.
    - GCNSwin: Swin Transformer backbone + two-layer GCN.
    - gcn_resnet101, gcn_swin_t, gcn_swin_b: Factory functions.
Inputs:
    images (B, 3, H, W) and optional label embedding / adjacency files.
Outputs:
    logits (B, C) computed as dot product of image features and GCN-refined
    label embeddings.
Notes:
    - The adjacency matrix is built from training co-occurrence statistics
      (gen_A / gen_adj from src/util).
    - Label embeddings (inp) are fixed buffers — they are not trained.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Parameter

from src.util import gen_A, gen_adj
from src.models.backbone import SwinBackbone, get_swin_backbone

# Graph Convolution Layer

class GraphConvolution(nn.Module):
    """
    Single graph convolution layer: Z = A * X * W.

    Implements the layer-wise propagation rule from:
        Kipf & Welling, "Semi-Supervised Classification with GCNs" (ICLR 2017).
        https://arxiv.org/abs/1609.02907

    Args:
        in_features (int): Dimension of input node features.
        out_features (int): Dimension of output node features.
        bias (bool): Whether to include a learnable bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight and bias with uniform distribution (fan-in scaling)."""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Propagate node features through the graph adjacency.

        Args:
            node_features (Tensor): Input node feature matrix.
            adjacency_matrix (Tensor): Normalized graph adjacency matrix.

        Returns:
            Tensor: Updated node features after one propagation step.

        Shape:
            node_features:    (C, D_in)
            adjacency_matrix: (C, C)
            output:           (C, D_out)
        """
        # Linear transform: project each node's features
        support = torch.matmul(node_features, self.weight)   # (C, D_out)
        # Graph propagation: aggregate from neighbors
        output = torch.matmul(adjacency_matrix, support)     # (C, D_out)

        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            f"({self.in_features} -> {self.out_features})"
        )


# GCN + ResNet-101

class GCNResnet(nn.Module):
    """
    Multi-label classifier combining ResNet-101 image features with GCN-refined
    label embeddings.

    Image features and label embeddings are matched via dot product to produce
    per-class logits. The GCN refines label embeddings using a learned co-
    occurrence adjacency matrix.

    Args:
        model: A torchvision ResNet instance used as the backbone.
        num_classes (int): Number of output classes (C).
        in_channel (int): Dimension of the input label embeddings (e.g. 300 for GloVe).
        t (float): Threshold for binarizing the co-occurrence matrix in gen_A.
        adj_file (str, optional): Path to pickled adjacency data (from gen_chexpert_data.py).
        inp_file (str, optional): Path to .npy label embedding file of shape (C, in_channel).
    """

    def __init__(
        self,
        model,
        num_classes: int,
        in_channel: int = 300,
        t: float = 0,
        adj_file: str = None,
        inp_file: str = None,
    ):
        super().__init__()

        # Shared backbone: ResNet feature extractor without FC head
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        self.num_classes = num_classes
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Two-layer GCN to refine label embeddings through co-occurrence structure
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        # Co-occurrence adjacency matrix (learned prior, not trained during forward)
        adjacency_matrix = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(adjacency_matrix).float())

        # ImageNet normalization constants (for reference / downstream use)
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # Label embeddings: fixed buffer (not trained), shape (C, in_channel)
        if inp_file:
            label_embeddings = torch.from_numpy(
                np.load(inp_file).astype(np.float32)
            )  # (C, in_channel)
        else:
            label_embeddings = torch.zeros(num_classes, in_channel)
        self.register_buffer("inp", label_embeddings)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract image features and match with GCN-refined label embeddings.

        Args:
            images (Tensor): Input images.

        Returns:
            Tensor: Per-class logits.

        Shape:
            images: (B, 3, H, W)
            output: (B, C)

        Notes:
            - Label embeddings are propagated through two GCN layers using the
              normalized co-occurrence adjacency matrix.
              (Giải thích: embedding nhãn được lan truyền qua đồ thị đồng xuất hiện)
            - Logits are the dot product of image features and label embeddings.
        """
        # Extract global image features from ResNet backbone
        image_features = self.features(images)          # (B, 2048, H', W')
        image_features = self.pooling(image_features)   # (B, 2048, 1, 1)
        image_features = image_features.view(
            image_features.size(0), -1
        )                                               # (B, 2048)

        # Normalize adjacency matrix for stable GCN propagation
        adjacency_matrix = gen_adj(self.A).detach()    # (C, C)

        # Propagate label embeddings through co-occurrence graph
        label_embeddings = self.relu(
            self.gc1(self.inp, adjacency_matrix)
        )                                               # (C, 1024)
        label_embeddings = self.relu(
            self.gc2(label_embeddings, adjacency_matrix)
        )                                               # (C, 2048)

        # Compute logits via dot product: each class has a learned direction in feature space
        label_embeddings = label_embeddings.transpose(0, 1)          # (2048, C)
        logits = torch.matmul(image_features, label_embeddings)      # (B, C)
        return logits

    def get_config_optim(self, lr: float, lrp: float) -> list[dict]:
        """
        Return optimizer parameter groups with separate learning rates.

        Args:
            lr (float): Base learning rate for GCN layers.
            lrp (float): Multiplier for backbone learning rate (typically < 1).

        Returns:
            list[dict]: Optimizer parameter groups.
        """
        return [
            {"params": self.features.parameters(), "lr": lr * lrp},
            {"params": self.gc1.parameters(), "lr": lr},
            {"params": self.gc2.parameters(), "lr": lr},
        ]


def gcn_resnet101(
    num_classes: int,
    t: float,
    pretrained: bool = False,
    adj_file: str = None,
    in_channel: int = 300,
    inp_file: str = None,
) -> GCNResnet:
    """
    Construct a GCNResnet with a ResNet-101 backbone.

    Args:
        num_classes (int): Number of output classes.
        t (float): Co-occurrence threshold for adjacency construction.
        pretrained (bool): Load ImageNet weights if True.
        adj_file (str, optional): Path to adjacency pickle file.
        in_channel (int): Label embedding input dimension.
        inp_file (str, optional): Path to label embedding .npy file.

    Returns:
        GCNResnet: Initialized model.
    """
    backbone = models.resnet101(pretrained=pretrained)
    return GCNResnet(
        backbone,
        num_classes,
        t=t,
        adj_file=adj_file,
        in_channel=in_channel,
        inp_file=inp_file,
    )


# GCN + Swin Transformer

class GCNSwin(nn.Module):
    """
    Multi-label classifier combining a Swin Transformer backbone with a
    two-layer GCN for label embedding propagation.

    Functionally equivalent to GCNResnet but uses SwinBackbone as the
    image feature extractor.

    Args:
        backbone (SwinBackbone): Pretrained Swin backbone that outputs (B, 2048).
        num_classes (int): Number of output classes (C).
        in_channel (int): Label embedding input dimension (e.g. 512 for BioMedCLIP).
        t (float): Threshold for building the binarized co-occurrence adjacency.
        adj_file (str, optional): Path to pickled adjacency data.
        inp_file (str, optional): Path to .npy label embedding file (C, in_channel).
    """

    def __init__(
        self,
        backbone: SwinBackbone,
        num_classes: int,
        in_channel: int = 512,
        t: float = 0,
        adj_file: str = None,
        inp_file: str = None,
    ):
        super().__init__()
        self.features = backbone       # SwinBackbone: (B, 3, H, W) → (B, 2048)
        self.num_classes = num_classes

        # Two-layer GCN: refine label embeddings using co-occurrence structure
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        # Fixed co-occurrence adjacency matrix (not a trainable parameter during fwd)
        adjacency_matrix = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(adjacency_matrix).float())

        # Label embeddings: fixed buffer (C, in_channel)
        if inp_file:
            label_embeddings = torch.from_numpy(
                np.load(inp_file).astype(np.float32)
            )  # (C, in_channel)
        else:
            label_embeddings = torch.zeros(num_classes, in_channel)
        self.register_buffer("inp", label_embeddings)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Swin features GCN-refined label embeddings → logits.

        Args:
            images (Tensor): Input images.

        Returns:
            Tensor: Per-class logits.

        Shape:
            images: (B, 3, H, W)
            output: (B, C)

        Notes:
            - Backbone already applies global average pooling and projection,
              so image_features are already (B, 2048).
            - GCN propagates label embeddings through the co-occurrence graph.
              (Giải thích: embedding nhãn lan truyền qua đồ thị đồng xuất hiện)
        """
        # Extract global image features via Swin backbone (includes pooling + projection)
        image_features = self.features(images)               # (B, 2048)

        # Normalize adjacency for GCN stability
        adjacency_matrix = gen_adj(self.A).detach()         # (C, C)

        # Propagate label embeddings through two GCN layers
        label_embeddings = self.relu(
            self.gc1(self.inp, adjacency_matrix)
        )                                                    # (C, 1024)
        label_embeddings = self.relu(
            self.gc2(label_embeddings, adjacency_matrix)
        )                                                    # (C, 2048)

        # Compute per-class logits via dot product
        label_embeddings = label_embeddings.transpose(0, 1)             # (2048, C)
        logits = torch.matmul(image_features, label_embeddings)         # (B, C)
        return logits

    def get_config_optim(self, lr: float, lrp: float) -> list[dict]:
        """
        Return optimizer parameter groups with backbone / GCN split learning rates.

        Args:
            lr (float): Base learning rate for GCN layers.
            lrp (float): Multiplier for backbone learning rate.

        Returns:
            list[dict]: Optimizer parameter groups.
        """
        return [
            {"params": self.features.parameters(), "lr": lr * lrp},
            {"params": self.gc1.parameters(), "lr": lr},
            {"params": self.gc2.parameters(), "lr": lr},
        ]


def gcn_swin_t(
    num_classes: int,
    t: float,
    pretrained: bool = True,
    adj_file: str = None,
    in_channel: int = 512,
    inp_file: str = None,
) -> GCNSwin:
    """
    Construct a GCNSwin with a Swin-Tiny backbone.

    Args:
        num_classes (int): Number of output classes.
        t (float): Co-occurrence threshold for adjacency construction.
        pretrained (bool): Load ImageNet weights if True.
        adj_file (str, optional): Path to adjacency pickle file.
        in_channel (int): Label embedding dimension.
        inp_file (str, optional): Path to label embedding .npy file.

    Returns:
        GCNSwin: Initialized model with Swin-Tiny backbone.
    """
    backbone = get_swin_backbone(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        out_dim=2048,
    )
    return GCNSwin(
        backbone,
        num_classes,
        t=t,
        adj_file=adj_file,
        in_channel=in_channel,
        inp_file=inp_file,
    )


def gcn_swin_b(
    num_classes: int,
    t: float,
    pretrained: bool = True,
    adj_file: str = None,
    in_channel: int = 512,
    inp_file: str = None,
) -> GCNSwin:
    """
    Construct a GCNSwin with a Swin-Base backbone.

    Args:
        num_classes (int): Number of output classes.
        t (float): Co-occurrence threshold for adjacency construction.
        pretrained (bool): Load ImageNet weights if True.
        adj_file (str, optional): Path to adjacency pickle file.
        in_channel (int): Label embedding dimension.
        inp_file (str, optional): Path to label embedding .npy file.

    Returns:
        GCNSwin: Initialized model with Swin-Base backbone.
    """
    backbone = get_swin_backbone(
        "swin_base_patch4_window7_224",
        pretrained=pretrained,
        out_dim=2048,
    )
    return GCNSwin(
        backbone,
        num_classes,
        t=t,
        adj_file=adj_file,
        in_channel=in_channel,
        inp_file=inp_file,
    )