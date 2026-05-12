"""
File: addgcn.py
Description:
    ADD-GCN (Attention-Driven Dynamic Graph Convolutional Network) for
    multi-label image classification. Combines a CNN backbone with a
    dynamic graph convolution module that builds label co-occurrence graphs
    from image features at inference time.
Main Components:
    - DynamicGraphConvolution: GCN layer with a static branch (learned adjacency)
      and a dynamic branch (image-driven adjacency constructed per forward pass).
    - ADDGCN: Full model combining ResNet-101 backbone, semantic-aware masking
      (SAM), and DynamicGraphConvolution for multi-label prediction.
    - addgcn_resnet101: Factory function for the standard ResNet-101 variant.
Inputs:
    images (B, 3, H, W)
Outputs:
    Two logit tensors (B, C) from the global and graph branches respectively.
Notes:
    - The final prediction is typically (out1 + out2) / 2.
    - mask_mat is initialized as the identity matrix; the diagonal extraction
      in forward() avoids a potentially fragile broadcast kernel.
"""

import torch
import torch.nn as nn
import torchvision


class DynamicGraphConvolution(nn.Module):
    """
    Graph convolution layer with both static and dynamic adjacency branches.

    The static branch learns a fixed adjacency from training (label co-occurrence
    prior). The dynamic branch constructs a per-image adjacency by comparing
    each node's features against a global context vector extracted from the batch.

    Args:
        in_features (int): Input feature dimension per node (D_in).
        out_features (int): Output feature dimension per node (D_out).
        num_nodes (int): Number of graph nodes (equals num_classes).
    """

    def __init__(self, in_features: int, out_features: int, num_nodes: int):
        super().__init__()

        # Static branch: learn a fixed label-pair affinity matrix
        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2),
        )

        # Dynamic branch: build per-image adjacency from global context
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        # Concat local + global → dynamic adjacency (B, num_nodes, num_nodes)
        self.conv_create_adj = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Propagate node features through the static (learned) adjacency.

        Args:
            node_features (Tensor): Node feature matrix.

        Returns:
            Tensor: Updated node features after static graph propagation.

        Shape:
            node_features: (B, D_in, C)
            output:        (B, D_out, C)
        """
        # Apply learned adjacency along the node dimension
        x = self.static_adj(node_features.transpose(1, 2))  # (B, C, C)
        x = self.static_weight(x.transpose(1, 2))            # (B, D_out, C)
        return x

    def forward_construct_dynamic_adj(
        self, node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Build a dynamic (image-conditioned) adjacency matrix from node features.

        Global context is computed by average-pooling all node features, then
        broadcast and concatenated with local node features to predict pairwise
        affinities via sigmoid.

        Args:
            node_features (Tensor): Current node feature matrix.

        Returns:
            Tensor: Soft adjacency matrix (values in [0, 1]).

        Shape:
            node_features: (B, D_in, C)
            output:        (B, C, C)
        """
        # Compute global context by pooling across nodes
        global_context = self.global_avg_pool(node_features)     # (B, D_in, 1)
        global_context = self.conv_global(global_context)
        global_context = self.bn_global(global_context)
        global_context = self.relu(global_context)
        global_context = global_context.expand_as(node_features)  # (B, D_in, C)

        # Concatenate global and local features, then predict adjacency
        combined = torch.cat((global_context, node_features), dim=1)  # (B, 2*D_in, C)
        dynamic_adj = self.conv_create_adj(combined)                   # (B, C, C)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(
        self,
        node_features: torch.Tensor,
        dynamic_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Propagate node features through the dynamic adjacency matrix.

        Args:
            node_features (Tensor): Node feature matrix.
            dynamic_adj (Tensor): Per-image soft adjacency matrix.

        Returns:
            Tensor: Updated node features.

        Shape:
            node_features: (B, D_in, C)
            dynamic_adj:   (B, C, C)
            output:        (B, D_out, C)
        """
        x = torch.matmul(node_features, dynamic_adj)  # (B, D_in, C)
        x = self.relu(x)
        x = self.dynamic_weight(x)                    # (B, D_out, C)
        x = self.relu(x)
        return x

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: static GCN residual + dynamic GCN refinement.

        Args:
            node_features (Tensor): Input node feature matrix.

        Returns:
            Tensor: Refined node features.

        Shape:
            node_features: (B, D_in, C)
            output:        (B, D_out, C)
        """
        # Static branch: residual update from learned adjacency
        static_out = self.forward_static_gcn(node_features)  # (B, D_out, C)
        node_features = node_features + static_out

        # Dynamic branch: build adjacency from current features, then propagate
        dynamic_adj = self.forward_construct_dynamic_adj(node_features)  # (B, C, C)
        node_features = self.forward_dynamic_gcn(node_features, dynamic_adj)  # (B, D_out, C)

        return node_features


class ADDGCN(nn.Module):
    """
    Attention-Driven Dynamic GCN multi-label classifier.

    Combines a ResNet backbone with:
      1. A global branch that predicts logits from top-1 spatial pooling.
      2. A SAM (Semantic Attention Module) branch that projects image features
         into class-conditioned node representations for graph reasoning.

    Args:
        backbone: A torchvision ResNet instance (used as a feature extractor).
        num_classes (int): Number of output classes (C).
    """

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()

        # Use ResNet layers as a shared feature extractor (drop FC head)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.num_classes = num_classes

        # Class activation maps: (B, C, H', W') spatial logits
        self.fc = nn.Conv2d(backbone.fc.in_features, num_classes, 1, bias=False)

        # Project image features from 2048-d to 1024-d for GCN input
        self.conv_transform = nn.Conv2d(2048, 1024, 1)
        self.relu = nn.LeakyReLU(0.2)

        # Dynamic graph convolution operating on class-conditioned node features
        self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)

        # Diagonal selector: initialized as identity — equivalent to out2[b, c, c].sum()
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())

        # Final per-class prediction from GCN output: (B, 1024, C) → (B, C)
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)

    def forward_feature(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps from the ResNet backbone.

        Args:
            images (Tensor): Input images.

        Returns:
            Tensor: Spatial feature maps.

        Shape:
            images: (B, 3, H, W)
            output: (B, 2048, H', W')
        """
        return self.features(images)

    def forward_classification_sm(
        self, image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute global branch logits via top-1 spatial pooling over class maps.

        Args:
            image_features (Tensor): Spatial feature maps from backbone.

        Returns:
            Tensor: Logits from global branch.

        Shape:
            image_features: (B, 2048, H', W')
            output:         (B, C)
        """
        # Generate spatial class activation maps
        class_maps = self.fc(image_features)       # (B, C, H', W')
        class_maps = class_maps.flatten(2)          # (B, C, H'*W')
        # Top-1 pooling: take maximum activation per class, then average over batch
        logits = class_maps.topk(1, dim=-1)[0].mean(dim=-1)  # (B, C)
        return logits

    def forward_sam(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Compute class-conditioned node features via Semantic Attention Module.

        Each class node aggregates image features weighted by its spatial attention
        map, producing a class-specific representation for graph reasoning.

        Args:
            image_features (Tensor): Spatial feature maps from backbone.

        Returns:
            Tensor: Class-conditioned node features for GCN input.

        Shape:
            image_features: (B, 2048, H', W')
            output:         (B, 1024, C)
        """
        # Compute soft spatial attention weights per class
        attention = self.fc(image_features)               # (B, C, H', W')
        attention = attention.flatten(2)                   # (B, C, H'*W')
        attention = torch.sigmoid(attention)
        attention = attention.transpose(1, 2).contiguous()  # (B, H'*W', C)

        # Project image features to 1024-d before aggregation
        node_features = self.conv_transform(image_features)   # (B, 1024, H', W')
        node_features = node_features.flatten(2).contiguous() # (B, 1024, H'*W')

        # Aggregate: weighted sum of spatial positions for each class node
        node_features = torch.matmul(node_features, attention)  # (B, 1024, C)
        return node_features

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass producing two sets of logits for ensemble prediction.

        Args:
            images (Tensor): Input images.

        Returns:
            tuple:
                - out1 (Tensor): Global branch logits (top-1 spatial pooling).
                - out2 (Tensor): GCN branch logits (graph-refined node features).

        Shape:
            images: (B, 3, H, W)
            out1:   (B, C)
            out2:   (B, C)

        Notes:
            - Typical usage: logits = (out1 + out2) / 2.0
            - out2 is extracted via diagonal of (B, C, C) — equivalent to
              selecting the per-class GCN output without an explicit identity mask.
        """
        image_features = self.forward_feature(images)  # (B, 2048, H', W')

        # Global branch: spatial pooling over class activation maps
        out1 = self.forward_classification_sm(image_features)  # (B, C)

        # GCN branch: class-conditioned nodes refined through dynamic graph
        node_features = self.forward_sam(image_features)       # (B, 1024, C)
        node_features = self.gcn(node_features)                # (B, 1024, C)
        node_features = node_features + self.forward_sam(image_features)

        # Project each node to a scalar logit; diagonal selects class-class entries
        out2 = self.last_linear(node_features)                # (B, C, C)
        out2 = out2.diagonal(dim1=1, dim2=2).contiguous()    # (B, C)

        return out1, out2

    def get_config_optim(self, lr: float, lrp: float) -> list[dict]:
        """
        Return parameter groups for optimizer with separate learning rates.

        The backbone uses a reduced learning rate (lr * lrp) to preserve
        pretrained ImageNet features. All other parameters use the full lr.

        Args:
            lr (float): Base learning rate for GCN and head parameters.
            lrp (float): Multiplier for backbone learning rate (typically < 1).

        Returns:
            list[dict]: Optimizer parameter groups.
        """
        backbone_param_ids = set(map(id, self.features.parameters()))
        non_backbone_params = filter(
            lambda p: id(p) not in backbone_param_ids, self.parameters()
        )
        return [
            {"params": self.features.parameters(), "lr": lr * lrp},
            {"params": non_backbone_params, "lr": lr},
        ]


def addgcn_resnet101(num_classes: int, pretrained: bool = True) -> ADDGCN:
    """
    Construct an ADDGCN model with a ResNet-101 backbone.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): If True, load ImageNet-pretrained ResNet-101 weights.

    Returns:
        ADDGCN: Initialized ADDGCN model.
    """
    weights = (
        torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        if pretrained
        else None
    )
    backbone = torchvision.models.resnet101(weights=weights)
    return ADDGCN(backbone, num_classes)