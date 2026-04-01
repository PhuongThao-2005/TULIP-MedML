import torchvision.models as models
from torch.nn import Parameter
from src.util import *
import torch
import torch.nn as nn
from src.models.backbone import SwinBackbone, get_swin_backbone

class GraphConvolution(nn.Module):
    """
    Simple GCN layer — https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None, inp_file=None):
        super(GCNResnet, self).__init__()
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

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        if inp_file:
            inp = torch.from_numpy(
                np.load(inp_file).astype(np.float32))  # (14, 300)
        else:
            inp = torch.zeros(num_classes, in_channel)
        self.register_buffer('inp', inp)

    def forward(self, feature):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)   # (B, 2048)

        adj = gen_adj(self.A).detach()
        x = self.gc1(self.inp, adj)    # (C, 1024)
        x = self.relu(x)
        x = self.gc2(x, adj)      # (C, 2048)
        x = self.relu(x)

        x = x.transpose(0, 1)           # (2048, C)
        x = torch.matmul(feature, x)    # (B, C)
        return x

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]


def gcn_resnet101(num_classes, t, pretrained=False, adj_file=None, in_channel=300, inp_file=None):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel, inp_file=inp_file)

class GCNSwin(nn.Module):
    def __init__(self, backbone, num_classes, in_channel=768, t=0,
                 adj_file=None, inp_file=None):
        super().__init__()
        self.features = backbone          # SwinBackbone → (B, 2048)
        self.num_classes = num_classes

        # GCN branch: word-vec → class embeddings
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        inp = (torch.from_numpy(np.load(inp_file).astype(np.float32))
               if inp_file else torch.zeros(num_classes, in_channel))
        self.register_buffer('inp', inp)

    def forward(self, x):
        feature = self.features(x)           # (B, 2048)  ← backbone đã pool+project

        adj = gen_adj(self.A).detach()
        z = self.relu(self.gc1(self.inp, adj))   # (C, 1024)
        z = self.relu(self.gc2(z, adj))          # (C, 2048)

        z = z.transpose(0, 1)                    # (2048, C)
        return torch.matmul(feature, z)          # (B, C)

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]
        
def gcn_swin_t(num_classes, t, pretrained=True, adj_file=None, in_channel=768, inp_file=None):
    backbone = get_swin_backbone(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        out_dim=2048,         # projection head align với GCN
    )
    return GCNSwin(backbone, num_classes, t=t, adj_file=adj_file, in_channel=in_channel, inp_file=inp_file)