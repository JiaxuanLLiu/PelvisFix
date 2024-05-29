import torch
import torch.nn as nn
from networks.NonLocal_self_attention import NONLocalBlock3D
from torch.nn import functional as F

class Structure_Contrast_module(nn.Module):
    def __init__(self, features, in_channels=1, num_classes=2, batch_normal=True):
        super(Structure_Contrast_module, self).__init__()
        self.features = features

        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.num_classes = num_classes

    def forward(self, x):
        # print("x", x.shape)
        x = self.features(x)
        return x

# cfgs = {'D':[64, 64, 'M', 128, 128, 'M', 'S2', 256, 256, 'M', 'S3', 512, 512, 'M']}
cfgs = {'D':['S2', 256, 'M']}

#--------------------------------------#
#   特征提取部分
#--------------------------------------#
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        elif v == 'S1':
            layers += [NONLocalBlock3D(in_channels=64, sub_sample=False, bn_layer=False)]
        elif v == 'S2':
            layers += [NONLocalBlock3D(in_channels=128)]
        elif v == 'S3':
            layers += [NONLocalBlock3D(in_channels=256)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def Structure_Contrast(**kwargs):
    feature = Structure_Contrast_module(make_layers(cfgs["D"], batch_norm=True), **kwargs)
    return feature