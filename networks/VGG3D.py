import torch
import torch.nn as nn
from networks.NonLocal_self_attention import NONLocalBlock3D
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bath_normal=False):
        super(DoubleConv, self).__init__()
        channels = out_channels

        layers = \
        [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if bath_normal: # 如果要添加BN层
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        # 构造序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bath_normal=False):
        super(TripleConv, self).__init__()
        channels = out_channels

        layers = \
        [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if bath_normal: # 如果要添加BN层
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        # 构造序列器
        self.triple_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DownSampling, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(x)

class VGG(nn.Module):
    def __init__(self, features, in_channels=1, num_classes=2, batch_normal=True):
        super(VGG, self).__init__()
        self.features = features

        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.num_classes = num_classes

        # --------------------------------------#
        #   平均池化到7x7大小
        # --------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # print("x", x.shape)
        x = self.features(x)
        # print("x", x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] #VGG16
# cfgs = {'D':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']} #标准
cfgs = {'D':[64, 64, 'M', 128, 128, 'M', 'S2', 256, 256, 'M', 'S3', 512, 512, 'M']}
# cfgs = {'D':[64, 64,'S1', 'M', 128, 128, 'M', 256, 256, 'M']}

# #--------------------------------------#
# #   特征提取部分
# #--------------------------------------#
# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 1
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
#         else:
#             conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv3d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
# def VGG16(**kwargs):
#     model = VGG(make_layers(cfgs["D"], batch_norm = False), **kwargs)
#     return model

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

def VGG16(**kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=True), **kwargs)
    return model