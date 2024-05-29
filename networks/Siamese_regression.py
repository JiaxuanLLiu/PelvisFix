import torch
import torch.nn as nn
import argparse

from networks.Resnet3D import generate_ResNet
from networks.VGG3D import VGG16
from networks.Transformer3D import Transformer3D

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=528, help="channels of feature map")
parser.add_argument("--dropout", type=int, default=0.1, help="dropout")
parser.add_argument("--nheads", type=int, default=8, help="number of heads")
parser.add_argument("--dim_feedforward", type=int, default=2048, help="dim_feedforwar")
parser.add_argument("--enc_layers", type=int, default=2, help="encorder layers")
parser.add_argument("--dec_layers", type=int, default=2, help="decorder layers")
parser.add_argument("--pre_norm", type=bool, default=False, help="")

opt = parser.parse_args()

class Siamese(nn.Module):
    def __init__(self, input_shape):
        super(Siamese, self).__init__()

        # model_depth = 18  # Resnet18
        # self.Resnet = generate_ResNet(model_depth)
        # # self.fully_connect1 = torch.nn.Linear(8192, 512)  # resnet18
        # # self.fully_connect2 = torch.nn.Linear(512, 64)
        # # # regression
        # # self.fully_connect3 = torch.nn.Linear(512, 1)
        # self.Projection = nn.Sequential(
        #     # torch.nn.Linear(131072, 512),
        #     torch.nn.Linear(512, 512),
        #     torch.nn.Linear(512, 64)
        # )
        # self.Regression_abs = nn.Sequential(
        #     # torch.nn.Linear(131072, 512),
        #     torch.nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.5),
        #     torch.nn.Linear(512, 1)
        # )
        # self.Regression_score = nn.Sequential(
        #     # torch.nn.Linear(131072, 512),
        #     torch.nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     torch.nn.Linear(512, 1)
        # )

        self.VGG3D = VGG16()
        del self.VGG3D.avgpool
        del self.VGG3D.classifier
        # # self.fully_connect1 = torch.nn.Linear(131072, 512)  # VGG3D
        # self.fully_connect1 = torch.nn.Linear(32768, 512)  # VGG3D
        # self.fully_connect2 = torch.nn.Linear(512, 64)
        # # regression
        # self.fully_connect3 = torch.nn.Linear(512, 1)

        ### 需要两次映射
        self.Projection1 = nn.Sequential(
            # torch.nn.Linear(131072, 512),
            torch.nn.Linear(32768, 512),
        )
        self.Projection2 = nn.Sequential(
            torch.nn.Linear(512, 64)
        )

        self.Regression_abs = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            torch.nn.Linear(512, 1)
        )

        self.Regression_score = nn.Sequential(
            # torch.nn.Linear(131072, 512),
            torch.nn.Linear(32768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            torch.nn.Linear(512, 1)
        )

        # self.backbone = build_backbone(opt)
        # self.transform = build_transformer(opt)
        # self.spineTransform = SpineTR(backbone=self.backbone, transformer=self.transform, num_classes=2, num_queries=1)
        # self.fully_connect1 = torch.nn.Linear(33792, 512)
        # self.fully_connect2 = torch.nn.Linear(512, 64)
        # self.fully_connect3 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        ###################### 提取特征 ####################
        # x1 = self.Resnet(x1)
        # x2 = self.Resnet(x2)

        x1 = self.VGG3D.features(x1)
        x2 = self.VGG3D.features(x2)

        # x1 = self.transform3D(x1)
        # x2 = self.transform3D(x2)

        # x1 = self.spineTransform(x1)
        # x2 = self.spineTransform(x2)

        # print("feature1", x1.shape)
        # print("feature2", x2.shape)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        # print("x1", x1.shape)
        ###################### Projection ####################
        # x1_p = self.fully_connect1(x1)
        # # x1_p = nn.Dropout(0.5)(x1_p)
        # # x1_p = nn.ReLU(inplace=True)(x1_p)
        # x1_p = self.fully_connect2(x1_p)
        x1_p_1 = self.Projection1(x1)
        x1_p_2 = self.Projection2(x1_p_1)

        # x2_p = self.fully_connect1(x2)
        # # x2_p = nn.Dropout(0.5)(x2_p)
        # # x2_p = nn.ReLU(inplace=True)(x2_p)
        # x2_p = self.fully_connect2(x2_p)
        x2_p_1 = self.Projection1(x2)
        x2_p_2 = self.Projection2(x2_p_1)

        ###################### Regression similarty #################
        x_abs = torch.abs(x1_p_1 - x2_p_1)
        # x_abs = self.fully_connect1(x_abs)
        # x_abs = nn.Dropout(0.5)(x_abs)
        # x_abs = nn.ReLU(inplace=True)(x_abs)
        # x_abs = self.fully_connect3(x_abs)
        x_abs = self.Regression_abs(x_abs)

        ###################### Regression type #################
        # x1_t = self.fully_connect1(x1)
        # x1_t = nn.Dropout(0.5)(x1_t)
        # x1_t = nn.ReLU(inplace=True)(x1_t)
        # x1_t = self.fully_connect3(x1_t)
        # x1_t = torch.sigmoid(x1_t)  ##将输出范围映射到0和1之间
        x1_t = self.Regression_score(x1)

        # x2_t = self.fully_connect1(x2)
        # x2_t = nn.Dropout(0.5)(x2_t)
        # x2_t = nn.ReLU(inplace=True)(x2_t)
        # x2_t = self.fully_connect3(x2_t)
        # x2_t = torch.sigmoid(x2_t)  ##将输出范围映射到0和1之间
        x2_t = self.Regression_score(x2)

        return x1_p_2, x2_p_2, x_abs, x1_t, x2_t

    # import torch
    # import torch.nn as nn
    # import argparse
    #
    # from networks.Resnet3D import generate_ResNet
    # from networks.VGG3D import VGG16
    # from networks.Transformer3D import Transformer3D
    #
    # from networks.SpineTransformer.Spine_Transformers import SpineTR
    # from networks.SpineTransformer.backbone import build_backbone
    # from networks.SpineTransformer.transformer import build_transformer
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--hidden_dim", type=int, default=528, help="channels of feature map")
    # parser.add_argument("--dropout", type=int, default=0.1, help="dropout")
    # parser.add_argument("--nheads", type=int, default=8, help="number of heads")
    # parser.add_argument("--dim_feedforward", type=int, default=2048, help="dim_feedforwar")
    # parser.add_argument("--enc_layers", type=int, default=2, help="encorder layers")
    # parser.add_argument("--dec_layers", type=int, default=2, help="decorder layers")
    # parser.add_argument("--pre_norm", type=bool, default=False, help="")
    #
    # opt = parser.parse_args()
    #
    # class Siamese(nn.Module):
    #     def __init__(self, input_shape):
    #         super(Siamese, self).__init__()
    #
    #         # model_depth = 18  # Resnet18
    #         # self.Resnet = generate_ResNet(model_depth)
    #         # self.fully_connect1 = torch.nn.Linear(8192, 512)  # resnet18
    #         # self.fully_connect2 = torch.nn.Linear(512, 64)
    #         # regression
    #         # self.fully_connect3 = torch.nn.Linear(512, 1)
    #
    #         self.VGG3D = VGG16()
    #         del self.VGG3D.avgpool
    #         del self.VGG3D.classifier
    #         # self.fully_connect1 = torch.nn.Linear(131072, 512)  # VGG3D
    #         self.fully_connect1 = torch.nn.Linear(32768, 512)  # VGG3D
    #         self.fully_connect2 = torch.nn.Linear(512, 64)
    #         # regression
    #         self.fully_connect3 = torch.nn.Linear(512, 1)
    #
    #         self.backbone = build_backbone(opt)
    #         self.transform = build_transformer(opt)
    #         self.spineTransform = SpineTR(backbone=self.backbone, transformer=self.transform, num_classes=2,
    #                                       num_queries=1)
    #         self.fully_connect1 = torch.nn.Linear(33792, 512)
    #         self.fully_connect2 = torch.nn.Linear(512, 64)
    #         self.fully_connect3 = torch.nn.Linear(512, 1)
    #
    #     def forward(self, x):
    #         x1, x2 = x
    #         ###################### 提取特征 ####################
    #         # x1 = self.Resnet(x1)
    #         # x2 = self.Resnet(x2)
    #
    #         # x1 = self.VGG3D.features(x1)
    #         # x2 = self.VGG3D.features(x2)
    #
    #         # x1 = self.transform3D(x1)
    #         # x2 = self.transform3D(x2)
    #
    #         x1 = self.spineTransform(x1)
    #         x2 = self.spineTransform(x2)
    #
    #         # print("feature1", x1.shape)
    #         # print("feature2", x2.shape)
    #
    #         x1 = torch.flatten(x1, 1)
    #         x2 = torch.flatten(x2, 1)
    #         # print("x1", x1.shape)
    #         ###################### Projection ####################
    #         x1_p = self.fully_connect1(x1)
    #         # x1_p = nn.Dropout(0.5)(x1_p)
    #         # x1_p = nn.ReLU(inplace=True)(x1_p)
    #         x1_p = self.fully_connect2(x1_p)
    #
    #         x2_p = self.fully_connect1(x2)
    #         # x2_p = nn.Dropout(0.5)(x2_p)
    #         # x2_p = nn.ReLU(inplace=True)(x2_p)
    #         x2_p = self.fully_connect2(x2_p)
    #
    #         ###################### Regression similarty #################
    #         x_abs = torch.abs(x1 - x2)
    #         x_abs = self.fully_connect1(x_abs)
    #         x_abs = nn.Dropout(0.5)(x_abs)
    #         x_abs = nn.ReLU(inplace=True)(x_abs)
    #         x_abs = self.fully_connect3(x_abs)
    #
    #         ###################### Regression type #################
    #         x1_t = self.fully_connect1(x1)
    #         x1_t = nn.Dropout(0.5)(x1_t)
    #         x1_t = nn.ReLU(inplace=True)(x1_t)
    #         x1_t = self.fully_connect3(x1_t)
    #         x1_t = torch.sigmoid(x1_t)  ##将输出范围映射到0和1之间
    #
    #         x2_t = self.fully_connect1(x2)
    #         x2_t = nn.Dropout(0.5)(x2_t)
    #         x2_t = nn.ReLU(inplace=True)(x2_t)
    #         x2_t = self.fully_connect3(x2_t)
    #         x2_t = torch.sigmoid(x2_t)  ##将输出范围映射到0和1之间
    #
    #         return x1_p, x2_p, x_abs, x1_t, x2_t