import torch
import torch.nn as nn
import argparse

from networks.Resnet3D import generate_ResNet
from networks.VGG3D import VGG16
from networks.Transformer3D import Transformer3D
from networks.SpineTransformer.Spine_Transformers import SpineTR
from networks.SpineTransformer.backbone import build_backbone
from networks.SpineTransformer.transformer import build_transformer

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

        # model_depth = 10  # Resnet34
        # self.Resnet = generate_ResNet(model_depth)

        self.VGG3D = VGG16()
        del self.VGG3D.avgpool
        del self.VGG3D.classifier
        # self.fully_connect1 = torch.nn.Linear(262144, 1024) #VGG3D
        # self.fully_connect2 = torch.nn.Linear(1024, 128)
        # classifier
        # self.fully_connect3 = nn.Linear(262144, 2, bias=True)

        # self.transform3D = Transformer3D(img_shape=input_shape)
        # self.fully_connect1 = torch.nn.Linear(262144, 1024)
        # self.fully_connect2 = torch.nn.Linear(1024, 128)
        # classifier
        # self.fully_connect3 = nn.Linear(262144, 2, bias=True)


        self.backbone = build_backbone(opt)
        self.transform = build_transformer(opt)
        self.spineTransform = SpineTR(backbone=self.backbone, transformer=self.transform, num_classes=2, num_queries=1)
        self.fully_connect1 = torch.nn.Linear(33792, 512)
        self.fully_connect2 = torch.nn.Linear(512, 128)
        self.fully_connect3 = torch.nn.Linear(512, 2)

    def forward(self, x):
        x1, x2 = x
        ###################### 提取特征 ####################
        # x1 = self.Resnet(x1)
        # x2 = self.Resnet(x2)
        # print("x1", x1.shape)

        # x1 = self.VGG3D.features(x1)
        # x2 = self.VGG3D.features(x2)
        # print("x1", x1.shape)
        #
        # x1 = self.transform3D(x1)
        # x2 = self.transform3D(x2)
        x1 = self.spineTransform(x1)
        x2 = self.spineTransform(x2)
        # print("x1", x1.shape)

        # print("feature1", x1.shape)
        # print("feature2", x2.shape)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        # print("x1", x1.shape)
        ###################### Projection ####################
        x1_p = self.fully_connect1(x1)
        x1_p = nn.ReLU(inplace=True)(x1_p)
        x1_p = self.fully_connect2(x1_p)

        x2_p = self.fully_connect1(x2)
        x2_p = nn.ReLU(inplace=True)(x2_p)
        x2_p = self.fully_connect2(x2_p)

        ###################### Classfication #################
        x1_c = self.fully_connect1(x1)
        x1_c = self.fully_connect3(x1_c)

        return x1_p, x2_p, x1_c