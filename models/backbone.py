#from PIL import Image
import requests
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torchvision.models import resnet50, resnet101
import torchvision.transforms as T

from .positional_encoding import PositionalEncodingSin

class Backbone(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # create ResNet-101 backbone
        self.backbone = resnet101()
        del self.backbone.layer4
        del self.backbone.avgpool
        del self.backbone.fc
        # create conversion layer
        self.conv = nn.Conv2d(1024, hidden_dim, 1)

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        # convert from 1024 to 256 feature planes for the transformer
        h = self.conv(x)
        shape = h.shape
        pos = PositionalEncodingSin.positionalencoding2d(shape[0],self.hidden_dim,shape[2],shape[3])
        
        return h, pos

class EmbeddingNetwork(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # create ResNet-101 backbone
        self.conv = nn.Conv2d(3, hidden_dim, 1)

    def forward(self, inputs):

        # convert from 3 to 256 feature planes for the transformer
        h = self.conv(inputs)
        shape = h.shape
        pos = PositionalEncodingSin.positionalencoding2d(shape[0],self.hidden_dim,shape[2],shape[3])
        
        return h, pos

class NoBackbone(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # create conversion layer
        self.conv = nn.Conv2d(3, hidden_dim, 2)

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # convert from 1024 to 256 feature planes for the transformer
        h = self.conv(inputs)
        shape = h.shape
        pos = PositionalEncodingSin.positionalencoding2d(shape[0],self.hidden_dim,shape[2],shape[3])
        
        return h, pos