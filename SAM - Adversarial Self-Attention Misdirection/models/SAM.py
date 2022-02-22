import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms.functional import *
import random

from models.unet import UNet
from models.ARViT import ARViT
from models.layers.layers import *

# from fastai.vision.all import *

__all__ = ['SAM']


class SAM(nn.Module):
    def __init__(self, enc_Layers=6, nhead=8, nclass=4, bs=90, hidden_dim=512, H=256, W=256, grid_l=16, gm_patch=16):
        super().__init__()
        
        #self.generator = UNet(in_channels,gen_classes,bilinear)
        self.model = ARViT(num_encoder_layers = enc_Layers, nhead=nhead, num_classes = nclass, batch_size=bs, hidden_dim=hidden_dim, image_h=H, image_w=W, grid_l=grid_l, gm_patch=gm_patch)
        
            
    def assertParams(self, rg):
        if rg == True:
            for name, param in self.model.named_parameters():
                fb = ["mask","pos"]
                if name not in fb:
                    assert param.requires_grad == True
                else:
                    assert param.requires_grad == False                 
        else:
            for name, param in self.model.named_parameters():
                assert param.requires_grad == False
    
    
    def paramsToUpdate(self, rg):
        if rg == False:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            fb = ["mask","pos"]
            for name, p in self.model.named_parameters(): 
                if name not in fb:
                    p.requires_grad_(True)         
        self.assertParams(rg)
            
    def forward(self, inputs):        
        
        if type(inputs) is tuple:
            #print("tuple",inputs[0].shape)
            input0=inputs[0]
        else:
            #print("single",inputs.shape)
            input0=inputs
        #print("input0", input0.shape)
        x, reduced_attn, sattn, gm = self.model(input0)
        
        return [x, reduced_attn, sattn, gm]
    

    

