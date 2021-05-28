import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone
from models.utils.losses import Attention_penalty_factor

from models.positional_encoding import PositionalEncodingSin
from models.backbone import Backbone
from models.encoder import EncoderModule
from models.unet import UNet

# from fastai.vision.all import *

__all__ = ['Joiner', 'create_mask', 'penalty_mask', 'pos_emb', 'GAN']


class GAN(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False, num_encoder_layers = 6, nhead=1, backbone = False, num_classes = 10, bypass=False, mask=None, pos_enc = "sin", batch_size=10, hidden_dim=256, image_h=32, image_w=32, grid_l=4, penalty_factor="1", alpha=1):
        super().__init__()
        self.generator = UNet(n_channels,n_classes,bilinear)
        self.model = Joiner(num_encoder_layers, nhead, backbone, num_classes, bypass, mask, pos_enc, batch_size, hidden_dim, image_h, image_w, grid_l, penalty_factor, alpha)
        self.noise_mode = True
        self.generator_mode = True
        
    def switcher():
        if self.noise_mode == True:
            self.noise_mode = False
        else:
            self.noise_mode = True
        
    def forward(self, inputs):
        
        input0 = inputs
        if self.noise_mode == True:
            inputs = self.generator(inputs)
            
        outputs, sattn, pattn = self.model(inputs)
        
        return [outputs, sattn, pattn, input0, inputs]
        
        
class Joiner(nn.Module):
    def __init__(self, num_encoder_layers = 6, nhead=1, backbone = False, num_classes = 10, bypass=False, mask=None, pos_enc = "sin", batch_size=10, hidden_dim=256, image_h=32, image_w=32, grid_l=4, penalty_factor="1", alpha=1):
        super().__init__()
        
        self.bypass = bypass
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pos_enc = pos_enc
        if backbone == True:
            self.f_map_h = image_h//16 
            self.f_map_w = image_w//16
            self.backbone = Backbone(hidden_dim=hidden_dim)
    
        self.fc1 = nn.Linear(self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)
            
            
    def forward(self, inputs, mask=Optional[Tensor], pos=Optional[Tensor], penalty_mask=Optional[Tensor]):
        #print("Passing through the model")
        #print(inputs.shape)
        h = self.backbone(inputs)

        x = h.flatten(1)

        x = self.fc1(x)
        #x = self.fc2(x)


        return [x, x, x]


def create_mask(batch_size, f_map_h, f_map_w):
    mask = torch.zeros((batch_size, f_map_h, f_map_w), dtype=torch.bool)
    return mask

def penalty_mask(batch_size, f_map_w, f_map_h, grid_l=4, penalty_factor="1", alpha=1):
    dist_matrix = Attention_penalty_factor.distance_matrix(batch_size, f_map_w, f_map_h, grid_l)
    grids_matrix = Attention_penalty_factor.grids_matrix(batch_size, f_map_w, f_map_h, grid_l)
    pf_matrix = Attention_penalty_factor.penalty_factor(dist_matrix, penalty_factor, alpha)

    penalty_mask = Attention_penalty_factor.penalty_matrix(batch_size, f_map_w, f_map_h, grids_matrix, pf_matrix, grid_l)
    
    return penalty_mask

def pos_emb(batch_size, hidden_dim, f_map_h, f_map_w):
    pos = PositionalEncodingSin.positionalencoding2d(batch_size,hidden_dim, f_map_h, f_map_w)
    return pos

def irregular_batch_size(testMask,testBatch):  
    mask_one = testMask[0]
    bs = testBatch.shape[0]
    bMask = []
    for b in range(bs):
        bMask.append(mask_one)
    d = torch.Tensor(testBatch.shape)
    newMask = torch.cat(bMask, out=d)
    newMask = newMask.view(testBatch.shape)
    return newMask
    
    

