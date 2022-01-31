import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms.functional import *
import random

from models.encoder2 import EncoderModule
from models.backbone import Backbone, NoBackbone
#from models.utils.losses import Attention_penalty_factor

from models.positional_encoding import PositionalEncodingSin
from models.unet import UNet
from models.layers.layers import *
from models.utils.distance_loss import Distance_Matrix2D

# from fastai.vision.all import *

__all__ = ['ARViT2D']

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2) 


class ARViT2D(nn.Module):
    def __init__(self, num_encoder_layers = 6, nhead=8, num_classes = 4, mask=None, pos_enc = "sin", batch_size=10, in_chans=3, hidden_dim=768, image_h=256, image_w=256, grid_l=16, gm_patch = 32, use_patches=True, attn_layer = 1,penalty_factor="2", alpha=4, beta=500, gamma=0.1):
        super().__init__()
        
        self.bs = batch_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pos_enc = pos_enc
        self.f_map_h = image_h//grid_l
        self.f_map_w = image_w//grid_l
        self.use_patches = use_patches
        
        self.pm = nn.Parameter(Distance_Matrix2D(bs=batch_size, width=image_w, height=image_h, grid_l=grid_l, penalty_factor=penalty_factor, alpha=alpha, beta=beta, gamma=gamma).pm,requires_grad=False)
        
        #self.backbone = ConvPatchEmbed(img_size=image_h, patch_size=grid_l, in_chans=in_chans, embed_dim=hidden_dim)
        self.backbone = PatchEmbed(img_size=image_h, patch_size=grid_l, in_chans=in_chans, embed_dim=hidden_dim)
        self.num_patches = self.backbone.num_patches
    
        self.encoder = EncoderModule(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers)
      
        #self.mlp = Mlp(in_features=self.hidden_dim * self.f_map_h * self.f_map_w, hidden_features=hidden_dim, out_features=self.num_classes)
        
        #self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.norm = LayerNorm(hidden_dim)
        #self.head = nn.Linear(hidden_dim, self.num_classes)
        
        self.head = nn.Linear(self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

        self.pos = nn.Parameter(pos_emb(batch_size, hidden_dim, self.f_map_h, self.f_map_w),requires_grad=False)
        self.mask = nn.Parameter(create_mask(batch_size, self.f_map_h, self.f_map_w),requires_grad=False)
        
        self.avgPool = nn.AvgPool2d(1, stride=1)
        #self.attn_map = nn.Parameter(torch.ones(batch_size, 64, 64))
            
        
    def paramsToUpdate(self, rg):

        fb = ["mask","pos","pm"]
        for name, p in self.named_parameters(): 
            if name not in fb:
                p.requires_grad_(rg)
            else:
                p.requires_grad_(False)
                
    def rescale(self, sattn):

        bs = sattn.shape[0]
        R = 256       
            
        attn = sattn.reshape(bs, R*R)
        low = attn.min(1, keepdim=True)[0]
        high = attn.max(1, keepdim=True)[0]
        attn = attn-low
        attn = attn/(high-low)
        attn = attn.reshape(bs,R,R)

        return attn
            
    def forward(self, inputs, mask=Optional[Tensor], pos=Optional[Tensor], penalty_mask=Optional[Tensor]):
                     
        bs = inputs.shape[0] 
        pm = self.pm
        if pm.shape[0]>=bs:
            pm = pm[:bs] 
        mask = self.mask
        #print(inputs.shape)
        if self.use_patches == False:
            h = self.backbone(inputs)
        else:
            h = self.backbone(inputs).permute(0,2,1)
            #print(h.shape)
            h = h.reshape(bs, self.hidden_dim, self.f_map_h, self.f_map_w)

        # used fixed sin positional encoding
        pos = self.pos
        if pos.shape[0] != h.shape[0]:
            pos = pos[0:h.shape[0],...]
            mask = mask[0:h.shape[0],...]

            
        #print(h.shape)
        att, sattn = self.encoder(src= 1*h, mask=mask, pos_embed=0.3*pos)
        #reduced_attn = self.rescale(sattn[self.attn_layer])
        #reduced_attn = self.rescale(sattn[1])
        #reduced_attn = sattn[1]
        reduced_attn = list(map(self.rescale, sattn))

        x = att
        
        x = self.norm(x)
        
        x = x.flatten(1)
        x = self.head(x)
        
        #x = self.global_pooling(x)
        
        #x = self.head( x.view(x.size(0), -1) )

        return [x, reduced_attn, sattn, pm]
    

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
    
def create_mask(batch_size, f_map_h, f_map_w):
    mask = torch.zeros((batch_size, f_map_h, f_map_w), dtype=torch.bool)
    return mask

