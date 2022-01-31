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

# from fastai.vision.all import *

__all__ = ['Joiner', 'create_mask', 'penalty_mask', 'pos_emb', 'GAN', 'ImageNetJoiner']


class GAN(nn.Module):
    def __init__(self, in_channels=3, gen_classes=3, bilinear=False, num_encoder_layers=8, nhead=8, use_patches=True, num_classes=4, bypass=False, mask=None, pos_enc="sin", batch_size=10, hidden_dim=768, image_h=32, image_w=32, grid_l=1, penalty_factor="1", alpha=1, data="CIFAR"):
        super().__init__()
        self.generator = UNet(in_channels,gen_classes,bilinear)
        if data=="CIFAR":
            self.model = Joiner(num_encoder_layers, nhead, use_patches, num_classes, bypass, mask, pos_enc, batch_size, hidden_dim, image_h, image_w, grid_l, penalty_factor, alpha)
        else:
            self.model = ImageNetJoiner(num_encoder_layers = num_encoder_layers, nhead=nhead, num_classes = num_classes, mask=mask, pos_enc = pos_enc, batch_size=batch_size, in_chans=in_channels, hidden_dim=hidden_dim, image_h=image_h, image_w=image_w, grid_l=grid_l, use_patches=use_patches, penalty_factor=penalty_factor, alpha=alpha)
        self.noise_mode = True
        self.generator_mode = True
        self.paramsToUpdate()
        
    def generatorSwitcher(self):
        if self.generator_mode == True:
            self.generator_mode = False
        else:
            self.generator_mode = True
        
    def noiseSwitcher(self):
        if self.noise_mode == True:
            self.noise_mode = False
        else:
            self.noise_mode = True
        
        
    def assertParams(self):
        if self.generator_mode == True:
            for name, param in self.generator.named_parameters():
                assert param.requires_grad == True
            for name, param in self.model.named_parameters():
                assert param.requires_grad == False
        else:
            for name, param in self.generator.named_parameters():
                assert param.requires_grad == False
            fb = ["mask","penalty_mask","pos"]
            for name, p in self.model.named_parameters(): 
                if name not in fb:
                    p.requires_grad_(True)    
    
    
    def paramsToUpdate(self):
        if self.generator_mode == True:
            for param in self.generator.parameters():
                param.requires_grad = True
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.generator.parameters():
                param.requires_grad = False
            fb = ["mask","penalty_mask","pos"]
            for name, p in self.model.named_parameters(): 
                if name not in fb:
                    p.requires_grad_(True)
                    
            self.assertParams()
            
            
    def forward(self, inputs):        
        
        input0 = inputs
        if self.noise_mode == True:
            inputs = self.generator(inputs)
            
        outputs, sattn = self.model(inputs)
        
        return [outputs, sattn, input0, inputs]
        
        
class Joiner(nn.Module):
    def __init__(self, num_encoder_layers = 6, nhead=1, use_patches = False, num_classes = 10, bypass=False, mask=None, pos_enc = "sin", batch_size=10, hidden_dim=256, image_h=32, image_w=32, grid_l=1, penalty_factor="1", alpha=1):
        super().__init__()
        
        self.bs = batch_size
        self.bypass = bypass
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pos_enc = pos_enc
        self.use_patches = use_patches
        if use_patches == False:
            self.f_map_h = image_h 
            self.f_map_w = image_w
            self.backbone = nn.Conv2d(3, hidden_dim, 1)
        else:
            self.f_map_h = image_h 
            self.f_map_w = image_w
            self.backbone = PatchEmbed(img_size=image_h, patch_size=grid_l, in_chans=3, embed_dim=hidden_dim)
            num_patches = self.backbone.num_patches
    
        self.encoder = EncoderModule(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers)

        if self.bypass == True:
            #self.mlp = Mlp(in_features=2*self.hidden_dim*self.f_map_h*self.f_map_w, hidden_features=hidden_dim, out_features=self.num_classes)
            self.fc1 = nn.Linear(2*self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

        else:
       
            #self.mlp = Mlp(in_features=self.hidden_dim * self.f_map_h * self.f_map_w, hidden_features=hidden_dim, out_features=self.num_classes)
            self.fc1 = nn.Linear(self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

            
        self.pos = nn.Parameter(pos_emb(batch_size, hidden_dim, self.f_map_h, self.f_map_w),requires_grad=False)
        self.mask = nn.Parameter(create_mask(batch_size, self.f_map_h, self.f_map_w),requires_grad=False)
        #self.penalty_mask = nn.Parameter(penalty_mask(batch_size, self.f_map_w, self.f_map_h, grid_l, penalty_factor, alpha),requires_grad=False)
        
        #For self-supervised Learning
        self.degrees = [0,90,180,270]
        self.labelDict = {0:0,90:1,180:2,270:3}
        
        # spatial positional encodings
        if self.pos_enc == "learned":
            self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
            self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
            
        
    def paramsToUpdate(self):

        fb = ["mask","penalty_mask","pos"]
        for name, p in self.named_parameters(): 
            if name not in fb:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
            
    def forward(self, inputs, mask=Optional[Tensor], pos=Optional[Tensor], penalty_mask=Optional[Tensor]):
        
        #if len(inputs) == 2:
        #    x0 = inputs[1]
        #    inputs = inputs[0]
        #else: x0 = inputs
        
        bs = inputs.shape[0]
        
        mask = self.mask

        if self.use_patches == False:
            h = self.backbone(inputs)
        else:
            h = self.backbone(inputs).permute(0,2,1)
            h = h.reshape(bs, self.hidden_dim, self.f_map_h, self.f_map_w)

        pos = self.pos
        if pos.shape[0] != h.shape[0]:
            pos = pos[0:h.shape[0],...]
            mask = mask[0:h.shape[0],...]
            #print(pos.shape)
            #print(mask.shape)

        #print("Shape h:",h.shape)
        #print("Shape pos:",pos.shape)
        att, sattn = self.encoder(src= 1*h, mask=mask, pos_embed=0.2*pos)
        #xattn = sattn.reshape(sattn.shape[:-1] + h.shape[-2:])

        #sattn = sattn.reshape(sattn.shape[:-2] + h.shape[-2:] + h.shape[-2:])

        #sattn = sattn.permute(0,3,4,1,2)

        x = att

        x = x.flatten(1)
        x = self.fc1(x)

        return [x, sattn]
    
    
class ImageNetJoiner(nn.Module):
    def __init__(self, num_encoder_layers = 8, nhead=8, num_classes = 10, mask=None, pos_enc = "sin", batch_size=10, in_chans=3, hidden_dim=768, image_h=256, image_w=256, grid_l=8, use_patches=True, penalty_factor="1", alpha=1):
        super().__init__()
        
        self.bs = batch_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pos_enc = pos_enc
        self.f_map_h = image_h//grid_l
        self.f_map_w = image_w//grid_l
        self.use_patches = use_patches
        
        self.backbone = PatchEmbed(img_size=image_h, patch_size=grid_l, in_chans=in_chans, embed_dim=hidden_dim)
        num_patches = self.backbone.num_patches
    
        self.encoder = EncoderModule(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers)
      
        #self.mlp = Mlp(in_features=self.hidden_dim * self.f_map_h * self.f_map_w, hidden_features=hidden_dim, out_features=self.num_classes)
        self.fc1 = nn.Linear(self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

        self.pos = nn.Parameter(pos_emb(batch_size, hidden_dim, self.f_map_h, self.f_map_w),requires_grad=False)
        self.mask = nn.Parameter(create_mask(batch_size, self.f_map_h, self.f_map_w),requires_grad=False)
        
        self.avgPool = nn.AvgPool2d(4, stride=4)
        #self.attn_map = nn.Parameter(torch.ones(batch_size, 64, 64))
            
        
    def paramsToUpdate(self):

        fb = ["mask","pos"]
        for name, p in self.named_parameters(): 
            if name not in fb:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
                
    def rescale(self, sattn, bs):

        a = self.avgPool(sattn.reshape(bs,1024,32,32))
        pattn = self.avgPool(a.reshape(bs,1024,64).permute(0, 2, 1).reshape(bs,64,32,32)).permute(0, 2, 3, 1).reshape(bs,64,64)
        attn = pattn
        for i in range(bs):
            attn[i] = attn[i] - attn[i].min()
            attn[i] = attn[i]/attn[i].max()
        
        attn = attn+0.0001
        
        return attn
            
    def forward(self, inputs, mask=Optional[Tensor], pos=Optional[Tensor], penalty_mask=Optional[Tensor]):
        
        bs = inputs.shape[0]        
        #penalty_mask = self.penalty_mask
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

        att, sattn = self.encoder(src= 1*h, mask=mask, pos_embed=0.3*pos)

        x = att
        x = x.flatten(1)
        x = self.fc1(x)
        #x = self.mlp(x)
        
        #attn = self.rescale(sattn[0],bs)

        return [x, sattn]#, attn]

#To lead the old model, remove the reshape and the attn
    
    
class RotationJoiner(nn.Module):
    def __init__(self, num_encoder_layers = 6, nhead=1, use_patches = False, num_classes = 10, mask=None, batch_size=10, hidden_dim=256, image_h=32, image_w=32, grid_l=1, penalty_factor="1", alpha=1):
        super().__init__()
        
        self.labels = torch.zeros(batch_size, dtype=torch.int64)
        self.bs = batch_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_patches = use_patches
        if use_patches == False:
            self.f_map_h = image_h 
            self.f_map_w = image_w
            self.backbone = nn.Conv2d(3, hidden_dim, 1)
        else:
            self.f_map_h = image_h 
            self.f_map_w = image_w
            self.backbone = PatchEmbed(img_size=image_h, patch_size=grid_l, in_chans=3, embed_dim=hidden_dim)
            num_patches = self.backbone.num_patches
    
        self.encoder = EncoderModule(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers)

        #self.mlp = Mlp(in_features=self.hidden_dim * self.f_map_h * self.f_map_w, hidden_features=hidden_dim, out_features=self.num_classes)
        self.fc1 = nn.Linear(self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

        self.pos = nn.Parameter(pos_emb(batch_size, hidden_dim, self.f_map_h, self.f_map_w),requires_grad=False)
        self.mask = nn.Parameter(create_mask(batch_size, self.f_map_h, self.f_map_w),requires_grad=False)
        
        #For self-supervised Learning
        self.degrees = [0,90,180,270]
        self.labelDict = {0:0,90:1,180:2,270:3}                   
       
    def paramsToUpdate(self):

        fb = ["mask","penalty_mask","pos"]
        for name, p in self.named_parameters(): 
            if name not in fb:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
            
    def forward(self, inputs, mask=Optional[Tensor], pos=Optional[Tensor], penalty_mask=Optional[Tensor]):
                
        bs = inputs.shape[0]
        self.labels = torch.zeros(bs, dtype=torch.int64)
        
        deg = random.choice(self.degrees)
        category = self.labelDict[deg]
        self.labels = self.labels+category
        inputs = rotate(inputs,deg)
        
        mask = self.mask

        if self.use_patches == False:
            h = self.backbone(inputs)
        else:
            h = self.backbone(inputs).permute(0,2,1)
            h = h.reshape(bs, self.hidden_dim, self.f_map_h, self.f_map_w)

        pos = self.pos
        if pos.shape[0] != h.shape[0]:
            pos = pos[0:h.shape[0],...]
            mask = mask[0:h.shape[0],...]


        att, sattn = self.encoder(src= 1*h, mask=mask, pos_embed=0.2*pos)

        x = att

        x = x.flatten(1)
        x = self.fc1(x)

        return [x, sattn, self.labels]
    
    
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
    
    

