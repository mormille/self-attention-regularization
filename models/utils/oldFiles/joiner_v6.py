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

__all__ = ['create_mask', 'penalty_mask', 'pos_emb', 'GAN', 'ImageNetJoiner','GM_Mask']


class GAN(nn.Module):
    def __init__(self, in_channels=3, gen_classes=3, bilinear=False, num_encoder_layers=8, nhead=8, use_patches=True, num_classes=4, bypass=False, mask=None, pos_enc="sin", batch_size=10, hidden_dim=768, image_h=32, image_w=32, grid_l=1, penalty_factor="1", alpha=1, data="CIFAR"):
        super().__init__()
        
        self.generator = UNet(in_channels,gen_classes,bilinear)
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
        
        return [x, reduced_attn, sattn, gm, input0, inputs]
    

    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)   
    
class LinearHead(nn.Module):
    def __init__(self, hidden_dim, f_map_h, f_map_w, num_classes):
        super().__init__()
        self.fcl = nn.Linear(hidden_dim * f_map_h * f_map_w, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        return self.fcl(x)   
    
class AdaptiveAvgPoolHead(nn.Module):
    def __init__(self, hidden_dim, f_map_h, f_map_w, num_classes):
        super().__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcl = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.global_pooling(x)
        return self.fcl( x.view(x.size(0), -1) )  


class ImageNetJoiner(nn.Module):
    def __init__(self, num_encoder_layers = 8, nhead=8, num_classes = 10, mask=None, pos_enc = "sin", batch_size=10, in_chans=3, hidden_dim=768, image_h=256, image_w=256, grid_l=8, gm_patch = 32, use_patches=True, attn_layer = 1,penalty_factor="1", alpha=1):
        super().__init__()
        
        self.bs = batch_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pos_enc = pos_enc
        self.f_map_h = image_h//grid_l
        self.f_map_w = image_w//grid_l
        self.use_patches = use_patches
        self.attn_layer = attn_layer
        self.gm_count = (image_w//gm_patch)**2
        
        self.gm_mask = GM_Mask(patch_size=gm_patch, width=image_w, height=image_h)
        
        #self.backbone = ConvPatchEmbed(img_size=image_h, patch_size=grid_l, in_chans=in_chans, embed_dim=hidden_dim)
        self.backbone = PatchEmbed2Step(img_size=image_h, patch_size=grid_l, in_chans=in_chans, embed_dim=hidden_dim)
        self.num_patches = self.backbone.num_patches
    
        self.encoder = EncoderModule(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers)
      
        #self.mlp = Mlp(in_features=self.hidden_dim * self.f_map_h * self.f_map_w, hidden_features=hidden_dim, out_features=self.num_classes)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.norm = LayerNorm(hidden_dim)
        #self.head = nn.Linear(hidden_dim, self.num_classes)
        
        self.head = nn.Linear(self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

        self.pos = nn.Parameter(pos_emb(batch_size, hidden_dim, self.f_map_h, self.f_map_w),requires_grad=False)
        self.mask = nn.Parameter(create_mask(batch_size, self.f_map_h, self.f_map_w),requires_grad=False)
        
        self.avgPool = nn.AvgPool2d(gm_patch//grid_l, stride=gm_patch//grid_l)
        #self.attn_map = nn.Parameter(torch.ones(batch_size, 64, 64))
            
        
    def paramsToUpdate(self, rg):

        fb = ["mask","pos"]
        for name, p in self.named_parameters(): 
            if name not in fb:
                p.requires_grad_(rg)
            else:
                p.requires_grad_(False)
                
    def rescale(self, sattn):

        ngm = self.gm_count*self.gm_count
        
        bs = sattn.shape[0]
        #print(sattn.shape)
        a = self.avgPool(sattn.reshape(bs,self.num_patches,self.f_map_h,self.f_map_w))
        #print(a.shape)
        attn = self.avgPool(a.reshape(bs,self.num_patches,self.gm_count).permute(0, 2, 1).reshape(bs,self.gm_count,self.f_map_h,self.f_map_w)).permute(0, 2, 3, 1).reshape(bs,self.gm_count,self.gm_count)
        
        #for i in range(bs):
        #    high = attn[i].max().item()
        #    low = attn[i].min().item()
        #    attn[i] = attn[i] - low
        #    attn[i] = attn[i]/(high-low)
            
        attn = attn.reshape(bs, self.gm_count*self.gm_count)
        low = attn.min(1, keepdim=True)[0]
        high = attn.max(1, keepdim=True)[0]
        attn = attn-low
        attn = attn/(high-low)
        attn = attn.reshape(bs,self.gm_count,self.gm_count)

        
        #attn = attn+0.0001

        return attn
            
    def forward(self, inputs, mask=Optional[Tensor], pos=Optional[Tensor], penalty_mask=Optional[Tensor]):
        
        if type(inputs) is tuple:
            #print("InModel - Generator input")
            inputs = inputs[0]
        #else:
        #    print("InModel - Orignial input")
        
        gm = self.gm_mask(inputs)
        
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
        #reduced_attn = self.rescale(sattn[self.attn_layer])
        reduced_attn = self.rescale(sattn[1])

        x = att
        
        x = self.norm(x)
        
        x = x.flatten(1)
        x = self.head(x)
        
        #x = self.global_pooling(x)
        #print(x.shape)
        
        #x = self.head( x.view(x.size(0), -1) )

        return [x, reduced_attn, sattn, gm]

#To lead the old model, remove the reshape and the attn
    
    
class GM_Mask(nn.Module):
    def __init__(self, patch_size=32, width=256, height=256):
        super().__init__()
        self.width = width #used
        self.height = height #used
        self.patch_size = patch_size #used
        self.qt_grids = (width//patch_size)**2 #used
        
        
    def img_patches(self, batch, patch_size):
        #torch.Tensor.unfold(dimension, size, step)
        #slices the images into grid_l*grid_l size patches
        #print(batch.shape)
        patches = batch.data.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        a, b, c, d, e, f, g = patches.shape
        patches = patches.reshape(a, c, d, e, f, g)
        #print(patches.shape)
        #print(patches.shape)
        return patches


    def grid_gram_matrix(self, patches):

        a, b, c, d, e, f = patches.shape
        # a=batch size
        # b=horizontal patches
        # c = vertical patches
        # d=number of feature maps
        # (e,f)=dimensions of a f. map (N=e*f)

        features = patches.reshape(a * b * c, d, e*f)  # resise F_XL into \hat F_XL
        #print(features.shape)
        # compute the gram product

        feat_t = features.permute(0,2,1)
        G = torch.bmm(features, feat_t)
        G = G.div(d * e * f).reshape(a, b, c, d, d)

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.

        return G
    
    def gram_dist_matrix(self, batch, grid_l):
        patches = self.img_patches(batch, grid_l)
        #print(patches.shape)
        Grid = self.grid_gram_matrix(patches)
        #print(Grid.shape)
        bs = batch.shape[0]

        g = Grid.reshape(bs,self.qt_grids,3,3)
        gh = g.unsqueeze(2)
        gv = g.unsqueeze(1)

        ghe = gh.repeat(1,1,self.qt_grids,1,1).reshape(bs,self.qt_grids*self.qt_grids,9).permute(0, 2, 1)
        gve = gv.repeat(1,self.qt_grids,1,1,1).reshape(bs,self.qt_grids*self.qt_grids,9).permute(0, 2, 1)

        pdist = nn.PairwiseDistance(p=0.1)
        output = pdist(ghe, gve).reshape(bs,self.qt_grids,self.qt_grids)

        #output -= output.min(1, keepdim=True)[0]
        #output /= output.max(1, keepdim=True)[0]
        
        output = output.view(output.size(0), -1)
        output -= output.min(1, keepdim=True)[0]
        output /= output.max(1, keepdim=True)[0]
        output = output.view(bs,self.qt_grids,self.qt_grids)

        return output
    
        
    def forward(self, batch):
        
        #batch = batch#.unsqueeze(0)
        
        dist_matrix = self.gram_dist_matrix(batch, self.patch_size)
          
        return dist_matrix
    
    
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
    
    

