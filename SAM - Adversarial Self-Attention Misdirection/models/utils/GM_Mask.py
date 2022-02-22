import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms.functional import *
import random

__all__ = ['GM_Mask']

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
    

