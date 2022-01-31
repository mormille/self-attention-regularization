
import copy
from typing import Optional, List

from fastai.distributed import *
from fastai.vision.all import *

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

__all__ = ['ARViT2D_Loss', 'Distance_loss','Distance_Matrix2D']


class Distance_Matrix2D(nn.Module):
    def __init__(self, bs, width=256, height=256, grid_l=16, penalty_factor="2", alpha=4, beta=500, gamma=0.1):
        super().__init__()
 
        self.pm = self.penalty_matrix(bs, width, height, grid_l, penalty_factor, alpha, beta, gamma)

    def distance_matrix(self, width=256, height=256, grid_l=16):

        w = width
        h = height
        qt_hor_grids = w//grid_l
        qt_ver_grids = h//grid_l
        qtd_grids = qt_hor_grids*qt_ver_grids
        c = 0
        grids = []
        for i in range(qtd_grids):
            hor_pos = i//qt_hor_grids
            ver_pos = c
            c = c+1
            grid = [hor_pos,ver_pos]
            grids.append(grid)
            if c == qt_ver_grids:
                c=0
        #gd = torch.tensor(np.array(grids))
        dist_grid = []
        for g in range(len(grids)):
            dist_pair_list = []
            for n in range(len(grids)):
                dist_pair_list.append(distance.cityblock(grids[g], grids[n]))
            dist_grid.append(dist_pair_list)

        dist_matrix = torch.tensor(np.array(dist_grid))

        return dist_matrix

    def penalty_weights(self, dist_matrix, penalty_factor="2", alpha=4, beta=500, gamma=0.1):
        if penalty_factor == "1":
            high = (dist_matrix.max(0, keepdim=True)[0][0]+1).reshape(256,1)
            pf_matrix = torch.div((dist_matrix+gamma),high)
            return pf_matrix
        if penalty_factor == "2":
            high = (dist_matrix.max(0, keepdim=True)[0][0]).reshape(256,1)/alpha
            a = torch.sub(dist_matrix,high)
            pf_matrix = torch.div(a,torch.sqrt(torch.square(a)+beta))
            return pf_matrix

    def penalty_matrix(self, bs, width=256, height=256, grid_l=16, penalty_factor="2", alpha=4, beta=500, gamma=0.1):
        dist_matrix = self.distance_matrix(width, height, grid_l)
        pf = self.penalty_weights(dist_matrix, penalty_factor, alpha, beta, gamma)
        stack = []
        for i in range(bs):
            stack.append(pf)
        pm = torch.stack(stack, dim=0)
        return pm

class Distance_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pm, sattn):
        #Computing the Distance Loss
        
        #if pm.shape[0]>=sattn.shape[0]:
        #    pm = pm[:sattn.shape[0]] 
        
        #loss = self.pm*sattn
        dist_loss = torch.sum(pm*sattn)#.float().mean()
        dist_loss[dist_loss <= 1] = 1

        Ld = TensorCategory(torch.log(dist_loss)) # ecalar number

        return Ld
    
    
class ARViT2D_Loss(nn.Module):
    def __init__(self,layer=None, sigma=0.01):
        super(ARViT2D_Loss, self).__init__()
        self.layer = layer
        self.sigma = sigma
        self.crossEntropy = nn.CrossEntropyLoss()
        self.Ld = Distance_loss()

    def forward(self, preds, label):

        classificationLoss = self.crossEntropy(preds[0],label)
        #print(classificationLoss)
        if self.layer != None:
            #print(self.layer)
            LD = self.Ld(preds[3],preds[1][self.layer])#.item()
        else:
            LD=0.0
        #print(self.beta*Latt)
        Lc = classificationLoss + self.sigma*LD
        #print(Lc)
        return Lc