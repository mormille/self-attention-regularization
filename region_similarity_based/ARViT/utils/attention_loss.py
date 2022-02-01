
import copy
from typing import Optional, List

from fastai.distributed import *
from fastai.vision.all import *

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

__all__ = ['CriticLoss','Curating_of_attention_loss']


class Curating_of_attention_loss(nn.Module):
    def __init__(self, bias=0.0, distortion=1):
        super().__init__()
 
        self.bias = bias
        self.d = distortion

    def penalty_factor(self, dist_matrix, alpha=1):

        pf_matrix = dist_matrix+self.bias
        return pf_matrix

    def forward(self, sattn, pattn):
        #Computing the Attention Loss
        
        pattn = self.penalty_factor(pattn)

        att_loss = sattn*pattn # (output2*label) (64x64 * 64x64)
        Latt = TensorCategory(torch.sum(att_loss)) # ecalar number

        return Latt
    
    
class CriticLoss(nn.Module):
    def __init__(self, layer=None, bias=0.0, beta=0.0002, sigma=1):
        super(CriticLoss, self).__init__()
        self.layer = layer
        self.beta = beta
        self.sigma = sigma
        self.crossEntropy = nn.CrossEntropyLoss()
        self.LCA = Curating_of_attention_loss(bias=bias)

    def forward(self, preds, label):

        classificationLoss = self.crossEntropy(preds[0],label)
        #print(classificationLoss)
        if self.layer != None:
            #print(self.layer)
            Latt = self.LCA(preds[1][self.layer], preds[3])#.item()
        else:
            Latt=0.0
        #print(self.beta*Latt)
        Lc = self.sigma*classificationLoss + self.beta*Latt
        #print(Lc)
        return Lc
    
