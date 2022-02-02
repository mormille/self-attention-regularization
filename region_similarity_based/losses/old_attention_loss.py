
import copy
from typing import Optional, List

from fastai.distributed import *
from fastai.vision.all import *

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np


__all__ = ['ARViT_Loss','Attention_loss']


class Attention_loss(nn.Module):
    def __init__(self, bias=0.01):
        super().__init__()
 
        self.bias = bias

    def penalty_factor(self, dist_matrix):

        pf_matrix = dist_matrix+self.bias
        return pf_matrix

    def forward(self, sattn, pattn):
        #Computing the Attention Loss
        
        pattn = self.penalty_factor(pattn)

        att_loss = sattn*pattn # (output2*label) (64x64 * 64x64)
        Latt = TensorCategory(torch.sum(att_loss)) # ecalar number

        return Latt
    
    
class ARViT_Loss(nn.Module):
    def __init__(self, layer=None, bias=0.01, lambda_=0.0002):
        super(ARViT_Loss, self).__init__()
        self.layer = layer
        self.lambda_ = lambda_
        self.crossEntropy = nn.CrossEntropyLoss()
        self.LCA = Attention_loss(bias=bias)

    def forward(self, preds, label):

        classificationLoss = self.crossEntropy(preds[0],label)
        #print(classificationLoss)
        if self.layer != None:
            #print(self.layer)
            Latt = self.LCA(preds[1][self.layer], preds[3])#.item()
        else:
            Latt=0.0
        #print(self.beta*Latt)
        Lc = classificationLoss + self.lambda_*Latt
        #print(Lc)
        return Lc
    
