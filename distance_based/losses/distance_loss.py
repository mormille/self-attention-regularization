
import copy
from typing import Optional, List

from fastai.distributed import *
from fastai.vision.all import *

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np


__all__ = ['ARViT2D_Loss', 'Distance_loss','ARViT2D_MultiLayer_Loss']



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
    def __init__(self,layer=None, lambda_=0.01):
        super(ARViT2D_Loss, self).__init__()
        self.layer = layer
        self.lambda_ = lambda_
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
        Lc = classificationLoss + self.lambda_*LD
        #print(Lc)
        return Lc
    
    
class ARViT2D_MultiLayer_Loss(nn.Module):
    def __init__(self, layers=[0,1,2,3,4,5], lambdas=[0.002,0.002,0.002,0.002,0.002,0.002]):
        super(ARViT2D_MultiLayer_Loss, self).__init__()
        self.layers = layers
        self.lambdas = lambdas
        self.crossEntropy = nn.CrossEntropyLoss()
        self.Ld = Distance_loss()

    def forward(self, preds, label):

        classificationLoss = self.crossEntropy(preds[0],label)
        #print(classificationLoss)
        LD = 0.0
        for i in range(len(self.layers)):
            LD = LD + self.lambdas[i]*self.Ld(preds[3], preds[1][self.layers[i]])#.item()

        #print(self.beta*Latt)
        Lc = classificationLoss + LD
        #print(Lc)
        return Lc