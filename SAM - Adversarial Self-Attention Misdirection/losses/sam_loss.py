
import copy
from typing import Optional, List

from fastai.distributed import *
from fastai.vision.all import *

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from losses.attention_loss import *

__all__ = ['Misdirection_loss','CriticLoss','GeneratorLoss']


class Misdirection_loss(nn.Module):
    def __init__(self, bias=-0.17):
        super().__init__()
 
        self.bias = bias

    def penalty_factor(self, dist_matrix):

        pf_matrix = (-dist_matrix + 1) + self.bias
        return pf_matrix

    def forward(self, sattn, pattn):
        #Computing the Attention Loss
        
        pattn = self.penalty_factor(pattn)

        #att_loss = sattn*pattn # (output2*label) (64x64 * 64x64)
        #Latt = TensorCategory(torch.sum(att_loss)) # ecalar number
        
        att_loss = torch.sum(sattn*pattn)#.float().mean()
        att_loss[att_loss <= 1] = 1

        Latt = TensorCategory(torch.log(att_loss)) # ecalar number

        return Latt

class CriticLoss(nn.Module):
    def __init__(self, layers=[0,1,2,3,4,5], bias=-0.17, lambdas=[0.002,0.002,0.002,0.002,0.002,0.002]):
        super(CriticLoss, self).__init__()
        self.layers = layers
        self.lambdas = lambdas
        self.crossEntropy = nn.CrossEntropyLoss()
        self.LCA = Attention_loss(bias=bias)

    def forward(self, preds, label):
        
        #Classification Loss
        classificationLoss = self.crossEntropy(preds[0],label)
        
        #Attention Loss
        Latt = 0.0
        for i in range(len(self.layers)):
            Latt = Latt + self.lambdas[i]*self.LCA(preds[1][self.layers[i]], preds[3])
        
        Lc = classificationLoss + Latt
        #print(Lc)
        return Lc
    
    
class GeneratorLoss(nn.Module):
    def __init__(self, beta=0.001, bias=-0.17, layers=[0,1,2,3,4,5], gammas=[0.0002,0.0002,0.0002,0.0002,0.0002,0.0002]):
        super().__init__()
        self.beta = beta
        self.gammas = gammas
        self.crossEntropy = nn.CrossEntropyLoss()
        self.LM = Misdirection_loss(bias=bias)
        self.MSE = nn.MSELoss()
        self.layers = layers
        
    def forward(self, output, image, target): #fake_pred, output, target

        #Classification Loss
        classificationLoss = self.crossEntropy(output[0],target)
        
        #Misdirection Loss
        if self.layers != None:
            Lm = 0.0
            for i in range(len(self.layers)):
                Lm = Lm + self.gammas[i]*self.LM(output[1][self.layers[i]], output[3])
        else:
            Lm = 0.0
        
        #MSE Reconstruction Loss
        Lrec = TensorCategory(self.MSE(image[0],image[1]))

        Lg = Lrec + Lm - self.beta*(classificationLoss)

        return Lg
    