
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from losses.attention_loss import *

from fastai.vision.all import *
from fastai.distributed import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback

__all__ = ['Accuracy','AL1','AL2','AL3','AL4','AL5','AL6','Cross_Entropy']


######################################################################
#Parameters
bias = -0.17
lambda_metric = 0.01

c_entropy = nn.CrossEntropyLoss() 
LCA2 = Attention_loss(bias=bias)
MSE = nn.MSELoss()

######################################################################
def Accuracy(preds,target): 

    _, pred = torch.max(preds[0], 1)

    return (pred == target).float().mean()
   
######################################################################

def AL1(preds,target):

    Latt = LCA2(preds[1][0], preds[3])
    
    return (lambda_metric*Latt).float().mean()

######################################################################

def AL2(preds,target):

    Latt = LCA2(preds[1][1], preds[3])
    
    return (lambda_metric*Latt).float().mean()

######################################################################

def AL3(preds,target):

    Latt = LCA2(preds[1][2], preds[3])
    
    return (lambda_metric*Latt).float().mean()

######################################################################

def AL4(preds,target):

    Latt = LCA2(preds[1][3], preds[3])
    
    return (lambda_metric*Latt).float().mean()

######################################################################

def AL5(preds,target):

    Latt = LCA2(preds[1][4], preds[3])
    
    return (lambda_metric*Latt).float().mean()

######################################################################

def AL6(preds,target):

    Latt = LCA2(preds[1][5], preds[3])
    
    return (lambda_metric*Latt).float().mean()

######################################################################

######################################################################


def Cross_Entropy(preds,target):

    Loss = c_entropy(preds[0], target)
    
    return (Loss).float().mean()

######################################################################

