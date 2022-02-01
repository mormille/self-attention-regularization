
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from ARViT2D.utils.distance_loss import Distance_loss

from fastai.vision.all import *
from fastai.distributed import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback

__all__ = ['Accuracy','DL1','DL2','DL3','DL4','DL5','DL6','Cross_Entropy']


######################################################################
#Parameters
beta_metric = 0.01
gamma_metric = 0.0005
sigma_metric = 0.01

c_entropy = nn.CrossEntropyLoss() 
LD = Distance_loss()
MSE = nn.MSELoss()

######################################################################
def Accuracy(preds,target): 

    _, pred = torch.max(preds[0], 1)

    return (pred == target).float().mean()

######################################################################

def DL1(preds,target):

    Latt = LD(preds[3], preds[1][0])
    
    return (sigma_metric*Latt).float().mean()

######################################################################

def DL2(preds,target):

    Latt = LD(preds[3], preds[1][1])
    
    return (sigma_metric*Latt).float().mean()

######################################################################

def DL3(preds,target):

    Latt = LD(preds[3], preds[1][2])
    
    return (sigma_metric*Latt).float().mean()

######################################################################

def DL4(preds,target):

    Latt = LD(preds[3], preds[1][3])
    
    return (sigma_metric*Latt).float().mean()

######################################################################

def DL5(preds,target):

    Latt = LD(preds[3], preds[1][4])
    
    return (sigma_metric*Latt).float().mean()

######################################################################

def DL6(preds,target):

    Latt = LD(preds[3], preds[1][5])
    
    return (sigma_metric*Latt).float().mean()

######################################################################

def Cross_Entropy(preds,target):

    Loss = c_entropy(preds[0], target)
    
    return (Loss).float().mean()

######################################################################
