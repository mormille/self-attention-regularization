
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone
from models.utils.new_losses import *
from models.utils.distance_loss import Distance_loss

from fastai.vision.all import *
from fastai.distributed import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback

__all__ = ['Accuracy','AL1','AL2','AL3','AL4','AL5','AL6','DL1','DL2','DL3','DL4','DL5','DL6', 'Adversarial_loss','Reconstruction_Loss','Cross_Entropy','Curating_Of_Attention_Loss']


######################################################################
#Parameters
beta_metric = 0.01
gamma_metric = 0.0005
sigma_metric = 1

c_entropy = nn.CrossEntropyLoss() 
LCA = Curating_of_attention_loss(bias=0.0)
LCA2 = Curating_of_attention_loss(bias=0.001)
LCA3 = Curating_of_attention_loss(bias=-0.3)
LD = Distance_loss()
MSE = nn.MSELoss()

######################################################################
def Accuracy(preds,target): 

    _, pred = torch.max(preds[0], 1)

    return (pred == target).float().mean()
   
######################################################################

def AL1(preds,target):

    Latt = LCA2(preds[1][0], preds[3])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def AL2(preds,target):

    Latt = LCA2(preds[1][1], preds[3])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def AL3(preds,target):

    Latt = LCA2(preds[1][2], preds[3])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def AL4(preds,target):

    Latt = LCA2(preds[1][3], preds[3])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def AL5(preds,target):

    Latt = LCA2(preds[1][4], preds[3])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def AL6(preds,target):

    Latt = LCA2(preds[1][5], preds[3])
    
    return (beta_metric*Latt).float().mean()

######################################################################

######################################################################

def DL1(preds,target):

    Latt = LD(preds[3], preds[1][0])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def DL2(preds,target):

    Latt = LD(preds[3], preds[1][1])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def DL3(preds,target):

    Latt = LD(preds[3], preds[1][2])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def DL4(preds,target):

    Latt = LD(preds[3], preds[1][3])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def DL5(preds,target):

    Latt = LD(preds[3], preds[1][4])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def DL6(preds,target):

    Latt = LD(preds[3], preds[1][5])
    
    return (beta_metric*Latt).float().mean()

######################################################################

def Cross_Entropy(preds,target):

    Loss = c_entropy(preds[0], target)
    
    return (Loss).float().mean()

######################################################################

def Adversarial_loss(preds,target,beta=beta_metric,sigma=sigma_metric,gamma=gamma_metric): 

    advLoss = -gamma*lossFunc(preds,target)

    return advLoss

######################################################################

def Reconstruction_Loss(preds,target,sigma=1):
    Lrec = sigma*MSE(preds[0],preds[1])
    
    return Lrec

######################################################################

def Curating_Of_Attention_Loss(preds,target):

    Loss = c_entropy(preds[0], target)
    
    return (Loss).float().mean()

