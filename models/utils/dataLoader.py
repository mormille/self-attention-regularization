from fastai.data import load
from fastai.vision.all import *
from fastai.vision.gan import *

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import *

from collections import defaultdict, deque
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch

from torchvision.datasets.imagenet import ImageNet

from models.utils.datasets import *

import webdataset as wds
from webdataset.dataset import Composable, IterableDataset, Shorthands
from itertools import islice

import random

__all__ = ['GramCifarLoader','LoadGramDataset','identity','void','ds_transform']

def GramCifarLoader(train_ds, valid_ds, batch_size): #
        
        train_dl = load.DataLoader(train_ds,batch_size=batch_size)
        valid_dl = load.DataLoader(valid_ds,batch_size=batch_size)
        dld = ImageDataLoaders(train_dl, valid_dl, device='cuda')
        
        return dld
    
def LoadGramDataset(train_path, valid_path, normalize=False, custom_transform=None):
    
    transform = ds_transform(normalize)
    
    if custom_transform!=None:
        transform = custom_transform
    
    train_ds = (
        wds.WebDataset(train_path)
        .shuffle(100)
        .decode("pil")
        .map(sample_decoder)
        .to_tuple("input.pyd", "output.pyd")
        .map_tuple(transform, identity)
    )

    valid_ds = (
        wds.WebDataset(valid_path)
        .shuffle(100)
        .decode("pil")
        .map(sample_decoder)
        .to_tuple("input.pyd", "output.pyd")
        .map_tuple(transform, identity)
    )
    
    train_ds.length = 50000
    valid_ds.length = 10000
    
    return train_ds, valid_ds
    
    
def identity(x):
    return x

def void(x):
    return None

def sample_decoder(sample):
    result = dict(__key__=sample["__key__"])
    for key, value in sample.items():
            result[key] = value
    return result

def ds_transform(normalize=False, H=32, W=32):
    if normalize == False:
        
        transform = T.Compose([
            T.Resize((H,W)),
            T.ToTensor()
        ])
        
        return transform
    
    else:
        
        transform = T.Compose([
            T.Resize((H,W)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return transform
    
    
    
    