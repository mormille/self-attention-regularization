#python -m torch.distributed.launch --nproc_per_node={num_gpus} launch.py

from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.gan import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback, ReduceLROnPlateau
from fastai import torch_core


from fastprogress import fastprogress
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import torchvision.transforms as T
import torch

from PIL import Image
import requests

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn

import argparse

from models.ARViT import ARViT
from losses.metrics import *
from losses.attention_loss import *
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
#print(args)
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
#PARAMETERS

#SPECIFICATIONS
H = 256
W= 256
bs =90
epochs = 90
lr = 5e-5

#HYPERPARAMETERS
reg_layers = [2,3]
grid_l = 16
gm_l = 16
nclass = 4
bias = -0.17 
lambdas = [0.002,0.002]

#SAVE FILE DETAILS
model_dir = Path.home()/'Luiz/saved_models/paper'
file_name = "ARViT-Basev2.pkl"
best_name = model_dir/'best/ARViT-Basev2'

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Loading the DataSets

path = Path.home()/'Luiz/gan_attention/data/Custom_ImageNet'

transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   n_inp=1,
                   get_items = get_image_files,
                   get_y     = parent_label,                  
                   splitter  = RandomSplitter(),
                   item_tfms = Resize(256),
                   batch_tfms= transform
                  )

dloader = dblock.dataloaders(path/"images", bs=bs)

# path = untar_data(URLs.IMAGENETTE)

# transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# dblock = DataBlock(blocks=(ImageBlock, CategoryBlock), 
#                  get_items=get_image_files, 
#                  splitter=RandomSplitter(),
#                  get_y=parent_label,
#                  item_tfms=Resize(H,W),
#                  batch_tfms=transform)

# dloader = dblock.dataloaders(path,bs=bs)

def new_empty():  
    return ()
dloader.new_empty = new_empty

#Defining the Loss Function
total_loss = ARViT_MultiLayer_Loss(layers= reg_layers, bias=bias, lambdas=lambdas)
#critic_loss = CriticValidationLoss()

plateau = ReduceLROnPlateau(monitor='valid_loss', patience=3)
save_best = SaveModelCallback(monitor='valid_loss', fname=best_name)

#Building the model
model = ARViT(num_encoder_layers = 6, nhead=8, num_classes = nclass, batch_size=bs, hidden_dim=512, image_h=H, image_w=W, grid_l=grid_l, gm_patch = gm_l)

#Wraping the Learner
learner = Learner(dloader, model, loss_func=total_loss, metrics=[Accuracy,AL1,AL2,AL3,AL4,AL5,AL6,Cross_Entropy], cbs=[save_best, plateau]).to_distributed(args.local_rank)

#with learner.distrib_ctx():
learner.fit_one_cycle(epochs, lr)

#model_dir = Path.home()/'Luiz/saved_models'
learner.export(model_dir/file_name)
