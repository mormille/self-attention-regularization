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

from ARViT2D.ARViT2D import ARViT2D
from losses.distance_loss import *
from losses.metrics import *

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
bs =50
epochs = 10
lr = 1e-4

#HYPERPARAMETERS
reg_layers = [0,1,2,3,4,5]
grid_l = 16
nclass = 10
alpha = 4
beta = 0.5 
gamma = 0.1
lambdas = [0.002,0.002,0.002,0.002,0.002,0.002]

#SAVE FILE DETAILS
model_dir = Path.home()/'Luiz/saved_models/AROB'
file_name = "ARViT2D-test.pkl"
best_name = model_dir/'best/ARViT2D-test'

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Loading the DataSets

# path = Path.home()/'Luiz/gan_attention/data/Custom_ImageNet'

# transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
#                    n_inp=1,
#                    get_items = get_image_files,
#                    get_y     = parent_label,                  
#                    splitter  = RandomSplitter(),
#                    item_tfms = Resize(256),
#                    batch_tfms= transform
#                   )

# dloader = dblock.dataloaders(path/"images", bs=bs)

path = untar_data(URLs.IMAGENETTE)

transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dblock = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=Resize(H,W),
                 batch_tfms=transform)

dloader = dblock.dataloaders(path,bs=bs)


def new_empty():  
    return ()
dloader.new_empty = new_empty

#Defining the Loss Function
total_loss = ARViT2D_MultiLayer_Loss(layers=reg_layers, lambdas=lambdas)

#plateau = ReduceLROnPlateau(monitor='valid_loss', patience=2)
save_best = SaveModelCallback(monitor='valid_loss', fname=best_name)

#Building the model
model = ARViT2D(num_encoder_layers = 6, nhead=12, num_classes = nclass, batch_size=bs, 
                hidden_dim=516, image_h=H, image_w=W, grid_l=grid_l, 
                penalty_factor="2", alpha=alpha, beta=beta, gamma=gamma)

#Wraping the Learner
learner = Learner(dloader, model, loss_func=total_loss, metrics=[Accuracy,DL1,DL2,DL3,DL4,DL5,DL6,Cross_Entropy], cbs=[save_best]).to_distributed(args.local_rank)

#Fitting the model
learner.fit_one_cycle(epochs, lr)

learner.export(model_dir/file_name)
