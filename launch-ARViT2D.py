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

from models.utils.ARViT2D import ARViT2D
from models.utils.distance_loss import ARViT2D_Loss
#from models.utils.joiner3 import ImageNetJoiner
#from models.utils.joiner_v5 import ImageNetJoiner
from models.utils.new_losses import *
from models.utils.metrics import *
from models.utils.dataLoader import *
from models.utils.datasets import *
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
#print(args)
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
#PARAMETERS

H = 256
W= 256
bs =80
grid_l = 16
gm_l = 32
nclass = 4
bias=0.001
#bias=0.001#-0.3
layer=1
epochs = 90

beta = 0.000005 #0.0002
beta_str = '5e-6'
gamma = 0.0005
sigma = 0.001

lr = 1e-4
lr_str = '1e-4'

model_dir = Path.home()/'Luiz/saved_models/AROB'
file_name = "ARViT2D-Base-6L.pkl"
best_name = model_dir/'best/ARViT2D-L2'

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
#dsets = dblock.datasets(path/"images")

dloader = dblock.dataloaders(path/"images", bs=bs)

def new_empty():  
    return ()
dloader.new_empty = new_empty

#Defining the Loss Function
critic_loss = ARViT2D_Loss(layer=layer, sigma=sigma)
#critic_loss = SingleLabelCriticLoss()

#plateau = ReduceLROnPlateau(monitor='valid_loss', patience=2)
save_best = SaveModelCallback(monitor='valid_loss', fname=best_name)

#Building the model
model = ARViT2D(num_encoder_layers = 6, nhead=12, num_classes = 4, mask=None, pos_enc = "sin", 
                batch_size=bs, in_chans=3, hidden_dim=516, image_h=256, image_w=256, grid_l=16, 
                gm_patch = 32, use_patches=True, attn_layer = 1,penalty_factor="2", alpha=4, 
                beta=500, gamma=0.1)

#Wraping the Learner
learner = Learner(dloader, model, loss_func=critic_loss, metrics=[Accuracy,DL1,DL2,DL3,DL4,DL5,DL6,Cross_Entropy], cbs=[save_best]).to_distributed(args.local_rank)

#Fitting the model
learner.fit_one_cycle(90, lr)

learner.export(model_dir/file_name)
