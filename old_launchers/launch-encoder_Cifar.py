#python -m torch.distributed.launch --nproc_per_node={num_gpus} launch.py

from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.gan import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback
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
from models.utils.joiner2 import Joiner
from models.utils.new_losses import *
from models.utils.metrics import _Accuracy
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

train_path  = "data/WebDataset-GramCifar/train/GramCifar-{0..4}.tar"
valid_path = "data/WebDataset-GramCifar/valid/GramCifar-0.tar"

H = 32
W= 32
bs = 5
grid_l = 2
nclass = 10
backbone = False
epochs = 50
epoch_list = [5,25,10,10]

lr = [5e-7,1e-7,5e-8,1e-8]
lr_str = '6e-7'

file_name = "Cifar_epochs_8Layers_learnedPos_"+str(epochs)+"epohcs_lr-"+lr_str+".pkl"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Creating the dataloader
path = untar_data(URLs.CIFAR)

transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=Resize(H,W),
                 batch_tfms=transform)

dloader = data.dataloaders(path,bs=bs) 

#Defining the Loss Function
critic_loss = SingleLabelCriticLoss()

#Building the model
model = Joiner(num_encoder_layers = 8, nhead=8, backbone = backbone, num_classes = nclass, bypass=False, pos_enc = "sin", hidden_dim=768, 
batch_size=bs, image_h=H, image_w=W, grid_l=grid_l,penalty_factor="1")

#Wraping the Learner
learner = Learner(dloader, model, loss_func=critic_loss, metrics=[_Accuracy]).to_distributed(args.local_rank)
for i in range(len(lr)):
    learner.fit(epoch_list[i], lr[i])

model_dir = Path.home()/'Luiz/saved_models'
learner.export(model_dir/file_name)
