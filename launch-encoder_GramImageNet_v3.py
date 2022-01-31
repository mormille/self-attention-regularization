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
#from models.utils.joiner3 import ImageNetJoiner
from models.utils.joiner_v5 import ImageNetJoiner
from models.utils.new_losses import *
from models.utils.metrics import Accuracy, Curating_Of_Attention_Loss, Curating_Of_Attention_Loss3, Cross_Entropy
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
bs =100
grid_l = 16
gm_l = 4
nclass = 4
pf = "3"
epochs = 90

beta = 0.0000005 #0.0002
beta_str = '5e-7'
gamma = 0.0005
sigma = 1.0

lr = 1e-4
lr_str = '1e-4'

file_name = "GramImageNet_Rotation_16x16grid_epochs-"+str(epochs)+"-beta-"+beta_str+"_PenaltyFactor3_Layer1_gm16.pkl"
#file_name = "GramImageNet_Rotation_16x16grid_epochs-"+str(epochs)+"_BaseModel.pkl"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Loading the DataSets

path = Path.home()/'Luiz/gan_attention/data/Custom_ImageNet'
save_path = 'data/Custom_ImageNet'

def get_gm(r):
    label = parent_label(r)
    a = attrgetter("name")
    rgex = RegexLabeller(pat = r'image(.*?).jpeg') 
    gm = torch.load(save_path+"/gramm/"+str(label)+"/gm"+rgex(a(r))+".pt")
    return gm, TensorCategory(int(label))

transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   n_inp=1,
                   get_items = get_image_files,
                   get_y     = parent_label,
                   #get_y     = get_gm,                   
                   splitter  = RandomSplitter(),
                   item_tfms = Resize(256),
                   #batch_tfms= Normalize.from_stats(*imagenet_stats)
                   batch_tfms= transform
                  )
dsets = dblock.datasets(path/"images")

dloader = dblock.dataloaders(path/"images", bs=bs)

def new_empty():  
    return ()
dloader.new_empty = new_empty

#Defining the Loss Function
critic_loss = CriticLoss(beta=beta, sigma=sigma, pf=pf)
#critic_loss = CriticValidationLoss()

plateau = ReduceLROnPlateau(monitor='valid_loss', patience=5)
save_best = SaveModelCallback(monitor='valid_loss', fname='pretrained/Best_Model_Layer1_ImageNet_Rotation_16x16grid_gm16_v3')

#Building the model
model = ImageNetJoiner(num_encoder_layers = 6, nhead=8, num_classes = nclass, batch_size=bs, hidden_dim=512, image_h=H, image_w=W, grid_l=grid_l, gm_patch = gm_l, attn_layer = 2)

#Wraping the Learner
learner = Learner(dloader, model, loss_func=critic_loss, metrics=[Accuracy,Curating_Of_Attention_Loss,Cross_Entropy], cbs=[save_best, plateau]).to_distributed(args.local_rank)

#with learner.distrib_ctx():
learner.fit_one_cycle(epochs, lr)

model_dir = Path.home()/'Luiz/saved_models'
learner.export(model_dir/file_name)
