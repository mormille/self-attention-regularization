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
from models.utils.joiner2 import Joiner, GAN
from models.utils.new_losses import *
from models.utils.metrics import Accuracy, Curating_Of_Attention_Loss, Reconstruction_Loss, Adversarial_loss
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

train_path  = "data/GramCifar/train/train_CustomCifar_4x4GridLabel_full.tar"
valid_path = "data/GramCifar/train/train_CustomCifar_4x4GridLabel_full.tar"

H = 32
W= 32
bs = 5
grid_l = 1
nclass = 4
use_patches = True
epochs = 50

beta = 1e-3 #0.001
beta_str = '1e-3'
gamma = 0.005#0.0005
sigma = 1.0

lr = 2e-7
lr_str = '2e-7'

file_name = "GramCifar_GAN_Rotation_epochs-"+str(epochs)+"-beta-"+beta_str+"_lr-"+lr_str+".pkl"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Loading the DataSets
#train_ds, valid_ds = LoadGramDataset(train_path, valid_path, normalize=False)
train_ds = torch.load(train_path)
valid_ds = torch.load(valid_path)
#Creating the dataloader
dloader = GramCifarLoader(train_ds, valid_ds, bs)

#Defining the Loss Function
critic_loss = CriticLoss(beta=beta, sigma=sigma)
gen_loss = GeneratorLoss(beta=beta, sigma=sigma, gamma=gamma)

#Building the model
gan = GAN(num_encoder_layers=8, nhead=8, use_patches=use_patches, num_classes=nclass, bypass=False, 
          hidden_dim=768, batch_size=bs, image_h=H, image_w=W, grid_l=grid_l,penalty_factor="1")

#Wraping the Critic Learner
critic_learner = Learner(dloader, gan, loss_func=critic_loss, metrics=[Accuracy,Curating_Of_Attention_Loss,Reconstruction_Loss]).to_distributed(args.local_rank)
#Wraping the Generator Learner
gen_learner = Learner(dloader, gan, loss_func=gen_loss, metrics=[Accuracy,Curating_Of_Attention_Loss,Adversarial_loss,Reconstruction_Loss]).to_distributed(args.local_rank)


for i in range(epochs):

    gan.noise_mode = True
    gan.generator_mode = True
    gan.paramsToUpdate()
    gen_learner.fit(1, 0.001)
    
    
    gan.noise_mode = True
    gan.generator_mode = False
    gan.paramsToUpdate()
    critic_learner.fit(1, lr)
    
    gan.noise_mode = False
    gan.generator_mode = False
    gan.paramsToUpdate()
    critic_learner.fit(1, lr)

model_dir = Path.home()/'Luiz/saved_models'
critic_learner.export(model_dir/file_name)
#torch.save(learner, "saved_models/"+file_name)
