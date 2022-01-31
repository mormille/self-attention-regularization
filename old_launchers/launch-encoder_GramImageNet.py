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
from models.utils.joiner3 import ImageNetJoiner
from models.utils.new_losses import *
from models.utils.metrics import Accuracy, Curating_Of_Attention_Loss, Curating_Of_Attention_Loss3
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

train_path  = "data/GramImageNet/train/Rotation_CustomImageNet_32x32GridLabel_1million_images.tar"
valid_path = "data/GramImageNet/Valid_Rotation_CustomImageNet_32x32GridLabel_1million_images.tar"

H = 256
W= 256
bs = 100
grid_l = 16
nclass = 4
pf = "2"
epochs = 4

beta = 0.000005 #0.0002
beta_str = '5e-6'
gamma = 0.0005
sigma = 1.0

lr = 2e-3
lr_str = '2e-3'

transform = T.Compose([
T.Resize((H,W)),
T.ToTensor(),
T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

file_name = "GramImageNet3_Rotation_epochs-"+str(epochs)+"-beta-"+beta_str+"_lr-"+lr_str+"PenaltyFactor2"+".pkl"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Loading the DataSets
#train_ds, valid_ds = LoadGramDataset(train_path, valid_path, normalize=False)
train_ds = torch.load(train_path)
valid_ds = torch.load(valid_path)
train_ds.transforms = transform
valid_ds.transforms = transform
#Creating the dataloader
dloader = GramCifarLoader(train_ds, valid_ds, bs)

#Defining the Loss Function
critic_loss = CriticLoss(beta=beta, sigma=sigma, enc_layer_idx=0, pf=pf)
#critic_loss = CriticValidationLoss()

#Building the model
model = ImageNetJoiner(num_encoder_layers = 6, nhead=8, num_classes = nclass, batch_size=bs, hidden_dim=384, image_h=H, image_w=W, grid_l=grid_l)

#Wraping the Learner
learner = Learner(dloader, model, loss_func=critic_loss, metrics=[Accuracy,Curating_Of_Attention_Loss]).to_distributed(args.local_rank)

learner.fit(epochs, 2e-7)

model_dir = Path.home()/'Luiz/saved_models'
learner.export(model_dir/file_name)
#torch.save(learner, "saved_models/"+file_name)
