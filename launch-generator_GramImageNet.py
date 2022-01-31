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
from models.utils.joiner3 import ImageNetJoiner
from models.utils.new_losses import *
from models.utils.metrics import *
from models.utils.dataLoader import *
from models.utils.datasets import *
from models.unet import UNet
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
bs =5
grid_l = 16
nclass = 4
pf = "3"
epochs = 90

beta = 0.00005 #0.0002
beta_str = '5e-6'
gamma = 0.0005
sigma = 1.0

lr = 1e-4
lr_str = '1e-4'

file_name = "GramImageNet_Rotation_16x16grid_epochs-"+str(epochs)+"-beta-"+beta_str+"_lr-"+lr_str+"PenaltyFactor3_Layer2.pkl"
#file_name = "BaseModel_16x16grid_epochs-"+str(epochs)+"_lr-"+lr_str+".pkl"

def get_gm(r):
    label = parent_label(r)
    a = attrgetter("name")
    rgex = RegexLabeller(pat = r'image(.*?).jpeg') 
    gm = torch.load(save_path+"/gramm/"+str(label)+"/gm"+rgex(a(r))+".pt")
    return gm, TensorCategory(int(label))

#Load Critic
critic_path = Path.home()/'Luiz/saved_models/GramImageNet_Rotation_16x16grid_epochs-90-beta-5e-6_lr-1e-4PenaltyFactor3_Layer2.pkl'

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Loading the DataSets

path = Path.home()/'Luiz/gan_attention/data/Custom_ImageNet'
save_path = 'data/Custom_ImageNet'

transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dblock = DataBlock(blocks    = (ImageBlock, [RegressionBlock, CategoryBlock]),
                   n_inp=1,
                   get_items = get_image_files,
                   #get_y     = parent_label,
                   get_y     = get_gm,                   
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
LCA = Curating_of_attention_loss(pf="2")
def GENAccuracy(preds,target): 
    with torch.no_grad():
        adv_pred = crt(output[0])

    _, pred = torch.max(adv_pred[0], 1)

    return (pred == target[1]).item().mean()

######################################################################

def GENCurating_Of_Attention_Loss(preds,target):
    with torch.no_grad():
        adv_pred = crt(output[0])

    Latt = LCA(adv_pred[1], adv_pred[3])

    return beta*(Latt.item().mean())
#critic_loss = CriticValidationLoss()

plateau = ReduceLROnPlateau(monitor='valid_loss', patience=5)
save_best = SaveModelCallback(monitor='valid_loss', fname='Best_Model_Layer2_ImageNet_Rotation_16x16grid')

#Building the model
model = UNet(n_channels=3, n_classes=3, bilinear=False)

#Wraping the Learner
learner = Learner(dloader, model, loss_func=gen_loss, metrics=[Reconstruction_Loss, gen_loss.Accuracy, gen_loss.Curating_Of_Attention_Loss], cbs=[save_best, plateau]).to_distributed(args.local_rank)

#with learner.distrib_ctx():
learner.fit_one_cycle(epochs, lr)

model_dir = Path.home()/'Luiz/saved_models'
learner.export(model_dir/file_name)
