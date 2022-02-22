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

H = 256
W= 256
bs = 5

model_dir = Path.home()/'Luiz/saved_models'

path1 = untar_data(URLs.IMAGENETTE)
path2 = untar_data(URLs.IMAGEWOOF)
path3 = untar_data(URLs.CIFAR)
path4 = untar_data(URLs.CIFAR_100)
path5 = untar_data(URLs.FLOWERS)


def flowers_loader(path):
    Path.BASE_PATH = path
    path.ls()
    df = pd.read_csv('data/flowers.csv')

    transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_x(r): return path/r['name']
    def get_y(r): return r['class']

    dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                       n_inp=1,
                       splitter=RandomSplitter(seed=42),
                       get_x= get_x,
                       get_y= get_y, 
                       item_tfms = Resize(256),
                       #batch_tfms= Normalize.from_stats(*imagenet_stats)
                       batch_tfms= transform
                      )

    dloader = dblock.dataloaders(df, bs=bs)
    return dloader



def load_model(fname, bs):
    #model = ARViT(num_encoder_layers = 6, nhead=8, num_classes = 4, 
    #                       batch_size=bs, hidden_dim=512, image_h=256, image_w=256, 
    #                       grid_l=16, gm_patch = 16)
    file_name = "ARViT-Basev2.pkl"
    net_dir = Path.home()/'Luiz/saved_models/paper'
    net = load_learner(net_dir/file_name, cpu=False)
    model = net.model
    
    model_path = fname + ".pth"
    model_dir = Path.home()/'Luiz/saved_models/paper/best'
    weights_dir = model_dir/model_path
    weights_dict = load_learner(weights_dir, cpu=False)
    model.load_state_dict(weights_dict)
    print("model loaded", fname)
        
    return model

def model_head(model, n_classes):
    model.head = nn.Linear(512*16*16, n_classes)
    #model.noise_mode = True
    #model.generator_mode = False

    trainable = ['head.weight','head.bias']
    for name, p in model.named_parameters():
        if name not in trainable:
            p.requires_grad = False
        else:
            p.requires_grad = True
    return model

def data_loader(path):
    transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                     get_items=get_image_files, 
                     splitter=RandomSplitter(),
                     get_y=parent_label,
                     item_tfms=Resize(H,W),
                     batch_tfms=transform)

    dloader = data.dataloaders(path,bs=bs)
    return dloader

def fine_tune_model(n_class, fname, bs, path_data, dataset = 'IMAGENETTE', epochs=[60,30], lr=5e-4, base_model = False):    
    
    model = load_model(fname, bs)
    model = model_head(model, n_class)
    
    if dataset == 'FLOWERS':
        dloader = flowers_loader(path_data)
    else:
        dloader = data_loader(path_data)
    #Defining the Loss Function
    finetune_loss = ARViT_CrossEntropy()
    
    save_dir = Path.home()/'Luiz/saved_models/paper/finetuned'
    name = fname+'_'+dataset
    sname = save_dir/name
        
    plateau = ReduceLROnPlateau(monitor='Accuracy', patience=3)
    save_best = SaveModelCallback(monitor='Accuracy', fname=sname)

    #Wraping the Learner
    learner = Learner(dloader, model, loss_func=finetune_loss, metrics=[Accuracy], cbs=[save_best, plateau]).to_distributed(args.local_rank)
    #learner.fit_one_cycle(50, 0.002)
    learner.fine_tune(epochs[0], base_lr=lr, freeze_epochs=epochs[1])
    

fine_tune_model(n_class=10, fname='ARViT-Basev2', bs=90, path_data=path1, dataset = 'IMAGENETTE')

#fine_tune_model(n_class=102, fname='ARViT-Base' , path_model='ARViT-Base.pkl', path_weights=None, path_data=path5, dataset = 'FLOWERS')  #DONE

#fine_tune_model(n_class=102, fname='ARViT-Base_Best' , path_model='ARViT-Base.pkl', path_weights='ARViT-Base.pth', path_data=path5, dataset = 'FLOWERS')  #DONE
    
    
#fine_tune_model(n_class=102, fname='ARViT-L1' , path_model='ARViT-L1.pkl', path_weights=None, path_data=path5, dataset = 'FLOWERS')  #DONE

#fine_tune_model(n_class=102, fname='ARViT-L1_Best' , path_model='ARViT-L1.pkl', path_weights='ARViT-L1.pth', path_data=path5, dataset = 'FLOWERS')  #DONE


#fine_tune_model(n_class=102, fname='ARViT-L2-G32' , path_model='ARViT-L2-G32.pkl', path_weights=None, path_data=path5, dataset = 'FLOWERS')  #DONE

#fine_tune_model(n_class=102, fname='ARViT-L2_Best' , path_model='ARViT-L2.pkl', path_weights='ARViT-L2.pth', path_data=path5, dataset = 'FLOWERS')  #DONE


#fine_tune_model(n_class=102, fname='ARViT-L3' , path_model='ARViT-L3.pkl', path_weights=None, path_data=path5, dataset = 'FLOWERS')  #DONE

#fine_tune_model(n_class=102, fname='ARViT-L3_Best' , path_model='ARViT-L3.pkl', path_weights='ARViT-L3.pth', path_data=path5, dataset = 'FLOWERS')  #DONE


#fine_tune_model(n_class=102, fname='ARViT-L4' , path_model='ARViT-L4.pkl', path_weights=None, path_data=path5, dataset = 'FLOWERS')  #DONE

#fine_tune_model(n_class=102, fname='ARViT-L4_Best' , path_model='ARViT-L4.pkl', path_weights='ARViT-L4.pth', path_data=path5, dataset = 'FLOWERS')  #DONE


#fine_tune_model(n_class=102, fname='ARViT-L5' , path_model='ARViT-L5.pkl', path_weights=None, path_data=path5, dataset = 'FLOWERS')  #DONE

#fine_tune_model(n_class=102, fname='ARViT-L5_Best' , path_model='ARViT-L5.pkl', path_weights='ARViT-L5.pth', path_data=path5, dataset = 'FLOWERS')  #DONE


#fine_tune_model(n_class=102, fname='ARViT-L6' , path_model='ARViT-L6.pkl', path_weights=None, path_data=path5, dataset = 'FLOWERS')  #DONE

#fine_tune_model(n_class=102, fname='ARViT-L6_Best' , path_model='ARViT-L6.pkl', path_weights='ARViT-L6.pth', path_data=path5, dataset = 'FLOWERS')  #DONE


