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
#from models.utils.joiner3 import *
#from models.utils.joiner_v5 import *
from models.utils.new_losses import *
from models.utils.metrics import Accuracy
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
bs = 80

model_dir = Path.home()/'Luiz/saved_models'

path1 = untar_data(URLs.IMAGENETTE)
path2 = untar_data(URLs.IMAGEWOOF)
path3 = untar_data(URLs.CIFAR)
path4 = untar_data(URLs.CIFAR_100)
path5 = untar_data(URLs.FLOWERS)

def get_gm(r):
    label = parent_label(r)
    a = attrgetter("name")
    rgex = RegexLabeller(pat = r'image(.*?).jpeg') 
    gm = torch.load(save_path+"/gramm/"+str(label)+"/gm"+rgex(a(r))+".pt")
    return gm, TensorCategory(int(label))

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



def load_model():
    #model = ARViT2D(num_encoder_layers = 12, nhead=12, num_classes = 4, mask=None, pos_enc = "sin", 
    #            batch_size=bs, in_chans=3, hidden_dim=516, image_h=256, image_w=256, grid_l=16, 
    #            gm_patch = 32, use_patches=True, attn_layer = 1,penalty_factor="2", alpha=4, 
    #            beta=500, gamma=0.1)

    model_dir = Path.home()/'Luiz/saved_models/AROB'
    net = load_learner(model_dir/'ARViT2D-Base-6L.pkl', cpu=False)
    weights_dir = model_dir/'best/ARViT2D-L2.pth'
    model = net.model
    #weights = torch.load(weights_dir)
    #model.pos = nn.Parameter(weights['pos'],requires_grad=False)
    #model.load_state_dict(weights)
    weights_dict = load_learner(weights_dir, cpu=False)
    model.load_state_dict(weights_dict)
    model = model.eval()
        
    return model

def model_head(model, n_classes):
    model.head = nn.Linear(516*16*16, n_classes)
    #model.noise_mode = True
    #model.generator_mode = False

    trainable = ['head.weight','head.bias']
    for name, p in model.named_parameters():
        if name not in trainable:
            p.requires_grad = False
        else:
            p.requires_grad = True

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

def fine_tune_model(n_class, fname, path_model, path_weights, path_data, dataset = 'IMAGENETTE', epochs=[60,30], lr=5e-4, base_model = False):    
    model = load_model()
    model_head(model, n_class)
    
    if dataset == 'FLOWERS':
        dloader = flowers_loader(path_data)
    else:
        dloader = data_loader(path_data)
    #Defining the Loss Function
    critic_loss = SingleLabelCriticLoss()
    
    save_dir = Path.home()/'Luiz/saved_models/AROB/finetuned'
    name = fname+'_'+dataset
    sname = save_dir/name
        
    plateau = ReduceLROnPlateau(monitor='Accuracy', patience=4)
    save_best = SaveModelCallback(monitor='Accuracy', fname=sname)

    #Wraping the Learner
    learner = Learner(dloader, model, loss_func=critic_loss, metrics=[Accuracy], cbs=[plateau, save_best]).to_distributed(args.local_rank)
    #learner.fit_one_cycle(50, 0.002)
    learner.fine_tune(epochs[0], base_lr=lr, freeze_epochs=epochs[1])
    

fine_tune_model(n_class=10, fname='ARViT2D-L2' , path_model='ARViT2D-Base.pkl', path_weights=None, path_data=path1, dataset = 'IMAGENETTE')  #DONE

fine_tune_model(n_class=10, fname='ARViT2D-L2' , path_model='ARViT2D-Base.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

fine_tune_model(n_class=10, fname='ARViT2D-L2' , path_model='ARViT2D-Base.pkl', path_weights=None, path_data=path3, dataset = 'CIFAR')  #DONE

fine_tune_model(n_class=100, fname='ARViT2D-L2' , path_model='ARViT2D-Base.pkl', path_weights=None, path_data=path4, dataset = 'CIFAR_100')  #DONE

fine_tune_model(n_class=102, fname='ARViT2D-L2' , path_model='ARViT2D-Base.pkl', path_weights=None, path_data=path5, dataset = 'FLOWERS')  #DONE

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


