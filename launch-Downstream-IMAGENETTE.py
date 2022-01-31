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
from models.utils.joiner3 import *
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
bs = 5

model_dir = Path.home()/'Luiz/saved_models'

path1 = untar_data(URLs.IMAGENETTE)
path2 = untar_data(URLs.IMAGEWOOF)
path3 = untar_data(URLs.CIFAR)
path4 = untar_data(URLs.CIFAR_100)


def load_model(model_path, best_model = None):
    model_dir = Path.home()/'Luiz/saved_models'
    net = load_learner(model_path, cpu=False)
    model = net.model
    model_name = 'Base_Model'

    if best_model != None:
        weight_dict = load_learner(best_model, cpu=False)
        model.load_state_dict(weight_dict)
        
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
    print(path_model, path_weights)
    print("Training on", dataset)
    
    load_dir = Path.home()/'Luiz/saved_models/paper'
    model_path = load_dir/path_model
    if path_weights != None:
        best_dir = Path.home()/'Luiz/saved_models/paper/best'
        weights_path = best_dir/path_weights
    else:
        weights_path = None
    model = load_model(model_path, weights_path)
    model_head(model, n_class)
        
    dloader = data_loader(path_data)
    #Defining the Loss Function
    critic_loss = SingleLabelCriticLoss()
    
    save_dir = Path.home()/'Luiz/saved_models/paper/finetuned'
    name = fname+'_'+dataset
    sname = save_dir/name
        
    plateau = ReduceLROnPlateau(monitor='Accuracy', patience=4)
    save_best = SaveModelCallback(monitor='Accuracy', fname=sname)

    #Wraping the Learner
    learner = Learner(dloader, model, loss_func=critic_loss, metrics=[Accuracy], cbs=[save_best, plateau]).to_distributed(args.local_rank)
    #learner.fit_one_cycle(50, 0.002)
    learner.fine_tune(epochs[0], base_lr=lr, freeze_epochs=epochs[1])
    

# fine_tune_model(n_class=10, fname='ARViT-Base' , path_model='ARViT-Base.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

# fine_tune_model(n_class=10, fname='ARViT-Base_Best' , path_model='ARViT-Base.pkl', path_weights='ARViT-Base.pth', path_data=path2, dataset = 'IMAGEWOOF')  #DONE
    
    
# fine_tune_model(n_class=10, fname='ARViT-L1' , path_model='ARViT-L1.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

# fine_tune_model(n_class=10, fname='ARViT-L1_Best' , path_model='ARViT-L1.pkl', path_weights='ARViT-L1.pth', path_data=path2, dataset = 'IMAGEWOOF')  #DONE


# fine_tune_model(n_class=10, fname='ARViT-L2' , path_model='ARViT-L2.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

# fine_tune_model(n_class=10, fname='ARViT-L2_Best' , path_model='ARViT-L2.pkl', path_weights='ARViT-L2.pth', path_data=path2, dataset = 'IMAGEWOOF')  #DONE


# fine_tune_model(n_class=10, fname='ARViT-L3' , path_model='ARViT-L3.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

# fine_tune_model(n_class=10, fname='ARViT-L3_Best' , path_model='ARViT-L3.pkl', path_weights='ARViT-L3.pth', path_data=path2, dataset = 'IMAGEWOOF')  #DONE


# fine_tune_model(n_class=10, fname='ARViT-L4' , path_model='ARViT-L4.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

# fine_tune_model(n_class=10, fname='ARViT-L4_Best' , path_model='ARViT-L4.pkl', path_weights='ARViT-L4.pth', path_data=path2, dataset = 'IMAGEWOOF')  #DONE


# fine_tune_model(n_class=10, fname='ARViT-L5' , path_model='ARViT-L5.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

# fine_tune_model(n_class=10, fname='ARViT-L5_Best' , path_model='ARViT-L5.pkl', path_weights='ARViT-L5.pth', path_data=path2, dataset = 'IMAGEWOOF')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L6-G32' , path_model='ARViT-L6-G32.pkl', path_weights=None, path_data=path1, dataset = 'IMAGENETTE')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L6-G32_Best' , path_model='ARViT-L6-G32.pkl', path_weights='ARViT-L6-G32.pth', path_data=path1, dataset = 'IMAGENETTE')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L6-G32' , path_model='ARViT-L6-G32.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L6-G32_Best' , path_model='ARViT-L6-G32.pkl', path_weights='ARViT-L6-G32.pth', path_data=path2, dataset = 'IMAGEWOOF')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L6-G32' , path_model='ARViT-L6-G32.pkl', path_weights=None, path_data=path3, dataset = 'CIFAR')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L6-G32_Best' , path_model='ARViT-L6-G32.pkl', path_weights='ARViT-L6-G32.pth', path_data=path3, dataset = 'CIFAR')  #DONE

fine_tune_model(n_class=100, fname='ARViT-L6-G32' , path_model='ARViT-L6-G32.pkl', path_weights=None, path_data=path4, dataset = 'CIFAR100')  #DONE

fine_tune_model(n_class=100, fname='ARViT-L6-G32_Best' , path_model='ARViT-L6-G32.pkl', path_weights='ARViT-L6-G32.pth', path_data=path4, dataset = 'CIFAR100')  #DONE


fine_tune_model(n_class=10, fname='ARViT-L1-G32' , path_model='ARViT-L1-G32.pkl', path_weights=None, path_data=path1, dataset = 'IMAGENETTE')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L1-G32_Best' , path_model='ARViT-L1-G32.pkl', path_weights='ARViT-L1-G32.pth', path_data=path1, dataset = 'IMAGENETTE')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L1-G32' , path_model='ARViT-L1-G32.pkl', path_weights=None, path_data=path2, dataset = 'IMAGEWOOF')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L1-G32_Best' , path_model='ARViT-L1-G32.pkl', path_weights='ARViT-L1-G32.pth', path_data=path2, dataset = 'IMAGEWOOF')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L1-G32' , path_model='ARViT-L1-G32.pkl', path_weights=None, path_data=path3, dataset = 'CIFAR')  #DONE

fine_tune_model(n_class=10, fname='ARViT-L1-G32_Best' , path_model='ARViT-L1-G32.pkl', path_weights='ARViT-L1-G32.pth', path_data=path3, dataset = 'CIFAR')  #DONE

fine_tune_model(n_class=100, fname='ARViT-L1-G32' , path_model='ARViT-L1-G32.pkl', path_weights=None, path_data=path4, dataset = 'CIFAR100')  #DONE

fine_tune_model(n_class=100, fname='ARViT-L1-G32_Best' , path_model='ARViT-L1-G32.pkl', path_weights='ARViT-L1-G32.pth', path_data=path4, dataset = 'CIFAR100')  #DONE

