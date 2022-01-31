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

H = 256
W= 256
bs = 5

model_dir = Path.home()/'Luiz/saved_models'
model_path1 = model_dir/'GramImageNet_Rotation_16x16grid_epochs-90-beta-5e-6_lr-3e-4PenaltyFactor3.pkl'
model_path2 = model_dir/'GramImageNet_Rotation_16x16grid_epochs-90-beta-5e-6_lr-1e-4PenaltyFactor3_Layer2.pkl'
model_path3 = model_dir/'GramImageNet_Rotation_16x16grid_epochs-90-beta-5e-6_lr-3e-4PenaltyFactor3_lastLayer.pkl'
best_model_path1 = model_dir/'Best_BaseModel_ImageNet_Rotation_16x16grid.pth'
best_model_path2 = model_dir/'Best_Model_Layer2_ImageNet_Rotation_16x16grid.pth'
best_model_path3 = model_dir/'Loss3_LastLayer_ImageNet_Rotation_16x16grid.pth'

path1 = untar_data(URLs.IMAGENETTE)
path2 = untar_data(URLs.IMAGEWOOF)
path3 = untar_data(URLs.CIFAR)
path4 = untar_data(URLs.CIFAR_100)

def get_gm(r):
    label = parent_label(r)
    a = attrgetter("name")
    rgex = RegexLabeller(pat = r'image(.*?).jpeg') 
    gm = torch.load(save_path+"/gramm/"+str(label)+"/gm"+rgex(a(r))+".pt")
    return gm, TensorCategory(int(label))

def paths(layer = 1, dataset = 'IMAGENETTE', base_model=False, best_model=False, pf="3", gm16=False): 
    model_dir = Path.home()/'Luiz/saved_models'
    if base_model == True:
        model_path = model_dir/'GramImageNet_Rotation_16x16grid_epochs-90_BaseModel.pkl'
    else:
        if gm16 == True:
            file_name = 'GramImageNet_Rotation_16x16grid_epochs-90-beta-5e-7_PenaltyFactor'+pf+'_Layer'+str(layer)+'_gm16.pkl'
        else:
            file_name = 'GramImageNet_Rotation_16x16grid_epochs-90-beta-5e-6_PenaltyFactor'+pf+'_Layer'+str(layer)+'.pkl'
        model_path = model_dir/file_name
        
    if best_model == True:
        weight_dir = Path.home()/'Luiz/gan_attention/models/pretrained/'
        if base_model == True:
            file_name = 'Best_Model_BaseModel_ImageNet_Rotation_16x16grid.pth'
        elif pf !="3":
            file_name = 'Best_Model_Layer'+str(layer)+'_ImageNet_Rotation_16x16grid_Loss'+pf+'.pth'
        else:
            if gm16 == True:
                file_name = 'Best_Model_Layer'+str(layer)+'_ImageNet_Rotation_16x16grid_gm16.pth'
            else:
                file_name = 'Best_Model_Layer'+str(layer)+'_ImageNet_Rotation_16x16grid.pth'
        weights_path = weight_dir/file_name
    else:
        weights_path = None
    
    if dataset == 'IMAGENETTE':
        data_path = untar_data(URLs.IMAGENETTE)
    elif dataset == 'IMAGEWOOF':
        data_path = untar_data(URLs.IMAGEWOOF)
    elif dataset == 'CIFAR':
        data_path = untar_data(URLs.CIFAR)
    elif dataset == 'CIFAR_100':
        data_path = untar_data(URLs.CIFAR_100)    
    elif dataset == 'CALTECH_101':    
        data_path = untar_data(URLs.CALTECH_101) 
    else:
        data_path = None
        
    return model_path, weights_path, data_path


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

def fine_tune_model(n_class, layer = 1, dataset = 'IMAGENETTE', base_model=False, best_model=False, epochs=[60,30], lr=5e-4, pf="3", gm16=False):
    path_model, weights_path, path_data = paths(layer, dataset, base_model, best_model, pf, gm16)
    print(path_model, weights_path, path_data)
    print("Training on", dataset)
    model = load_model(path_model, weights_path)
    model_head(model, n_class)
        
    dloader = data_loader(path_data)
    #Defining the Loss Function
    critic_loss = SingleLabelCriticLoss()
    
    if base_model == True:
        layer = 'BASE_MODEL'
        print(layer)
    
    if best_model != False:
        name = 'LAYER_'+str(layer)+'_LOSS_'+pf+'_' + '_BestModel_'
    else:
        name = 'LAYER_'+str(layer)+'_LOSS_'+pf+'_'

    plateau = ReduceLROnPlateau(monitor='_Accuracy', patience=4)
    save_best = SaveModelCallback(monitor='_Accuracy', fname='finetuned/'+name+'_FineTuned__'+dataset+'__BestWeights_gm16')

    #Wraping the Learner
    learner = Learner(dloader, model, loss_func=critic_loss, metrics=[_Accuracy], cbs=[save_best, plateau]).to_distributed(args.local_rank)
    #learner.fit_one_cycle(50, 0.002)
    learner.fine_tune(epochs[0], base_lr=lr, freeze_epochs=epochs[1])
    

fine_tune_model(10, layer = 3, dataset = 'IMAGENETTE', base_model=False, best_model=False, pf = "3", gm16 = True)  #DONE
fine_tune_model(10, layer = 3, dataset = 'IMAGENETTE', base_model=False, best_model=True, pf = "3", gm16 = True)   #DONE

fine_tune_model(10, layer = 3, dataset = 'IMAGEWOOF', base_model=False, best_model=False, pf = "3", gm16 = True)   #DONE
fine_tune_model(10, layer = 3, dataset = 'IMAGEWOOF', base_model=False, best_model=True, pf = "3", gm16 = True)    #DONE

#fine_tune_model(10, layer = 2, dataset = 'CIFAR', base_model=False, best_model=False, pf = "3", gm16 = True)  #DONE
#fine_tune_model(10, layer = 2, dataset = 'CIFAR', base_model=False, best_model=True, pf = "3", gm16 = True)   #DONE

#fine_tune_model(100, layer = 2, dataset = 'CIFAR_100', base_model=False, best_model=False, pf = "3", gm16 = True)   #DONE
#fine_tune_model(100, layer = 2, dataset = 'CIFAR_100', base_model=False, best_model=True, pf = "3", gm16 = True)    #DONE



