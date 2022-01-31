from fastai.vision.all import *
from fastai.distributed import *
from fastai.metrics import error_rate
from fastai.callback.tracker import SaveModelCallback

from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import torchvision.transforms as T
import torch
from torch.nn.parallel import DistributedDataParallel
from torchvision.transforms.functional import *

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


H = 256
W= 256
bs = 5


def paths(layer = 1, dataset = 'IMAGENETTE', base_model=False, best_model=False): 
    model_dir = Path.home()/'Luiz/saved_models'
    if base_model == False:
        model_path = model_dir/'BaseModel_ImageNet_Rotation_16x16grid_epochs-90.pkl'
    else:
        model_path = model_dir/'GramImageNet_Rotation_16x16grid_epochs-90-beta-5e-6_PenaltyFactor3_Layer'+str(layer)+'.pkl'
        
    if best_model == False:
        weight_dir = Path.home()/'Luiz/gan_attention/models/finetuned/'
        file_name = 'LAYER_'+str(layer)+'_LOSS_3__FineTuned__'+dataset+'__BestWeights.pth'
        weights_path = weight_dir/file_name
    else:
        weight_dir = Path.home()/'Luiz/gan_attention/models/finetuned/'
        file_name = 'LAYER_'+str(layer)+'_LOSS_3__BestModel__FineTuned__'+dataset+'__BestWeights.pth'
        weights_path = weight_dir/file_name
    
    if dataset == 'IMAGENETTE':
        data_path = untar_data(URLs.IMAGENETTE)
    elif dataset == 'IMAGEWOOF':
        data_path = untar_data(URLs.IMAGEWOOF)
    elif dataset == 'CIFAR':
        data_path = untar_data(URLs.CIFAR)
    elif dataset == 'CIFAR_100':
        data_path = untar_data(URLs.CIFAR_100)    
    else:
        data_path = None
        
    return model_path, weights_path, data_path


def get_gm(r):
    label = parent_label(r)
    a = attrgetter("name")
    rgex = RegexLabeller(pat = r'image(.*?).jpeg') 
    gm = torch.load(save_path+"/gramm/"+str(label)+"/gm"+rgex(a(r))+".pt")
    return gm, TensorCategory(int(label))

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

def model_test(layer = 1, dataset = 'IMAGENETTE', base_model=False, best_model=False, n_class, epochs=1, lr=1e-9):
    path_model, weights_path, path_data = paths(layer, dataset, base_model, best_model)
    #print("Training on", dataset_name)
    model = load_model(path_model, None)
    model_head(model, n_class)
    if best_model != None:
        weight_dict = load_learner(weights_path, cpu=False)
        model.load_state_dict(weight_dict)
        print("weights loaded")
        
    dloader = data_loader(path_data)
    #Defining the Loss Function
    critic_loss = SingleLabelCriticLoss()
    
    #Wraping the Learner
    learner = Learner(dloader, model, loss_func=critic_loss, metrics=[_Accuracy])
    learner.fit_one_cycle(epochs, lr)
    #learner.fine_tune(60, base_lr=5e-4, freeze_epochs=30)