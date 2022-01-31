#python -m torch.distributed.launch --nproc_per_node={num_gpus} launch.py

from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.gan import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback, ReduceLROnPlateau
from fastai import torch_core


from fastprogress import fastprogress
from torchvision import datasets, transforms, models
import os
import copy
import torchvision.transforms as T
import torch

import numpy as np
import torch.nn.functional as F
from torch import nn

import argparse
from models.utils.new_losses import *
from models.utils.metrics import _Accuracy
from models.utils.dataLoader import *
from models.utils.datasets import *
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
#print(args)
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

#PARAMETERS

def get_gm(r):
    label = parent_label(r)
    a = attrgetter("name")
    rgex = RegexLabeller(pat = r'image(.*?).jpeg') 
    gm = torch.load(save_path+"/gramm/"+str(label)+"/gm"+rgex(a(r))+".pt")
    return gm, TensorCategory(int(label))


model_dir = Path.home()/'Luiz/saved_models'
net = load_learner(model_dir/'BaseModel_ImageNet_Rotation_16x16grid_epochs-90-beta-5e-6_lr-3e-4PenaltyFactor2.pkl', cpu=False)
model = net.model
model_name = 'Base_Model'

best_model = False
if best_model == True:
    #weight_dict = load_learner(model_dir/'Best_BaseModel_ImageNet_Rotation_16x16grid.pth', cpu=False)
    model.load_state_dict(load_learner(model_dir/'Best_BaseModel_ImageNet_Rotation_16x16grid.pth', cpu=False))

model.head = nn.Linear(512*16*16, 1000)
#model.noise_mode = True
#model.generator_mode = False

trainable = ['head.weight','head.bias']
for name, p in model.named_parameters():
    if name not in trainable:
        p.requires_grad = False
    else:
        p.requires_grad = True
        
#for name, p in model.generator.named_parameters():
#    p.requires_grad = False
        
H = 256
W= 256
bs = 100

path = Path.home()/'Luiz/gan_attention/data/ImageNet'
save_path = 'data/ImageNet'

transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dblock = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=Resize(H,W),
                 batch_tfms=transform)
dsets = dblock.datasets(path)

dloader = dblock.dataloaders(path, bs=bs)


#Defining the Loss Function
critic_loss = SingleLabelCriticLoss()

plateau = ReduceLROnPlateau(monitor='_Accuracy', patience=3)
save_best = SaveModelCallback(monitor='_Accuracy', fname='finetuned/IMAGENET_FineTuned'+model_name+'BestWeights')

#Wraping the Learner
learner = Learner(dloader, model, loss_func=critic_loss, metrics=[_Accuracy], cbs=[save_best,plateau]).to_distributed(args.local_rank)

#learner.fit_one_cycle(50, 0.002)

learner.fine_tune(50, base_lr=2e-3, freeze_epochs=10)

#fb = ["mask","penalty_mask","pos"]
#for name, p in model.named_parameters():
#    if name not in fb:
#        p.requires_grad_(True)
    
#learner.fit(30,5e-9)

#model_dir = Path.home()/'Luiz/saved_models/downstream'
#learner.export(model_dir/file_name)

