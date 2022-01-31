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
from models.utils.new_losses import SingleLabelCriticLoss
from models.Vit import _create_vision_transformer
from models.utils.metrics import _Accuracy
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

#train_path  = "data/WebDataset-GramCifar/train/GramCifar-{0..4}.tar"
#valid_path = "data/WebDataset-GramCifar/valid/GramCifar-0.tar"

def _before_fit(self):
    opt_kwargs = { 'find_unused_parameters' : DistributedTrainer.fup } if DistributedTrainer.fup is not None else {}
    self.learn.model = DistributedDataParallel(
        nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model,
        device_ids=[self.cuda_id], output_device=self.cuda_id, find_unused_parameters=True, **opt_kwargs)
    self.old_dls = list(self.dls)
    self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
    if rank_distrib(): self.learn.logger=noop
DistributedTrainer.before_fit = _before_fit

H = 32
W= 32
bs = 12
ps = 1
nclass = 10
epochs = 40
epoch_list = [5,5,10,10,10]

lr = [1e-6,5e-7,1e-7,5e-8,1e-8]
lr_str = '1e-6'

file_name = "Vit_StdCifar_epochs-"+str(epochs)+"_lr-"+lr_str+".pkl"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Creating the dataloader
path = untar_data(URLs.CIFAR)

transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=Resize(H,W),
                 batch_tfms=transform)

dloader = data.dataloaders(path,bs=bs) 

#Defining the Loss Function
critic_loss = SingleLabelCriticLoss()

#Building the model
model = _create_vision_transformer('vit_base_patch16_224',pretrained=False, img_size=H, patch_size=ps, 
                                 in_chans=3, num_classes=nclass, depth=6, num_heads=6)

#Wraping the Learner
learner = Learner(dloader, model, loss_func=critic_loss, metrics=[_Accuracy]).to_distributed(args.local_rank)
for i in range(len(lr)):
    learner.fit(epoch_list[i], lr[i])

model_dir = Path.home()/'Luiz/saved_models'
learner.export(model_dir/file_name)
