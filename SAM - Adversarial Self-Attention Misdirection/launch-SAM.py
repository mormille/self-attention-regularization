#python -m torch.distributed.launch --nproc_per_node={num_gpus} launch.py

from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.gan import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback, ReduceLROnPlateau
from fastai import torch_core


# from fastprogress import fastprogress
# from torchvision import datasets, transforms, models
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision.transforms as T
# import torch
# import torch.nn.functional as F
# from torch import nn
# import argparse

from models.unet import UNet
from models.SAM import SAM
from models.utils.fastai_gan import *
from losses.attention_loss import *
from losses.sam_loss import *
from torch.nn.parallel import DistributedDataParallel

# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int)
# args = parser.parse_args()
# torch.cuda.set_device(args.local_rank)
# torch.distributed.init_process_group(backend='nccl', init_method='env://')

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#SAVE FILE DETAILS
####################################
model_dir = Path.home()/'Luiz/saved_models/SAM'
file_name = "SAM_v0.pkl"
best_name = model_dir/'best/SAM_v0'
####################################


#Hyperparameters
####################################
epochs=90
lr=5e-5
####################################
bs=20
H=W=256
nclass=10
grid_l=gm_l=16
####################################

#Dataloader
####################################
path = Path.home()/'Luiz/gan_attention/data/Custom_ImageNet'
transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   n_inp=1,
                   get_items = get_image_files,
                   get_y     = parent_label,                  
                   splitter  = RandomSplitter(),
                   item_tfms = Resize(256),
                   batch_tfms= transform
                  )
dloader = dblock.dataloaders(path/"images", bs=bs)
####################################

#Models
####################################
gen = UNet(n_channels=3, n_classes=3, bilinear=False)
crt = SAM(enc_Layers=6, nhead=8, nclass=nclass, bs=bs, hidden_dim=512, H=H, W=W, grid_l=grid_l, gm_patch=gm_l)
####################################

#Losses
####################################
generator_loss = GeneratorLoss()
critic_loss = CriticLoss()
####################################

#Personalizing FASTAI GAN Module
####################################
Learner._do_one_batch = __do_one_batch
####################################

#Metrics
####################################
####################################
def Acc(preds,target): 
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds)
        _, pred = torch.max(fakePreds[0], 1)
        return (pred == target).float().mean()
    else:
        _, pred = torch.max(preds[0], 1)

        return (pred == target).float().mean()

MSE = nn.MSELoss()    
def Lrec(preds,target):
    if len(preds) == 2:
        Lrec = MSE(preds[0],preds[1]).float().mean()
    else:
        Lrec = 0.000
  
    return Lrec

LCA = Attention_loss()
def La1(preds,target,layer=0):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        Latt = LCA(fakePreds[1][layer], fakePreds[3])
        return (0.01*Latt).float().mean()
    else:
        Latt = LCA(preds[1][layer], preds[3])   
        return (0.01*Latt).float().mean()
    
def La2(preds,target,layer=1):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        Latt = LCA(fakePreds[1][layer], fakePreds[3])
        return (0.01*Latt).float().mean()
    else:
        Latt = LCA(preds[1][layer], preds[3])   
        return (0.01*Latt).float().mean()

def La3(preds,target,layer=2):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        Latt = LCA(fakePreds[1][layer], fakePreds[3])
        return (0.01*Latt).float().mean()
    else:
        Latt = LCA(preds[1][layer], preds[3])   
        return (0.01*Latt).float().mean()
    
def La4(preds,target,layer=3):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        Latt = LCA(fakePreds[1][layer], fakePreds[3])
        return (0.01*Latt).float().mean()
    else:
        Latt = LCA(preds[1][layer], preds[3])   
        return (0.01*Latt).float().mean()
    
def La5(preds,target,layer=4):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        Latt = LCA(fakePreds[1][layer], fakePreds[3])
        return (0.01*Latt).float().mean()
    else:
        Latt = LCA(preds[1][layer], preds[3])   
        return (0.01*Latt).float().mean()
    
def La6(preds,target,layer=5):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        Latt = LCA(fakePreds[1][layer], fakePreds[3])
        return (0.01*Latt).float().mean()
    else:
        Latt = LCA(preds[1][layer], preds[3])   
        return (0.01*Latt).float().mean()

c_entropy = nn.CrossEntropyLoss() 
def CrossEnt(preds,target):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds)
        Loss = c_entropy(fakePreds[0], target)
        return (Loss).float().mean()
    else:
        Loss = c_entropy(preds[0], target)
        return (Loss).float().mean()

LM = Misdirection_loss()
def Lm(preds,target,layers=[0,1,2,3,4,5],gammas=[0.0002,0.0002,0.0002,0.0002,0.0002,0.0002]):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        Lm = 0.0
        for i in range(len(layers)):
            Lm = Lm + gammas[i]*LM(fakePreds[1][layers[i]], fakePreds[3])
        return (Lm).float().mean()
    else:
        Lm = 0.0
        for i in range(len(layers)):
            Lm = Lm + gammas[i]*LM(preds[1][layers[i]],preds[3])
        return (Lm).float().mean()
####################################
####################################
    
#Learner
####################################
plateau = ReduceLROnPlateau(monitor='CrossEnt', patience=3)
save_best = SaveModelCallback(monitor='CrossEnt', fname=best_name)

learner = GANLearner(dloader,gen,crt,generator_loss,critic_loss,gen_first=False, metrics=[Acc,CrossEnt,Lm,Lrec,La1,La2,La3,La4,La5,La6]).to_fp16()

with learner.distrib_ctx(): learner.fit_one_cycle(epochs, lr)

learner.export(model_dir/file_name)
