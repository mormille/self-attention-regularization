#python -m torch.distributed.launch --nproc_per_node={num_gpus} launch.py

from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.tracker import SaveModelCallback
from fastprogress import fastprogress
import torch
import argparse
from models.utils.joiner2 import Joiner
from models.utils.losses import CriticLoss
from models.utils.metrics import Accuracy



parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

H = 32
W= 32
bs = 5
nclass = 10
backbone = False
epochs = 60

beta = 0.005
gamma = 0.0005
sigma = 1.0

seed = 1234

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


model = Joiner(num_encoder_layers = 6, nhead=8, backbone = backbone, num_classes = nclass, bypass=False, hidden_dim=256, 
          batch_size=bs, image_h=H, image_w=W,grid_l=4,penalty_factor="2")


#path = './data/ImageNetRotation1k/'
#path = untar_data(URLs.IMAGENETTE_320)
path = untar_data(URLs.CIFAR)

transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=Resize(H,W),
                 batch_tfms=transform)

dloader = data.dataloaders(path,bs=bs) 

# def load_data(bs):
#     path = rank0_first(lambda: untar_data(URLs.CIFAR)) 
    
#     item_tfms = [ Resize((H,W), method='squish')]
#     transform = [*aug_transforms(), Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

#     datablock = DataBlock(blocks=(ImageBlock, CategoryBlock), 
#                      get_items=get_image_files, 
#                      splitter=RandomSplitter(),
#                      get_y=parent_label,
#                      item_tfms=item_tfms,
#                      batch_tfms=transform)

#     return datablock.dataloaders(path/'train', bs=bs, num_workers=num_cpus())

#data = load_data(20)
critic_loss = CriticLoss(beta,sigma)

learner = Learner(dloader, model, loss_func=critic_loss, metrics=[Accuracy]).to_distributed(args.local_rank)
learner.fit(epochs, 2e-7)

model_dir = Path.home()/'Luiz/saved_models'
learner.export(model_dir/'Cifar_Transformer_Test6.pkl')