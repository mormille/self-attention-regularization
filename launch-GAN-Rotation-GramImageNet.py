#python -m torch.distributed.launch --nproc_per_node={num_gpus} launch.py

from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.gan import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback
from fastai import torch_core


from fastprogress import fastprogress
import torch
import argparse
#from models.utils.gan_joiner import GAN
from models.utils.joiner3 import *
from models.utils.new_losses import *
#from models.utils.metrics import *
from models.utils.misc import *
from models.unet import UNet
from models.utils.datasets import *

from torchvision import datasets, transforms, models
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

H = 256
W= 256
bs = 32
grid_l = 16
nclass = 4
pf = "3"
epochs = 2

beta = 0.00001 #0.0002
beta_str = '5e-6'
gamma = 0.0005
sigma = 1.0

lr = 2e-4
lr_str = '2e-4'

file_name = "GAN_GramImageNet_Rotation_16x16grid_epochs-"+str(epochs)+"-beta-"+beta_str+"_lr-"+lr_str+"PenaltyFactor3.pkl"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#path = './data/ImageNetRotation1k/'
#path = untar_data(URLs.IMAGENETTE_320)
path = Path.home()/'Luiz/gan_attention/data/Custom_ImageNet'
save_path = 'data/Custom_ImageNet'

def get_gm(r):
    label = parent_label(r)
    a = attrgetter("name")
    rgex = RegexLabeller(pat = r'image(.*?).jpeg') 
    gm = torch.load(save_path+"/gramm/"+str(label)+"/gm"+rgex(a(r))+".pt")
    return gm, TensorCategory(int(label))

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

gen = UNet(n_channels=3, n_classes=3, bilinear=False)
crt = ImageNetJoiner(num_encoder_layers = 6, nhead=8, num_classes = nclass, batch_size=bs, hidden_dim=512, image_h=H, image_w=W, grid_l=grid_l)

generator_loss = GeneratorLoss(beta, gamma,sigma)
critic_loss = CriticLoss(beta=beta, sigma=sigma, enc_layer_idx=0, pf=pf)



def _before_fit(self):
    opt_kwargs = { 'find_unused_parameters' : DistributedTrainer.fup } if DistributedTrainer.fup is not None else {}
    self.learn.model = DistributedDataParallel(
        nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model,
        device_ids=[self.cuda_id], output_device=self.cuda_id, find_unused_parameters=True, **opt_kwargs)
    self.old_dls = list(self.dls)
    self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
    if rank_distrib(): self.learn.logger=noop
DistributedTrainer.before_fit = _before_fit

def _set_freeze_model(m, rg):
    if type(m) == ImageNetJoiner:
        m.paramsToUpdate()
    else:
        for p in m.parameters(): p.requires_grad_(rg)
set_freeze_model = _set_freeze_model

class _GANModule(Module):
    "Wrapper around a `generator` and a `critic` to create a GAN."
    def __init__(self, generator=None, critic=None, gen_mode=False):
        if generator is not None: self.generator=generator
        if critic    is not None: self.critic   =critic
        store_attr('gen_mode')

    def forward(self, *args):
        #print(self.gen_mode)
        return self.generator(*args) if self.gen_mode else self.critic(*args)

    def switch(self, gen_mode=None):
        "Put the module in generator mode if `gen_mode`, in critic mode otherwise."
        self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
GANModule = _GANModule

class _GANLoss(GANModule):
    "Wrapper around `crit_loss_func` and `gen_loss_func`"
    def __init__(self, gen_loss_func, crit_loss_func, gan_model):
        super().__init__()
        store_attr('gen_loss_func,crit_loss_func,gan_model')

    def generator(self, input, output, target):
        "Evaluate the `output` with the critic then uses `self.gen_loss_func`"
        fake_pred = self.gan_model.critic(output[0])
        self.gen_loss = self.gen_loss_func(fake_pred, output, target)
        return self.gen_loss

    def critic(self, input, real_pred, target):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.crit_loss_func`."
        #print(input.shape)
        with torch.no_grad():
            fake = self.gan_model.generator(input)#.requires_grad_(False)
        fake_pred = self.gan_model.critic(fake[0])
        self.crit_loss = self.crit_loss_func(real_pred, target) + self.crit_loss_func(fake_pred, target)
        return self.crit_loss

GANLoss = _GANLoss

@delegates()
class _GANLearner(Learner):
    "A `Learner` suitable for GANs."
    def __init__(self, dls, generator, critic, gen_loss_func, crit_loss_func, switcher=None, gen_first=False,
                 switch_eval=True, show_img=True, clip=None, cbs=None, metrics=None, **kwargs):
        #print("Creating Learner")
        gan = GANModule(generator, critic)
        loss_func = GANLoss(gen_loss_func, crit_loss_func, gan)
        if switcher is None: switcher = FixedGANSwitcher(n_crit=5, n_gen=1)
        trainer = GANTrainer(clip=clip, switch_eval=switch_eval, gen_first=gen_first, show_img=show_img)
        cbs = L(cbs) + L(trainer, switcher)
        metrics = L(metrics) + L(*LossMetrics('gen_loss,crit_loss'))
        super().__init__(dls, gan, loss_func=loss_func, cbs=cbs, metrics=metrics, **kwargs)

    @classmethod
    def from_learners(cls, gen_learn, crit_learn, switcher=None, weights_gen=None, **kwargs):
        "Create a GAN from `learn_gen` and `learn_crit`."
        losses = gan_loss_from_func(gen_learn.loss_func, crit_learn.loss_func, weights_gen=weights_gen)
        return cls(gen_learn.dls, gen_learn.model, crit_learn.model, *losses, switcher=switcher, **kwargs)

    @classmethod
    def wgan(cls, dls, generator, critic, switcher=None, clip=0.01, switch_eval=False, **kwargs):
        "Create a WGAN from `data`, `generator` and `critic`."
        return cls(dls, generator, critic, _tk_mean, _tk_diff, switcher=switcher, clip=clip, switch_eval=switch_eval, **kwargs)

GANLearner = _GANLearner


def _before_batch(self):
    "Clamp the weights with `self.clip` if it's not None, set the correct input/target."
    if self.training and self.clip is not None:
        for p in self.critic.parameters(): p.data.clamp_(-self.clip, self.clip)
    if not self.gen_mode:
        (self.learn.xb,self.learn.yb) = (self.xb,self.yb)
GANTrainer.before_batch = _before_batch


def __set_trainable(self):
    train_model = self.generator if     self.gen_mode else self.critic
    loss_model  = self.generator if not self.gen_mode else self.critic
    set_freeze_model(train_model, True)
    set_freeze_model(loss_model, False)
    if self.switch_eval:
        train_model.train()
        loss_model.eval()
GANTrainer._set_trainable = __set_trainable


def __do_one_batch(self):
    #print("type x", type(*self.xb))
    #print("type y", type(*self.yb))
    self.pred = self.model(*self.xb)
    self('after_pred')
    if len(self.yb):
        self.loss_grad = self.loss_func(*self.xb, self.pred, *self.yb)
        self.loss = self.loss_grad.clone()
    self('after_loss')
    if not self.training or not len(self.yb): return
    self('before_backward')
    self.loss_grad.backward()
    self._with_events(self.opt.step, 'step', CancelStepException)
    self.opt.zero_grad()
Learner._do_one_batch = __do_one_batch


####################################
####################################
###########################
############################

class _FixedGANSwitcher(Callback):
    "Switcher to do `n_crit` iterations of the critic then `n_gen` iterations of the generator."
    run_after = GANTrainer
    def __init__(self, n_crit=1, n_gen=1): store_attr('n_crit,n_gen')
    def before_train(self): self.n_c,self.n_g = 0,0

    def after_batch(self):
        "Switch the model if necessary."
        #print("After Batch")
        if not self.training: return
        if self.learn.gan_trainer.gen_mode:
            self.n_g += 1
            n_iter,n_in,n_out = self.n_gen,self.n_c,self.n_g
        else:
            #print("After batch Else")
            self.n_c += 1
            n_iter,n_in,n_out = self.n_crit,self.n_g,self.n_c
        target = n_iter if isinstance(n_iter, int) else n_iter(n_in)
        #print(target)
        #print(n_out)
        if target == n_out:
            self.learn.gan_trainer.switch()
            self.n_c,self.n_g = 0,0
FixedGANSwitcher = _FixedGANSwitcher

def _switch(self, gen_mode=None):
    "Switch the model and loss function, if `gen_mode` is provided, in the desired mode."
    self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
    self._set_trainable()
    #self.model.switch(gen_mode)
    #self.loss_func.switch(gen_mode)
GANTrainer.switch = _switch

def __set_trainable(self):
    train_model = self.generator if     self.gen_mode else self.critic
    loss_model  = self.generator if not self.gen_mode else self.critic
    set_freeze_model(train_model, True)
    set_freeze_model(loss_model, False)
    if self.switch_eval:
        train_model.train()
        loss_model.eval()
GANTrainer._set_trainable = __set_trainable

def _accumulate(self, learn):
        bs = find_bs(learn.yb)
        if self.attr == 'gen_loss':
            self.total += learn.to_detach(getattr(learn.loss_func, self.attr, 0))*bs
        else:
            self.total += learn.to_detach(getattr(learn.loss_func, self.attr, 0))*bs
        self.count += bs
LossMetric.accumulate = _accumulate

def _to_detach(self,b, cpu=True, gather=True):
    b = self._to_detach(b, cpu, gather)
    #print("B ==========================================", len(b))
    def _inner(b):
        if torch.is_tensor(b) == True:
            if b.ndim>0:
                # for each rank, compute overflow of read idxs vs self.n and accumulate them to unpad totals after gathering
                n = sum([min(0,max(-len(b)//self.world_size,
                                   self.n-(self.i+r*self.n_padded//self.world_size))) for r in range(self.world_size)])
                b = b[:n or None]
            return b
        else:
            return b
    return apply(_inner,b) if gather and all(hasattr(self,o) for o in ('i','n','n_padded')) else b
DistributedDL.to_detach = _to_detach



#metrics

def GAN_Accuracy(preds,target): 
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        _, pred = torch.max(fakePreds[0], 1)
        return (pred.cuda() == target[1].cuda()).float().mean()
    else:
        _, pred = torch.max(preds[0], 1)
        return (pred.cuda() == target[1].cuda()).float().mean()
    
MSE = nn.MSELoss()    
def GAN_Reconstruction_Loss(preds,target,sigma=1):
    if len(preds) == 2:
        Lrec = sigma*MSE(preds[0],preds[1])
    else:
        Lrec = 0.000
  
    return Lrec

LCA2 = Curating_of_attention_loss(pf="2")
def GAN_Curating_Of_Attention_Loss(preds,target):
    if len(preds) == 2:
        fakePreds = learner.gan_trainer.critic(preds[0])
        Latt = LCA2(fakePreds[1], fakePreds[3])
        return (beta*Latt).float().mean()
    else:
        Latt = LCA2(preds[1], preds[3])
    
        return (beta*Latt).float().mean()

#metrics=[Accuracy,Reconstruction_Loss]

#print("Number of Training Images:", len(dld.train)*bs)
#print("Number of Validation Images:", len(dld.valid)*bs)
#print("Batch Size:", bs)

#plateau = ReduceLROnPlateau(monitor='Accuracy', patience=5)
#save_best = SaveModelCallback(monitor='train_loss', fname='Best_BaseModel_ImageNet_Rotation_16x16grid')

learner = GANLearner(dloader,gen,crt,generator_loss,critic_loss,gen_first=True,metrics=[GAN_Accuracy,GAN_Reconstruction_Loss,GAN_Curating_Of_Attention_Loss]).to_distributed(args.local_rank)

learner.fit_one_cycle(epochs, 2e-4)

model_dir = Path.home()/'Luiz/saved_models'
learner.export(model_dir/'Cifar_Gan_Test.pkl')
