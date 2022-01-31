
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone
from models.unet import UNet


from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.gan import *
from fastai.callback.tracker import SaveModelCallback
from fastai import torch_core


__all__ = ['_GANModule','_GANLoss','_FixedGANSwitcher','_before_batch','_switch','_set_freeze_model','_GANLearner']

class _GANModule(Module):
    "Wrapper around a `generator` and a `critic` to create a GAN."
    def __init__(self, generator=None, critic=None, gen_mode=False):
        #print("Custom GAN Module")
        if generator is not None: self.generator=generator
        if critic    is not None: self.critic   =critic
        store_attr('gen_mode')

    def forward(self, *args):
        #print(*args)
        return self.generator(*args) if self.gen_mode else self.critic(*args)

    def switch(self, gen_mode=None):
        "Put the module in generator mode if `gen_mode`, in critic mode otherwise."
        self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
        
        

class _GANLoss(GANModule):
    "Wrapper around `crit_loss_func` and `gen_loss_func`"
    def __init__(self, gen_loss_func, crit_loss_func, gan_model):
        super().__init__()
        store_attr('gen_loss_func,crit_loss_func,gan_model')

    def generator(self, output, target):
        "Evaluate the `output` with the critic then uses `self.gen_loss_func`"
        fake_pred = self.gan_model.critic(output)
        self.gen_loss = self.gen_loss_func(fake_pred, target)
        return self.gen_loss

    def critic(self, real_pred, input):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.crit_loss_func`."
        #print("GANLoss - Critic Loss")
        for param in self.gan_model.generator.parameters():
            param.requires_grad_(False)
        fake = self.gan_model.generator(real_pred[4])
        fake_pred = self.gan_model.critic(fake)
        self.crit_loss = self.crit_loss_func(real_pred, input) + self.crit_loss_func(fake_pred, input)
        return self.crit_loss
    
    
    
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

            
def _before_batch(self):
    "Clamp the weights with `self.clip` if it's not None, set the correct input/target."
    if self.training and self.clip is not None:
        for p in self.critic.parameters(): p.data.clamp_(-self.clip, self.clip)
    if not self.gen_mode:
        (self.learn.xb,self.learn.yb) = (self.xb,self.yb)



def _switch(self, gen_mode=None):
    "Switch the model and loss function, if `gen_mode` is provided, in the desired mode."
    self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
    self._set_trainable()
    self.model.switch(gen_mode)
    self.loss_func.switch(gen_mode)
    
    
    
def _set_freeze_model(m, rg):
    if type(m) == Joiner:
        m.paramsToUpdate()
    else:
        for p in m.parameters(): p.requires_grad_(rg)
            
            
class _GANLearner(Learner):
    "A `Learner` suitable for GANs."
    def __init__(self, dls, generator, critic, gen_loss_func, crit_loss_func, switcher=None, gen_first=False,
                 switch_eval=True, show_img=True, clip=None, cbs=None, metrics=None, **kwargs):
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
    
    
