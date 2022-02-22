from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.gan import *
from fastai.metrics import *
from fastai.callback.tracker import SaveModelCallback
from fastai import torch_core


__all__ = ['_before_fit','_set_freeze_model','_GANModule','_GANLoss','_GANLearner','_before_batch','__set_trainable','__do_one_batch','_FixedGANSwitcher','_switch','__set_trainable','_accumulate','_to_detach']

def _before_fit(self):
    opt_kwargs = { 'find_unused_parameters' : DistributedTrainer.fup } if DistributedTrainer.fup is not None else {}
    self.learn.model = DistributedDataParallel(
        nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model,
        device_ids=[self.cuda_id], output_device=self.cuda_id, find_unused_parameters=True, **opt_kwargs)
    self.old_dls = list(self.dls)
    self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
    if rank_distrib(): self.learn.logger=noop

def _set_freeze_model(m, rg):
        m.paramsToUpdate(rg)

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


def _before_batch(self):
    "Clamp the weights with `self.clip` if it's not None, set the correct input/target."
    if self.training and self.clip is not None:
        for p in self.critic.parameters(): p.data.clamp_(-self.clip, self.clip)
    if not self.gen_mode:
        (self.learn.xb,self.learn.yb) = (self.xb,self.yb)


def __set_trainable(self):
    train_model = self.generator if     self.gen_mode else self.critic
    loss_model  = self.generator if not self.gen_mode else self.critic
    set_freeze_model(train_model, True)
    set_freeze_model(loss_model, False)
    if self.switch_eval:
        train_model.train()
        loss_model.eval()


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


def _switch(self, gen_mode=None):
    "Switch the model and loss function, if `gen_mode` is provided, in the desired mode."
    self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
    self._set_trainable()
    #self.model.switch(gen_mode)
    #self.loss_func.switch(gen_mode)

def __set_trainable(self):
    train_model = self.generator if     self.gen_mode else self.critic
    loss_model  = self.generator if not self.gen_mode else self.critic
    set_freeze_model(train_model, True)
    set_freeze_model(loss_model, False)
    if self.switch_eval:
        train_model.train()
        loss_model.eval()


def _accumulate(self, learn):
        bs = find_bs(learn.yb)
        if self.attr == 'gen_loss':
            self.total += learn.to_detach(getattr(learn.loss_func, self.attr, 0))*bs
        else:
            self.total += learn.to_detach(getattr(learn.loss_func, self.attr, 0))*bs
        self.count += bs

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
