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

from .models.backbone import Backbone
from .models.encoder import EncoderModule
from .models.joiner import Joiner
from .models.losses import Attention_penalty_factor, Generator_loss
from .models.unet import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_transform(H,W, normalize:bool=True):
    if normalize == True:
        transform = T.Compose([
        T.Resize((H,W)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform
    else:
        transform = T.Compose([
        T.Resize((H,W)),
        T.ToTensor()
        ])
        return transform



def train_model(model, inputs, labels, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    
    real_inputs = inputs.to(device) # allow gpu use
    labels = labels.to(device) # allow gpu use
    # Pass real images through discriminator
    real_preds, pattn = model(real_inputs)
    #real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = model_criterion(real_preds, labels)
    real_score = torch.mean(real_preds).item()
    
    _, preds_real = torch.max(outputs, 1)
    running_real_corrects = torch.sum(preds_real == labels.data)
    
    # Generate fake images
    noised_inputs = generator(real_inputs)

    # Pass fake images through discriminator
    noised_preds, noised_pattn = model(noised_inputs)
    noised_loss = model_criterion(noised_preds, labels)
    noised_score = torch.mean(noised_preds).item()
    
    _, preds_noised = torch.max(outputs, 1)
    running_noised_corrects = torch.sum(preds_noised == labels.data)

    # Update discriminator weights
    loss = real_loss + noised_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, noised_score, running_real_corrects, running_noised_correct

def train_generator(inputs, labels, opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    real_inputs = inputs.to(device) # allow gpu use
    labels = labels.to(device) # allow gpu use
    
    # Generate fake images
    noised_inputs = generator(real_inputs)
    
    # Try to fool the discriminator
    preds, pattn = model(noised_inputs)
    model_loss = model_criterion(preds, labels)
    loss = gen_criterion(pattn, noised_inputs, real_inputs, model_loss)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

def fit(model, generator, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #torch.cuda.empty_cache()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, att, feat = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            torch.save(model.state_dict(),'test/epoch_' + str(epoch) + '.pth')
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),'test/TOP_PERFORMANCE_epoch_' + str(epoch) + '.pth')
    return model

