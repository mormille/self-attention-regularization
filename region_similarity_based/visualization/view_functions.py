from fastai.vision.all import *
from fastai.distributed import *
from fastai.metrics import error_rate
from fastai.callback.tracker import SaveModelCallback
import argparse
from timeit import default_timer as timer

from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import torchvision.transforms as T
import torch
from torchvision.transforms.functional import *

from PIL import Image
import requests
import imageio

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn

import ipywidgets as widgets
from IPython.display import display, clear_output
import math

from ARViT.ARViT import *


__all__ = ['run_load_model','get_att_maps','show_attention_maps', 'AttentionVisualizer','get_gm','get_gm_view','view_gm','plot_patch_gm','plot3channels','view_sattn']

W = H = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
T.Resize((H,W)),
T.ToTensor()#,
#T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


transform2 = T.Compose([
T.Resize((H,W))
])

def get_gm(r):
    label = parent_label(r)
    a = attrgetter("name")
    rgex = RegexLabeller(pat = r'image(.*?).jpeg') 
    gm = torch.load(save_path+"/gramm/"+str(label)+"/gm"+rgex(a(r))+".pt")
    return gm, TensorCategory(int(label))


def get_att_maps(model, k):
    start = timer()
    im = Image.open('visualization/sample_images/image'+str(k)+'.jpeg')
    img = transform(im).unsqueeze(0).to(device)

    outputs, attn, sattn, gm  = model(img.to(device))

    a = 16
    maps = [sattn[0].reshape(1,a,a,a,a),#.permute(0,3,4,1,2)
    sattn[1].reshape(1,a,a,a,a),#.permute(0,3,4,1,2)
    sattn[2].reshape(1,a,a,a,a),#.permute(0,3,4,1,2)
    sattn[3].reshape(1,a,a,a,a),#.permute(0,3,4,1,2)
    sattn[4].reshape(1,a,a,a,a),#.permute(0,3,4,1,2)
    sattn[5].reshape(1,a,a,a,a)]#.permute(0,3,4,1,2)

    return maps, im

def get_gm_view(k):
    im = Image.open('visualization/sample_images/image'+str(k)+'.jpeg')
    img = transform(im).unsqueeze(0).to(device)
    GM = GM_Mask(patch_size=16, width=256, height=256)
    gm = GM(img)

    return im, gm

def view_gm(k,color='copper'):
    im, gm = get_gm_view(k)
    fig, axs = plt.subplots(1, 2, figsize=(12,8))
    axs[0].imshow(im)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(gm[0].cpu().detach().numpy(),cmap=color, interpolation='nearest')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.show()

def show_attention_maps(im,i,j,maps_list, color='cividis'):
    
    pix_h = i
    pix_w = j
    maps = [maps_list[0][0,pix_h,pix_w,...].cpu().detach().numpy(),
    maps_list[1][0,pix_h,pix_w,...].cpu().detach().numpy(),
    maps_list[2][0,pix_h,pix_w,...].cpu().detach().numpy(),
    maps_list[3][0,pix_h,pix_w,...].cpu().detach().numpy(),
    maps_list[4][0,pix_h,pix_w,...].cpu().detach().numpy(),
    maps_list[5][0,pix_h,pix_w,...].cpu().detach().numpy()]
    
    #start = timer()
    fig3 = plt.figure(constrained_layout=False, figsize=(16,7))
    fig3.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, hspace=0, wspace=0)
    gs = fig3.add_gridspec(2, 5)

    f3_ax1 = fig3.add_subplot(gs[:, :2])
    f3_ax1.set_title('Input Image', fontsize='xx-large')
    f3_ax1.imshow(im)
    f3_ax1.set_xticks([])
    f3_ax1.set_yticks([])
    f3_ax2 = fig3.add_subplot(gs[0, 2])
    f3_ax2.set_title('Layer 1', fontsize='x-large')
    f3_ax2.imshow(maps[0],cmap=color, interpolation='nearest')
    f3_ax2.set_xticks([])
    f3_ax2.set_yticks([])
    f3_ax3 = fig3.add_subplot(gs[0, 3])
    f3_ax3.set_title('Layer 2', fontsize='x-large')
    f3_ax3.imshow(maps[1],cmap=color, interpolation='nearest')
    f3_ax3.set_xticks([])
    f3_ax3.set_yticks([])
    f3_ax4 = fig3.add_subplot(gs[0, 4])
    f3_ax4.set_title('Layer 3', fontsize='x-large')
    f3_ax4.imshow(maps[2],cmap=color, interpolation='nearest')
    f3_ax4.set_xticks([])
    f3_ax4.set_yticks([])
    f3_ax5 = fig3.add_subplot(gs[1, 2])
    f3_ax5.set_title('Layer 4', fontsize='x-large')
    f3_ax5.imshow(maps[3],cmap=color, interpolation='nearest')
    f3_ax5.set_xticks([])
    f3_ax5.set_yticks([])
    f3_ax6 = fig3.add_subplot(gs[1, 3])
    f3_ax6.set_title('Layer 5', fontsize='x-large')
    f3_ax6.imshow(maps[4],cmap=color, interpolation='nearest')
    f3_ax6.set_xticks([])
    f3_ax6.set_yticks([])
    f3_ax7 = fig3.add_subplot(gs[1, 4])
    f3_ax7.set_title('Layer 6', fontsize='x-large')
    f3_ax7.imshow(maps[5],cmap=color, interpolation='nearest')
    f3_ax7.set_xticks([])
    f3_ax7.set_yticks([])

    fig3.subplots_adjust(hspace=0.0)
    title = 'Attention maps for position ('+str(i)+','+str(j)+')'
    fig3.suptitle(title, fontsize=28)
    
    
def get_att_out(model, k):
    start = timer()
    im = Image.open('visualization/sample_images/image'+str(k)+'.jpeg')
    img = transform(im).unsqueeze(0).to(device)

    outputs, attn, sattn, gm  = model(img.to(device))

    return sattn, im

def view_sattn(model, k, layer=3, color='copper',div=100):
    sattn, im = get_att_out(model,k)
    
    sattn_map = sattn[layer][0]
    threshold = torch.sum(sattn_map)/div
    
    fig, axs = plt.subplots(1, 2, figsize=(12,8))
    axs[0].imshow(im)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(sattn[layer][0].cpu().detach().numpy(),cmap=color, interpolation='nearest',vmax=threshold)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.show()

def paths(layer = 1, base_model=False, best_model=False, pf="3", gm16 = False): 
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
        
        
    return model_path, weights_path

def load_model(model_path, weights_path = None):
    model_dir = Path.home()/'Luiz/saved_models'
    net = load_learner(model_path, cpu=False)
    model = net.model.eval()

    if weights_path != None:
        weight_dict = load_learner(weights_path, cpu=False)
        model.load_state_dict(weight_dict)
    
    model.to(device)
    
    return model

def run_load_model(layer,base_model,best_model,pf,gm16=False):
    model_path, weight_path = paths(layer, base_model, best_model, pf, gm16)
    #print(model_path)
    #print(weight_path)
    model = load_model(model_path,weight_path)
    return model

class AttentionVisualizer:
    def __init__(self, model, attn_layer,img_url = None,img_idx = None, transform=transform, color='cividis'):
        self.model = model
        self.transform = transform

        self.url = ""
        self.cur_url = None
        self.pil_img = None
        self.tensor_img = None

        self.conv_features = None
        self.enc_attn_weights = None
        self.dec_attn_weights = None
        self.attn_layer = attn_layer
        self.img_url = img_url
        self.img_idx = img_idx
        self.color = color

        self.setup_widgets()

    def setup_widgets(self):
        self.sliders = [
            widgets.Text(
                value=self.img_url,
                placeholder='Type something',
                description='URL (ENTER):',
                disabled=False,
                continuous_update=False,
                layout=widgets.Layout(width='100%')
            ),
            widgets.FloatSlider(min=0, max=0.99,
                        step=0.02, description='X coordinate', value=0.72,
                        continuous_update=False,
                        layout=widgets.Layout(width='50%')
                        ),
            widgets.FloatSlider(min=0, max=0.99,
                        step=0.02, description='Y coordinate', value=0.40,
                        continuous_update=False,
                        layout=widgets.Layout(width='50%')),
            widgets.Checkbox(
              value=False,
              description='Direction of self attention',
              disabled=False,
              indent=False,
              layout=widgets.Layout(width='50%'),
          ),
            widgets.Checkbox(
              value=True,
              description='Show red dot in attention',
              disabled=False,
              indent=False,
              layout=widgets.Layout(width='50%'),
          )
        ]
        self.o = widgets.Output()

    def compute_features(self, img):
        model = self.model
        # use lists to store the outputs via up-values
        outputs, dec_attn_weights, enc_attn_weights, gm = model(img.to(device))
        #conv_features, enc_attn_weights, dec_attn_weights = model(img.to(device))

        # don't need the list anymore
        self.conv_features = outputs
        self.dec_attn_weights = dec_attn_weights
        # get the HxW shape of the feature maps of the CNN
        #print(enc_attn_weights.shape)
        shape = enc_attn_weights[self.attn_layer].shape[-2:]
        # and reshape the self-attention to a more interpretable shape
        self.enc_attn_weights = enc_attn_weights[self.attn_layer][0].reshape(16,16,16,16)
        #print(self.enc_attn_weights.shape)
    
    def compute_on_image(self, url):
        if url != self.url:
            self.url = url
            if self.img_idx != None:
                #print("Loading from file")
                path = '/home/atsumilab/Luiz/self-attention-regularization/region_similarity_based/visualization/'
                self.pil_img = Image.open(path+'sample_images/image'+str(self.img_idx)+'.jpeg')
            else:
                #print("Loading from the web")
                self.pil_img = transform2(Image.open(requests.get(url, stream=True).raw))
            # mean-std normalize the input image (batch-size: 1)
            self.tensor_img = self.transform(self.pil_img).unsqueeze(0)
            self.compute_features(self.tensor_img)
    
    def update_chart(self, change):
        with self.o:
            clear_output()

            # j and i are the x and y coordinates of where to look at
            # sattn_dir is which direction to consider in the self-attention matrix
            # sattn_dot displays a red dot or not in the self-attention map
            url, j, i, sattn_dir, sattn_dot = [s.value for s in self.sliders]

            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 7))
            self.compute_on_image(url)

            # convert reference point to absolute coordinates
            j = int(j * self.tensor_img.shape[-1])
            i = int(i * self.tensor_img.shape[-2])

            # how much was the original image upsampled before feeding it to the model
            scale = self.pil_img.height / self.tensor_img.shape[-2]

            # compute the downsampling factor for the model
            # it should be 32 for standard DETR and 16 for DC5
            sattn = self.enc_attn_weights
            fact = 2 ** round(math.log2(self.tensor_img.shape[-1] / sattn.shape[-1]))

            # round the position at the downsampling factor
            x = ((j // fact) + 0.5) * fact
            y = ((i // fact) + 0.5) * fact

            axs[0].imshow(self.pil_img)
            axs[0].axis('off')
            axs[0].add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))

            idx = (i // fact, j // fact)
            
            if sattn_dir:
                sattn_map = sattn[idx[0], idx[1], ...]
                threshold = torch.sum(sattn_map)/100
            else:
                sattn_map = sattn[idx[0], idx[1],...]
                threshold = torch.sum(sattn_map)/100
            axs[1].imshow(sattn_map.cpu().detach().numpy(), cmap=self.color, interpolation='nearest')#,vmax=threshold)
            if sattn_dot:
                axs[1].add_patch(plt.Circle((idx[1],idx[0]), 1, color='r'))
            axs[1].axis('off')
            axs[1].set_title(f'self-attention{(i, j)}')

            plt.show()
        
    def run(self):
        for s in self.sliders:
            s.observe(self.update_chart, 'value')
        self.update_chart(None)
        url, x, y, d, sattn_d = self.sliders
        res = widgets.VBox(
            [
            url,
            widgets.HBox([x, y]),
            widgets.HBox([d, sattn_d]),
            self.o
          ])
        return res
    
def plot_patch_gm(k, color = 'bone', color2 = 'Blues'):
    im = Image.open('visualization/sample_images/image'+str(k)+'.jpeg')
    img = transform(im).unsqueeze(0)
    GM = GM_Mask(patch_size=16, width=256, height=256)
    patch = GM.img_patches(img,16)
    gms = GM.grid_gram_matrix(patch)
    gm = GM(img)
    patches=patch.reshape(256,3,16,16)
    maps = gms.reshape(256,3,3)
    
    fig3 = plt.figure(constrained_layout=False, figsize=(24,16))
    fig3.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, hspace=0, wspace=0)
    gs = fig3.add_gridspec(5, 6)

    f3_ax = fig3.add_subplot(gs[:, :2])
    #f3_ax.set_title('Input Image', fontsize='xx-large')
    f3_ax.imshow(im)
    f3_ax.set_xticks([])
    f3_ax.set_yticks([])

    f3_ax0 = fig3.add_subplot(gs[0, 2])
    #f3_ax0.set_title('x1', fontsize='x-large')
    f3_ax0.imshow(patches[0].permute(1, 2, 0),cmap=color, interpolation='nearest')
    f3_ax0.set_xticks([])
    f3_ax0.set_yticks([])
    f3_ax1 = fig3.add_subplot(gs[0, 3])
    #f3_ax1.set_title('G1', fontsize='x-large')
    f3_ax1.imshow(maps[0],cmap=color, interpolation='nearest')
    f3_ax1.set_xticks([])
    f3_ax1.set_yticks([])

    f3_ax2 = fig3.add_subplot(gs[1, 2])
    #f3_ax2.set_title('Layer 1', fontsize='x-large')
    f3_ax2.imshow(patches[1].permute(1, 2, 0),cmap=color, interpolation='nearest')
    f3_ax2.set_xticks([])
    f3_ax2.set_yticks([])
    f3_ax3 = fig3.add_subplot(gs[1, 3])
    #f3_ax3.set_title('Layer 2', fontsize='x-large')
    f3_ax3.imshow(maps[1],cmap=color, interpolation='nearest')
    f3_ax3.set_xticks([])
    f3_ax3.set_yticks([])

    f3_ax4 = fig3.add_subplot(gs[2, 2])
    #f3_ax4.set_title('Layer 3', fontsize='x-large')
    f3_ax4.imshow(patches[2].permute(1, 2, 0),cmap=color, interpolation='nearest')
    f3_ax4.set_xticks([])
    f3_ax4.set_yticks([])
    f3_ax5 = fig3.add_subplot(gs[2, 3])
    #f3_ax5.set_title('Layer 4', fontsize='x-large')
    f3_ax5.imshow(maps[2],cmap=color, interpolation='nearest')
    f3_ax5.set_xticks([])
    f3_ax5.set_yticks([])

    f3_ax6 = fig3.add_subplot(gs[3, 2])
    #f3_ax6.set_title('Layer 5', fontsize='x-large')
    f3_ax6.imshow(patches[254].permute(1, 2, 0),cmap=color, interpolation='nearest')
    f3_ax6.set_xticks([])
    f3_ax6.set_yticks([])
    f3_ax7 = fig3.add_subplot(gs[3, 3])
    #f3_ax7.set_title('Layer 6', fontsize='x-large')
    f3_ax7.imshow(maps[254],cmap=color, interpolation='nearest')
    f3_ax7.set_xticks([])
    f3_ax7.set_yticks([])

    f3_ax8 = fig3.add_subplot(gs[4, 2])
    #f3_ax8.set_title('Layer 1', fontsize='x-large')
    f3_ax8.imshow(patches[255].permute(1, 2, 0),cmap=color, interpolation='nearest')
    f3_ax8.set_xticks([])
    f3_ax8.set_yticks([])
    f3_ax9 = fig3.add_subplot(gs[4, 3])
    #f3_ax9.set_title('Layer 2', fontsize='x-large')
    f3_ax9.imshow(maps[255],cmap=color, interpolation='nearest')
    f3_ax9.set_xticks([])
    f3_ax9.set_yticks([])

    f3_ax10 = fig3.add_subplot(gs[:, 4:])
    #f3_ax10.set_title('Input Image', fontsize='xx-large')
    f3_ax10.imshow(gm[0],cmap=color2, interpolation='nearest')
    f3_ax10.set_xticks([])
    f3_ax10.set_yticks([])
    
def plot3channels(k):
    im = imageio.imread('visualization/sample_images/image'+str(k)+'.jpeg')
    figure, plots = plt.subplots(ncols=3, nrows=1, figsize=(20, 14))
    for i, subplot in zip(range(3), plots):
        temp = np.zeros(im.shape, dtype='uint8')
        temp[:,:,i] = im[:,:,i]
        subplot.imshow(temp)
        subplot.set_axis_off()
    plt.show()