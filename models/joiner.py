import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .encoder import EncoderModule
from .decoder import DecoderModule
from .backbone import Backbone, NoBackbone
from .losses import Attention_penalty_factor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Joiner(nn.Module):
    def __init__(self, backbone, encoder, num_classes = 10, batch_size=1, hidden_dim=512, image_h=200, image_w=200, grid_l=3, penalty_factor="1", alpha=1):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.f_map_h = image_h//15 
        self.f_map_w = image_w//15

        self.dist_matrix = Attention_penalty_factor.distance_matrix(batch_size, self.f_map_w, self.f_map_h, grid_l)
        self.grids_matrix = Attention_penalty_factor.grids_matrix(batch_size, self.f_map_w, self.f_map_h, grid_l)
        self.pf_matrix = Attention_penalty_factor.penalty_factor(self.dist_matrix, penalty_factor, alpha)

        self.penalty_mask = Attention_penalty_factor.penalty_matrix(batch_size, self.f_map_w, self.f_map_h, self.grids_matrix, self.pf_matrix, grid_l)
        #self.penalty_mask = Attention_penalty_factor.penalty_mask(batch_size, self.f_map_w, self.f_map_h, grid_l, penalty_factor, alpha)

        self.backbone = backbone
        self.encoder = encoder

        self.fc1 = nn.Linear(2*self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)
        #self.fc2 = nn.Linear(8192, 1024)
        #self.fc3 = nn.Linear(1024, 128)
        #self.fc4 = nn.Linear(128, 10)


    def forward(self, inputs):

        h, pos = self.backbone(inputs)
        print(h.shape)
        att, sattn = self.encoder(src=h, pos_embed=pos.to(device))

        print(att.shape)
        print(sattn.shape)
   

        sattn = sattn.reshape(sattn.shape[:-2] + h.shape[-2:] + h.shape[-2:])
        sattn = sattn.permute(0,3,4,1,2)
        print(sattn.shape)
        pattn = sattn*self.penalty_mask.to(device)

        x = torch.cat([h,att],1)
        #print(x.shape)
        x = x.reshape(-1, 2*self.hidden_dim * self.f_map_h * self.f_map_w)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc1(x)
        return x, sattn, pattn

class TransConv(nn.Module):
    def __init__(self, backbone, encoder, num_classes = 10, batch_size=1, hidden_dim=512, image_h=200, image_w=200, grid_l=3, penalty_factor="1", alpha=1):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.f_map_h = image_h//15 
        self.f_map_w = image_w//15

        self.dist_matrix = Attention_penalty_factor.distance_matrix(batch_size, self.f_map_w, self.f_map_h, grid_l)
        self.grids_matrix = Attention_penalty_factor.grids_matrix(batch_size, self.f_map_w, self.f_map_h, grid_l)
        self.pf_matrix = Attention_penalty_factor.penalty_factor(self.dist_matrix, penalty_factor, alpha)

        self.penalty_mask = Attention_penalty_factor.penalty_matrix(batch_size, self.f_map_w, self.f_map_h, self.grids_matrix, self.pf_matrix, grid_l)
        #self.penalty_mask = Attention_penalty_factor.penalty_mask(batch_size, self.f_map_w, self.f_map_h, grid_l, penalty_factor, alpha)

        self.backbone = backbone
        self.encoder = encoder

        self.self_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, nhead=8, dropout=0.05)

        self.fc1 = nn.Linear(144*12*12, self.num_classes)
        #self.fc2 = nn.Linear(8192, 1024)
        #self.fc3 = nn.Linear(1024, 128)
        #self.fc4 = nn.Linear(128, 10)


    def forward(self, inputs):

        h, pos = self.backbone(inputs)

        src = h.flatten(2).permute(2, 0, 1)
        #print(h.shape)
        sattn = self.encoder(src)

        #print(att.shape)
        #print(sattn.shape)

        #x = sattn.reshape(sattn.shape[:-1] + h.shape[-2:]) 

        #sattn = sattn.reshape(sattn.shape[:-2] + h.shape[-2:] + h.shape[-2:])
        #sattn = sattn.permute(0,3,4,1,2)
        #print(sattn.shape)
        #pattn = sattn*self.penalty_mask.to(device)

        #x = torch.cat([h,att],1)
        #print(x.shape)
        #x = x.reshape(-1, 144 * 12 * 12)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.fc1(x)
        return sattn


