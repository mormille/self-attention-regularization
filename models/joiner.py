import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .encoder import EncoderModule
from .decoder import DecoderModule
from .backbone import Backbone, NoBackbone
from .losses import Attention_penalty_factor

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class Joiner(nn.Module):
    def __init__(self, backbone, encoder, num_classes = 10, bypass=False, conv_backbone = False, pos_enc = "sin", batch_size=10, hidden_dim=512, image_h=30, image_w=30, grid_l=5, penalty_factor="1", alpha=1):
        super().__init__()
        
        self.bypass = bypass
        self.pos_enc = pos_enc
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        if conv_backbone == True:
            self.f_map_h = image_h//15 
            self.f_map_w = image_w//15
        else:
            self.f_map_h = image_h 
            self.f_map_w = image_w

        self.dist_matrix = Attention_penalty_factor.distance_matrix(batch_size, self.f_map_w, self.f_map_h, grid_l)
        self.grids_matrix = Attention_penalty_factor.grids_matrix(batch_size, self.f_map_w, self.f_map_h, grid_l)
        self.pf_matrix = Attention_penalty_factor.penalty_factor(self.dist_matrix, penalty_factor, alpha)

        self.penalty_mask = Attention_penalty_factor.penalty_matrix(batch_size, self.f_map_w, self.f_map_h, self.grids_matrix, self.pf_matrix, grid_l)
        #self.penalty_mask = Attention_penalty_factor.penalty_mask(batch_size, self.f_map_w, self.f_map_h, grid_l, penalty_factor, alpha)

        self.backbone = backbone
        self.encoder = encoder

        if self.bypass == True:

            self.fc1 = nn.Linear(2*self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)

        else:
            self.fc1 = nn.Linear(self.hidden_dim * self.f_map_h * self.f_map_w, self.num_classes)
            #self.fc2 = nn.Linear(512,self.num_classes)

        # spatial positional encodings
        if self.pos_enc == "learned":
            self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
            self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):

        h, pos = self.backbone(inputs)
        #h = [bs, 512, 12, 12])

        if self.pos_enc == "learned":
            # construct positional encodings
            H, W = h.shape[-2:]
            pos = torch.cat([
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(1)

        att, sattn = self.encoder(src= 0.2*h, pos_embed=pos.to(device))
        #xattn = sattn.reshape(sattn.shape[:-1] + h.shape[-2:])

        sattn = sattn.reshape(sattn.shape[:-2] + h.shape[-2:] + h.shape[-2:])

        sattn = sattn.permute(0,3,4,1,2)

        pattn = sattn*self.penalty_mask.to(device)

        if self.bypass == True:

            x = torch.cat([h,att],1)

        else:

            x = att


        x = x.flatten(1)

        x = self.fc1(x)
        #x = self.fc2(x)


        return x, sattn, pattn




