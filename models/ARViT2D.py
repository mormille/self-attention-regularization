""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor

from .layers.layers import PatchEmbed, Mlp, trunc_normal_#, lecun_normal_
from .layers.drop import DropPath
from .utils.distance_loss import Distance_loss



class CustomTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        #self.linear1 = nn.Linear(d_model, dim_feedforward)
        #self.dropout = nn.Dropout(dropout)
        #self.linear2 = nn.Linear(dim_feedforward, d_model)

        #self.norm1 = nn.LayerNorm(d_model)
        #self.norm2 = nn.LayerNorm(d_model)
        #self.dropout1 = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        
        #print("src shape:", src.shape)
        q = k = v = self.with_pos_embed(src, pos)
        #v = src
        #print(src.shape)
        #print(q.shape)
        #print(k.shape)
        src, sattn = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        #sattn = src2[1]
        #print("attn shape:", sattn.shape)
        #print("src-2 shape:", src.shape)
        #src2 = src2[0]

        #src = src + self.dropout1(src2)
        #src = self.norm1(src)
        #src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        #src = src + self.dropout2(src2)
        #src = self.norm2(src)
        return src, sattn

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        #src2 = self.norm1(src)
        q = k = v = self.with_pos_embed(src, pos)
        src, sattn = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        
        #sattn = src2[1]
        #print(sattn.shape)
        #src2 = src2[0]
        #src = src + self.dropout1(src2)
        #src2 = self.norm2(src)
        #src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        #src = src + self.dropout2(src2)
        return src, sattn

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    
    
class CustomBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.1,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CustomTransformerEncoderLayer(dim, nhead=num_heads, dim_feedforward=dim, dropout=attn_drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        #print('x:', x.shape)
        src, attn = self.attn(self.norm1(x))
        #print('attn:', attn.shape)
        x = x + self.drop_path(src)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
    
class TransformerEncoder2D(nn.Module):

    def __init__(self, depth, norm, dim, num_heads, mlp_ratio, qkv_bias, drop, 
                attn_drop, norm_layer, act_layer, drop_path_rate):
        super().__init__()
        #self.layers = _get_clones(encoder_layer, num_layers)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        #self.block = CustomBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #        attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        
        self.layers = nn.Sequential(*[CustomBlock(dim=dim, num_heads=num_heads, 
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                    attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer, 
                                    act_layer=act_layer) for i in range(depth)])
        
        self.num_layers = depth

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        #print(src.shape)
        output = src
        #print("Output", len(output))
        attn = []
        for layer in self.layers:
            output, sattn = layer(output)
            attn.append(sattn)

        return output, attn


class VisionTransformer2D(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.gm_mask = GM_Mask(patch_size=patch_size, width=img_size, height=img_size)
        self.avgPool = nn.AvgPool2d(1, stride=1)
        self.patch_size = patch_size
        self.num_patches = int((img_size/patch_size)*(img_size/patch_size))

        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = TransformerEncoder2D(depth, None, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                           drop=drop_rate,attn_drop=attn_drop_rate, norm_layer=norm_layer, 
                                           act_layer=act_layer, drop_path_rate=drop_path_rate)
        
#         self.blocks = nn.Sequential(*[
#             CustomBlock(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
#                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def rescale(self, sattn):
        #print(sattn.shape)
        bs = sattn.shape[0]
        sattn = sattn[:,1:,1:]
        #print(sattn.shape)
        R = 256
        
        a = self.avgPool(sattn.reshape(bs,self.num_patches,self.patch_size,self.patch_size))
        attn = self.avgPool(a.reshape(bs,self.num_patches,R).permute(0, 2, 1).reshape(bs,R,self.patch_size,self.patch_size)).permute(0, 2, 3, 1).reshape(bs,R,R)  
            
        attn = attn.reshape(bs, R*R)
        low = attn.min(1, keepdim=True)[0]
        high = attn.max(1, keepdim=True)[0]
        attn = attn-low
        attn = attn/(high-low)
        attn = attn.reshape(bs,R,R)

        
        #attn = attn+0.0001

        return attn
            
            
    def forward_features(self, x):
        x = self.patch_embed(x)
        #print(x.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        
        # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        #print(x.permute(1, 0, 2).shape)#.flatten(2).permute(2, 0, 1).shape))
        x, sattn = self.blocks(x.permute(1, 0, 2))
        x = self.norm(x.permute(1, 0, 2))
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), sattn
        else:
            return x[:, 0], x[:, 1], sattn

    def forward(self, x):
        gm = self.gm_mask(x)
        x, sattn = self.forward_features(x)
        reduced_attn = list(map(self.rescale, sattn))
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist, sattn, reduced_attn, gm
            else:
                return (x + x_dist) / 2 , sattn, reduced_attn, gm
        else:
            x = self.head(x)
        return x, reduced_attn, sattn, gm


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict

class GM_Mask(nn.Module):
    def __init__(self, patch_size=32, width=256, height=256):
        super().__init__()
        self.width = width #used
        self.height = height #used
        self.patch_size = patch_size #used
        self.qt_grids = (width//patch_size)**2 #used
        
        
    def img_patches(self, batch, patch_size):
        #torch.Tensor.unfold(dimension, size, step)
        #slices the images into grid_l*grid_l size patches
        #print(batch.shape)
        patches = batch.data.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        a, b, c, d, e, f, g = patches.shape
        patches = patches.reshape(a, c, d, e, f, g)
        #print(patches.shape)
        #print(patches.shape)
        return patches


    def grid_gram_matrix(self, patches):

        a, b, c, d, e, f = patches.shape
        # a=batch size
        # b=horizontal patches
        # c = vertical patches
        # d=number of feature maps
        # (e,f)=dimensions of a f. map (N=e*f)

        features = patches.reshape(a * b * c, d, e*f)  # resise F_XL into \hat F_XL
        #print(features.shape)
        # compute the gram product

        feat_t = features.permute(0,2,1)
        G = torch.bmm(features, feat_t)
        G = G.div(d * e * f).reshape(a, b, c, d, d)

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.

        return G
    
    def gram_dist_matrix(self, batch, grid_l):
        patches = self.img_patches(batch, grid_l)
        #print(patches.shape)
        Grid = self.grid_gram_matrix(patches)
        #print(Grid.shape)
        bs = batch.shape[0]

        g = Grid.reshape(bs,self.qt_grids,3,3)
        gh = g.unsqueeze(2)
        gv = g.unsqueeze(1)

        ghe = gh.repeat(1,1,self.qt_grids,1,1).reshape(bs,self.qt_grids*self.qt_grids,9).permute(0, 2, 1)
        gve = gv.repeat(1,self.qt_grids,1,1,1).reshape(bs,self.qt_grids*self.qt_grids,9).permute(0, 2, 1)

        pdist = nn.PairwiseDistance(p=0.1)
        output = pdist(ghe, gve).reshape(bs,self.qt_grids,self.qt_grids)

        #output -= output.min(1, keepdim=True)[0]
        #output /= output.max(1, keepdim=True)[0]
        
        output = output.view(output.size(0), -1)
        output -= output.min(1, keepdim=True)[0]
        output /= output.max(1, keepdim=True)[0]
        output = output.view(bs,self.qt_grids,self.qt_grids)

        return output
    
        
    def forward(self, batch):
        
        #batch = batch#.unsqueeze(0)
        
        dist_matrix = self.gram_dist_matrix(batch, self.patch_size)
          
        return dist_matrix
    
    
def create_mask(batch_size, f_map_h, f_map_w):
    mask = torch.zeros((batch_size, f_map_h, f_map_w), dtype=torch.bool)
    return mask

def penalty_mask(batch_size, f_map_w, f_map_h, grid_l=4, penalty_factor="1", alpha=1):
    dist_matrix = Attention_penalty_factor.distance_matrix(batch_size, f_map_w, f_map_h, grid_l)
    grids_matrix = Attention_penalty_factor.grids_matrix(batch_size, f_map_w, f_map_h, grid_l)
    pf_matrix = Attention_penalty_factor.penalty_factor(dist_matrix, penalty_factor, alpha)

    penalty_mask = Attention_penalty_factor.penalty_matrix(batch_size, f_map_w, f_map_h, grids_matrix, pf_matrix, grid_l)
    
    return penalty_mask

def pos_emb(batch_size, hidden_dim, f_map_h, f_map_w):
    pos = PositionalEncodingSin.positionalencoding2d(batch_size,hidden_dim, f_map_h, f_map_w)
    return pos

def irregular_batch_size(testMask,testBatch):  
    mask_one = testMask[0]
    bs = testBatch.shape[0]
    bMask = []
    for b in range(bs):
        bMask.append(mask_one)
    d = torch.Tensor(testBatch.shape)
    newMask = torch.cat(bMask, out=d)
    newMask = newMask.view(testBatch.shape)
    return newMask