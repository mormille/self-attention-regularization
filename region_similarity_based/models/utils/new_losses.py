
import copy
from typing import Optional, List

from fastai.distributed import *
from fastai.vision.all import *

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from models.encoder import EncoderModule
from models.backbone import Backbone, NoBackbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

__all__ = ['Curating_of_attention_mask', 'GeneratorLoss', 'CriticLoss','GanLossWrapper','Curating_of_attention_loss',
           'CriticValidationLoss','SingleLabelCriticLoss']


class Curating_of_attention_mask(nn.Module):
    def __init__(self, patch_size=16, width=320, height=320, map_width=20, map_height=20, grid_l=1, pf="1"):
        super().__init__()
        self.grid_l = grid_l
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.map_width = map_width
        self.map_height = map_height
        self.pf = pf
        self.grids_list = self.grids_list(self.width, self.height, self.patch_size)
        self.grids_matrix = self.grids_matrix(self.map_width, self.map_height, self.grid_l)
        
        
    def grids_list(self, width, height, patch_size): #COMPUTED ONCE BEFORE TRAINING
        w = width
        h = height
        qt_hor_grids = w//patch_size
        qt_ver_grids = h//patch_size
        qtd_grids = qt_hor_grids*qt_ver_grids
        c = 0
        grids_list = []
        for i in range(qtd_grids):
            hor_pos = i//qt_hor_grids
            ver_pos = c
            c = c+1
            grid = [hor_pos,ver_pos]
            grids_list.append(grid)
            if c == qt_ver_grids:
                c=0
        return grids_list
        
        
    def grids_matrix(self, map_width, map_height, grid_l):

        w = map_width
        h = map_height
        len_input_seq = h*w
        qt_hor_grids = w//grid_l
        qt_ver_grids = h//grid_l


        grid_list = []
        for i in range(h):
            row_grid_list = []
            preliminar_ver_grid = i//grid_l
            if preliminar_ver_grid != 0:
                preliminar_ver_grid = preliminar_ver_grid*qt_hor_grids

            for h in range(w):
                preliminar_grid = h//grid_l+preliminar_ver_grid
                row_grid_list.append(preliminar_grid)

            grid_list.append(row_grid_list)
        grid_matrix = torch.tensor(np.array(grid_list))

        return grid_matrix
        
        
    def img_patches(self, batch, patch_size):
        #torch.Tensor.unfold(dimension, size, step)
        #slices the images into grid_l*grid_l size patches
        patches = batch.data.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        a, b, c, d, e, f, g = patches.shape
        patches = patches.reshape(a, c, d, e, f, g)
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

        G = torch.mm(features[0], features[0].t())

        for i in range(1,a*b*c):
            g = torch.mm(features[i], features[i].t())
            G= torch.cat((G, g), 0)


        G = G.div(d * e * f).reshape(a, b * c, d, d)

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.

        return G



    def gram_dist_matrix(self, batch, grid_l):
        patches = self.img_patches(batch, grid_l)
        #print(patches.shape)
        Grid = self.grid_gram_matrix(patches)
        #print(Grid.shape)
        bs = batch.shape[0]
        #print(bs)
        MSE = nn.MSELoss()

        #Grid = Grid.reshape(Grid.shape[0],Grid.shape[1]*Grid.shape[2],Grid.shape[3],Grid.shape[4])
        gmg_shape = Grid.shape
        #print(gmg_shape)
        
        
        mse_grid = []
        for k in range(gmg_shape[0]):
            dist_grid = []
            for g in range(gmg_shape[1]):
                dist_pair_list = []
                for n in range(gmg_shape[1]):
                    dist_pair_list.append(MSE(Grid[k][g], Grid[k][n]))
                dist_grid.append(dist_pair_list)
            mse_grid.append(dist_grid)

        dist_matrix = torch.tensor(np.array(mse_grid))
        for i in range(bs):
            dist_matrix[i] -= dist_matrix[i].min()
            dist_matrix[i] /= dist_matrix[i].max()

        return dist_matrix


    def penalty_matrix(self, map_width, map_height, grid_matrix, dist_matrix, grid_l, pf): #unused due to lack of performance
        bs,_,_ = dist_matrix.shape
        pep = []
        for s in range(bs):
            pf_matrix = self.penalty_factor(dist_matrix[s], penalty_factor=pf, alpha=1)
            w = map_width
            h = map_height

            qt_hor_grids = w//grid_l
            qt_ver_grids = h//grid_l
            qtd_grids = qt_hor_grids*qt_ver_grids

            penalty_mask = []
            for i in range(qtd_grids):
                ref_column = pf_matrix[i]
                p_matrix = grid_matrix.type(torch.FloatTensor)
                for j in range(1,len(ref_column)):
                    #print(float(j))
                    p_matrix[p_matrix==j]=float(ref_column[j])
                p_matrix[p_matrix==0]=float(ref_column[0])
                penalty_mask.append(p_matrix)

            #print(len(penalty_mask))    

            penalty_enc = []
            for i in range(h):
                penalty_row = []
                for j in range(w):
                    #print(grid_matrix[i,j])
                    #print(penalty_mask[grid_matrix[i,j]].shape)
                    penalty_row.append(penalty_mask[grid_matrix[i,j]])
                    #print(len(penalty_row))
                generic_tensor = Tensor(h,w)
                penalty_row_tensor = torch.cat(penalty_row, out=generic_tensor)
                penalty_enc.append(penalty_row_tensor)
                #print(penalty_row_tensor.shape)
                #break

            b = torch.Tensor(h, w, h, w)
            c=torch.cat(penalty_enc, out=b)
            c = c.view(h, w, h, w)
            pep.append(c)

        d = torch.Tensor(bs, h, w, h, w)
        penalty_encoding_pattern = torch.cat(pep, out=d)
        penalty_encoding_pattern = penalty_encoding_pattern.view(bs, h, w, h, w)

        return penalty_encoding_pattern
        
    def forward(self, batch):
        
        batch = batch.unsqueeze(0)
        
        dist_matrix = self.gram_dist_matrix(batch, self.patch_size)
        
        return dist_matrix
    

class Curating_of_attention_loss(nn.Module):
    def __init__(self, bias=0.0, distortion=1):
        super().__init__()
 
        self.bias = bias
        self.d = distortion

    def penalty_factor(self, dist_matrix, alpha=1):

        pf_matrix = dist_matrix+self.bias
        return pf_matrix

    def forward(self, sattn, pattn):
        #Computing the Attention Loss
        
        pattn = self.penalty_factor(pattn)

        att_loss = sattn*pattn # (output2*label) (64x64 * 64x64)
        Latt = TensorCategory(torch.sum(att_loss)) # ecalar number

        return Latt
    
    
class CriticLoss(nn.Module):
    def __init__(self, layer=None, bias=0.0, beta=0.0002, sigma=1):
        super(CriticLoss, self).__init__()
        self.layer = layer
        self.beta = beta
        self.sigma = sigma
        self.crossEntropy = nn.CrossEntropyLoss()
        self.LCA = Curating_of_attention_loss(bias=bias)

    def forward(self, preds, label):

        classificationLoss = self.crossEntropy(preds[0],label)
        #print(classificationLoss)
        if self.layer != None:
            #print(self.layer)
            Latt = self.LCA(preds[1][self.layer], preds[3])#.item()
        else:
            Latt=0.0
        #print(self.beta*Latt)
        Lc = self.sigma*classificationLoss + self.beta*Latt
        #print(Lc)
        return Lc
    
class CriticValidationLoss(nn.Module):
    def __init__(self):
        super(CriticValidationLoss, self).__init__()

    def forward(self, preds, label):
        #print("Critic Loss")
        crossEntropy = nn.CrossEntropyLoss()
        classificationLoss = crossEntropy(preds[0], label[1])      
        
        return classificationLoss
    
class SingleLabelCriticLoss(nn.Module):
    def __init__(self):
        super(SingleLabelCriticLoss, self).__init__()
        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self, preds, label):
        #print("Critic Loss")
        #crossEntropy = nn.CrossEntropyLoss()
        classificationLoss = self.crossEntropy(preds[0], label)
        
        return classificationLoss
    
    
class GeneratorLoss(nn.Module):
    def __init__(self,  beta=0.005, sigma=1, gamma=0.0005):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.crossEntropy = nn.CrossEntropyLoss()
        self.LCA = Curating_of_attention_loss(pf="2", pool=2, grid_l=16, img_size=256)
        self.MSE = nn.MSELoss()
        

    def forward(self, output, image, target): #fake_pred, output, target
        #print("Generator Loss")
        
        #crossEntropy = nn.CrossEntropyLoss()
        #passing sigma to both CLoss and ALoss to help produce images that hinder the attention maps
        
        classificationLoss = self.sigma*(self.crossEntropy(output[0],target))

        #LCA = Curating_of_attention_loss()
        Latt = self.beta*(self.LCA(output[1], output[3]).item())
        
        #MSE = nn.MSELoss()
        Lrec = self.MSE(image[0],image[1]).item()

        Lg = Lrec - self.gamma*(classificationLoss+Latt)

        return Lg

    
class GanLossWrapper(nn.Module):
    def __init__(self, beta=0.0000005, gamma=0.005,sigma=1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.generator_mode = True
        self.generatorLoss = GeneratorLoss(beta=self.beta, gamma=self.gamma, sigma=self.sigma)
        self.criticLoss = CriticLoss(beta=self.beta, sigma=self.sigma)

    def forward(self, output, target):
        #print(len(output))
        if len(output) == 4:
            #print("Generator Loss")
            loss = self.generatorLoss(output, target)
            
        else:
            #print("Critic Loss")
            loss = self.criticLoss(output, target)
            
        return loss
        
        
    
def img_patches(img_t, grid_l):
    #torch.Tensor.unfold(dimension, size, step)
    #slices the images into grid_l*grid_l size patches
    patches = img_t.data.unfold(1, 3, 3).unfold(2, grid_l, grid_l).unfold(3, grid_l, grid_l)
    a, b, c, d, e, f, g = patches.shape
    patches = patches.reshape(a, c, d, e, f, g)
    #print(patches.shape)
    return patches


def grid_gram_matrix(input):

    a, b, c, d, e, f = p.shape
    # a=batch size
    # b=horizontal patches
    # c = vertical patches
    # d=number of feature maps
    # (e,f)=dimensions of a f. map (N=e*f)

    features = p.reshape(a * b * c, d, e*f)  # resise F_XL into \hat F_XL
    #print(features.shape)
    # compute the gram product

    G = torch.mm(features[0], features[0].t())

    for i in range(1,a*b*c):
        g = torch.mm(features[i], features[i].t())
        G= torch.cat((G, g), 0)


    G = G.div(d * e * f).reshape(a, b, c, d, d)

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    
    return G