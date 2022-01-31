
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

__all__ = ['Attention_penalty_factor', 'Curating_of_attention_loss', 'GeneratorLoss', 'CriticLoss','GanLossWrapper']

class Attention_penalty_factor(nn.Module):
    def __init__(self, ):
        super().__init__()

    def distance_matrix(batch_size, width, height, grid_l=3):
        
        w = width
        h = height
        qt_hor_grids = w//grid_l
        qt_ver_grids = h//grid_l
        qtd_grids = qt_hor_grids*qt_ver_grids
        c = 0
        grids = []
        for i in range(qtd_grids):
            hor_pos = i//qt_hor_grids
            ver_pos = c
            c = c+1
            grid = [hor_pos,ver_pos]
            grids.append(grid)
            if c == qt_ver_grids:
                c=0

        dist_grid = []
        for g in range(len(grids)):
            dist_pair_list = []
            for n in range(len(grids)):
                dist_pair_list.append(distance.cityblock(grids[g], grids[n]))
            dist_grid.append(dist_pair_list)
            
        dist_matrix = torch.tensor(np.array(dist_grid))

        return dist_matrix

    def grids_matrix(batch_size, width, height, grid_l=3):
        
        w = width
        h = height
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
    
    def penalty_factor(dist_matrix, penalty_factor="1", alpha=1):
        if penalty_factor == "1" or penalty_factor =="distraction":
            pf_matrix = (1/(dist_matrix+1))**alpha
            return pf_matrix
        if penalty_factor == "2" or penalty_factor =="misdirection":
            pf_matrix = alpha*((torch.max(dist_matrix)//2)-dist_matrix+0.1)**3
            return pf_matrix


    def penalty_matrix(batch_size, width, height, grid_matrix, pf_matrix, grid_l=3):
        
        w = width
        h = height

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

        pep = []
        for b in range(batch_size):
            pep.append(c)

        d = torch.Tensor(batch_size, h, w, h, w)
        penalty_encoding_pattern = torch.cat(pep, out=d)
        penalty_encoding_pattern = penalty_encoding_pattern.view(batch_size, h, w, h, w)

        return penalty_encoding_pattern

    def penalty_mask(batch_size, width, height, grid_l=3, penalty_factor="1", alpha=1):
        dist_matrix = distance_matrix(batch_size, width, height, grid_l)
        grids_matrix = grids_matrix(batch_size, width, height, grid_l)
        pf_matrix = penalty_factor(dist_matrix, penalty_factor, alpha)

        mask = penalty_matrix(batch_size, width, height,grids_matrix,pf_matrix)
        #UNFINISHED
        return mask

class Curating_of_attention_loss(nn.Module):
    def __init__(self, ):
        super().__init__()
 
    def forward(self, pattn):
        #Computing the Attention Loss
        Latt = torch.sum(pattn)

        return Latt


class CriticLoss(nn.Module):
    def __init__(self, beta=0.0000005, sigma=1):
        super(CriticLoss, self).__init__()
        self.beta = beta
        self.sigma = sigma

    def forward(self, preds, label):
        #print("Critic Loss")
        crossEntropy = nn.CrossEntropyLoss()
        classificationLoss = crossEntropy(preds[0], label)

        LCA = Curating_of_attention_loss()
        Latt = LCA(preds[2])
        
        Lc = self.sigma*classificationLoss - self.beta*Latt
        
        return Lc
    

#[x, sattn, pattn, inputs, x0]
    
class GeneratorLoss(nn.Module):
    def __init__(self, beta=0.0000005, gamma=0.005,sigma=1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, output, target):
        #print("Generator Loss")
        LCA = Curating_of_attention_loss()
        Latt = TensorCategory(self.beta*LCA(output[2]))

        crossEntropy = nn.CrossEntropyLoss()
        model_loss = self.gamma*crossEntropy(output[0],target)

        MSE = nn.MSELoss()
        Lrec = TensorCategory(self.sigma*MSE(output[3],output[4]))

        Lg = Lrec + model_loss + Latt

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