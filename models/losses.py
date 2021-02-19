
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from scipy.spatial import distance
import numpy as np

from .encoder import EncoderModule
from .backbone import Backbone, NoBackbone

class Attention_penalty_factor(nn.Module):
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
                p_matrix[p_matrix==j]=ref_column[j]
            p_matrix[p_matrix==0]=ref_column[0]
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
    def __init__(self, grid_l=3):
        super(LCA, self).__init__()
        self.grid_l = grid_l
 
    def forward(self, inputs):        
        A = inputs
        qt_hor_grids = A.shape[2]//grid_l
        qt_ver_grids = A.shape[3]//grid_l
        grid_temp = A.shape[2]*A.shape[3]//self.grid_l
        temp = A.view(A.shape[0], A.shape[1], grid_temp, self.grid_l)
        #print(temp.shape)
        ind  = np.arange(grid_temp)
        #print(ind.shape)
        ind2 = ind.reshape(A.shape[2], grid_temp//A.shape[2]).T.reshape(grid_temp//grid_l, grid_l)
        #print(ind2.shape)
        grids = temp[:, :, ind2, :]
        #print(B.shape)

        
        return grids