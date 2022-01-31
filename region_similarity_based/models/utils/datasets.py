from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import *

from collections import defaultdict, deque
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch

#from classy_vision.dataset import ClassyDataset
#from classy_vision.tasks import ClassificationTask

#from classy_vision.dataset import ClassyDataset, register_dataset
#from classy_vision.dataset.transforms import ClassyTransform, build_transforms
from torchvision.datasets.imagenet import ImageNet

import time
import random
from models.utils.new_losses import *

__all__ = ['CustomCifar','CustomImageNet','CustomImageNetRotation','ImagenetRotation','ImagenetOverlapedRotation','ImagenetColorization']


class CustomCifar(datasets.CIFAR10):
    def __init__(self, path, transforms, width=32, height=32, patch_size = 2, map_width = 16, map_height = 16, grid_l=1, pf="1",size=None, start_idx = 0, train=True):
        super().__init__(path)
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.map_width = map_width
        self.map_height = map_height
        self.grid_l = grid_l
        self.pf = pf
        self.transforms = transforms
        self.train = train
        self.start_time = time.time()
        if size == None:
            if train == True:
                self.size = 50000
            else:
                self.size = 10000
        else:
            self.size = size
        self.start_idx = start_idx
        self.indexes = self.create_idx()
        self.attention_mask = Curating_of_attention_mask(self.patch_size, self.width, self.height, self.map_width, self.map_height, self.grid_l, self.pf)
        self.pattn_dict = self.create_labels()

#     def create_idx(self):
#         indexes = {}
#         i = 0
#         max_len = 50000
#         if self.train != True:
#             max_len = 10000
#         while len(indexes) < (self.size):
#             r=random.randint(0,max_len)
#             if r not in indexes.values(): 
#                 indexes[i] = r
#                 i +=1

#         return indexes

    def new_empty(self):
        
        return ()

    def create_idx(self):
        indexes = {}
        for i in range(self.size):
            indexes[i] = i+ self.start_idx

        return indexes
    
    def create_labels(self):
        
        transform = T.Compose([
        T.Resize((self.height,self.width)),
        T.ToTensor()
        ])
        
        attention_labels = {}
        for i in range(len(self.indexes)):
            index = self.indexes[i]
            im, _ = super().__getitem__(index)
            pattn = self.attention_mask(self.transforms(im))
            attention_labels[i] = pattn
            
            if i%50 == 0:
                time_elapsed = time.time() - self.start_time
                print('{} images complete in {:.0f}m {:.0f}s'.format(i, time_elapsed // 60, time_elapsed % 60))
                #self.start_time = time.time()
            
        return attention_labels

    def __len__(self):
         return len(self.indexes)
        
    def __getitem__(self, index):

        #index = self.indexes[index]
        im, label = super().__getitem__(self.indexes[index])
        pattn = self.pattn_dict[index]
        return self.transforms(im),(pattn,label)


##self, path, transforms, width=320, height=320, patch_size = 16, map_width = 20, map_height = 20, grid_l=1, size=None, start_idx = 0, train=True

class CustomImageNet(datasets.ImageNet):
    def __init__(self, path, transforms, width=256, height=256, patch_size = 32, map_width = 256, map_height = 256, grid_l=1, pf="1",size=None, start_idx = 0, train=True):
        super().__init__(path)
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.map_width = map_width
        self.map_height = map_height
        self.grid_l = grid_l
        self.pf = pf
        self.transforms = transforms
        self.train = train
        self.start_time = time.time()
        if size == None:
            if train == True:
                self.size = 1000000
            else:
                self.size = 200000
        else:
            self.size = size
        self.start_idx = start_idx
        self.indexes = self.create_idx()
        self.attention_mask = Curating_of_attention_mask(self.patch_size, self.width, self.height, self.map_width, self.map_height, self.grid_l, self.pf)
        self.pattn_dict = self.create_labels()

    def new_empty(self):
        
        return ()

    def create_idx(self):
        indexes = {}
        for i in range(self.size):
            indexes[i] = i+ self.start_idx

        return indexes
    
    def create_labels(self):
        
        transform = T.Compose([
        T.Resize((self.height,self.width)),
        T.ToTensor()
        ])
        
        attention_labels = {}
        for i in range(len(self.indexes)):
            index = self.indexes[i]
            im, _ = super().__getitem__(index)
            pattn = self.attention_mask(self.transforms(im))
            attention_labels[i] = pattn
            
            if i%50 == 0:
                time_elapsed = time.time() - self.start_time
                print('{} images complete in {:.0f}m {:.0f}s'.format(i, time_elapsed // 60, time_elapsed % 60))
                #self.start_time = time.time()
            
        return attention_labels

    def __len__(self):
         return len(self.indexes)
        
    def __getitem__(self, index):

        #index = self.indexes[index]
        im, label = super().__getitem__(self.indexes[index])
        pattn = self.pattn_dict[index]
        return self.transforms(im),(pattn,label)


    
class CustomImageNetRotation(datasets.ImageNet):
    def __init__(self, path, transforms, width=256, height=256, patch_size = 32, map_width = 256, map_height = 256, grid_l=1, pf="1",size=None, start_idx = 0, train=True):
        super().__init__(path)
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.map_width = map_width
        self.map_height = map_height
        self.grid_l = grid_l
        self.pf = pf
        self.transforms = transforms
        self.train = train
        self.start_time = time.time()
        if size == None:
            if train == True:
                self.size = 1000000
            else:
                self.size = 200000
        else:
            self.size = size
        self.start_idx = start_idx
        self.indexes = self.create_idx()
        self.degrees = [0,90,180,270]
        self.labelDict = {0:0,90:1,180:2,270:3}
        self.attention_mask = Curating_of_attention_mask(self.patch_size, self.width, self.height, self.map_width, self.map_height, self.grid_l, self.pf)
        self.pattn_dict, self.rotation_dict = self.create_labels()

    def new_empty(self):
        
        return ()

    def create_idx(self):
        indexes = {}
        for i in range(self.size):
            indexes[i] = i+ self.start_idx

        return indexes
    
    def create_labels(self):
        
        transform = T.Compose([
        T.Resize((self.height,self.width)),
        T.ToTensor()
        ])
        
        attention_labels = {}
        rotation_labels = {}
        for i in range(len(self.indexes)):
            index = self.indexes[i]
            im, _ = super().__getitem__(index)
            deg = random.choice(self.degrees)
            pattn = self.attention_mask(rotate(self.transforms(im),deg))
            attention_labels[i] = pattn
            rotation_labels[i] = deg
            
            if i%50 == 0:
                time_elapsed = time.time() - self.start_time
                print('{} images complete in {:.0f}m {:.0f}s'.format(i, time_elapsed // 60, time_elapsed % 60))
                #self.start_time = time.time()
            
        return attention_labels, rotation_labels

    def __len__(self):
         return len(self.indexes)
        
    def __getitem__(self, index):

        #index = self.indexes[index]
        im, _ = super().__getitem__(self.indexes[index])
        pattn = self.pattn_dict[index]
        deg = self.rotation_dict[index]
        label = self.labelDict[deg]
        
        return rotate(self.transforms(im),deg),(pattn,label)
    

class ImagenetRotation(datasets.ImageNet):
    def __init__(self, path, transforms, size, train=True):
        super().__init__(path)
        self.transforms = transforms
        self.size = size
        self.degrees = [0,90,180,270]
        self.labelDict = {0:0,90:1,180:2,270:3}
        self.deg = {}
        self.indexes = self.create_idx()

    def create_idx(self):
        indexes = {}
        self.deg = {}
        i = 0
        while len(indexes) < (self.size):
            r=random.randint(0,1281166)
            if r not in indexes.values() : 
                indexes[i] = r
                self.deg[i] = random.choice(self.degrees)
                i +=1

        return indexes

    def new_empty(self):
        
        return ()
    
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        ind = self.indexes[index]
        im, _ = super().__getitem__(ind)
        dgr = self.deg[index]
        self.category = self.labelDict[dgr] 
        return rotate(self.transforms(im),dgr), self.category

class ImagenetOverlapedRotation(datasets.ImageNet):
    def __init__(self, path, transforms, size, train=True):
        super().__init__(path)
        self.transforms = transforms
        self.size = size
        self.degrees = [0,90,180,270]
        self.labelDict = {0:0,90:1,180:2,270:3}
        self.indexes = self.create_idx()

    def create_idx(self):
        indexes = {}
        idx_list = []
        i = 0
        while i < (self.size/4):
            r=random.randint(0,1281166)
            if r not in idx_list: 
                idx_list.append(r)
                i +=1
        d = 0    
        k = 0
        while d < 4:
            for v in idx_list:
                indexes[k]=v
                k+=1
            d+=1

        return indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        key = index
        index = self.indexes[index]
        im, _ = super().__getitem__(index)

        if key < self.size/4:
            dgr = 0
        elif key >= self.size/4 and key < self.size/2:
            dgr = 90
        elif key >= self.size/2 and key < 3*self.size/4:
            dgr = 180
        else:
            dgr = 270

        self.category = self.labelDict[dgr]
        
        return rotate(self.transforms(im),dgr), self.category

class ImagenetColorization(datasets.ImageNet):
    def __init__(self, path, transforms, size, train=True):
        super().__init__(path)
        self.transforms = transforms
        self.size = size
        self.indexes = self.create_idx()

    def create_idx(self):
        indexes = {}
        i = 0
        while len(indexes) < (self.size):
            r=random.randint(0,1281166)
            if r not in indexes.values(): 
                indexes[i] = r
                i +=1

        return indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        index = self.indexes[index]
        im, _ = super().__getitem__(index)
        grayscale = T.Compose([transforms.Grayscale(num_output_channels=3)])
        grayImg = grayscale(im)
        return self.transforms(grayImg), self.transforms(im)