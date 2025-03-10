import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter

import numpy as np

import os

import random



class parameters:
    def __init__(self):
        self.k_period = 5
        self.adjust_scale = 10

        self.Is_lr_decay = True
        self.gamma = 0.998

        self.weight_scale = 1

        self.batch_size = 8

        self.times_n = 1
        self.num_sample = 2000*self.times_n
        self.num_train = int(self.num_sample*0.8)
        self.num_test = int(self.num_sample*0.2)

        self.epoches = 6400
        self.learning_rate = 1e-3

        self.width = 40

        self.times_w = 2



        # regularization

        # regularization parameter
        self.weight_decay = 0

        # L^p norm
        self.p_norm = 2


        self.merge = 0
        
        
def setup_seed(seed):
    
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True
    
    

def weights_init(shape):
    # initial the weights in the first layer as [1, 1, ..., 1, -1, -1, ..., -1]

    # get the half of the shape
    shape_np = np.array(shape)
    shape_half = np.divide(shape_np,2)
    shape_half = np.ceil(shape_half).tolist()

    W = torch.cat((torch.ones(int(shape_half[0])), -1*torch.ones(int(shape_half[0]))))

    return torch.reshape(W, shape)


def bias_init(shape, j, times_w, merge):
    
    shape_half = int(np.divide(shape, times_w))
    
    
    bias_each = torch.linspace(-(1+merge), 1+merge, shape_half)
    
    return bias_each.repeat(times_w)


def coefficient_init_random(shape, j, times_w, merge, weight_scale):
    
    
    t = torch.linspace(-1 / weight_scale, 1 / weight_scale, shape[1])

    # random shuffle the sorted weight
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    
    return torch.reshape(t, shape)
    
    