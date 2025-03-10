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
        # num_sample = 8000
        # num_sample = 20
        self.num_train = int(self.num_sample*0.8)
        self.num_test = int(self.num_sample*0.2)

        self.epoches = 6400
        self.learning_rate = 1e-3

        # width_vector = [80, 160, 320, 640, 1280, 2560]
#         self.width_vector = [20, 80, 320]
        self.width = 40
        # width_vector = [40]

        self.times_w = 2



        # regularization

        # regularization parameter
        self.weight_decay = 0

        # L^p norm
        self.p_norm = 2


        self.merge = 0
        
        
def setup_seed(seed):
    
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
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
    
#     setup_seed(2022)
#     setup_seed(int(2022+ (1000*j)))

#     Coe = torch.rand(1)
#     if j == 1:
#     print('Before: bias'+str(j)+': '+str(Coe))
    
    shape_half = int(np.divide(shape, times_w))
    
#     bias_each = torch.empty(shape_half)
#     nn.init.uniform_(bias_each, a = -(1+merge), b = 1+merge)
    
#     Coe = torch.rand(1)
#     if j == 1:
#     print('random: '+str(bias_each))
#     print('After: bias'+str(j)+': '+str(Coe))
    
    bias_each = torch.linspace(-(1+merge), 1+merge, shape_half)
    
#     bias_each.repeat(shape)
    
#     print(bias_each.repeat(times).shape)
#     return torch.cat((bias_each, bias_each, bias_each, bias_each, bias_each, bias_each, bias_each, bias_each))
    return bias_each.repeat(times_w)


def coefficient_init_random(shape, j, times_w, merge, weight_scale):
    
#     setup_seed(2022)
#     setup_seed(int(2022+ (1000*j)))
    
#     Coe = torch.rand(4)
#     if j == 1:
#         print('Before: coe'+str(j)+': '+str(Coe))
    
#     shape_half = int(np.divide(shape[1], times_w))
# #     shape_np = np.array(shape)
# #     shape_half = np.divide(shape_np,2)
# #     shape_half = int(np.ceil(shape_half).tolist())
    
    
#     coe_each = torch.empty(shape_half)
#     nn.init.uniform_(coe_each, a = -(1+merge), b = 1+merge)
    
#     Coe = torch.rand(4)
#     if j == 1:
#         print('After: coe'+str(j)+': '+str(Coe))

    
#     return coe_each.repeat(times_w)
    
    t = torch.linspace(-1 / weight_scale, 1 / weight_scale, shape[1])

#     # initialize the weight as sorted.
#     return torch.reshape(t, shape)

    # random shuffle the sorted weight
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    
    return torch.reshape(t, shape)
    
    