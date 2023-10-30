import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

from Resnet import *    

from ms_layer import MS_resnet_layer 


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SortMean(object):
    def __init__(self, num_batch):
        self.reset(num_batch)
    
    def reset(self,num_batch):
        self.rate_list = []
        self.num = num_batch
        self.avg = 0

    def update(self,val):
        if len(self.rate_list)<self.num:
            self.rate_list.append(val)
        else:  
            self.rate_list.append(val)
            self.rate_list.pop(0)
        self.avg = np.mean(self.rate_list)

class SortMove(object):
    def __init__(self, weight=0.9):
        self.reset(weight)
        
    def reset(self,weight):
        self.avg = 0
        self.weight = weight
        self.flag = False

    def update(self,val):
        if self.flag:
            self.avg = self.weight*self.avg + (1-self.weight)*val
        else: self.avg = val

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)



def load_ms_layer(model_name,classes_nums, pretrain=True, require_grad=True):

    print('==> Building model..')
    if model_name == 'resnet50_ms':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = MS_resnet_layer(net,50, 512, classes_nums)
    elif model_name == 'resnet18_ms':
        net = resnet18(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = MS_resnet_layer(net,18, 512, classes_nums)
    elif model_name == 'resnet101_ms':
        net = resnet101(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = MS_resnet_layer(net,101, 512, classes_nums)
    elif model_name == 'resnet152_ms':
        net = resnet152(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = MS_resnet_layer(net,152, 512, classes_nums)

    return net