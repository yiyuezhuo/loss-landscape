# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:01:56 2018

@author: yiyuezhuo
"""

import cifar10.models.resnet as resnet

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import numpy as np

import h5py

net = resnet.ResNet56()
model_file = r'cifar10\trained_nets\resnet56_sgd_lr=0.1_bs=128_wd=0.0005\model_300.t7'

def inspect(_storage, _loc):
    global storage,loc
    storage = _storage
    loc = _loc
    return _storage

#stored = torch.load(model_file, map_location=lambda storage, loc: storage)
stored = torch.load(model_file, map_location=inspect)

print("acc: {}".format(stored['acc']))
print("epoch: {}".format(stored['epoch']))
print("keys: {}".format(stored.keys()))

if 'state_dict' in stored.keys():
    net.load_state_dict(stored['state_dict'])
else:
    net.load_state_dict(stored)


def make_direction_1d(weights, ignore_bn=True):
    dl = []
    for w in weights:
        if len(w.shape) == 4: # 2d conv (out, input, width, height)
            noise = torch.randn(w.shape)
            for ww, no in zip(w, noise): # filter fold
                no.div_(no.norm())
                no.mul_(ww.norm())
            dl.append(noise)
        else: # batch norm shift and scale parameters
            if ignore_bn:
                dl.append(w.data.clone())
            else:
                noise = torch.randn(w.shape)
                noise.div_(noise.norm())
                noise.mul_(w.norm())
                dl.append(noise)
    return dl

def set_weights(net, weights):
    for p in net.parameters():
        p.data = weights.data

def apply_direction_1d(net, weights, direcs, k):
    for p,w,d in zip(net.parameters(), weights, direcs):
        p.data = w.data + d.data * k
        
def apply_direction_2d(net, weights, direcs, k, direcs2, k2):
    for p,w,d,d2 in zip(net.parameters(), weights, direcs, direcs2):
        p.data = w.data + d.data * k + d2.data * k2


#torch.manual_seed(43)
torch.manual_seed(123)


origin_weights = []
for p in net.parameters():
    origin_weights.append(p.clone())
    
direcs =  make_direction_1d(origin_weights)
direcs2 = make_direction_1d(origin_weights)


normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


data_folder = 'cifar10/data'
trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                       download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=False)
'''
testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                       download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=False)
# 使用train 的loss才能反应迭代时候算法看到的情况
'''

use_cuda = torch.cuda.is_available()
#use_cuda = False
# CPU time: 10min  vs GPU time: 20s

if use_cuda:
    net.cuda()
    origin_weights = [w.cuda() for w in origin_weights]
    direcs = [d.cuda() for d in direcs]
    direcs2 = [d.cuda() for d in direcs2]
    
net.eval()


def eval_loss(net, train_loader):
    
    total_size = 0
    total_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for inputs, targets in train_loader:
        batch_size = inputs.size(0)
        total_size += batch_size
        
        if use_cuda:
            inputs,targets = inputs.cuda(), targets.cuda()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()*batch_size
        
    return total_loss / total_size


def eval_direcs(net, train_loader, direcs, start = -0.5, end=0.5, n_iter=10):
    loss_l = []
    for k in np.linspace(start, end, n_iter):
        apply_direction_1d(net, origin_weights, direcs, k)
        loss = eval_loss(net, train_loader)
        loss_l.append(loss)
        print('k:{} s:{}'.format(k, loss))
    return loss_l

def setup_surface_1d(net, direcs, surface_name = 'surface.h5', 
                     direcs_name = 'direcs.pt',
                     start = -0.5, end=0.5, n_iter=50):
    
    torch.save(direcs, direcs_name)
    
    f = h5py.File(surface_name)
    f['direcs_name'] = direcs_name
    f['start'] = start
    f['end'] = end
    f['n_iter'] = n_iter
    kl = np.linspace(start,end,n_iter)
    f['kl'] = kl
    f['succ'] = ~np.ma.make_mask(kl)
    f['loss'] = np.zeros_like(kl)
    f.close()
    
def process_surface_1d(net, surface_name = 'surface.h5'):
    f = h5py.File(surface_name)
    direcs = torch.load(f['direcs_name'].value)
    
    kl = f['kl'][:]
    
    for i,k in enumerate(kl):
        if f['succ'][i]:
            print('skip idxL {} k: {}'.format(i,k))
            continue
        apply_direction_1d(net, origin_weights, direcs, k)
        loss = eval_loss(net, train_loader)
        f['loss'][i] = loss
        f['mask'][i] = True
        print('write idx: {} k: {} loss: {}'.format(i,k,loss))
    f.close()
        
def setup_surface_2d(net, direcs, direcs2,
                     surface_name = 'surface2d.h5',
                     direcs_name = 'direcs.pt', direcs2_name = 'direcs2.pt',
                     start_x = -1.0, end_x=1.0, n_iter_x=51,
                     start_y = -1.0, end_y=1.0, n_iter_y=51):
    
    torch.save(direcs, direcs_name)
    torch.save(direcs2, direcs2_name)
    
    f = h5py.File(surface_name)
    
    f['direcs_name'] = direcs_name
    f['direcs2_name'] = direcs2_name
    
    f['start_x'] = start_x
    f['end_x'] = end_x
    f['n_iter_x'] = n_iter_x
    
    f['start_y'] = start_y
    f['end_y'] = end_y
    f['n_iter_y'] = n_iter_y
    
    _kx = np.linspace(start_x,end_x,n_iter_x)
    _ky = np.linspace(start_y,end_y,n_iter_y)
    kx,ky = np.meshgrid(_kx, _ky, indexing='ij')
    
    f['kx'] = kx
    f['ky'] = ky
    
    f['succ'] = ~np.ma.make_mask(kx)
    f['loss'] = np.zeros_like(kx)
    f.close()
    
def process_surface_2d(net, surface_name = 'surface2d.h5'):
    f = h5py.File(surface_name)
    direcs = torch.load(f['direcs_name'].value)
    direcs2 = torch.load(f['direcs2_name'].value)
    
    kx = f['kx'][:]
    ky = f['ky'][:]
    
    n_iter_x = int(f['n_iter_x'].value)
    n_iter_y = int(f['n_iter_y'].value)
    
    for i in range(n_iter_x):
        for j in range(n_iter_y):
            k = kx[i,j]
            k2 = ky[i,j]
            
            if f['succ'][i,j]:
                print('skip idx {},{} k: {},{} loss: {}'.format(i,j,k,k2,f['loss'][i,j]))
                continue
            apply_direction_2d(net, origin_weights, direcs, k, direcs2, k2)
            loss = eval_loss(net, train_loader)
            f['loss'][i,j] = loss
            f['succ'][i,j] = True
            print('write idx: {},{} k: {},{} loss: {}'.format(i,j,k,k2,loss))
    f.close()

'''

import matplotlib.pyplot as plt

setup_surface_1d(net, direcs)
process_surface_1d(net)

f = h5py.File('surface.h5')
plt.plot(f['loss'])
f.close()

'''
'''
import matplotlib.pyplot as plt

setup_surface_2d(net, direcs, direcs2)
process_surface_2d(net)

f = h5py.File('surface2d.h5')
Z = f['loss'][:]
_X = np.linspace(f['start_x'].value,f['end_x'].value,f['n_iter_x'].value)
_Y = np.linspace(f['start_y'].value,f['end_y'].value,f['n_iter_y'].value)
X,Y = np.meshgrid(_X,_Y,indexing='ij')

vmin, vmax, vlevel = 0.1, 10, 0.5
fig = plt.figure(figsize=(16,9))
CS = plt.contour(X, Y, Z, cmpa='summer', levels=np.arange(vmin, vmax, vlevel))
plt.clabel(CS, inline=1, fontsize=8)

f.close()

'''
