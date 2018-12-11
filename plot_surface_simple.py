# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 20:41:45 2018

@author: yiyuezhuo
"""

import socket
import copy
import h5py
import os

import torch
import torch.nn as nn

import numpy as np

import model_loader
import net_plotter

def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"

def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already setted up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file




class args():
    cuda = True
    batch_size = 128
    ngpu = 1 
    
    dataset = 'cifar10'
    datapath = 'cifar10/data'
    data_split = 1
    split_idx = 0
    
    #model = 'resnet56'\
    model = 'vgg9'
    model_file  = 'cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7'
    model_file2 = 'cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1/model_300.t7'
    
    loss_name = 'crossentropy'
    
    x = '-0.5:1.5:401'
    y = None
    
    plot = True
    
comm, rank, nproc = None, 0, 1 # No mpi setting    
    
if args.cuda:
    if not torch.cuda.is_available():
        raise Exception('User selected cuda option, but cuda is not available on this machine')
    gpu_count = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpu_count)
    print('Rank %d use GPU %d of %d GPUs on %s' %
          (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]

net = model_loader.load(args.dataset, args.model, args.model_file)
w = net_plotter.get_weights(net) # initial parameters
s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
if args.ngpu > 1:
    # data parallel with multiple GPUs on a single node
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))


dir_file = net_plotter.name_direction_file(args) # name the direction file
if rank == 0:
    net_plotter.setup_direction(args, dir_file, net)

surf_file = name_surface_file(args, dir_file)
if rank == 0:
    setup_surface_file(args, surf_file, dir_file)
