# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:14:05 2018

@author: yiyuezhuo
"""

import h5py

f = h5py.File('h5py_test.h5','w')
f['yyz'] = 'yyz'
assert False