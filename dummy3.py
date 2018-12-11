# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:16:21 2018

@author: yiyuezhuo
"""

import time
import h5py
import numpy as np

f = h5py.File('endless.h5')
if 'count' in f.keys():
    print('reading...')
    print(f'count: {f["count"].value}')
    for i in range(1,f['count'].value):
        print(f'{i}: {f[str(i)].value}')
else:
    f['count'] = 0
    f['count_list'] = []
print('writing...')
while True:
    f[str(f['count'].value)] = -1
    f['count'].write_direct(np.array(f['count'].value + 1))
    time.sleep(1)
f.close()