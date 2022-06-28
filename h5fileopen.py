# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 08:39:19 2022

@author: ramra
"""

import h5py

f = h5py.File('best_model.h5' , 'r')

print(list(f.keys()))

dset = f['top_level_model_weights']

print(type(dset))