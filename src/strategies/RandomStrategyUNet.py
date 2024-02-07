#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:07:40 2023

@author: bernifoellmer
"""

import random
from strategies.ALStrategy import ALStrategy

class RandomStrategy(ALStrategy):
    def __init__(self, name='RANDOM'):
        self.name = name
    
    def query(self, opts, folderDict, manager, data_query, CLSample=None, NumSamples=10, batchsize=None, pred_class=['XMask'], save_uc=False, previous=None):
        idx = [x for x in range(len(data_query))]
        random.shuffle(idx)
        idx = idx[0:NumSamples]
        samples = [data_query[i] for i in idx]
        return samples
