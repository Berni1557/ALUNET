#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:07:40 2023

@author: bernifoellmer
"""

from strategies.ALStrategy import ALStrategy

class FullStrategy(ALStrategy):
    def __init__(self, name='FULL'):
        self.name = name
    
    def query(self, opts, folderDict, manager, data_query, CLSample=None, NumSamples=None, batchsize=None, pred_class=['XMask'], save_uc=False, previous=None):
        samples = data_query
        return samples