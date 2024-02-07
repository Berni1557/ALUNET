# -*- coding: utf-8 -*-

import os, sys
#from abc import ABC, abstractmethod
#import pandas as pd
#import h5py
#import numpy as np
#from helper.helper import splitFolderPath
from utils.YAML import YAML, YAML_MODE
#from helper.helper import splitFolderPath, splitFilePath
#from sklearn.model_selection import train_test_split
from collections import defaultdict

class DatasetBaseModel():
 
    def __init__(self):
        self.props = defaultdict(lambda: None)
        super().__init__()
        
    def yml_process(self, settingsfilepath, mode):
        # Save yml file
        if mode==YAML_MODE.SAVE:
            self.yml.save(self.props, settingsfilepath)
            
        # Update yml file
        if mode==YAML_MODE.UPDATE:
            self.props = self.yml.update(self.props, settingsfilepath)
            
        # Load yml file
        if mode==YAML_MODE.LOAD:
            self.props = self.yml.load(settingsfilepath)
