# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.SaveState import SaveState
from utils.TensorboardViewer import TensorboardViewer


class TensorboardViewerTorch(TensorboardViewer):

    def __init__(self, settingsfilepath, mode=SaveState.SAVE, update_time=None):
        TensorboardViewer.__init__(self, settingsfilepath, mode=mode, update_time=update_time)
        self.step = 0
        self.epoch = 0
        self.log_dir = ''
        self.update_time = update_time
        
    def next_step(self):
        self.step = self.step + 1
        
    def next_epoch(self):
        self.epoch = self.epoch + 1
        
    def __get_iter__(self, iterator='step'):
        if iterator=='step':
            return self.step
        elif iterator=='epoch':
            return self.epoch
        else:
            raise ValueError('Iterator: ' + iterator + ' does not exist.')

    def start_log(self, log_dir, comment=''):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment, flush_secs=60)
        
    def add_scalar(self, label='Loss/train', value=0.1, iterator='step'):
        i = self.__get_iter__(iterator)
        self.writer.add_scalar(label, value, global_step=i)
        
    def add_histogram(self, name, hist, iterator='step'):
        i = self.__get_iter__(iterator)
        self.writer.add_histogram(name, hist, global_step=i)
        
    def add_image(self, imagename, image, iterator='step', dataformats='HW'):
        i = self.__get_iter__(iterator)
        self.writer.add_image(imagename, image, global_step=i, dataformats=dataformats)
        
    def add_images(self, imagesname, images, iterator='step'):
        i = self.__get_iter__(iterator)
        self.writer.add_image(imagesname, images, global_step=i)
        
    def add_text(self, name, txt, iterator='step'):
        i = self.__get_iter__(iterator)
        self.writer.add_text(name, txt, global_step=i)
        
    def add_embedding(self, embeddings, metadata, label_img=None, iterator='step', tag='default'):
        i = self.__get_iter__(iterator)
        #self.writer.add_embedding(embeddings, metadata=metadata, label_img=label_img, global_step=i)
#        import keyword
#        meta = []
#        while len(meta)<3:
#            meta = meta+keyword.kwlist # get some strings
#        meta = meta[:3]
        self.writer.add_embedding(embeddings, metadata=metadata, label_img=label_img ,global_step=i, tag=tag)
        #self.deleteLastNewlineTSV(self.log_dir)
        
    def deleteLastNewlineTSV(self, logfolder):
        cfiles = []
        for root, dirs, files in os.walk(logfolder):
            for file in files:
                if file.endswith('.tsv'):
                    cfiles.append(os.path.join(root, file))
        for file in cfiles:
            self.deleteLastNewline(file)

    def deleteLastNewline(self, filepath):
        # Read in the file
        with open(filepath, 'r') as file :
            filedata = file.read()
        # Replace the target string
        if filedata[-1]=='\n':
            filedata = filedata[0:-1]
        # Write the file out again
        with open(filepath, 'w') as file:
            file.write(filedata)
            
    def close(self):
        self.writer.close()
        

#import pandas as pd
#f = pd.read_table('H:/cloud/cloud_data/Projects/DL/Code/src/logs/model_training02_02_06_2020_20_52_09/00016/default/metadata.tsv', sep='\t')
        
#import keyword
#import torch
#meta = []
#while len(meta)<100:
#    meta = meta+keyword.kwlist # get some strings
#meta = meta[:100]
#
#for i, v in enumerate(meta):
#    meta[i] = v+str(i)
#
#label_img = torch.rand(100, 3, 10, 32)
#for i in range(100):
#    label_img[i]*=i/100.0
#
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter(log_dir='H:/cloud/cloud_data/Projects/DL/Code/src/logs/tmp01')
#writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
#writer.add_embedding(torch.randn(100, 5), label_img=label_img)
#
#writer.add_embedding(torch.randn(100, 5), metadata=meta, global_step=0)
#writer.add_embedding(torch.randn(100, 5), metadata=meta, global_step=1)
#writer.add_embedding(torch.randn(100, 5), metadata=meta, global_step=2)
