
import os, sys
import numpy as np
import torch
from torch import nn
from scipy import stats
from tqdm import tqdm
import random
from scipy.stats import entropy
import pdb
from scipy import stats
import pandas as pd
import math
import time
from strategies.ALStrategy import ALStrategy
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry')
sys.path.append('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry')

def enable_dropout(model, drop_rate=0.5):
    for m in model.modules():
        if m.__class__.__name__.startswith('ConvDropoutNormReLU'):
            m.p=drop_rate
            m.train()


def disable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p=0.0
            m.eval()
                    
class BALDStrategy(ALStrategy):
    def __init__(self, name='BALDStrategy'):
        self.name = name

    def query(self, opts, folderDict, man, data_query, CLSample, NumSamples=10, pred_class='XMaskPred', batchsize=None, save_uc=False, device='cuda', previous=True, NumRounds=10):
        
        # self=strategy

        # init model
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        enable_dropout(net.model['unet'].network, 0.01)
        soft1 = torch.nn.Softmax(dim=1)
        
        # Order samples by image
        if opts.dim==2:
            imagenames = list(np.unique([s.F['imagename'] for s in data_query]))
            data_load = pd.DataFrame()
            for imn in imagenames:
                data_im = [{'imagename': s.F['imagename'], 'slice': int(s.F['slice']), 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in data_query if s.F['imagename']==imn]
                data_load = pd.concat([data_load, pd.DataFrame(data_im)])
            data_load.reset_index(inplace=True)
        else:
            imagenames = list(np.unique([s.F['imagename'] for s in data_query]))
            data_load = pd.DataFrame()
            for imn in imagenames:
                data_im = [{'imagename': s.F['imagename'], 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in data_query if s.F['imagename']==imn]
                data_load = pd.concat([data_load, pd.DataFrame(data_im)])
            data_load.reset_index(inplace=True)
        
        # Create dataloader
        dataloader_train = net.model['unet'].get_dataloaders_BF(data_load, batch_size=8)
        NumBatches = math.ceil(len(data_load)/dataloader_train.data_loader.batch_size)
        uncertanty=[]
        uncertantyA=[]
        properties=[]
        device='cuda'
        pbar = tqdm(total=NumBatches)
        pbar.set_description("Compute entropy")
        for b in range(NumBatches):
            pbar.update()
            batch = next(dataloader_train)
            data = batch['data']
            props = batch['properties']
            data = data.to(device, non_blocking=True)
            entA=[]
            entB=[]
            for r in range(NumRounds):
                out = net.model['unet'].network(data)
                for i in range(len(out)): out[i] = out[i].detach_().cpu()
                outs = soft1(out[0])
                entOut = entropy(outs, base=2, axis=1)
                entA.append(np.expand_dims(outs, axis=0))
                entB.append(np.expand_dims(entOut, axis=0))
            entA = np.vstack(entA).mean(axis=0)
            entA = entropy(entA, base=2, axis=1)
            entB = np.vstack(entB)
            entB = entB.mean(axis=0)
            entBald = entA-entB
            if opts.dim==2:
                uncertanty = uncertanty + list(np.mean(entBald, axis=(1,2)))
            else:
                uncertanty = uncertanty + list(np.mean(entBald, dim=(1,2,3)))
            properties = properties + props
            if save_uc: uncertantyA.append(entBald)
        pbar.close()
        disable_dropout(net.model['unet'].network)
        
        
        if save_uc: uncertantyA=np.vstack(uncertantyA)

        # Create idxs to sort samples
        idxs=[]
        for prop in properties:
            for i,s in enumerate(data_query):
                sID = str(s.F['ID']) + '_' + str(s.F['IDP'])
                propID = str(prop['ID']) + '_' + str(prop['IDP'])
                if sID==propID:
                    idxs.append(i)
                    continue
                
        # Create patches  
        idx = np.argsort(uncertanty)[::-1]
        samples=[]
        for i in idx[0:NumSamples]:
            patch=data_query[idxs[i]]
            patch.F['uncertanty'] = uncertanty[i]
            if save_uc:
                patch.F['uncertantyMap'] = uncertantyA[i]
            key_class = list(properties[i]['class_locations'].keys())[0]
            patch.F['classLocations']=properties[i]['class_locations'][key_class]
            samples.append(patch)

        # Set uncertainty to all samples
        for i in idx:
            data_query[idxs[i]].F['uncertanty'] = uncertanty[i]
            
        return samples

