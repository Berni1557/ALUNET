
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
import json
from strategies.ALStrategy import ALStrategy
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry')
sys.path.append('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry')

class EntropyStrategy(ALStrategy):
    def __init__(self, name='EntropyStrategy'):
        self.name = name

    def query(self, opts, folderDict, man, data_query, CLSample, NumSamples=10, pred_class='XMaskPred', batchsize=None, save_uc=False, device='cuda', previous=True):
        
        # self=strategy
        data_query = data_query
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
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
        entropies=[]
        entropiesA=[]
        properties=[]
        device='cuda'
        pbar = tqdm(total=NumBatches)
        pbar.set_description("Compute entropy")
        for b in range(NumBatches):
            pbar.update()
            batch = next(dataloader_train)
            data = batch['data']
            #target = batch['target']
            props = batch['properties']
            data = data.to(device, non_blocking=True)
            out = net.model['unet'].network(data)
            for i in range(len(out)): out[i] = out[i].detach_().cpu()
            outs = soft1(out[0])
            entA = entropy(outs, base=2, axis=1)
            if save_uc:
                entropiesA.append(entA)
            if opts.dim==2:
                ent = np.mean(entA, axis=(1,2))
            else:
                ent = np.mean(entA, axis=(1,2,3))
            entropies = entropies + list(ent)
            properties = properties + props
        pbar.close()
        
        if save_uc: entropiesA=np.vstack(entropiesA)

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
        idx = np.argsort(entropies)[::-1]
        samples=[]
        for i in idx[0:NumSamples]:
            patch=data_query[idxs[i]]
            #p=opts.CLPatch()
            #p.ID=s.ID
            #p.F=s.F.copy()
            patch.F['entropy'] = entropies[i]
            if save_uc:
                patch.F['entropyMap'] = entropiesA[i]
            # !!! Check this is always true
            key_class = list(properties[i]['class_locations'].keys())[0]
            patch.F['classLocations']=properties[i]['class_locations'][key_class]
            #p.F['patch']=0
            samples.append(patch)
        
        return samples
