
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
from sklearn.metrics import pairwise_distances
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry')
sys.path.append('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry')

class CORESETStrategy(ALStrategy):
    def __init__(self, name='CORESETStrategy'):
        self.name = name
        self.activation = {}
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
        
    def query(self, opts, folderDict, man, data_query, CLSample, NumSamples=10, pred_class='XMaskPred', batchsize=None, save_uc=False, device='cuda', previous=True):
        
        # self=strategy
        data_uc = data_query
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        data_train = man.datasets['train'].data
        
        # Set embedding hook
        for layer, mod in net.model['unet'].network.named_modules():
            if 'encoder' in layer and 'convs' in layer and 'nonlin' in layer:
                mod_core=mod
        mod_core.register_forward_hook(self.get_activation('core'))
       
        
        # Extract embeddings
        X_ul = self.getEmbeddings(opts, data_uc, net)
        X_set = self.getEmbeddings(opts, data_train, net)
        
        # CoreSet
        idx = self.furthest_first(X_ul, X_set, NumSamples)
        
        # Create patches      
        samples=[]
        for i in idx:
            patch=data_uc[i]
            samples.append(patch)

        if False:
            dataset = opts.CLDataset()
            dataset.load_dataset_mds3D(man, opts, folderDict, samples)
            opts.CLSample.predict(samples, opts, net)
            import matplotlib.pyplot as plt
            for s in samples[0:10]:
                s.plotSample(plotlist=['XImage', 'P', 'Y'], color=True)
                #plt.imshow(s.F['uncertantyMap'][1,:,:])
                #plt.show()
                #plt.imshow(s.F['uncertantyMap'][2,:,:])
                
        return samples


    def getEmbeddings(self, opts, data, net):

        # Order samples by image
        if opts.dim==2:
            imagenames = list(np.unique([s.F['imagename'] for s in data]))
            data_load = pd.DataFrame()
            for imn in imagenames:
                data_im = [{'imagename': s.F['imagename'], 'slice': int(s.F['slice']), 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in data if s.F['imagename']==imn]
                data_load = pd.concat([data_load, pd.DataFrame(data_im)])
            data_load.reset_index(inplace=True)
        else:
            imagenames = list(np.unique([s.F['imagename'] for s in data]))
            data_load = pd.DataFrame()
            for imn in imagenames:
                data_im = [{'imagename': s.F['imagename'], 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in data if s.F['imagename']==imn]
                data_load = pd.concat([data_load, pd.DataFrame(data_im)])
            data_load.reset_index(inplace=True)
        
        # Create dataloader
        batch_size=8
        dataloader_train = net.model['unet'].get_dataloaders_BF(data_load, batch_size=batch_size)
        NumBatches = math.ceil(len(data_load)/dataloader_train.data_loader.batch_size)
        embeddings=[]
        properties=[]
        device='cuda'
        
        # Extract embeddings
        pbar = tqdm(total=NumBatches)
        pbar.set_description("Compute embedding")
        for b in range(NumBatches):
            pbar.update()
            batch = next(dataloader_train)
            dataB = batch['data']
            #target = batch['target']
            props = batch['properties']
            dataB = dataB.to(device, non_blocking=True)
            out = net.model['unet'].network(dataB)
            emb = self.activation['core'].reshape(dataB.shape[0],-1)
            embeddings.append(emb.cpu())
            for i in range(len(out)): out[i] = out[i].detach_().cpu()
            properties = properties + props
        pbar.close()

        # Create idxs to sort samples
        idxs=[]
        for prop in properties:
            for i,s in enumerate(data):
                sID = str(s.F['ID']) + '_' + str(s.F['IDP'])
                propID = str(prop['ID']) + '_' + str(prop['IDP'])
                if sID==propID:
                    idxs.append(i)
                    continue
        
        # Order embeddings
        embeddings = torch.vstack(embeddings)
        Xemb = torch.zeros(embeddings.shape)
        for i, idx in enumerate(idxs):
            Xemb[idx] = embeddings[i]
        Xemb = Xemb.numpy()
        
        return Xemb
    
    def furthest_first(self, X_ul, X_set, n):
    
            #sys.exit()

            m = np.shape(X_ul)[0]
            if np.shape(X_set)[0] == 0:
                min_dist = np.tile(float("inf"), m)
            else:
                dist_ctr = pairwise_distances(X_ul, X_set)
                min_dist = np.amin(dist_ctr, axis=1)
    
            idxs = []
    
            for i in range(n):
                idx = min_dist.argmax()
                idxs.append(idx)
                dist_new_ctr = pairwise_distances(X_ul, X_ul[[idx], :])
                for j in range(m):
                    min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
    
            return idxs
