#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:07:40 2023

@author: bernifoellmer
"""

import os, sys
# Set system path
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry')
sys.path.append('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry')
import random
import torch
from tqdm import tqdm
import math
import numpy as np
from nngeometry.object import PMatKFAC, PMatDiag, PMatBlockDiag, PMatDense, PVector, PMatLowRank, PMatImplicit, PMatEKFAC, PVector
from nngeometry.layercollection import LayerCollection
import pandas as pd
from strategies.ALStrategy import ALStrategy
from utils.ALUNet import UNetPatch
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from kneed import KneeLocator
from utils.ct import CTImage, CTRef

def enable_dropout(model, drop_rate=0.5):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p=drop_rate
            m.train()
        if m.__class__.__name__.startswith('ConvDropoutNormReLU'):
            m.p=drop_rate
            m.train()
            
def disable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p=0.0
            m.eval()
        if m.__class__.__name__.startswith('ConvDropoutNormReLU'):
            m.p=0.0
            m.eval()

def similarity_fim(data0, data1, FI):
    S=torch.zeros((len(data0), len(data1)))
    G0 = torch.vstack([s.F['grad']*FI.cpu() for s in data0])
    G1 = torch.vstack([s.F['grad'] for s in data1]).transpose(0,1)
    S = torch.matmul(G0,G1)/len(FI)
    Sn = S.numpy()
    return Sn
             
            
class USIMCStrategy(ALStrategy):
    def __init__(self, name='USIMCStrategy'):
        self.name = name
        self.dtype=torch.FloatTensor

    def query(self, opts, folderDict, man, data_query, NumMCD=10, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False):
        
        # self=strategy
        layer_used='mid'
        NumSamplesQ=NumSamples*5
        NumParams=10000
        save_uc=False
        dropout_rate=0.1
        
        # Compute uncertainty
        data_uc = data_query
        data_uc = self.getUncertaintyV(opts, folderDict, man, data_uc, opts.CLPatch, NumSamples=len(data_query), batchsize=500, pred_class=['XMaskPred'], previous=previous, save_uc=save_uc, dropout_rate=dropout_rate)
        
        # Extract weights
        data_train = man.datasets['train'].data
        imagenames = np.unique([s.F['imagename'] for s in data_train])
        fp_nnunet = opts.fp_nnunet
        fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
        fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        label_ignore = load_json(os.path.join(fp_nnUNet_preprocessed, dname, 'dataset.json'))['labels']['ignore']
        weight = np.zeros(label_ignore)
        for im in imagenames:
            #print(im)
            fip_image = os.path.join(fp_nnUNet_raw, dname, 'labelsTr', im)
            arr = CTImage(fip_image).image()
            for c in range(label_ignore):   
                weight[c] = weight[c] + (arr==c).sum()/arr.size
        weight = 1/weight
        weight[0] = 0
        weight = weight/weight.sum()

        # Weighted uncertainty
        uncertantyV = np.array([s.F['uncertanty'] for s in data_uc])
        #uncertanty = weight[0]*uncertantyV[:,0]+weight[1]*uncertantyV[:,1]+weight[2]*uncertantyV[:,2]
        uncertanty = np.zeros(uncertantyV.shape[0])
        for i in range(label_ignore):
            uncertanty = uncertanty+weight[i]*uncertantyV[:,i]
        prop = uncertanty/uncertanty.sum()
        
        # Get Network and layers
        net, layer_collection, layers = self.getNetLayer(opts, man, folderDict, layer_used, previous)

        # Compute F
        #data_F = np.random.choice(data_uc, size=min(NumFIMMax, len(data_uc)), replace=False, p=prop)
        #FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, previous)
        idxG = np.array([i for i in range(NumParams)])
        
        # Compute gradients of data_QU and data_uc
        self.grad_data(opts, folderDict, man, net, data_uc, layer_collection, previous, idxG=idxG, fisherG=True)

        # Estimate number of cluster
        x = np.array([i for i in range(len(uncertanty))])
        y = sorted(uncertanty)[::-1]
        kneedle = KneeLocator(x, y, S=10.0, curve="convex", direction="decreasing")
        NumSamplesQ = int(min(max(kneedle.knee, NumSamplesQ), len(data_uc)))
        # kneedle.plot_knee()
        
        #data_Q = np.random.choice(data_uc, size=min(NumSamplesQ, len(data_uc)), replace=False, p=prop)
        data_Q = np.random.choice(data_uc, size=min(NumSamplesQ, len(data_uc)), replace=True, p=prop)
        data = np.vstack([s.F['grad'].float() for s in data_uc])
        queryData = np.vstack([s.F['grad'].float() for s in data_Q])
        n = len(data)
        num_queries = len(queryData)
        #mode='dense'
        #lambdaValfloat=1.0
        if len(data_uc)<1000:
            optimizer = 'NaiveGreedy'
        else:
            optimizer = 'StochasticGreedy'
        stopIfZeroGain = False
        stopIfNegativeGain = False
        verbose = False
        show_progress = False
        epsilon = 0.01
        budget = NumSamples
        queryDiversityEta = 1.0
        obj = FacilityLocationVariantMutualInformationFunction(n, num_queries, query_sijs=None, data=data, queryData=queryData, metric='cosine', queryDiversityEta=queryDiversityEta)
        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
        greedyIndices = [x[0] for x in greedyList]
        data_A = [data_uc[i] for i in greedyIndices]       
        
        for s in data_A:
            if 'grad' in s.F: del s.F['grad']
            if 'uncertanty' in s.F: del s.F['uncertanty']
        for s in data_query:
            if 'grad' in s.F: del s.F['grad']
            if 'uncertanty' in s.F: del s.F['uncertanty']
        for s in data_Q:
            if 'grad' in s.F: del s.F['grad']
            if 'uncertanty' in s.F: del s.F['uncertanty']
        return data_A
    
    
    def grad_data(self, opts, folderDict, man, net, data, layer_collection, previous, idxG=None, idx_class=None, append=False, fisherG=False):
        # Load previous networks
        
        # data = data_query
        #loss_fisher = opts.CLSample.loss_fisher
        loss_fisher = UNetPatch.loss_fisher
        batchsize = 500
        num = math.ceil(len(data)/batchsize)
        pbar = tqdm(total=num)
        pbar.set_description("Compute gradients")
        for i in range(num):
            print('Batch: ' + str(i) + ' / ' + str(num))
            pbar.update()
            batch = data[i*batchsize:(i+1)*batchsize]  
            _ = opts.CLPatch.samplesToloader(batch, opts, net, reshapeOrg=False, dtype=self.dtype, batch_size=1)
            #idx_output = None
            loss_func = loss_fisher(net, idx_output=None, idx_class=None)
            params = layer_collection.get_parameters_BF(net.model['unet'].network)
            for s in batch:
                #pbar.update()
                imaged = s.X['XImage']
                if fisherG:
                    loss_ce = loss_func(imaged)
                    grad_params = torch.autograd.grad(loss_ce[0].sum(), params, create_graph=True)
                else:
                    outputi = net.model['unet'].network(imaged)
                    targeti = [torch.argmax(x, dim=1, keepdim=True) for x in outputi]
                    lossi = net.model['unet'].loss(outputi, targeti)
                    grad_params = torch.autograd.grad(lossi, params, create_graph=True)
                fce = torch.hstack([torch.reshape(grad, (-1,)) for grad in grad_params]).detach().cpu()
                if append:
                    if idxG is not None:
                        fce=fce[idxG[idx_class]]
                    if s.F['grad'] is None:
                        s.F['grad'] = fce.double()
                    else: 
                        s.F['grad'] = torch.hstack([s.F['grad'], fce.double()])
                else:
                    if idxG is not None:
                        fce=fce[idxG]
                    s.F['grad'] = fce.double()
            for s in batch:
                del s.X['XImage']
                del s.Y['XMask']
                del s.P['XMaskPred']
        pbar.close()
        
        
    def getNetLayer(self, opts, man, folderDict, layer_used, previous):
        # Extract layers
        net = man.load_model(opts, folderDict, previous=previous)
        if layer_used=='mid':
            for layer, mod in net.model['unet'].network.named_modules():
                if 'encoder' in layer and 'convs' in layer and '1.conv' in layer:
                    mid_layer=layer
            layers=[mid_layer]
        elif layer_used=='decoder':
            layers=[]
            for layer, mod in net.model['unet'].network.named_modules():
                if 'decoder' in layer and ('seg' in layer or 'convs' in layer):
                    layers.append(layer)
        elif layer_used=='last':
            layers=[]
            for layer, mod in net.model['unet'].network.named_modules():
                if 'seg' in layer:
                    layers.append(layer)
        elif layer_used=='manual':
            layers=[]
            layers.append('decoder.stages.4.convs.1.conv')
            layers.append('decoder.stages.5.convs.1.conv') 
            layers.append('decoder.stages.6.convs.1.conv') 
        layer_collection = LayerCollection.from_model_BF(net.model['unet'].network, ignore_unsupported_layers=True,layer_in=layers)
        return net, layer_collection, layers

        
    def computeF(self, opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, previous, idxG=None, idx_class=None, fisherG=False):
        
        eps=1e-20

        # Compute gradients
        self.grad_data(opts, folderDict, man, net, data_F, layer_collection, previous, idxG=idxG, idx_class=idx_class, fisherG=fisherG)
        
        params = layer_collection.get_parameters_BF(net.model['unet'].network)
        num_params = int(np.sum([x.numel() for x in params]))
        gradEmb = torch.zeros(data_F[0].F['grad'].shape)
        for s in data_F:
            gradEmb = gradEmb+(data_F[0].F['grad']*data_F[0].F['grad'])
        gradEmb = (gradEmb/len(data_F)).numpy().astype('float64')
        
        population = [x for x in range(len(gradEmb))]
        prop = gradEmb/gradEmb.sum()
        NumParamsSel = min(NumParams, (gradEmb>eps).sum())
        if idxG is None:
            idxG = np.random.choice(population, size=min(num_params, NumParamsSel), replace=False, p=prop)
            F=torch.from_numpy(gradEmb[idxG])
        else:
            F=torch.from_numpy(gradEmb)
        #idxG = np.random.choice(population, size=min(num_params, NumParamsSel), replace=False)
        
        #F=torch.from_numpy(gradEmb[idxG])
        FI=(1/F).cuda()
        
        return FI, idxG


    def getUncertaintyV(self, opts, folderDict, man, data_query, CLSample, NumSamples=10, pred_class='XMaskPred', batchsize=None, save_uc=False, device='cuda', previous=True, NumRounds=10, NumSamplesMax=15000, dropout_rate=0.01):
        
        # self=strategy

        # init model
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        enable_dropout(net.model['unet'].network, dropout_rate)
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
        uncertantyL=[]
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
            # Iterate over monte carlo rounds
            rounds=[]
            for r in range(NumRounds):
                out = net.model['unet'].network(data)
                for i in range(len(out)): out[i] = out[i].detach_().cpu()
                outs = soft1(out[0])
                rounds.append(torch.unsqueeze(outs, dim=0).clone())
                #rounds.append(outs.clone())
            roundsT = torch.vstack(rounds)
            varT = torch.var(roundsT, dim=0)
            if opts.dim==2:
                uncertanty = uncertanty + list(torch.mean(varT, dim=(2,3)).numpy())
            else:
                uncertanty = uncertanty + list(torch.mean(varT, dim=(2,3,4)).numpy())
            properties = properties + props
            if save_uc: uncertantyL.append(varT)
            
            
        pbar.close()
        disable_dropout(net.model['unet'].network)
        
        if save_uc: uncertantyL=np.vstack(uncertantyL)

        # Create idxs to sort samples
        idxs=[]
        for prop in properties:
            for i,s in enumerate(data_query):
                sID = str(s.F['ID']) + '_' + str(s.F['IDP'])
                propID = str(prop['ID']) + '_' + str(prop['IDP'])
                if sID==propID:
                    idxs.append(i)
                    continue
                
        uncertantyA = np.vstack(uncertanty)
        uncertantyM = uncertantyA.mean(axis=1)
                
        # Create patches  
        idx = np.argsort(uncertantyM)[::-1]
        samples=[]
        for i in idx[0:NumSamples]:
            patch=data_query[idxs[i]]
            patch.F['uncertanty'] = uncertantyA[i]
            if save_uc:
                patch.F['uncertantyMap'] = uncertantyL[i]
            # !!! Check this is always true
            key_class = list(properties[i]['class_locations'].keys())[0]
            patch.F['classLocations']=properties[i]['class_locations'][key_class]
            samples.append(patch)

        # Set uncertainty to all samples
        for i in idx:
            #data_query[idxs[i]].F['uncertanty'] = uncertantyM[i]
            data_query[idxs[i]].F['uncertanty'] = uncertantyA[i]
            
        return samples

