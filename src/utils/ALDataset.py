#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:07:40 2023

@author: bernifoellmer
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils.ALManager import ALManager
from utils.ALSample import ALSample
from utils.DataframeBaseModel import DataframeBaseModel, defaultdict
from utils.DataAccess import DataAccess
from abc import ABC, abstractmethod
from utils.ct import CTImage, CTRef

class ALDataset(ABC):
    def __init__(self, name='', NumChannelsIn=1, NumChannelsOut=1, Tasknames=['XMask']):
        self.name = name
        self.NumChannelsIn=NumChannelsIn
        self.NumChannelsOut=NumChannelsOut
        self.NumTasks=1
        self.Tasknames=Tasknames
    
    def create_dataset(self, opts):
        pass

        
    def load_dataset_hdf5(self, folderDict, data, debug=True):
        # self = dataset

        # Create DataAccess
        dataaccess = DataAccess()
        hdf5filepath = folderDict['fip_hdf5_all']
        # Get index
        IDs = [s.ID for s in data]
        
        # Load image by index
        datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, debug=debug)

        pbar = tqdm(total=len(data))
        pbar.set_description("Loading image")
        for i,s in enumerate(data):

            pbar.update(1)
            s.X = {'XImage': torch.from_numpy(datadict['X']['XImage'][i:i+1,:])}
            s.Y = {k:torch.from_numpy(datadict['Y'][k][i:i+1]) for k in datadict['Y']}
            s.features = {k: datadict['F'][k][i:i+1] for k in datadict['F']}
        pbar.close()
        
        del datadict
        
    # @abstractmethod
    # def save_hdf5_all(self, opts, folderDict):
    #     pass
    
    # @abstractmethod
    # def load_hdf5_all(self, folderDict, data):
    #     pass

class ALDatasetMulti(ALDataset):
    def __init__(self, name='', NumChannelsIn=1, NumChannelsOut=[4, 6, 16], Tasknames=['XRegion', 'XMain', 'XSegment']):
        ALDataset.__init__(self, name='', NumChannelsIn=NumChannelsIn, NumChannelsOut=NumChannelsOut)
        self.name = name
        self.NumTasks=len(NumChannelsOut)
        self.Tasknames=Tasknames
    
    def create_dataset(self, opts):
        pass
    
    # @abstractmethod
    # def save_hdf5_all(self, opts, folderDict):

    def save_hdf5_all(self, opts, folderDict):
        # self = manager
        
        #fp_dataset = folderDict['fp_dataset']
        #fp_dataset = os.path.join(opts.fp_modules, opts.dataset, 'data')
        fp_patches = os.path.join(opts.fp_active, 'data', 'patches')
        
        # Split dataset in train, test, valid
        df_patches = pd.read_pickle(os.path.join(fp_patches, 'patches.pkl'))
        
        #masknames = for x in list(df_patches.maskname)
        
        NumTasks = len(df_patches.maskname[0])
        
        imagenames = sorted(df_patches.imagename.unique())
        masknames=[]
        for i in range(NumTasks):
            masknames.append(sorted(list(set([x[i] for x in list(df_patches.maskname)]))))
            
        #filenames_label = sorted(list(set([x[0] for x in list(df_patches.maskname)])))
        #filenames_lesion = sorted(list(set([x[1] for x in list(df_patches.maskname)])))
        #filenames_label = df_patches.maskname.unique()
        #filenames_lesion = df_patches.filename_lesion.unique()
        dataset = opts.CLDataset()
        NumChannelsIn = dataset.NumChannelsIn
        pad = int((NumChannelsIn-1)/2)
        
        dataaccess = DataAccess()
        hdf5filepath = folderDict['fip_hdf5_all']
        
        pbar = tqdm(total=len(imagenames))
        pbar.set_description("Loading image " )
        for i in range(len(imagenames)):
            pbar.update(1)
            fip_image = os.path.join(folderDict['fp_images'], imagenames[i])
            image_arr = CTImage(fip_image).image()
            image_arr_pad = np.zeros((image_arr.shape[0]+2*pad, image_arr.shape[1], image_arr.shape[2]))
            image_arr_pad[pad:-pad] = image_arr
            mask_arr=[]
            for j in range(NumTasks):
                fip_mask = os.path.join(folderDict['fp_references_org'], masknames[j][i])
                mask_arr.append(CTRef(fip_mask).ref())
            df_image = df_patches[df_patches.imagename==imagenames[i]]
            df_image.reset_index(inplace=True)
            Ximage = np.zeros((len(df_image),NumChannelsIn,512,512))
            Xmask = [np.zeros((len(df_image),1,512,512)) for k in range(NumTasks)]
            XID = np.zeros((len(df_image),1))
            for index, row in df_image.iterrows():
                #Ximage[index]=image_arr[index-pad:index+pad+1,:]
                Ximage[index]=image_arr_pad[index:index+2*pad+1,:]
                for j in range(NumTasks):
                    Xmask[j][index]=mask_arr[j][index,:]
                XID[index]=row.ID
            datadict = defaultdict(None,{'XID': [XID], 'Ximage': [Ximage], 'Xmask': Xmask})
            dataaccess.save(hdf5filepath, datadict)
        pbar.close()
            


    def load_hdf5_all(self, folderDict, data):
        # self = dataset
        print('in123')
        
        # Create DataAccess
        dataaccess = DataAccess()
        hdf5filepath = folderDict['fip_hdf5_all']

        # Load ID list
        datadict = dict(None,{'XID': []})
        XID=list(dataaccess.read(hdf5filepath, datadict=datadict)['ID'][0][:,0])
        
        # Get index
        idx = [XID.index(s.ID) for s in data]
        
        # Load image by index
        datadict = defaultdict(None,{'XID': [], 'Ximage': [], 'Xmask': []})
        datadict = dataaccess.readIdx(hdf5filepath, datadict=datadict, idx=idx)
        
        pbar = tqdm(total=len(data))
        pbar.set_description("Loading image")
        for i,s in enumerate(data):
            #if s.ID==1163:
            #    sys.exit()
            pbar.update(1)
            s.image = torch.FloatTensor(datadict['Ximage'][0][i:i+1,:])
            s.mask=[]
            for j in range(self.NumTasks):
                s.mask.append(torch.FloatTensor(datadict['Xmask'][j][i:i+1,:]))
            #Xmask = torch.FloatTensor(datadict['Xmask'][0][i:i+1,:])
            #Xlesion = torch.FloatTensor(datadict['Xlesion'][0][i:i+1,:])
            #s.mask = [Xlabel, Xlesion]
        pbar.close()
        del datadict
        
    def load_dataset_hdf5(self, folderDict, data, debug=True):
        # self = dataset
        #print('in123')
        
        # Create DataAccess
        dataaccess = DataAccess()
        hdf5filepath = folderDict['fip_hdf5_all']

        # Load ID list
        #datadict = dict({'XID': []})
        #print('hdf5filepath123', hdf5filepath)
        #XID=list(dataaccess.read_dict(hdf5filepath, keys_select=['ID'])['ID'])
        #print('XID', XID)
        
        # Get index
        #idx = [XID.index(s.ID) for s in data]
        IDs = [s.ID for s in data]
        
        # Load image by index
        #datadict = defaultdict(None,{'XID': [], 'XImage': [], 'XMask': []})
        #datadict = dataaccess.readIdx(hdf5filepath, datadict=datadict, idx=idx)
        datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, debug=debug)
        #print('datadictS', datadict.keys())
        #print('datadictSX', datadict['X'].keys())
        #print('datadictSY', datadict['Y'].keys())
        
        #print('F1234', datadict['F'].keys())
        #sys.exit()
        
        pbar = tqdm(total=len(data))
        pbar.set_description("Loading image")
        for i,s in enumerate(data):
            #if s.ID==1163:
            #    sys.exit()
            pbar.update(1)
            s.X = {'XImage': torch.from_numpy(datadict['X']['XImage'][i:i+1,:])}
            #s.mask=[]
            #for task in self.Tasknames:
                #s.mask.append(torch.FloatTensor(datadict[task][i:i+1,:]))
            s.Y = {k:torch.from_numpy(datadict['Y'][k][i:i+1]) for k in datadict['Y']}
            s.features = {k: datadict['F'][k][i:i+1] for k in datadict['F']}
            #print('s.features', s.features)
                #s.mask.append(torch.FloatTensor(datadict['XRegion'][i:i+1,:]))
                #s.mask.append(torch.FloatTensor(datadict['XMain'][i:i+1,:]))
                #s.mask.append(torch.FloatTensor(datadict['XSegment'][i:i+1,:]))
            #Xmask = torch.FloatTensor(datadict['Xmask'][0][i:i+1,:])
            #Xlesion = torch.FloatTensor(datadict['Xlesion'][0][i:i+1,:])
            #s.mask = [Xlabel, Xlesion]
        pbar.close()
        
        del datadict