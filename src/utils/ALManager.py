#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:33:30 2023

@author: bernifoellmer
"""

import os, sys
import shutil
import numpy as np
from glob import glob
from distutils.dir_util import copy_tree
import pandas as pd
from utils.ALSample import ALSample, ALSampleMulti
from utils.DataAccess import DataAccess
from utils.helper import splitFilePath, splitFolderPath
import random
import time

class SALDataset:
    """
    SALDataset
    """
    
    name = None
    data = []
    loadDataFlag = False

    def __init__(self, name):
        self.name = name
        self.fip_hdf5 = None
        
    def save(self, fp_manager, save_dict, save_class=ALSample, hdf5=False):
        # self=manager.datasets['train']
        fp_dataset = os.path.join(fp_manager, self.name)
        os.makedirs(fp_dataset, exist_ok=True)
        fp_data = os.path.join(fp_dataset, 'data')
        self.fip_hdf5 = os.path.join(fp_data, 'sl.hdf5')
        if os.path.exists(fp_data):
            shutil.rmtree(fp_data)
        os.makedirs(fp_data, exist_ok=True)
        save_class.save(self.data, fp_data, save_dict, dataset_name=self.name, hdf5=hdf5)

        
    def load(self, fp_manager, load_dict, load_class=ALSample, hdf5=False):
        fp_dataset = os.path.join(fp_manager, self.name)
        fp_data = os.path.join(fp_dataset, 'data')
        if os.path.exists(fp_data):
            self.data = load_class.load(fp_data, load_dict=load_dict, load_class=load_class, dataset_name=self.name, hdf5=hdf5)
            #self.data = Sample.load(fp_data, load_image=load_image, load_label=load_label, load_mask=load_mask, load_prediction=load_prediction, randomize=randomize, NumSamples=NumSamples, load_fisher=load_fisher)
    
    def delete(self, data):
        ID_list = [int(s.ID) for s in data]
        data_new=[]
        for s in self.data:
            if int(s.ID) not in ID_list:
                data_new.append(s)
        self.data = data_new

    def deleteImage(self):
        for s in self.data:
            s.deleteImage()
            
class ALManager:
    """
    ALManager
    """
    
    folderDict = dict()
    split = dict()
    datasetnames = ['train', 'valid', 'test', 'query', 'labeled', 'unlabeled']
    datasets = dict({'train': SALDataset('train'), 
                     'valid': SALDataset('valid'), 
                     'test': SALDataset('test'), 
                     'query': SALDataset('query'), 
                     'labeled': SALDataset('labeled'), 
                     'unlabeled': SALDataset('unlabeled'),
                     'false': SALDataset('false'),
                     'action': SALDataset('action')})
    
    def __init__(self, fp_dataset):
        self.fp_dataset = fp_dataset
        self.datasets = dict({'train': SALDataset('train'), 
                     'valid': SALDataset('valid'), 
                     'test': SALDataset('test'), 
                     'query': SALDataset('query')})

    def info(self):
        print('Dataset info: ' + self.folderDict['name'])
        for key in self.datasets:
            print('Number ' + key + ': ' + str(len(self.datasets[key].data)))
          
    def updateDatasetPath(self, fp_manager):
        for key in self.datasets:
            fp_data = os.path.join(os.path.join(fp_manager, self.datasets[key].name), 'data')
            self.datasets[key].fip_hdf5 = os.path.join(fp_data, 'sl.hdf5')
            
        
    def createALFolderpath(self, fp_active=None, fip_split=None, fp_images=None, fp_references_org=None, method=None, NewVersion=False, VersionUse=None):
        
        # self = manager

        # Create data folders
        folderDict=dict()
        # Create fp_active
        os.makedirs(fp_active, exist_ok=True)
        ALFolderpathMethod = os.path.join(fp_active, method)
        
        # Copy initial folder of method
        print('ALFolderpathMethod123', ALFolderpathMethod)
        #print('method123', method)
        if not os.path.isdir(ALFolderpathMethod) and not method=='INIT':
            fp_init01 = os.path.join(fp_active, 'INIT', 'INIT_V01')
            os.makedirs(ALFolderpathMethod, exist_ok=True)
            fp_method01 = os.path.join(fp_active, method, method+'_V01')
            copy_tree(fp_init01, fp_method01)
            folderDict['copy_init'] = True
            #print('fp_init01', fp_init01)
            #print('fp_method01', fp_method01)
            time.sleep(1)
        else:
            folderDict['copy_init'] = False
        
        #print('copy_init12345', folderDict['copy_init'])
        #sys.exit()
            
            
        # Read version folder
        folders = sorted(glob(ALFolderpathMethod + '/*'))
        if VersionUse is not None:
            version = VersionUse
            version_prev = version-1
            ALFolderpathMethodVersion = os.path.join(ALFolderpathMethod, method + '_V' + str(version).zfill(2))
            ALFolderpathMethodVersion_prev = os.path.join(ALFolderpathMethod, method + '_V' + str(version_prev).zfill(2))
            if not os.path.isdir(ALFolderpathMethodVersion):
                return None
        elif len(folders)==0:
            version = 1     # First version
            ALFolderpathMethodVersion_prev = ''
        else:
            if NewVersion:
                version_prev = int(folders[-1][-2:])
                version = int(folders[-1][-2:]) + 1
            else:
                version = int(folders[-1][-2:])   
                version_prev = version-1
            ALFolderpathMethodVersion_prev = os.path.join(ALFolderpathMethod, method + '_V' + str(version_prev).zfill(2))
        
        # Create ALFolderpathMethodVersion
        ALFolderpathMethodVersion = os.path.join(ALFolderpathMethod, method + '_V' + str(version).zfill(2))

        # Create folderDict
        folderDict['version'] = version
        folderDict['name'] = method + '_V' + str(version).zfill(2)
        folderDict['name_prev'] = method + '_V' + str(version-1).zfill(2)
        folderDict['fp_images'] = fp_images
        folderDict['fp_dataset'] = os.path.join(ALFolderpathMethodVersion, 'data')
        folderDict['fip_hdf5_all'] = os.path.join(fp_active, 'hdf5_all.hdf5')
        folderDict['fp_action'] = os.path.join(ALFolderpathMethodVersion, 'action')
        folderDict['fp_references_org'] = fp_references_org
        folderDict['fp_references'] = os.path.join(ALFolderpathMethodVersion, 'references')
        folderDict['fp_results'] = os.path.join(ALFolderpathMethodVersion, 'results')
        folderDict['fp_predict'] = os.path.join(ALFolderpathMethodVersion, 'predict')
        folderDict['fp_log'] = os.path.join(ALFolderpathMethodVersion, 'log')
        folderDict['modelpath'] = os.path.join(ALFolderpathMethodVersion, 'model')
        folderDict['modelpath_prev'] = os.path.join(ALFolderpathMethodVersion_prev, 'model')
        folderDict['fp_manager'] = os.path.join(ALFolderpathMethodVersion, 'manager')
        folderDict['fp_manager_prev'] = os.path.join(ALFolderpathMethodVersion_prev, 'manager')
        
        fip_action_previous = []
        for v in range(version,1,-1):
            fp_al = os.path.join(ALFolderpathMethod, method + '_V' + str(v).zfill(2))
            fip = os.path.join(fp_al, 'manager', 'action', 'data', 'sl.hdf5')
            fip_action_previous.append(fip)
        folderDict['fip_action_previous'] = fip_action_previous
        self.folderDict = folderDict
        
        # Create folders
        os.makedirs(folderDict['fp_dataset'], exist_ok=True)
        os.makedirs(folderDict['fp_references'], exist_ok=True)
        os.makedirs(folderDict['fp_results'], exist_ok=True)
        os.makedirs(folderDict['fp_log'], exist_ok=True)
        os.makedirs(folderDict['fp_predict'], exist_ok=True)
        os.makedirs(folderDict['modelpath'], exist_ok=True)
        os.makedirs(folderDict['fp_manager'], exist_ok=True)
        
        # Copy references
        if NewVersion:
            if version==1:
                pass
            # Copy manager data
            if version>1:
                copy_tree(folderDict['fp_manager_prev'], folderDict['fp_manager'])
                
        self.updateDatasetPath(fp_manager=folderDict['fp_manager'])


        return folderDict

   
    def init_datasets(self, opts, exclude=['unlabeled'], TrainRatio=0.85, ValidRatio=0.15, NumPreSelect=None):
        
        # self = manager
        
        # Split dataset in train, test, valid
        #df_patches = pd.read_pickle(os.path.join(self.fp_dataset, 'patches/patches.pkl'))
        fp_patches = os.path.join(opts.fp_active, 'data', 'patches')
        df_patches = pd.read_pickle(os.path.join(fp_patches, 'patches.pkl'))
        

        imagenames = df_patches.imagename.unique()
        if NumPreSelect is not None:
            imagenames = imagenames[0:NumPreSelect]
            
        np.random.shuffle(imagenames)
        NumTrain = int(np.round(len(imagenames)*TrainRatio))
        NumValid = int(np.round(len(imagenames)*ValidRatio))
        NumTest = len(imagenames)-NumTrain-NumValid
        imagenames_train = imagenames[0:NumTrain]
        imagenames_valid = imagenames[NumTrain:NumTrain+NumValid]
        imagenames_test = imagenames[NumTrain+NumValid:]

        split=dict()
        split['train']=[]
        split['query']=list(imagenames_train)
        split['valid']=list(imagenames_valid)
        split['test']=list(imagenames_test)
        split['false']=[]
        split['action']=[]
        split['center']=[]
        self.split=split
            
        # Init dataset
        for key, value in self.datasets.items():
            if key not in exclude:
                imagenames = self.split[key]
                datasetList=[]
                for i,imagename in enumerate(imagenames):
                    df_image = df_patches[df_patches.imagename==imagename]
                    for index, row in df_image.iterrows():
                        #ID_str = row.ID
                        s = ALSample()
                        s.ID = row.ID
                        s.imagename = imagename
                        #s.maskname = row.maskname
                        s.slice = row.slice
                        s.fip_image = os.path.join(self.folderDict['fp_images'], row.imagename + '.mhd')
                        s.fip_mask = os.path.join(self.folderDict['fp_references'], row.maskname + '.mhd')
                        s.fip_pred = os.path.join(self.folderDict['fp_predict'], row.imagename + '.mhd')
                        datasetList.append(s)
                self.datasets[key].data = datasetList
            
    # def init_datasets(self, opts, exclude=['unlabeled'], TrainRatio=0.85, ValidRatio=0.15, NumPreSelect=None):
        
    #     # self = manager
        
    #     # Split dataset in train, test, valid
    #     #df_patches = pd.read_pickle(os.path.join(self.fp_dataset, 'patches/patches.pkl'))
    #     fp_patches = os.path.join(opts.fp_active, 'data', 'patches')
    #     df_patches = pd.read_pickle(os.path.join(fp_patches, 'patches.pkl'))
        

    #     imagenames = df_patches.imagename.unique()
    #     if NumPreSelect is not None:
    #         imagenames = imagenames[0:NumPreSelect]
            
    #     np.random.shuffle(imagenames)
    #     NumTrain = int(np.round(len(imagenames)*TrainRatio))
    #     NumValid = int(np.round(len(imagenames)*ValidRatio))
    #     NumTest = len(imagenames)-NumTrain-NumValid
    #     imagenames_train = imagenames[0:NumTrain]
    #     imagenames_valid = imagenames[NumTrain:NumTrain+NumValid]
    #     imagenames_test = imagenames[NumTrain+NumValid:]

    #     split=dict()
    #     split['train']=[]
    #     split['query']=list(imagenames_train)
    #     split['valid']=list(imagenames_valid)
    #     split['test']=list(imagenames_test)
    #     split['false']=[]
    #     split['action']=[]
    #     split['center']=[]
    #     self.split=split
            
    #     # Init dataset
    #     for key, value in self.datasets.items():
    #         if key not in exclude:
    #             imagenames = self.split[key]
    #             datasetList=[]
    #             for i,imagename in enumerate(imagenames):
    #                 df_image = df_patches[df_patches.imagename==imagename]
    #                 for index, row in df_image.iterrows():
    #                     #ID_str = row.ID
    #                     s = ALSample()
    #                     s.ID = row.ID
    #                     s.imagename = imagename
    #                     #s.maskname = row.maskname
    #                     s.slice = row.slice
    #                     s.fip_image = os.path.join(self.folderDict['fp_images'], row.imagename + '.mhd')
    #                     s.fip_mask = os.path.join(self.folderDict['fp_references'], row.maskname + '.mhd')
    #                     s.fip_pred = os.path.join(self.folderDict['fp_predict'], row.imagename + '.mhd')
    #                     datasetList.append(s)
    #             self.datasets[key].data = datasetList
        
    def save(self, fp_manager=None, exclude=[], include=[], save_dict={}, save_class=ALSample, hdf5=False):
        # self = manager
        if exclude and include:
            raise ValueError('Parameter exclude and include can not be defined together')
        if fp_manager is None:
            fp_manager = self.folderDict['fp_manager']
        for key, value in self.datasets.items():
            if key not in exclude and (key in include or not include):
                os.makedirs(fp_manager, exist_ok=True)
                value.save(fp_manager, save_dict, save_class=save_class, hdf5=hdf5)
                
    def load(self, fp_manager=None, exclude=['unlabeled'], include=[], load_dict={}, load_class=ALSample, hdf5=False):
        # self=manager
        if fp_manager is None:
            fp_manager = self.folderDict['fp_manager']
        for key, value in self.datasets.items():
            if key not in exclude and (key in include or not include):
                value.load(fp_manager, load_dict=load_dict, load_class=load_class, hdf5=hdf5)

    def getRandom(self, dataset='train', NumSamples=10, remove=True, clone=False):
        data = self.datasets[dataset].data
        idx = [x for x in range(len(data))]
        random.shuffle(idx)
        idx = idx[0:NumSamples]
        samples=[]
        data_new=[]
        for i in range(len(data)):
            if i in idx:
                if clone:
                    samples.append(data[i].clone(data[i]))
                else:
                    samples.append(data[i])
            else:
                data_new.append(data[i])
        if remove:
            self.datasets[dataset].data = data_new
        return samples
    
    # def train(self,name_training='training'):
    #     self._train()

    # def train(self,name_training='training'):
    #     # self=manager
        
    #     # Load data from hdf5
    #     hdf5filepath = os.path.join(self.folderDict['fp_dataset'], 'dataset_hdf5', 'dataset.hdf5')
    #     dataaccess = DataAccess()
    #     Xlabel, Ylabel = dataaccess.read_labels(hdf5filepath)
        
    #     settingsfilepath = os.path.join(self.folderDict['fp_dataset'], 'ALLiverDataset.yml')
    #     dataframe = ALLiverDataframe(settingsfilepath, overwrite=False, folderpathData=None, dataname='')
    #     dataframe.props['batch_size'] = 8
    #     dataframe.props['sampling'] = 'oversampling'
    #     dataframe.Xlabel = Xlabel
    #     dataframe.Ylabel = Ylabel
    
    #     # Create model
    #     settingsfilepath_model = os.path.join(CConfig['srcpath'], 'modules', 'ALLiver', 'ALLiverModel.yml')
    #     net = ALLiverModel(settingsfilepath_model, overwrite=True)
    #     params={'epochs': 500, 'lr': 0.001, 'gamma': 0.97, 'step_size': 5, 'batch_size': dataframe.props['batch_size'], 
    #             'hdf5filepath': hdf5filepath, 'device': 'cuda'}
    #     net.create_unet_multi_task(params=params)
    #     settingsfilepath_tf = '/mnt/SSD2/cloud_data/Projects/DL/Code/src/visualizer/TensorboardViewerTorch.yml'
    #     #log_dir_tf='/mnt/SSD2/cloud_data/Projects/DL/Code/src/modules/ALLiver/logs'
    #     log_dir_tf=self.folderDict['fp_log']
        
    #     net.initLog(settingsfilepath=settingsfilepath_tf, log_dir_tf=log_dir_tf, name_training = name_training, mode=SaveState.LOAD)
        
    #     # Load pretrained model
    #     net.props['NumSamplesTrainLoad'] = 1000
    #     net.props['NumSamplesValidLoad'] = 2000
    #     net.props['loadPretrained'] = False
    #     net.props['savePretrained'] = True
    #     #net.props['loadPretrainedFolderPath'] = '/mnt/SSD2/cloud_data/Projects/DL/Code/src/modules/ALLiver/pretrained/'
    #     #net.props['savePretrainedFolderPath'] = '/mnt/SSD2/cloud_data/Projects/DL/Code/src/modules/ALLiver/pretrained/'
    #     net.props['loadPretrainedFolderPath'] = os.path.join(self.folderDict['modelpath'])
    #     net.props['savePretrainedFolderPath'] = os.path.join(self.folderDict['modelpath'])
    #     #net.props['resultsFolder'] = '/mnt/SSD2/cloud_data/Projects/DL/Code/src/modules/ALLiver/results'
    #     net.props['resultsFolder'] = self.folderDict['fp_results']
    #     self=net
        
    #     # Train model
    #     net.train_unet_multi_task(dataframe, name_training=name_training, saveData=SaveState.SAVE, params=params)


class ALManagerMulti(ALManager):
    """
    ALManager
    """
    
    def __init__(self, fp_dataset):
        ALManager.__init__(self, fp_dataset)


    def init_datasets(self, opts, exclude=['unlabeled'], TrainRatio=0.85, ValidRatio=0.15, NumPreSelect=None):
        
        # self = manager

        hdf5filepath = os.path.join(opts.fp_active, 'hdf5_all_X.hdf5')
        
        # Create DataAccess
        dataaccess = DataAccess()
        data=dataaccess.read_dict(hdf5filepath, keys_select=['ID', 'imagename', 'maskname','slice', 'train', 'valid', 'test'])
        
        # # Extract image names
        # imagenames_hdf5 = list(data['F']['imagename'])
        # imagenames_unique = list(np.unique(imagenames_hdf5))
        
        # # Split images into train, valid, test
        # np.random.shuffle(imagenames_unique)
        # NumTrain = int(np.round(len(imagenames_unique)*TrainRatio))
        # NumValid = int(np.round(len(imagenames_unique)*ValidRatio))
        # NumTest = len(imagenames_unique)-NumTrain-NumValid
        # imagenames_train = imagenames_unique[0:NumTrain]
        # imagenames_valid = imagenames_unique[NumTrain:NumTrain+NumValid]
        # imagenames_test = imagenames_unique[NumTrain+NumValid:]

        # split=dict()
        # split['train']=[]
        # split['query']=list(imagenames_train)
        # split['valid']=list(imagenames_valid)
        # split['test']=list(imagenames_test)
        # split['false']=[]
        # split['action']=[]
        # split['center']=[]
        # self.split=split

        # # Init dataset
        # for key, value in self.datasets.items():
        #     if key not in exclude:
        #         imagenames = self.split[key]
        #         datasetList=[]
        #         for i,im in enumerate(imagenames_hdf5):
        #             if im in imagenames:
        #                 s = ALSampleMulti()
        #                 s.name = str(data['ID'][i])
        #                 s.ID = int(data['ID'][i])
        #                 s.imagename = data['F']['imagename'][i]
        #                 s.maskname = data['F']['maskname'][i]
        #                 s.F = dict({k:data['F'][k][i] for k in data['F']})
        #                 datasetList.append(s)
        #         self.datasets[key].data = datasetList 
            
 