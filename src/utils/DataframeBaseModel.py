# -*- coding: utf-8 -*-

import os, sys
import h5py
#from config.config import CConfig
#datapath = CConfig['datapath']
from utils.YAML import YAML, YAML_MODE
from utils.DataAccess import DataAccess
import pandas as pd
import numpy as np
#from helper.YAML import YAML, YAML_MODE
from sklearn.model_selection import train_test_split
from collections import defaultdict
from scipy import stats
from utils.helper import splitFolderPath, splitFilePath
import math
import scipy.ndimage
import random
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import time
import cv2
import imutils
import psutil
#from abc import ABC, abstractmethod

class DataframeBaseModel:
    """
    DataframeBaseModel model
    """
    
    props = dict()
    settingsfilepath = ''
    #props = dict(folderpathData = datapath + '/Fabric3DCNN/tmp')
    yml = YAML()
    #patches_train = None
    #patches_valid = None
    #patches_test = None
    #folderpathData = ''
    
    def __init__(self, props=None, settingsfilepath='', mode=YAML_MODE.DEFAULT, device=None):
        self.props = props
        self.settingsfilepath = settingsfilepath
        self.yml = YAML()
        self.model = None
        #self.ImReader = ImageReader()
        #self.dataname='exp'
        self.device = device
        self.generator = {}
        self.augmenter_train = DataAugmenter()
        self.augmenter_valid = DataAugmenter()
        self.augmenter_test = DataAugmenter()
        self.augmenter_predict = DataAugmenter()
        
        # Save yml file
        if mode==YAML_MODE.SAVE:
            self.yml.save(self.props, settingsfilepath)
            
        # Update yml file
        if mode==YAML_MODE.UPDATE:
            self.props = self.yml.update(self.props, settingsfilepath)
            
        # Load yml file
        if mode==YAML_MODE.LOAD:
            self.props = self.yml.load(settingsfilepath)
            
    def label_update(self, data):
        # Update Xlabel
        Xlabel = dict()
        for i,x in enumerate(data['Xlabel']):
            s = str(x)
            Xlabel[s[2:-1]] = i

        # Update Ylabel
        Ylabel = dict()
        for i,x in enumerate(data['Ylabel']):
            Ylabel[str(x)[2:-1]] = i
        return Xlabel, Ylabel

    def saveData(self, hdf5filepath, datadict):
        """ Save datadict in hdf5 file
            
        :param hdf5filepath:  Filepath to hdf5 file
        :type hdf5filepath: str
        :param datadict:  Dictionary containing lists, where each list in the dictionary contains a numpy array. Straing array have the data type 'np.string_' e.g. np.array(list(x), dtype = np.string_))
        :type datadict: dict
        """
        
        def convertH5PYTypeSave(x):
            if type(x) == np.ndarray:
                if x.dtype.kind == 'U':
                    x = list(x)
                    x = [np.string_(i) for i in x]
                    #x = np.string_(x)
                return x
            else:
                return x
    
        #  = {'Xtrain': Xtrain, 'Xvalid': Xvalid, 'Xtest': Xtest, 'Ytrain': Ytrain, 'Yvalid': Yvalid, 'Ytest': Ytest, 'Xlabel': Xlabel, 'Ylabel': Ylabel}
        
        # Create folder for hdf5 file
        folderpath_hdf5, _, _ = splitFilePath(hdf5filepath)
        if not os.path.isdir(folderpath_hdf5):
            os.mkdir(folderpath_hdf5)
        
        # Open hdf5 file from writing
        #self.fileHDF5 = h5py.File(hdf5filepath, 'w')
        self.fileHDF5 = h5py.File(hdf5filepath, 'a')
        
        # Iterate over dictionary entries (list)
        for key in datadict:
            if not key in self.fileHDF5.keys():
                self.fileHDF5[key] = len(datadict[key])     # Save number of data arrays in the list of the key
            # Iterate over data in lists
            for i,c in enumerate(datadict[key]):
                key_c = key + '_' + str(i)
                d = convertH5PYTypeSave(datadict[key][i])
                print('datadict[key][i]', datadict[key][i])
                #self.fileHDF5[key_c] = d     # Save data to hdf5 file. Key is constructed from dict key name und list position index
                #self.fileHDF5[key_c].append(d)
                
                if not key_c in self.fileHDF5.keys():
                    print('out')
                    #self.fileHDF5[key_c] = d
                    print('TYPE', type(d))
                    if type(d)==list:
                        maxshape=(None)
                    elif type(d)==str:
                        maxshape=None
                    else:
                        maxshape=tuple([None] + list(d.shape[1:]))
                    self.fileHDF5.create_dataset(key_c, data=d, maxshape=maxshape, chunks=True)
                else:
                    print('in')
                    #self.fileHDF5[key_c] = d
                    
                    if type(d)==list:
                        s=len(d)
                    elif type(d)==str:
                        s=None
                    else:
                        s=d.shape[0]
                        
                    self.fileHDF5[key_c].resize((self.fileHDF5[key_c].shape[0] + s), axis = 0)
                    
                    
        self.fileHDF5.close()
        
        
    def loadData(self, hdf5filepath, datadict, NumSamples=None, excludeKeySamples=['Xlabel', 'Ylabel'], shuffle=True):
        """ Load datadict from hdf5 file
            
        :param hdf5filepath:  Filepath to hdf5 file
        :type hdf5filepath: str
        :param datadict:  Dictionary contains None
        :type datadict: dict
        """        
        
        # Open hdf5 file for reading
        self.fileHDF5 = h5py.File(hdf5filepath, 'r')
        datalist=[]
        
#        # Create index
#        key = list(datadict.keys())[0]
#        key_c = key + '_' + str(0)
#        data_set = self.fileHDF5[key_c]
#        NumSamplesMin = min(data_set.shape[0], NumSamples)
#        idx=[i for i in range(data_set.shape[0])]
#        if shuffle:
#            random.shuffle(idx)
#        idx=sorted(idx[0:NumSamplesMin])
        #print('idx', idx)
        
        # Iterate over dictionary entries (list)
        for key in datadict:
            NumCol = np.array(self.fileHDF5.get(key))   # Extract number of data array in the liost
            datalist_i=[]
            # Iterate over data in lists
            for i in range(NumCol):
                key_c = key + '_' + str(i)
                # Read subset of data ik key is not in excludeKeySamples list and NumSamples is not None
                if NumSamples is not None and key not in excludeKeySamples:
                    data_set = self.fileHDF5[key_c]
                    
                    #print('data_set shape', data_set.shape)
                    #print('key_c', key_c)
                    
                            # Create index
                    #key = list(datadict.keys())[0]
                    #key_c = key + '_' + str(0)
                    #data_set = self.fileHDF5[key_c]
                    NumSamplesMin = min(data_set.shape[0], NumSamples)
                    idx=[i for i in range(data_set.shape[0])]
                    if shuffle:
                        random.shuffle(idx)
                    idx=sorted(idx[0:NumSamplesMin])
                    
                    data = data_set[idx]
                    #print('idx', idx)

                    #data = data_set[0:NumSamplesMin]
                else:
                    data_set = self.fileHDF5[key_c]
                    data = data_set[()]
                datalist_i.append(data)
            datalist.append(datalist_i)

        # Update Xlabel
        Xlabel = dict()
        for i,x in enumerate(datalist[-2]):
            Xlabel[str(x)] = i
        datalist[-2] = Xlabel
        
        # Update Ylabel
        Ylabel = dict()
        for i,x in enumerate(datalist[-1]):
            Ylabel[str(x)] = i
        datalist[-1] = Ylabel
            
        self.fileHDF5.close()
        return datalist
    
    def shuffle(self, patches):
        return patches.reindex(np.random.permutation(patches.index))

    def split_ID_init(self, df_in, split_dataset=[0.6, 0.2, 0.2], IDTag=None, shuffle=True):
        
        df = df_in.copy()
        # Extract ID
        IDunique = np.unique(df[IDTag])
        idx = [i for i in range(0,len(IDunique))]
        
        # Normalize split
        split = [float(i)/sum(split_dataset) for i in split_dataset]
        
        # Shuffle
        if shuffle:
            random.shuffle(idx)
            
        # Split num samples
        NumTest = round(len(idx)*split[2])
        NumValid = round(len(idx)*split[1])
        NumTrain = len(idx) - NumTest - NumValid
        
        # Split ID
        idx_train_unique = IDunique[idx[0:NumTrain]]
        idx_valid_unique = IDunique[idx[NumTrain:NumTrain+NumValid]]
        idx_test_unique = IDunique[idx[NumTrain+NumValid:NumTrain+NumValid+NumTest]]

        return idx_train_unique, idx_valid_unique, idx_test_unique

    def split_manager(self, X, Y, shuffle=True, manager=None, idx_filename=None, idx_SLICE=None):
        
        idx_train=[]
        idx_valid=[]
        idx_test=[]
        for i in range(len(Y[0])):
            filename = Y[idx_filename][i]
            SLICE = Y[idx_SLICE][i]
            if manager.checkInDataset(dataset='train', filename=filename, SLICE=SLICE):
                idx_train.append(i)
            if manager.checkInDataset(dataset='valid', filename=filename, SLICE=SLICE):
                idx_valid.append(i)
            if manager.checkInDataset(dataset='test', filename=filename, SLICE=SLICE):
                idx_test.append(i).append(i)
        idx_train = np.array(idx_train)
        idx_valid = np.array(idx_valid)
        idx_test = np.array(idx_test)
        
        # print('idx_train', idx_train)
        # print('idx_valid', idx_valid)
        # print('idx_test', idx_test)

        # ID = list(Y[idx_label])
        # idx_train = [i in idx_train_unique for i in ID]
        # idx_valid = [i in idx_valid_unique for i in ID]
        # idx_test = [i in idx_test_unique for i in ID]
        
        # Split by index
        if len(idx_train)>0:
            Xtrain = [x[idx_train] for x in X]
            Ytrain = [y[idx_train] for y in Y]
        else:
            Xtrain = [np.array(()) for x in X]
            Ytrain = [np.array(()) for y in Y]

        if len(idx_valid)>0:
            Xvalid = [x[idx_valid] for x in X]
            Yvalid = [y[idx_valid] for y in Y]
        else:
            Xvalid = [np.array(()) for x in X]
            Yvalid = [np.array(()) for y in Y]
            
        if len(idx_test)>0:
            Xtest = [x[idx_test] for x in X]
            Ytest = [y[idx_test] for y in Y]
        else:
            Xtest = [np.array(()) for x in X]
            Ytest = [np.array(()) for y in Y]  
            
        # Shuffle Xtrain 
        if shuffle:
            idx = np.arange(Xtrain[0].shape[0])
            np.random.shuffle(idx)
            Xtrain = [x[idx] for x in Xtrain]
            Ytrain = [x[idx] for x in Ytrain]
        
        # Shuffle Xvalid
        if shuffle:
            idx = np.arange(Xvalid[0].shape[0])
            np.random.shuffle(idx)
            Xvalid = [x[idx] for x in Xvalid]
            Yvalid = [x[idx] for x in Yvalid]
        
        # Shuffle Xtest
        if shuffle:
            idx = np.arange(Xtest[0].shape[0])
            np.random.shuffle(idx)
            Xtest = [x[idx] for x in Xtest]
            Ytest = [x[idx] for x in Ytest]

        return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest
    
    def split_ID(self, X, Y, idx_train_unique, idx_valid_unique, idx_test_unique, idx_label, shuffle=True):
        
        ID = list(Y[idx_label])
        idx_train = [i in idx_train_unique for i in ID]
        idx_valid = [i in idx_valid_unique for i in ID]
        idx_test = [i in idx_test_unique for i in ID]
        
        # Split by index
        if len(idx_train)>0:
            Xtrain = [x[idx_train] for x in X]
            Ytrain = [y[idx_train] for y in Y]
        else:
            Xtrain = [np.array(()) for x in X]
            Ytrain = [np.array(()) for y in Y]

        if len(idx_valid)>0:
            Xvalid = [x[idx_valid] for x in X]
            Yvalid = [y[idx_valid] for y in Y]
        else:
            Xvalid = [np.array(()) for x in X]
            Yvalid = [np.array(()) for y in Y]
            
        if len(idx_test)>0:
            Xtest = [x[idx_test] for x in X]
            Ytest = [y[idx_test] for y in Y]
        else:
            Xtest = [np.array(()) for x in X]
            Ytest = [np.array(()) for y in Y]  
            
        # Shuffle Xtrain 
        if shuffle:
            idx = np.arange(Xtrain[0].shape[0])
            np.random.shuffle(idx)
            Xtrain = [x[idx] for x in Xtrain]
            Ytrain = [x[idx] for x in Ytrain]
        
        # Shuffle Xvalid
        if shuffle:
            idx = np.arange(Xvalid[0].shape[0])
            np.random.shuffle(idx)
            Xvalid = [x[idx] for x in Xvalid]
            Yvalid = [x[idx] for x in Yvalid]
        
        # Shuffle Xtest
        if shuffle:
            idx = np.arange(Xtest[0].shape[0])
            np.random.shuffle(idx)
            Xtest = [x[idx] for x in Xtest]
            Ytest = [x[idx] for x in Ytest]

        return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest
    
    def split_df(self, patches, split_dataset=[0.6, 0.2, 0.2]):
        """ Split pandasdataframe
            
        :param patches: Pandas dataframe with data
        :type patches: pd.dataframe
        :param split_dataset:  List of split ratio [train, valid, test]
        :type split_dataset: list
        """
        
        split = [float(i)/sum(split_dataset) for i in split_dataset]
        p_train = split[0]
        #p_valid = split[1]
        p_test = split[2]
        p_train_valid = p_train/(1.0-p_test)
        
        if p_test < 1:
            X_test, X_train_valid = train_test_split(patches,train_size=p_test, shuffle=False)
        else:
            X_test = patches
            X_train_valid = None
                        
        if p_train_valid < 1 and isinstance(X_train_valid, pd.DataFrame):
            X_train, X_valid = train_test_split(X_train_valid,train_size=p_train_valid, shuffle=False)
        else:
            X_train = X_train_valid
            X_valid = None
        return X_train, X_valid, X_test

    
    def split_df(self, patches, split_dataset=[0.6, 0.2, 0.2]):
        """ Split pandasdataframe
            
        :param patches: Pandas dataframe with data
        :type patches: pd.dataframe
        :param split_dataset:  List of split ratio [train, valid, test]
        :type split_dataset: list
        """
        
        split = [float(i)/sum(split_dataset) for i in split_dataset]
        p_train = split[0]
        #p_valid = split[1]
        p_test = split[2]
        p_train_valid = p_train/(1.0-p_test)
        
        if p_test < 1:
            X_test, X_train_valid = train_test_split(patches,train_size=p_test, shuffle=False)
        else:
            X_test = patches
            X_train_valid = None
                        
        if p_train_valid < 1 and isinstance(X_train_valid, pd.DataFrame):
            X_train, X_valid = train_test_split(X_train_valid,train_size=p_train_valid, shuffle=False)
        else:
            X_train = X_train_valid
            X_valid = None
        return X_train, X_valid, X_test
    
    
    def split_rnn_np(self, X, Y, split_dataset = [0.6, 0.2, 0.2], shuffle=True):
        """ Split X and Y numpy arrays
            
        :param X: Input data
        :type X: np.array
        :param Y:  Output data
        :type Y: np.array
        :param split_dataset:  List of split ratio [train, valid, test]
        :type split_dataset: list
        :param shuffle: Shuffle dtata before splitting
        :type shuffle: boll
        """
        
        # Shuffle data
        if shuffle:
            idx = np.arange(X[0].shape[1])
            np.random.shuffle(idx)
            X = [x[:,idx] for x in X]
            Y = [x[idx] for x in Y]
        
        def split(X, split_dataset, idx_batch):
            split = [float(i)/sum(split_dataset) for i in split_dataset]
            num = X.shape[idx_batch]

            split = [float(i)/sum(split_dataset) for i in split_dataset]
            p_train = int(round(split[0] * num))
            p_test = int(round(split[2] * num))
            p_train_valid = (p_train/(num - p_test))
            
            if p_test < num:
                X=np.swapaxes(X, 0, idx_batch)
                X_test, X_train_valid = train_test_split(X, train_size=p_test, shuffle=False)
                X_test=np.swapaxes(X_test, 0, idx_batch)
                X_train_valid=np.swapaxes(X_train_valid, 0, idx_batch)
            else:
                X_test = X
                X_train_valid = None
                            
            if p_train_valid < num:
                X_train_valid=np.swapaxes(X_train_valid, 0, idx_batch)
                X_train, X_valid = train_test_split(X_train_valid, train_size=p_train_valid, shuffle=False)
                X_train=np.swapaxes(X_train, 0, idx_batch)
                X_valid=np.swapaxes(X_valid, 0, idx_batch)
                X_train_valid=np.swapaxes(X_train_valid, 0, idx_batch)
            else:
                X_train = X_train_valid
                X_valid = None
            return X_train, X_valid, X_test
        
        X = [split(x, split_dataset, idx_batch=1) for x in X]
        Y = [split(y, split_dataset, idx_batch=0) for y in Y]
        
        Xtrain = [x[0] for x in X]
        Xvalid = [x[1] for x in X]
        Xtest = [x[2] for x in X]
        Ytrain = [y[0] for y in Y]
        Yvalid = [y[1] for y in Y]
        Ytest = [y[2] for y in Y]
        
        return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest
    
    def split_np(self, X, Y, split_dataset = [0.6, 0.2, 0.2], shuffle=True):
        """ Split X and Y numpy arrays
            
        :param X: Input data
        :type X: np.array
        :param Y:  Output data
        :type Y: np.array
        :param split_dataset:  List of split ratio [train, valid, test]
        :type split_dataset: list
        :param shuffle: Shuffle dtata before splitting
        :type shuffle: boll
        """
        
        # Shuffle data
        if shuffle:
            idx = np.arange(X[0].shape[0])
            np.random.shuffle(idx)
            X = [x[idx] for x in X]
            Y = [x[idx] for x in Y]

        
        def split(X, split_dataset):
            split = [float(i)/sum(split_dataset) for i in split_dataset]
            num = X.shape[0]

            split = [float(i)/sum(split_dataset) for i in split_dataset]
            p_train = int(round(split[0] * num))
            p_test = int(round(split[2] * num))
            p_train_valid = (p_train/(num - p_test))
            
            if p_test < num:
                X_test, X_train_valid = train_test_split(X, train_size=p_test, shuffle=False)
            else:
                X_test = X
                X_train_valid = None
                            
            if p_train_valid < num:
                X_train, X_valid = train_test_split(X_train_valid, train_size=p_train_valid, shuffle=False)
            else:
                X_train = X_train_valid
                X_valid = None
            return X_train, X_valid, X_test
        
        X = [split(x, split_dataset) for x in X]
        Y = [split(y, split_dataset) for y in Y]
        
        Xtrain = [x[0] for x in X]
        Xvalid = [x[1] for x in X]
        Xtest = [x[2] for x in X]
        Ytrain = [y[0] for y in Y]
        Yvalid = [y[1] for y in Y]
        Ytest = [y[2] for y in Y]
        
        return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest

    def split_patients(self, X, Y, split, PatientID_idx=2):
        """ Split X and Y numpy arrays
            
        :param X: Input data
        :type X: np.array
        :param Y:  Output data
        :type Y: np.array
        :param split_dataset:  List of split ratio [train, valid, test]
        :type split_dataset: list
        :param shuffle: Shuffle dtata before splitting
        :type shuffle: boll
        """
        
        # X=self.X
        # Y=self.Y
            
        # idx_train = 
        idx_train=[]
        idx_valid=[]
        idx_test=[]
        patients = list(Y[PatientID_idx])
        for i,pat in enumerate(patients):
            if type(pat)==np.bytes_:
                pat = pat.decode('UTF-8')
            if pat in split['TRAIN']:
                idx_train.append(i)
            if pat in split['VALID']:
                idx_valid.append(i)
            if pat in split['TEST']:
                idx_test.append(i)

        Xtrain = [x[idx_train] for x in X]
        Ytrain = [y[idx_train] for y in Y]
        Xvalid = [x[idx_valid] for x in X]
        Yvalid = [y[idx_valid] for y in Y]
        Xtest = [x[idx_test] for x in X]
        Ytest = [y[idx_test] for y in Y]

        return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest
    

    def split_ID(self, X, Y, split, ID_idx=0):
        """ Split X and Y numpy arrays
            
        :param X: Input data
        :type X: np.array
        :param Y:  Output data
        :type Y: np.array
        :param split_dataset:  List of split ratio [train, valid, test]
        :type split_dataset: list
        :param shuffle: Shuffle dtata before splitting
        :type shuffle: boll
        """
        
        # X=self.X
        # Y=self.Y
            
        # idx_train = 
        idx_train=[]
        idx_valid=[]
        idx_test=[]
        ID_list = list(Y[ID_idx])
        for i,patid in enumerate(ID_list):
            if patid in split['TRAIN']:
                idx_train.append(i)
            if patid in split['VALID']:
                idx_valid.append(i)
            if patid in split['TEST']:
                idx_test.append(i)

        Xtrain = [x[idx_train] for x in X]
        Ytrain = [y[idx_train] for y in Y]
        Xvalid = [x[idx_valid] for x in X]
        Yvalid = [y[idx_valid] for y in Y]
        Xtest = [x[idx_test] for x in X]
        Ytest = [y[idx_test] for y in Y]

        return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest
    
    def switch_dim(self, X, dim_input='NWHC', dim_ioutput='NCWH'):
        if dim_input=='NWHC' and dim_ioutput=='NCWH':
            Xout = [np.swapaxes(x,1,3) for x in X]
            Xout = [np.swapaxes(x,2,3) for x in Xout]
            return Xout
        elif dim_input=='NCWH' and dim_ioutput=='NWHC':
            Xout = [np.swapaxes(x,1,3) for x in X]
            Xout = [np.swapaxes(x,2,3) for x in Xout]
            return Xout
        else:
            raise ValueError('Combination of inputand outpu dimensions not defined.')
    
    def to_one_hot(self, v, num_classes):
        return np.squeeze(np.eye(num_classes)[v.reshape(-1)])
    
    def from_one_hot(self, v):
        return v.argmax(1)
    
    def createGenerator(self, generatorname, X, Y, Xlabel, Ylabel, mode='train'):
        generator = getattr(self, generatorname, None)
        self.generator[mode] = generator(X, Y, Xlabel, Ylabel)
        if self.generator[mode] is None:
            raise ValueError('The generatorname: ' + generatorname + ' does not exist.')

    # def createIterativeGeneratorTrain(self, generatorname, hdf5filepath, NumSamples=100, batch_size=None, shuffle=True):

    #     mode='train'

    #     # Reset generator
    #     self.generator[mode] = None
        
    #     datadict = defaultdict(None, {'Xtrain': [], 'Ytrain': [], 'Xlabel': [], 'Ylabel': []})
    #     dataaccess = DataAccess()
    #     data = dataaccess.readRandomized(hdf5filepath, datadict=datadict, NumSamples=NumSamples, shuffle=shuffle)

    #     X = data['Xtrain']
    #     Y = data['Ytrain']
    #     XlabelArray = data['Xlabel']
    #     YlabelArray = data['Ylabel']
        
    #     self.NumSamplesTrain = X[0].shape[0]

    #     # Update Xlabel
    #     self.Xlabel = dict()
    #     for i,x in enumerate(XlabelArray):
    #         s = str(x)
    #         self.Xlabel[s[2:-1]] = i

    #     # Update Ylabel
    #     self.Ylabel = dict()
    #     for i,x in enumerate(YlabelArray):
    #         self.Ylabel[str(x)[2:-1]] = i

    #     self.generator[mode] = None
        
    #     generator = getattr(self, generatorname, None)
    #     self.generator[mode] = generator(X, Y, self.Xlabel, self.Ylabel, batch_size=batch_size)
    #     if self.generator[mode] is None:
    #         raise ValueError('The generatorname: ' + generatorname + ' does not exist.')

    # def createIterativeGeneratorValid(self, generatorname, hdf5filepath, NumSamples=100, batch_size=None, shuffle=True):
        
    #     mode='valid'
    #     # Reset generator
    #     self.generator[mode] = None
        
    #     datadict = defaultdict(None, {'Xvalid': [], 'Yvalid': [], 'Xlabel': [], 'Ylabel': []})
    #     dataaccess = DataAccess()
    #     data = dataaccess.readRandomized(hdf5filepath, datadict=datadict, NumSamples=NumSamples, shuffle=shuffle)

    #     X = data['Xvalid']
    #     Y = data['Yvalid']
    #     XlabelArray = data['Xlabel']
    #     YlabelArray = data['Ylabel']
        

    #     self.NumSamplesValid = X[0].shape[0]

    #     # Update Xlabel
    #     self.Xlabel = dict()
    #     for i,x in enumerate(XlabelArray):
    #         s = str(x)
    #         self.Xlabel[s[2:-1]] = i

    #     # Update Ylabel
    #     self.Ylabel = dict()
    #     for i,x in enumerate(YlabelArray):
    #         self.Ylabel[str(x)[2:-1]] = i

    #     generator = getattr(self, generatorname, None)
    #     self.generator[mode] = generator(X, Y, self.Xlabel, self.Ylabel, batch_size=batch_size)
    #     if self.generator[mode] is None:
    #         raise ValueError('The generatorname: ' + generatorname + ' does not exist.')
        
    # def createIterativeGeneratorTest(self, generatorname, hdf5filepath, NumSamples=100, batch_size=None, shuffle=True):
        
    #     mode='test'
        
    #     # Reset generator
    #     self.generator[mode] = None
        
    #     datadict = defaultdict(None, {'Xtest': [], 'Ytest': [], 'Xlabel': [], 'Ylabel': []})
    #     dataaccess = DataAccess()
    #     data = dataaccess.readRandomized(hdf5filepath, datadict=datadict, NumSamples=NumSamples, shuffle=shuffle)
        
    #     if len(data['Xtest'])==0:
    #         raise ValueError('Can not test model because the test dataset is empty.')
            
    #     X = data['Xtest']
    #     Y = data['Ytest']
    #     XlabelArray = data['Xlabel']
    #     YlabelArray = data['Ylabel']
        
    #     self.NumSamplesTest = X[0].shape[0]
        
    #     # Update Xlabel
    #     self.Xlabel = dict()
    #     for i,x in enumerate(XlabelArray):
    #         s = str(x)
    #         self.Xlabel[s[2:-1]] = i

    #     # Update Ylabel
    #     self.Ylabel = dict()
    #     for i,x in enumerate(YlabelArray):
    #         self.Ylabel[str(x)[2:-1]] = i

    #     generator = getattr(self, generatorname, None)
    #     self.generator[mode] = generator(X, Y, self.Xlabel, self.Ylabel, batch_size=batch_size)
    #     if self.generator[mode] is None:
    #         raise ValueError('The generatorname: ' + generatorname + ' does not exist.')

           
    def getSamples(self, NumSamples=100, mode='train'):
        X, Y = next(self.generator[mode])
        batch_size = X[0].shape[0]
        NumNext = int(np.ceil(NumSamples/batch_size)-1)
        for i in range(NumNext):
            Xt, Yt = next(self.generator[mode])
            X = [np.concatenate([x, xt]) for x, xt in zip(X, Xt)]
            Y = [np.concatenate([y, yt]) for y, yt in zip(Y, Yt)]
        X = [x[0:NumSamples] for x in X]
        Y = [y[0:NumSamples] for y in Y]
        return X, Y
    
    # def loadImages(self, images=[], mode='test', generatorname='generator_active'):
        
    #     # Create generator
    #     generator = getattr(self, generatorname, None)
    #     self.generator[mode] = generator(X, Y, self.Xlabel, self.Ylabel, batch_size=None)
    #     if self.generator[mode] is None:
    #         raise ValueError('The generatorname: ' + generatorname + ' does not exist.')

class DataAugmenter:
    """
    DataAugmenter
    """
    
    def __init__(self):
        self.order_data = []
        self.props_data = defaultdict(lambda: None)
        self.order_batch = []
        self.props_batch = defaultdict(lambda: None)
        self.idx=0
        self.done=False
        
    def selectBatch(self, X, Y, batchsize, method='random_sampling', params={'equal_change_idx': 0, 'equal_pos_idx': 1}):
        
        # if method=='sorted_index_sampling':
        #     if self.done:
        #         return None, None
        #     else:
        #         Xout = [x[self.idx:self.idx+batchsize] for x in X]
        #         Yout = [y[self.idx:self.idx+batchsize] for y in Y]
        #         self.idx=self.idx+batchsize
        #         if self.idx>=len(X[0]):
        #             self.done=True
        #         return Xout, Yout
        
        # if method=='sorted_sampling':
        #     X = [x[0:batchsize] for x in X]
        #     Y = [y[0:batchsize] for y in Y]
        #     return X, Y
        
        if method=='random_sampling':
            idx = [i for i in range(list(X.values())[0].shape[0])]
            np.random.shuffle(idx)
            Xb={}
            Yb={}
            for k in X: Xb[k]=X[k][idx[0:batchsize]]
            for k in Y: Yb[k]=Y[k][idx[0:batchsize]]
            return Xb, Yb

        if method=='replay_sampling':
            N0 = int(batchsize/2)
            N1 = batchsize-N0
            # Select samples with propability -1
            prop = params['prop']
            #print('prop123', prop)
            idx0 = np.where(prop==-1)[0]
            np.random.shuffle(idx0)
            idx0 = idx0[0:N0]
            #print('idx0123', idx0)
            # Select samples with propability >-1
            prop[prop==-1] = 0
            #prop = prop/prop.sum()
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(prop)), prop))
            idx1 = customDist.rvs(size=N1)
            #print('idx1123', idx1)
            idx = np.concatenate((idx0, idx1))
            np.random.shuffle(idx)
            Xb={}
            Yb={}
            for k in X: Xb[k]=X[k][idx]
            for k in Y: Yb[k]=Y[k][idx]
            #sys.exit()
            return Xb, Yb   
        
        # if method=='random_sampling_rnn':
        #     indicies = [i for i in range(X[0].shape[1])]
        #     np.random.shuffle(indicies)
        #     X = [x[indicies[:,0:batchsize]] for x in X]
        #     Y = [y[indicies[:,0:batchsize]] for y in Y]
        #     return X, Y
        
        # if method=='RandomOverSampler':
        #     indicies = np.expand_dims(np.array([i for i in range(X[0].shape[0])]),1)
        #     ros = RandomOverSampler(params['RandomOverSamplerRatio'], sampling_strategy='minority')
        #     idx_label = params['RandomOverSamplerIdx']
        #     indicies, _ = ros.fit_resample(indicies, Y[idx_label])
        #     indicies = indicies[:,0]
        #     np.random.shuffle(indicies)
        #     X = [x[indicies[0:batchsize]] for x in X]
        #     Y = [y[indicies[0:batchsize]] for y in Y]
        #     return X, Y

        if method=='RandomUnderSampler':
            idx = np.expand_dims(np.array([i for i in range(list(X.values())[0].shape[0])]),1)
            rus = RandomUnderSampler(sampling_strategy=0.3)
            idx_rus, _ = rus.fit_resample(idx, Y[params['target_key']])
            np.random.shuffle(idx_rus)
            Xb={}
            Yb={}
            for k in X: Xb[k]=X[k][idx_rus[0:batchsize,0]]
            for k in Y: Yb[k]=Y[k][idx_rus[0:batchsize,0]]
            return Xb, Yb
        
        # if method=='change_position_sampling':

        #     equal_change_idx = params['equal_change_idx']
        #     equal_pos_idx = params['equal_pos_idx']
            
        #     equal_change_ratio = 0.0

        #     NumEqualChange = int(round(batchsize * equal_change_ratio))
        #     NumEqualPos = batchsize - NumEqualChange
            
        #     idx_equal_change = np.argwhere(Y[equal_change_idx] == 1)
        #     idx_equal_pos = np.argwhere(Y[equal_pos_idx] == 1)
            
        #     np.random.shuffle(idx_equal_change)
        #     np.random.shuffle(idx_equal_pos)
            
        #     Xchange = [x[idx_equal_change[0:NumEqualChange,0]] for x in X]
        #     Xpos = [x[idx_equal_pos[0:NumEqualPos,0]] for x in X]
        #     Ychange = [x[idx_equal_change[0:NumEqualChange,0]] for x in Y]
        #     Ypos = [x[idx_equal_pos[0:NumEqualPos,0]] for x in Y]
            
        #     Xout = [np.concatenate([x, y]) for x, y in zip(Xchange, Xpos)]
        #     Yout = [np.concatenate([x, y]) for x, y in zip(Ychange, Ypos)]

        #     return Xout, Yout
        # else:
        #     raise ValueError('Batch selection method: ' + method + ' does not exist.')
        
    def apply_data(self, X, Y):
        # Apply data augmentation
        for augmethod in self.order_data:
            X, Y = augmethod(X, Y)
        return X, Y

    def apply_batch(self, X, Y):
        for augmethod in self.order_batch:
            X, Y = augmethod(X, Y)
        return X, Y
        
    def shuffle_aug(self):
        def shuffle(X, Y):
            indicies = [i for i in range(list(X.values())[0].shape[0])]
            np.random.shuffle(indicies)
            for k in X: X[k]=X[k][indicies]
            for k in Y: Y[k]=Y[k][indicies]
            return X, Y
        return shuffle
    
    def rotate90_aug(self, X_idx=[], Y_idx=[]):
        def rotate90(X, Y):
            def rot(X, index_rot):
                for i, idx in enumerate(index_rot):
                    X[i,:] = np.rot90(X[i,:], idx, axes=(1,2))
                return X
            batch_size = X[0].shape[0]
            index_rot = list(np.random.choice(4, batch_size))
            for idx in X_idx:
                X[idx] = rot(X[idx], index_rot)
            for idx in Y_idx:
                Y[idx] = rot(Y[idx], index_rot)
            return X, Y  
        return rotate90  

    def rotate_aug(self, X_idx=[], Y_idx=[], angle_min=-10, angle_max=10):
        def rotate(X, Y):
            def rot(X, angles):
                for i, ang in enumerate(angles):
                    for c in range(X.shape[1]):
                        X[i,c,:] = imutils.rotate(X[i,c,:], angle=ang)
                return X
            batch_size = X[0].shape[0]
            angles = [random.uniform(angle_min, angle_max) for i in range(batch_size)]
            for idx in X_idx:
                X[idx] = rot(X[idx], angles)
            for idx in Y_idx:
                Y[idx] = rot(Y[idx], angles)
            return X, Y  
        return rotate  

    def translate_aug(self, X_idx=[], Y_idx=[], X_border=[], Y_border=[], x_std=10, pixel=True):
        def translate(X, Y):
            def trans(X, x_pos, y_pos, border):
                for i, ang in enumerate(x_pos):
                    for c in range(X.shape[1]):
                        # get the width and height
                        height, width = X[i,c,:].shape[:2]
                        # create the translation matrix
                        tx = x_pos[i]
                        ty = y_pos[i]
                        translation_matrix = np.float32([
                            [1, 0, tx],
                            [0, 1, ty]
                        ])
                        # apply translation
                        final_size = (width, height)   
                        #print('X123', X.shape)
                        #print('border123', border[c])
                        #print('translation_matrix123', translation_matrix)
                        X[i,c,:,:] = cv2.warpAffine(X[i,c,:,:], translation_matrix, final_size, borderValue=border[c])
                return X
            batch_size = X[list(X.keys())[0]].shape[0]
            if not pixel:
                x_pos = [np.random.normal(0, x_std) for i in range(batch_size)]
                y_pos = [np.random.normal(0, x_std) for i in range(batch_size)]
            else:
                x_pos = [np.round(np.random.normal(0, x_std)) for i in range(batch_size)]
                y_pos = [np.round(np.random.normal(0, x_std)) for i in range(batch_size)]
            
            
            for idx, border_idx in zip(X_idx, X_border):
                X[idx] = trans(X[idx], x_pos, y_pos, border_idx)
            for idx, border_idx in zip(Y_idx, Y_border):
                Y[idx] = trans(Y[idx], x_pos, y_pos, border_idx)
            return X, Y  
        return translate  
    
    def flip_aug(self, X_idx=[], Y_idx=[]):
        def flip(X, Y):
            def flipX(X, index_flip):
                for i, idx in enumerate(index_flip):
                    for c in range(X.shape[1]):
                        if idx == 0:
                            X[i,c,:] = np.fliplr(X[i,c,:])
                        else:
                            X[i,c,:] = np.flipud(X[i,c,:])
                return X
            
            batch_size = X[0].shape[0]
            index_flip = list(np.random.choice(2, batch_size))
            
            for idx in X_idx:
                X[idx] = flipX(X[idx], index_flip)
            for idx in Y_idx:
                Y[idx] = flipX(Y[idx], index_flip)
            return X, Y
        
        
        return flip
    
    def filter_aug(self, X_idx=[], Y_idx=[0,1], filter_value=[1,2]):
        def filter_func(X, Y):
            for idx in X_idx:
                idx_select = np.isin(X[idx],np.array(filter_value))
                X = [x[idx_select] for x in X]
                Y = [x[idx_select] for x in Y]
            for idx in Y_idx:
                idx_select = np.isin(Y[idx],np.array(filter_value))
                X = [x[idx_select] for x in X]
                Y = [x[idx_select] for x in Y]
            return X, Y
            
        return filter_func

    
    def noise_aug(self, mean=0.0, std=0.05, X_idx=[], Y_idx=[], mode='normal'):
        """ Add gaussian noise to data
            
        :param mean:  Gaussian noise mean
        :type mean: float
        :param std:  Gaussian noise standart deviation
        :type std: float
        :param X_idx:  List if inices of data list X where gaussian noise is applied
        :type X_idx: list
        :param Y_idx:  List if inices of data list Y where gaussian noise is applied
        :type Y_idx: list
        :param mode:  Type of noise ('normal' - Gaussian noise)
        :type mode: str
        """    
        def noise(X,Y):
            if mode=='normal':
                # Add noise to X data
                Xout=X
                for i in X_idx:
                    Xi = X[i]
                    n = np.random.normal(mean, std, Xi.shape)
                    Xout[i] = Xi + n
                    
                # Add noise to Y data
                Yout=Y
                for i in Y_idx:
                    Yi = Y[i]
                    n = np.random.normal(mean, std, Yi.shape)
                    Yout[i] = Yi + n
                return Xout, Yout 
            else:
                raise ValueError('Noise type: ' + mode + ' does not exist.')
        return noise


    def crop_half_aug(self, X_idx=[0,1], Y_idx=[6]):
        
        def crop_half(X,Y):

            for i in X_idx:
                w = round(X[i].shape[2]/2)
                h = round(X[i].shape[3]/2)
                for j in range(X[i].shape[0]):
                    #print('j', j)
                    for c in range(X[i].shape[1]):
                        X[i][j,c,:,:] = scipy.ndimage.zoom(X[i][j,c,0:w,0:h], 2, order=1)
            for i in Y_idx:
                w = round(Y[i].shape[2]/2)
                h = round(Y[i].shape[3]/2)
                for j in range(Y[i].shape[0]):
                    #print('j', j)
                    for c in range(Y[i].shape[1]):
                        Y[i][j,c,:,:] = scipy.ndimage.zoom(Y[i][j,c,0:w,0:h], 2, order=1)
                return X, Y
        return crop_half

    def normalize_minmax_aug(self, Xmin, Xmax, X_idx=[0,1], Y_idx=[6], clip=False):
        
        def normalize_minmax(X,Y):        
            # Normalize X data
            for key in X_idx:
                Zn = (X[key] - Xmin) / (Xmax - Xmin)
                if clip:
                    Zn[Zn<0]=0.0
                    Zn[Zn>1]=1.0
                X[key] = Zn
            # Normalize Y data
            for key in Y_idx:
                Zn = (Y[key] - Xmin) / (Xmax - Xmin)
                if clip:
                    Zn[Zn<0]=0.0
                    Zn[Zn>1]=1.0
                Y[key] = Zn
            return X, Y    
        
        return normalize_minmax
    
    def normalize_scale_aug(self, norm_min=-1, norm_max=1, perc=0, X_idx=[0,1], Y_idx=[6]):
        
        def normalize_scale(X,Y):
                        
            # Normalize X data
            for i in X_idx:
                for c in range(X[i].shape[1]):
                    Xi_min = np.percentile(X[i][:,c,:,:], perc)
                    Xi_max = np.percentile(X[i][:,c,:,:], 100-perc)
                    #print('Xi_max', Xi_max)
                    #print('Xi_min', Xi_min)
                    Zn = (X[i][:,c,:,:] - Xi_min) / (Xi_max - Xi_min)
                    Z = Zn * (norm_max - norm_min) + norm_min
                    X[i][:,c,:,:] = Z
                
            # Normalize Y data
            for i in Y_idx:
                for c in range(Y[i].shape[1]):
                    Yi_min = np.percentile(Y[i][:,c,:,:], perc)
                    Yi_max = np.percentile(Y[i][:,c,:,:], 100-perc)
                    Zn = (Y[i][:,c,:,:] - Yi_min) / (Yi_max - Yi_min)
                    Z = Zn * (norm_max - norm_min) + norm_min
                    Y[i][:,c,:,:] = Z
            return X, Y    
        
        return normalize_scale
    
    def normalize_scale_inv(self, X, norm_min=0, norm_max=255, perc=3):
        eps=1e-7
        for c in range(X.shape[1]):
            Xi_min = np.percentile(X[:,c,:,:], perc)
            Xi_max = np.percentile(X[:,c,:,:], 100-perc)
            Zn = (X[:,c,:,:] - Xi_min) / (Xi_max - Xi_min + eps)
            Z = Zn * (norm_max - norm_min) + norm_min
            X[:,c,:,:] = Z
        return X

    def normalize_channel_aug(self, X_idx=[0,1], Y_idx=[6]):

        def normalize_channel(X,Y):
            
            # Normalize X
            self.Xmean = [None] * len(X_idx)
            self.Xstd = [None] * len(X_idx)
            for i, idx in enumerate(X_idx):
                Xi = np.swapaxes(X[idx],1,3)
                Z=np.zeros(Xi.shape)
                self.Xmean[i] = [None] * Xi.shape[3]
                self.Xstd[i] = [None] * Xi.shape[3]
                for c in range(Xi.shape[3]):
                    mean = np.mean(Xi[:,:,:,c])
                    std = np.std(Xi[:,:,:,c])
                    self.Xmean[i][c] = mean
                    self.Xstd[i][c] = std
                    if std == 0:
                        Z[:,:,:,c] = (Xi[:,:,:,c]-mean)
                    else:
                        Z[:,:,:,c] = (Xi[:,:,:,c]-mean)/std
                    Xi[:,:,:,c] = Z[:,:,:,c]
                Xi = np.swapaxes(Xi,1,3)
                X[idx] = Xi
            
            # Normalize X
            self.Ymean = [None] * len(Y_idx)
            self.Ystd = [None] * len(Y_idx)
            for i, idx in enumerate(Y_idx):
                Yi = np.swapaxes(Y[idx],1,3)
                Z=np.zeros(Yi.shape)
                self.Ymean[i] = [None] * Yi.shape[3]
                self.Ystd[i] = [None] * Yi.shape[3]
                for c in range(Yi.shape[3]):
                    mean = np.mean(Yi[:,:,:,c])
                    std = np.std(Yi[:,:,:,c])
                    self.Ymean[i][c] = mean
                    self.Ystd[i][c] = std
                    if std == 0:
                        Z[:,:,:,c] = (Yi[:,:,:,c]-mean)
                    else:
                        Z[:,:,:,c] = (Yi[:,:,:,c]-mean)/std
                    Yi[:,:,:,c] = Z[:,:,:,c]
                Yi = np.swapaxes(Yi,1,3)
                Y[idx] = Yi
            return X, Y
        return normalize_channel

    def normalize_channel_inv(self, X, means, stds):
        Xout = X
        for c in range(len(means)):
            Xout[:,c,:,:] = X[:,c,:,:] * stds[c] + means[c]
        return Xout
    
    def selectChannels_aug(self, channels=[], idx=[0]):
        def selectChannels(X,Y):
            for i in idx:
                Xi = X[i]
                Xic = Xi[:, channels, :, :]
                X[i] = Xic
            return X, Y
        return selectChannels
    
    def normalize_255_aug(self, X_idx=[], Y_idx=[]):
        def normalize_255(X,Y):
            Xout=[]
            for i in X_idx:
                X[i] = X[i] / 255
            for i in Y_idx:
                Y[i] = Y[i] / 255
            return X, Y
        return normalize_255
    
    def normalize_255_inv(self, X, Y, idx=[0]):
        Xout=[]
        for i in idx:
            Xi=X[i]
            Z = Xi * 255
            Xout.append(Z)
        return Xout
    
    def normalize_5perc_channel_inv(self, X, Y, idx=[0]):
        
        Xt = np.swapaxes(X[0],1,3)
        X=[]
        X.append(Xt)
        
        Xout=[]
        for i in idx:
            Xi=X[i]
            if Xi.shape[3] > 24:
                raise ValueError('Number of channels is grater than 24. Number of channels must be in the the third dimension of the data.')
            Z=np.zeros(Xi.shape)
            for c in range(Xi.shape[3]):
                mean = self.Xmean[i][c]
                std = self.Xstd[i][c]
                Z[:,:,:,c] = (Xi[:,:,:,c] * std) + mean
            Xout.append(Z)
            
        Xt = np.swapaxes(Xout[0],1,3)
        Xout=[]
        Xout.append(Xt)
        return Xout
