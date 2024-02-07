# -*- coding: utf-8 -*-

import os
import sys
from utils.config import CConfig
from utils.YAML import YAML
import h5py
import numpy as np
from datetime import datetime
from collections import defaultdict
from utils.DataframeBaseModel import DataframeBaseModel
from utils.helper import splitFolderPath, splitFilePath
from utils.TensorboardViewerTorch import TensorboardViewerTorch
from xgboost import XGBClassifier
from utils.SaveState import SaveState
import torch 
from torch import nn
from glob import glob
import json

# def isNaN(num):
#     return num != num

def isNaN(value):
    if isinstance(value, np.ndarray):
        out = bool(np.isnan(value).any())
    elif isinstance(value,  torch.Tensor):
        out = bool(torch.isnan(value).any())
    else:
        out = (value != value) or (value is None)
    return out

class DLBaseModel():
    """
    Deep learning base model
    """
    
    model = None
    file_writer = None
    props = None
    folderpathData = ''
    model = None
    epoch_bias = 0
    step_bias = 0
    tuner = False
    
    def __init__(self, settingsfilepath, overwrite=False, props=None):
        """ Initialize model
            
        :param settingsfilepath:  Filepath to settings file
        :type settingsfilepath: str
        :param overwrite:  True - overwrite settings file with default settings, False - do not overwrite settings file
        :type overwrite: bool
        :param props:  Dictionary with props
        :type props: defaultdict
        """
        
        self.settingsfilepath = settingsfilepath                            #: Filepath to settings file
        self.yml = YAML()                                                   #: YAML connection
        self.model = None
        self.props = props
        self.cddLossAcc = CDDLossAcc()
        self.cddLossAccValid = CDDLossAcc()
        self.cddLossAccTest = CDDLossAcc()
        self.time_str = ''
        
        #if not self.props['settingsfilepath_tf']:
        #    self.props['settingsfilepath_tf'] = 'H:/cloud/cloud_data/Projects/DL/Code/src/visualizer/TensorboardViewerTorch.yml'
        
        # Update time string
        self.updateTimeStr()
        
        # Load or save settings in yml file
        if os.path.isfile(settingsfilepath) and not overwrite:
            self.props = self.yml.load(settingsfilepath)
            self.props = defaultdict(lambda: None, self.props)
        else:
            # Create folder if not exist
            fp_settingsfilepath = os.path.join(splitFilePath(settingsfilepath)[0], splitFilePath(settingsfilepath)[1])
            os.makedirs(fp_settingsfilepath, exist_ok=True)
            self.yml.save(self.props, settingsfilepath)
        if props['folderpathData']:
            self.folderpathData = props['folderpathData']
            
        # Initialize tesorboard visualizer
        #log_dir_tf = 'H:/cloud/cloud_data/Projects/DL/Code/src/logs'
        #settingsfilepath_tf = 'H:/cloud/cloud_data/Projects/DL/Code/src/visualizer/TensorboardViewerTorch.yml'
        
        
    def initLog(self, settingsfilepath='H:/cloud/cloud_data/Projects/DL/Code/src/visualizer/TensorboardViewerTorch.yml', log_dir_tf='H:/cloud/cloud_data/Projects/DL/Code/src/logs', name_training = 'training01', mode=SaveState.LOAD):
        
        log_dir_tf = self.createLogDir(log_dir_tf, name_training)
        self.visualizer = TensorboardViewerTorch(settingsfilepath, mode)
        self.visualizer.start_log(log_dir_tf, comment='start_log01')
        self.log_dir_tf = log_dir_tf

     
    def checkParams(self, params, paramnames):
        """ Check model parameter
        
        Check if input params for the model contain all required parameters specified in paramnames
        and checkbs if params contains not required parameter which are not specified in paramnames.
    
        :param parameter:  Dictionary of all model parameter
        :type parameter: dict
        :param paramnames:  List of parameter names
        :type paramnames: list
        """
        for key, value in params.items():
            if key not in paramnames:
                raise ValueError('Parameter: ' + key + ' not needed.')
        for key_name in paramnames:
            foundkey=False
            for key, value in params.items():
                if key_name == key:
                    foundkey=True
            if not foundkey:
                raise ValueError('Parameter: ' + key_name + ' is missing.')
        
    def getData(self, ModelDataframe, name_training, saveData=SaveState.CREATE, params=defaultdict(lambda: None), fitFromFolder=False, loadPretrainedWeights=False, NumFilterTrain=None):
        Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = self.train_data(ModelDataframe, name_training, saveData=saveData, params=params, NumFilterTrain=NumFilterTrain)
        return Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest
    
    def createDataset(self, ModelDataframe, name_training, saveData=SaveState.CREATE, params=defaultdict(lambda: None), fitFromFolder=False, loadPretrainedWeights=False):
        Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = self.train_data(ModelDataframe, name_training, saveData=saveData, params=params)
            
    def loadPretrainedWeights(self, pretrained_weights=None):
        """
        Load pretrained tensorflow model
        
        Args:
            pretrained_weights: If true, load from Pretrained_weights path described in the setting file else no pretrained weights are loaded
        """
        if not pretrained_weights:
            if os.path.isfile(self.props['Pretrained_weights']):
                pretrained_weights = self.props['Pretrained_weights']
                print('Loading:', pretrained_weights)
                self.model.load_weights(pretrained_weights)
            else:
                raise FileExistsError('DL: Pretrained_weights file ' + self.props['Pretrained_weights'] + ' does not exist.')
    
    def loadDataTMP(self):
        hf = h5py.File(self.folderpathData, 'r')
        Xdata = np.array(hf.get('Xdata'))
        Ydata = np.array(hf.get('Ydata'))
        Xvalid = np.array(hf.get('Xvalid'))
        Yvalid = np.array(hf.get('Yvalid'))
        hf.close()
        return Xdata, Ydata, Xvalid, Yvalid
    
    def saveDataTMP(self, Xdata, Ydata, Xvalid, Yvalid):
        hf = h5py.File(self.folderpathData, 'w')
        hf.create_dataset('Xdata', data=Xdata)
        hf.create_dataset('Ydata', data=Ydata)
        hf.create_dataset('Xvalid', data=Xvalid)
        hf.create_dataset('Yvalid', data=Yvalid)
        hf.close()
        
    def updateTimeStr(self):
        now = datetime.now()
        now_str = now.strftime("%m_%d_%Y_%H_%M_%S")
        self.time_str = now_str
        
    def createLogDir(self, log_dir, name_training):
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        now_str = self.time_str
        log_dir_tf = log_dir + '/model_' + name_training + '_' + now_str
        if not os.path.isdir(log_dir_tf):
            os.mkdir(log_dir_tf)
        return log_dir_tf
    
    def to_one_hot(self, v, num_classes):
        return np.squeeze(np.eye(num_classes)[v.reshape(-1)])
    
    def from_one_hot(self, v):
        return v.argmax(1)

    def updatePropsLocalPath(self, datapath, pathnames):
        for pname in pathnames:
            self.props[pname] = datapath + self.props[pname] 
            
    def saveModel(self, modelfolderpath, modelname='CCD', epoch=0, optimizer=None, scheduler=None, params_train=None):
        # Save model
        if self.model:
            epoch_str = "{:07n}".format(epoch)
            if not os.path.exists(modelfolderpath):
                os.mkdir(modelfolderpath)     
            for key in self.model:
                modelfilename = modelname + '_' + epoch_str 
                modelpath_m = os.path.join(modelfolderpath, modelfilename + '_' + key + '.pt')
                print('Saveing model ' + modelpath_m)
                if isinstance(self.model[key], XGBClassifier):
                    self.model[key].save_model(modelpath_m)
                else:
                    torch.save(self.model[key].state_dict(), modelpath_m)
        else:
            raise ValueError('Model not initialized.')
            
        # Save learning rate scheduler
        if scheduler is not None:
            modelpath_scheduler = os.path.join(modelfolderpath, modelfilename + '_' + key + '_scheduler.pt')
            torch.save(optimizer.state_dict(), modelpath_scheduler)
        
        # Save optimizer (adam)
        if optimizer is not None:
            modelpath_opt = os.path.join(modelfolderpath, modelfilename + '_' + key + '_opt.pt')
            torch.save(optimizer.state_dict(), modelpath_opt)
        
        # Save params_train
        if params_train is not None:
            modelpath_params_train = os.path.join(modelfolderpath, modelfilename + '_' + key + '_params.json')
            with open(modelpath_params_train, 'w') as outfile:
                json.dump(params_train, outfile)

    def loadModel(self, modelfolderpath=None, epoch=None, optimizer=None, scheduler=None, params_train=None, load_model=True, name='unet'):
        
        # self=net
        
        try:
            if modelfolderpath is None:
                modelfolderpath = self.props['loadPretrainedFilePath']
    
            # Load pretrained files and extract maximum epoch
            files = glob(os.path.join(modelfolderpath, '*'+name+'.pt'))
            #print('files123', os.path.join(modelfolderpath, '*'+name+'.pt'))
            epochlist=[]
            filenameList=[]
            for f in files:
                _, filename, _ = splitFilePath(f)
                file = filename.split("_")
                filenameList.append(filename)
                epochlist.append(int(file[-2]))
                
            # Set epoch_bias
            if epoch is None:
                idx = np.argmax(epochlist)
                self.epoch_bias = epochlist[idx]
                modelname='_'.join(filenameList[idx].split("_")[0:-2])
            else:
                self.epoch_bias = 0
    
            epoch_str = "{:07n}".format(self.epoch_bias)
            # Load model
            if load_model:
                if self.model:
                    
                    # Check if pretrained folder or file exist
                    if not (os.path.isfile(modelfolderpath) or os.path.isdir(modelfolderpath)):
                        raise ValueError('Folderpath to pretrained model ' + modelfolderpath + ' does not exist.')
                                                        
                    # Load all sub networks
                    for key in self.model:
                        modelpath_m = os.path.join(modelfolderpath, modelname + '_' + epoch_str + '_' + key + '.pt')                   
                        print('modelpath_m123', modelpath_m)
                        if os.path.exists(modelpath_m):
                            print('Loading model ' + modelpath_m)
                            if isinstance(self.model[key], XGBClassifier):
                                self.model[key].load_model(modelpath_m)
                            else:
                                #self.model[key].load_state_dict(torch.load(modelpath_m))
                                self.model[key].load_state_dict(torch.load(modelpath_m), strict=False)
                        else:
                            raise ValueError('Model path ' + os.path.join(modelfolderpath, modelname) + ' does not exist.')
                else:
                    raise ValueError('Model not initialized.')
                
            # Load scheduler
            if scheduler is not None:
                key = list(self.model.keys())[0]
                modelpath_scheduler = os.path.join(modelfolderpath, modelname + '_' + epoch_str + '_' + key + '_scheduler.pt')                   
                scheduler.load_state_dict(torch.load(modelpath_scheduler))
                print('Loading scheduler: ' + modelpath_scheduler)
            
            # Load optimizer (adam)
            if optimizer is not None:
                key = list(self.model.keys())[0]
                modelpath_opt = os.path.join(modelfolderpath, modelname + '_' + epoch_str + '_' + key + '_opt.pt')                   
                optimizer.load_state_dict(torch.load(modelpath_opt))
                print('Loading optimizer: ' + modelpath_opt)
                
            # Load params_train
            if params_train is not None:
                key = list(self.model.keys())[0]
                modelpath_params_train = os.path.join(modelfolderpath, modelname + '_' + epoch_str + '_' + key + '_params.json')  
                with open(modelpath_params_train) as json_file:
                    params_train = json.load(json_file)
        except:
            raise ValueError('Modelfolderpath not found:', modelfolderpath)

        return self.model, optimizer, scheduler, params_train
            
    def printModelState(self):
        print('Printing model state')
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size()) 
            
    def count_parameters(self, model):
        for key in model.keys():
            m = model[key]
            num = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print('Number of model parameters ', key, ':', num)
            
    def predictSubset(self, modelname, X, NumSamples=100):
        num = round(X.shape[0] / NumSamples)
        pos = list(np.round(np.linspace(0, X.shape[0], num)))
        XpredList = []
        for i in range(len(pos)-1):
            Xpred = self.model[modelname](X[int(pos[i]):int(pos[i+1]),:])
            XpredList.append(Xpred.cpu().data.numpy())
        Xout = np.vstack(XpredList)
        return Xout
            

def initWeights(layer=nn.Conv2d, method='xavier', params={'mean': 0.0, 'std': 0.01}):
    """ Initialize model weights
    
    :param layer: Torch layer object
    :type layer: nn
    :param method:  Initialization method
    :type method: str
    :param params:  Parameter for weight initialization
    :type method: dict
    """
    
    def xavier_normal(m):
        if isinstance(m, layer):
            # Check parameter
            if params['mean'] is not None and params['std'] is not None:
                m.weight.data.normal_(params['mean'], params['std'])
                m.bias.data.zero_()
            else:
                raise ValueError('Parameter for weight initialization not found.')
                
    if method=='xavier':
        func = xavier_normal
    else:
        raise ValueError('Initialization method ' + method + ' does not exist.')
            
    return func

class acc():
    """
    acc - Accuracy object
    """
    def __init__(self, name='', value=None, updateValue=True, updateCounter=True, counter_init=0):
        self.__name = name
        self.__counter_init = counter_init
        self.__counter = counter_init
        self.__value = value
        self.__updateCounter = updateCounter
        self.__updateValue = updateValue
    
    def update(self, value):
        # if self.__name=='confusion_sum':
        #     print('confusion_sum1234', value)
        #     print('confusion_sumtype', type(value))
        if type(value) == list:
            # if self.__name=='confusion_sum':
            #     print('confusion___value', type(self.__value))
            if type(self.__value) == list:
                if self.__updateValue:
                    # if self.__name=='confusion_sum':
                    #     print('confusion_sum12345', value)
                    self.__value = [i+j for i,j in zip(self.__value, value)]
                else:
                    # if self.__name=='confusion_sum':
                    #     print('confusion_sum123456', value)
                    self.__value = value
                    #sys.exit()
            else:
                self.__value = value
        elif type(value)==torch.Tensor:
            if self.__updateValue:
                #if not isNaN(value.item()):
                if not isNaN(value):
                    if self.__value is None: 
                        self.__value = value.clone()
                    else:
                        self.__value += value.clone()
            else:
                if not isNaN(value.item()):
                    self.__value = value.item()
        elif type(value)==np.ndarray:
            # if self.__name=='confusion_sum':
            #     print('confusion_sum1234', value)
            if self.__updateValue:
                #if not isNaN(value).any():
                if not np.isnan(value).any():
                    if self.__value is None: 
                        self.__value = value.copy()
                    else:
                        self.__value += value.copy()
            else:
                if not isNaN(value):
                    self.__value = value
        else:
            if self.__updateValue:
                #print('value123', value)
                #print('__value123', self.__value)
                if not isNaN(value):
                    if self.__value is None: 
                        self.__value = value
                    else:
                        self.__value += value
            else:
                if not isNaN(value):
                    self.__value = value
        
        # if self.__name=='confusion':
        #     print('self.__counter145', self.__counter)
        if self.__updateCounter and not isNaN(value):
            self.__counter += 1
            # if self.__name=='confusion':
            #     print('self.__counter1456', self.__counter)
            
    def print_acc(self):
        # if self.__name=='confusion_sum':
        #     print('__counter123', self.__counter)
        # if self.__name=='confusion':
        #     print('__counter1234', self.__counter)
        if self.__counter>0:
            if type(self.__value) == list:
                value = [v/self.__counter for v in self.__value]
            else:
                value = self.__value / self.__counter
            print(self.__name + ':', value)
        else:
            print(self.__name + ':', 'No values!')
            
    def value(self):
        #if self.__value is None: return None
        #if self.__counter>0:
        if self.__counter>0 and self.__value is not None:
            if type(self.__value) == list:
                value = [v/self.__counter for v in self.__value]
            else:
                value = self.__value / self.__counter
        else:
            #value = np.NaN           
            value = None    
        return value
    
    def reset(self, value=None):
        self.__counter = self.__counter_init
        self.__value = value

        
    def updateSettings(self, updateValue=True, updateCounter=True, counter=0):
        self.__updateValue = updateValue
        self.__updateCounter = updateCounter
        self.__counter = counter
        self.__counter_init = counter
            
class loss():
    """
    loss - Losses object
    """
    def __init__(self, name='', value=0, updateValue=True, updateCounter=True, counter_init=0):
        self.__name = name
        self.__counter_init = counter_init
        self.__counter = counter_init
        self.__value = value
        self.__updateCounter = updateCounter
        self.__updateValue = updateValue
    
    def update(self, value):
        if not isNaN(value.item()):
            if self.__updateValue:
                #self.__value += value.item()
                if self.__value is None: 
                    self.__value = value.item()
                else:
                    self.__value += value.item()
            if self.__updateCounter:
                self.__counter += 1 
    
    def print_loss(self):
        if self.__counter>0:
            value = self.__value / self.__counter
            print(self.__name + ':', value)
        else:
            print(self.__name + ':', 'No values!')

    def value(self):
        #if self.__value is None: return None
        if self.__counter>0 and self.__value is not None:
            value = self.__value / self.__counter
        else:
            #value = np.NaN
            value = None 
        return value
    
    def reset(self, value=None):
        self.__counter = self.__counter_init
        self.__value = value

    def updateSettings(self, updateValue=True, updateCounter=True, counter=0):
        self.__updateValue = updateValue
        self.__updateCounter = updateCounter
        self.__counter = counter
        self.__counter_init = counter

            
class CDDLossAcc():
    """
    CDD_losses - Loss and accuracy handler
    """

    def __init__(self, losses_list=[], acc_list=[]):
        self.init(losses_list=[], acc_list=[])
        
    def init(self, losses_list=[], acc_list=[]):
        self.losses = defaultdict(lambda: None)
        for key in losses_list:
            self.losses[key] = loss(name=key)
        self.acc = defaultdict(lambda: None)
        for key in acc_list:
            self.acc[key] = acc(name=key)
            
    def sum_losses(self, losses):
        for key in losses.keys():
            if self.losses[key] is not None:
                self.losses[key].update(losses[key])
            else:
                raise ValueError('Key loss: ' + key + ' does not exist.')

    def sum_acc(self, acc):

        for key in acc.keys():
            if self.acc[key] is not None:
                self.acc[key].update(acc[key])
            else:
                raise ValueError('Key acc: ' + key + ' does not exist.')

    def print_losses(self, keys_print=None):
        print('-------------- Losses --------------')
        for key in self.losses.keys():
            if keys_print is None or key in keys_print:
                self.losses[key].print_loss()

    def print_acc(self, keys_print=None):
        print('-------------- Accuracies --------------')
        for key in self.acc.keys():
            if keys_print is None or key in keys_print:
                self.acc[key].print_acc()
            
    def get_acc(self, key):
        if self.acc[key] is not None:
            return self.acc[key].value()
        else:
            raise ValueError('Key acc: ' + key + ' does not exist.')

    def get_loss(self, key):
        if self.losses[key] is not None:
            return self.losses[key].value()
        else:
            raise ValueError('Key loss: ' + key + ' does not exist.')

    def visualize_loss(self, visualizer, keys_vis=None):
        for key in self.losses:
            if keys_vis is None or key in keys_vis:
                value = self.losses[key].value()
                if not type(value) == np.ndarray and not type(value) == list:
                    if value is None: value = np.nan
                    visualizer.add_scalar(key, value)

    def visualize_acc(self, visualizer, keys_vis=None):
        for key in self.acc:
            if keys_vis is None or key in keys_vis:
                value = self.acc[key].value()
                if not type(value) == torch.Tensor and not type(value) == np.ndarray and not type(value) == list:
                    if value is None: value = np.nan
                    visualizer.add_scalar(key, value)

    def reset_losses(self):
        for key in self.losses:
            self.losses[key].reset()

    def reset_acc(self):
        for key in self.acc:
            self.acc[key].reset()
