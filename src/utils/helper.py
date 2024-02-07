# -*- coding: utf-8 -*-
import os
import ntpath
from dirsync import sync
import numpy as np
import sys
from collections import defaultdict
from glob import glob
import SimpleITK as sitk
from SimpleITK import ConnectedComponentImageFilter
import torch

def round_num(x):
    """Rounds a number such that numbers such as X.5 are uprounded

    Parameters
    ----------
    x : float
        Number to be rounded
        
    Returns
    ----------
    int
        Rounded number
    """
    
    return int(np.floor(x + 0.5))
                        
def splitFilePath(filepath):
    """Split filepath into folderpath, filename and file extension

    Parameters
    ----------
    filepath : str
        Filepath
        
    Returns
    ----------
    tuple of str
        Returns tuple of (folderpath, filename, file_extension)
    """
    
    folderpath, _ = ntpath.split(filepath)
    head, file_extension = os.path.splitext(filepath)
    folderpath, filename = ntpath.split(head)
    return folderpath, filename, file_extension

def splitFolderPath(folderpath):
    """Split filepath into folderpath of parent folder and foldername

    Parameters
    ----------
    folderpath : str
        Folderpath
        
    Returns
    ----------
    tuple of str
        Returns tuple of (folderpath_parent, foldername)
    """
    
    folderpath_parent, foldername = ntpath.split(folderpath)
    return folderpath_parent, foldername

def replaceFileExt(filepath, ext='.txt'):
    """Rounds a number such that numbers such as X.5 are uprounded

    Parameters
    ----------
    filepath : str
        Extension which replaced original file extension of filepath
        
    Returns
    ----------
    str
        Filepath with replaced file extension
    """
    
    folderpath, filename, file_extension = splitFilePath(filepath)
    return os.path.join(folderpath,filename + ext)

def syncPC(sourcedir='//tsclient/H/cloud/cloud_data/Projects/DL/Code', targetdir='H:/cloud/cloud_data/Projects/DL/Code'):
    """Synchronize folders

    Parameters
    ----------
    sourcedir : str
        Source directory
    targetdir : str
        Target directory      
    """
    sync(sourcedir, targetdir, 'sync', verbose=True)

def syncPCFolders(sourcedir='//tsclient/H/cloud/cloud_data/Projects/DL/Code', targetdir='H:/cloud/cloud_data/Projects/DL/Code', folders=[]):
    """Synchronize subfolders in a folder 

    Parameters
    ----------
    sourcedir : str
        Source directory
    targetdir : str
        Target directory    
    folders : list of str
        List of folder names that are synchronized
    """
    for f in folders:
        sourcedirF = os.path.join(sourcedir, f)
        targetdirF = os.path.join(targetdir, f)
        sync(sourcedirF, targetdirF, 'sync', verbose=True) 

def compute_one_hot_np_3(x, dim=1):
    """Compute one hot marix by using the maximum value

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (N,C,W,H)
    dim : int
        Dimension used to compute one-hot encoded matrix
        
    Returns
    ----------
    ndarray
        One hot encoded array of shape (N,C,W,H)
    """
    shape = (x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1])
    xmax = np.argmax(x,axis=dim)
    xmaxr = xmax.reshape(-1)
    xhot = np.eye(x.shape[1])[xmaxr]
    xhot = xhot.reshape(shape)
    xhot = np.swapaxes(xhot,1,4)
    xhot = np.swapaxes(xhot,2,4)
    one_hot = np.swapaxes(xhot,3,4)
    return one_hot

def compute_one_hot_np(x, dim=1):
    """Compute one hot marix by using the maximum value

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (N,C,W,H)
    dim : int
        Dimension used to compute one-hot encoded matrix
        
    Returns
    ----------
    ndarray
        One hot encoded array of shape (N,C,W,H)
    """
    shape = (x.shape[0], x.shape[2], x.shape[3], x.shape[1])
    xmax = np.argmax(x,axis=dim)
    xmaxr = xmax.reshape(-1)
    xhot = np.eye(x.shape[1])[xmaxr]
    xhot = xhot.reshape(shape)
    xhot = np.swapaxes(xhot,1,3)
    one_hot = np.swapaxes(xhot,2,3)
    return one_hot
        
# def compute_one_hot_noclass_np(x, dim=1):
#     """Compute one hot marix by using the maximum value
#     """
#     #idx = x.sum()
#     shape = (x.shape[0], x.shape[2], x.shape[3], x.shape[1])
#     xmax = np.argmax(x,axis=1)
#     xmax_value = np.max(x,axis=1)
#     xmax_idx = (xmax_value==0).reshape(-1)
#     xmaxr = xmax.reshape(-1)
#     xhot = np.eye(x.shape[1])[xmaxr]
#     xhot[xmax_idx] = np.zeros(xhot.shape)[xmax_idx]
#     xhot = xhot.reshape(shape)
#     xhot = np.swapaxes(xhot,1,3)
#     one_hot = np.swapaxes(xhot,2,3)
#     return one_hot


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_best = False
        
    def __call__(self, val_loss, model):
        

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.save_best = False
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.save_best = True
        # '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        # self.val_loss_min = val_loss


def compute_sample_weight_torch(y, class_weight=None, eps=1e-8):
    # """Estimate sample weights by class for unbalanced datasets.
    # Similar to sklearn.utils.class_weight.compute_sample_weight.
    # Default behavior to balance classes but class_weight can be passed
    # as argument.
    # The weights will be normalized so their sum is equals to the number of
    # samples. This normalization isn't present in sklearn's version.
    # Args:
    #     y: 1D tensor
    #         Array of original class labels per sample.
    #     class_weight: 1D FloatTensor
    #         Class weights where its idx correspond to class value
    #     eps: float
    #         Value for numerical stability
    # Returns:
    #     weights: 1D FloatTensor
    #        Sample weights as applied to the original y.
    # Examples:
    # >>> y = torch.FloatTensor([1, 0, 0, 0])
    # >>> compute_sample_weight(y)
    #     tensor([ 2.0000,  0.6667,  0.6667,  0.6667])
    # >>> compute_sample_weight(y, class_weight=torch.FloatTensor([.4, .6]))
    #     tensor([ 1.3333,  0.8889,  0.8889,  0.8889])
    # """

    y = y.long()
    batch_size = y.size(0)

    if class_weight is None:
        n_classes = y.unique().size(0)
    else:
        n_classes = class_weight.size(0)

    y_onehot = torch.zeros(batch_size, n_classes, dtype=torch.float, device=y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    if class_weight is None:
        class_weight = 1 / (y_onehot.sum(dim=0) + eps)
        """
        classes available in y will have weight = 1/n_members
        while classes not present in y will have weight = 1/eps
        force weight of non available classes to 0
        """
        class_weight[class_weight > 1] = 0

    weights = torch.mm(y_onehot, class_weight.view(-1, 1)).squeeze()
    weights = batch_size * weights / (torch.sum(weights) + eps)
    return weights


def compute_class_weight_torch(y, n_classes, class_weight=None, eps=1e-8):
    # """Estimate class weights for unbalanced datasets.
    # Similar to sklearn.utils.class_weight.compute_class_weight.
    # Default behavior to balance classes but class_weight can be passed
    # as argument.
    # The weights will be normalized so their sum is one.
    # Args:
    #     y: 1D tensor
    #         Array of original class labels per sample.
    #     class_weight: 1D FloatTensor
    #         Class weights where its idx correspond to class value
    #     eps: float
    #         Value for numerical stability
    # Returns:
    #     class_weights: 1D FloatTensor
    #        vector with weight for each class.
    # Examples:
    # >>> y = torch.FloatTensor([1, 0, 0, 0, 1, 2, 1, 1])
    # >>> compute_class_weight(y)
    #     tensor([ 0.3333,  0.2500,  1.0000])
    # >>> compute_class_weight(y, class_weight=torch.FloatTensor([.4, .5, .1]))
    #     tensor([ 0.4000,  0.5000,  0.1000])
    # Computing weight in forward pass:
    # >>> for i, (X, y) in enumerate(dataloader):
    # >>>     y = y.float()
    # >>>     optimizer.zero_grad()
    # >>>     y_pred = model(X).squeeze()
    # >>>     weights = compute_class_weight(y)
    # >>>     loss = binary_cross_entropy(y_pred, y, weights)
    # >>>     loss.backward()
    # >>>     optimizer.step()
    # """

    y = y.long()
    batch_size = y.size(0)

    # if class_weight is None:
    #     n_classes = y.unique().size(0)
    # else:
    #     n_classes = class_weight.size(0)

    y_onehot = torch.zeros(batch_size, n_classes, dtype=torch.float, device=y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    if class_weight is None:
        class_weight = 1 / (y_onehot.sum(dim=0) + eps)
        """
        classes available in y will have weight = 1/n_members
        while classes not present in y will have weight = 1/eps
        force weight of non available classes to 0
        """
        class_weight[class_weight > 1] = 0

    # normalize weigths
    class_weight /= class_weight.sum()
    return class_weight


def compute_one_hot_torch(x, dim=1):
    """Compute one hot marix by using the maximum value
    """
    one_hot = torch.zeros(x.shape, dtype=torch.float32).to(x.device)
    one_hot = one_hot.scatter_(1,torch.argmax(x, dim, keepdim=True) , 1)
    return one_hot


def save_dcm_as_mhd(folderpath_src, filepath_dst):
    """ Save dicom folder as mhd image file
    
    """
    
    filepathDcm = glob(folderpath_src + '/*.dcm')[0]
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(filepathDcm)
    file_reader.ReadImageInformation()
    series_ID = file_reader.GetMetaData('0020|000e')
    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folderpath_src, series_ID)
    data = sitk.ReadImage(sorted_file_names)
    sitk.WriteImage(data, filepath_dst, True)
    
class ddict(defaultdict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__