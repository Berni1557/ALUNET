# -*- coding: utf-8 -*-
import os, sys
import torch
#from torch import nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
#from ct.Sample import Sample, defaultdict
import math
#from helper.helper import splitFilePath, splitFolderPath
#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
from utils.DataAccess import DataAccess
from utils.ct import CTImage, CTRef
#from sklearn.utils import shuffle
#from utils.helper import compute_one_hot_np
import copy
from glob import glob

class ALSample():
    
    """
    Active learning sample
    """
    
    dtype = torch.FloatTensor
    #XArrays = ['XImage', 'XMask', 'XPred', 'XWeight', 'XRegion', 'XPseudo', 'XRefine']
    #saveX=['XImage']
    #saveY=['XMask', 'XPred', 'XWeight', 'XRegion', 'XPseudo', 'XRefine']
    da = ['X', 'Y', 'P']
        
    def __init__(self):
        super().__init__()
        self.name = ''
        self.ID = None
        self.fip_image = None
        self.fip_mask = None
        self.fip_pred = None
        self.X=dict()   # [1,C,W,H]
        self.Y=dict()
        self.P=dict()
        self.refined = False
        self.info = dict()
        self.msg = ''
        self.status = ''
        self.command = ''
        self.F = dict()
        self.info = defaultdict(lambda: None, {})
        self.save_dict = defaultdict(None, {})
        self.load_dict = defaultdict(None, {})
        #self.da = ['X', 'Y', 'P']


    def getXY(self, key):
        for k in self.da:
            D = getattr(self, k, None)
            if key in D: return D[key]
        return None
        
    # def clone(s):
    #     sc = ALSample()
    #     sc.name = s.name
    #     sc.ID = s.ID
    #     sc.fip_image = s.fip_image
    #     sc.fip_mask = s.fip_mask
    #     sc.fip_pred = s.fip_pred
    #     sc.refined = s.refined
    #     sc.info = s.info.copy()
    #     sc.msg = s.msg
    #     sc.status = s.status
    #     sc.command = s.command
    #     sc.features = s.features.copy()
    #     for key in s.XArrays:
    #         if getattr(s, key, None) is not None: 
    #             setattr(sc, key, copy.deepcopy(getattr(s, key)))
    #     return sc
    
    @classmethod
    def clone(cls, s):
        sc = cls()
        sc.name = s.name
        sc.ID = s.ID
        sc.fip_image = s.fip_image
        sc.fip_mask = s.fip_mask
        sc.fip_pred = s.fip_pred
        sc.refined = s.refined
        sc.info = s.info.copy()
        sc.msg = s.msg
        sc.status = s.status
        sc.command = s.command
        sc.F = s.F.copy()
        for key in s.da:
            D0 = getattr(s, key, None)
            D1 = getattr(sc, key, None)
            for k in D0: D1[k]=copy.deepcopy(D0[k])
        return sc
    
    @staticmethod
    def getSampleByID(sl, ID):
        for s in sl:
            if s.F['ID']==ID:
                return s
        return None
        
    def deleteImage(self):
        for k in self.da:
            D = getattr(self, k, None)
            for l in D: D[l]=None
            
        # for key in self.XArrays:
        #     if key in list(self.X.keys()):
        #         self.X[key]=None
        #     if key in list(self.Y.keys()):
        #         self.Y[key]=None
        # self.image = None
        # self.mask = None
        # self.pred = None
        # self.weight = None
        # self.region = None
        # self.pseudo = None
        # self.refine = None

    def cpu(self, keys=[]):
        for k in self.da:
            D = getattr(self, k, None)
            for l in D:
                if D[l] is not None: 
                    D[l] = D[l].to('cpu')  
            
        # for key in self.X:
        #     if not keys or key in keys:
        #         if self.X[key] is not None:
        #             self.X[key] = self.X[key].to('cpu')  
        # for key in self.Y:
        #     if not keys or key in keys:
        #         if self.Y[key] is not None:
        #             self.Y[key] = self.Y[key].to('cpu')  
                
            # if getattr(self, key, None) is not None: 
            #     getattr(self, key, None).to('cpu')        
    # def cpu(self):
    #     for key in self.XArrays:
    #         if getattr(self, key, None) is not None: 
    #             getattr(self, key, None).to('cpu')
                
        # if self.image is not None: self.image = self.image.to('cpu')
        # if self.mask is not None: self.mask = self.mask.to('cpu')
        # if self.pred is not None: self.pred = self.pred.to('cpu')
        # if self.weight is not None: self.weight = self.weight.to('cpu')
        # if self.region is not None: self.region = self.region.to('cpu')
        # if self.pseudo is not None: self.pseudo = self.pseudo.to('cpu')
        # if self.refine is not None: self.refine = self.refine.to('cpu')
        
    def cuda(self, keys=[]):
        for k in self.da:
            D = getattr(self, k, None)
            for l in D:
                if D[l] is not None:
                    D[l] = D[l].to('cuda')
            
    # def cuda(self, keys=[]):
    #     for key in self.X:
    #         if not keys or key in key:
    #             self.X[key].to('cuda')  
    #     for key in self.Y:
    #         if not keys or key in key:
    #             self.Y[key].to('cuda')  
                
    # def cuda(self):
    #     for key in self.XArrays:
    #         if getattr(self, key, None) is not None: 
    #             getattr(self, key, None).to('cuda')
                
        # if self.image is not None: self.image = self.image.to('cuda')
        # if self.mask is not None: self.mask = self.mask.to('cuda')
        # if self.pred is not None: self.pred = self.pred.to('cuda')
        # if self.weight is not None: self.weight = self.weight.to('cuda')
        # if self.region is not None: self.region = self.region.to('cuda')
        # if self.pseudo is not None: self.pseudo = self.pseudo.to('cuda')
        # if self.refine is not None: self.refine = self.refine.to('cuda')
        
    def save_any(self):
        b=False
        for key in self.XArrays:
            b = b or self.save_dict[key]
        return b
        #return self.save_dict['image'] or self.save_dict['mask'] or self.save_dict['pred'] or self.save_dict['weight'] or self.save_dict['region'] or self.save_dict['pseudo'] or self.save_dict['refine']


    # def plotSample(self, plotmode='image, mask, pred, weight, region, pseudo, refine', save=False, folderpath='', name='', color=False, title=True, format_im='svg', dpi=300):
    #     self.cpu()
    #     filepath = os.path.join(folderpath, name)
    #     soft=nn.Softmax(dim=1)
        
    #     plotlist=['XImage', 'XMask', 'Xpred', 'XWeight', 'XRegion', 'XPseudo', 'XRefine']
    #     softlist=['Xpred']
    #     for pl in plotlist:
    #         if pl in plotmode:
    #             im = getattr(self, pl, None)
    #             if im is not None:
    #                 image = im.data.numpy()
    #                 if pl in softlist: image = soft(image)
    #                 image = image[0,0,:,:]
    #                 plt.imshow(image, cmap='gray')
    #                 name = self.name + '_' + str(self.ID)
    #                 if title: plt.title(name) 
    #                 if save:
    #                     plt.savefig(filepath + '_image.'+format_im, format=format_im, dpi=dpi)
    #                 else:
    #                     plt.show()

    def __plot_image(self, image, name, title, color=False, save=False, filepath=None, format_im=None, dpi=300):
        if color:
            #print('image1234', image.shape)
            im = np.zeros((image.shape[2], image.shape[3]))
            for c in range(image.shape[1]):
                im = im + (c+1) * image[0,c,:,:]
            plt.imshow(im)
            plt.imshow(im, cmap='Accent', interpolation='nearest')
            if title: plt.title(name) 
            if save: plt.savefig(filepath + name + format_im, format=format_im, dpi=dpi)
            plt.show()
        else:
            if title: plt.title(name) 
            if save: plt.savefig(filepath + name + format_im, format=format_im, dpi=dpi)
            plt.imshow(image[0,0,:,:], cmap='gray')
            plt.show()
            
    def plotSample(self, plotlist=['XImage', 'XMask', 'XPred', 'XWeight', 'XRegion', 'XPseudo', 'XRefine'], save=False, fp='', name='', color=False, title=True, format_im='svg', dpi=300):
        self.cpu()
        filepath = os.path.join(fp, name)

        for d in self.da:
            if d in plotlist:
                plotlist = plotlist + list(getattr(self, d, None).keys())

        for pl in plotlist:
            name = self.name + '_' + pl  + '_' + str(int(self.ID))
            im = self.getXY(pl)
            if im is not None:
                image = im.data.numpy()
                if pl=='XImage' and len(image.shape)==4:
                    idx = int((image.shape[1]-1)/2)
                    image = image[:,idx:idx+1]
                    self.__plot_image(image, name, title, color=False, filepath=filepath, save=save)
                else:
                    image = image[:,1:2]
                    self.__plot_image(image, name, title, color=color, filepath=filepath, save=save)
                    
                        
    @staticmethod
    def samplesToloader(data, dtype=torch.DoubleTensor, batch_size=1):
        data_leader = []
        for s in data:
            image = s.image_norm()
            data_leader.append([image[0].type(dtype), s.mask.type(dtype), s.weight.type(dtype)])
        loader= torch.utils.data.DataLoader(dataset=data_leader, batch_size=batch_size, shuffle=False, pin_memory=False)
        return loader

    def image_norm(self, Xmin=-2000, Xmax=1300, clip=False):
        image=self.X['XImage'].clone()
        image = (image - Xmin) / (Xmax - Xmin)
        if clip:
            image[image<0]=0.0
            image[image>1]=1.0
        return image
    
    @staticmethod
    def predict(sl, net):
        net.model['unet'].eval()
        net.model['unet'].float()
        with torch.no_grad():
            pbar = tqdm(total=len(sl))
            pbar.set_description("Prediction of samples")
            for s in sl:
                pbar.update(1)
                image = s.image_norm().cuda().clone()
                s.pred = net.model['unet'](image).detach().clone()
                image.detach()
                del image
            pbar.close()   


        
    
    @classmethod
    def save(cls, sl, fp_data, save_dict, dataset_name='', hdf5=False):
        if len(sl)>0:
            if hdf5:
                cls.save_hdf5(sl=sl, fp_data=fp_data, save_dict=save_dict, dataset_name=dataset_name)
            else:
                cls.save_pkl(sl=sl, fp_data=fp_data, save_dict=save_dict, dataset_name=dataset_name)


        
    @classmethod
    def save_pkl(cls, sl, fp_data, save_dict, dataset_name='', hdf5=False):
        
        # if "label_mask_fim" in save_dict:
        #     save_dict['label_mask_weak_fim'] = save_dict.pop('label_mask_fim')
        # if "label_mask_var" in save_dict:
        #     save_dict['label_mask_weak_var'] = save_dict.pop('label_mask_var')
        # if "pseudo_mask" in save_dict:
        #     save_dict['pseudo_mask_weak'] = save_dict.pop('pseudo_mask')
        # if "label_refine_fim" in save_dict:
        #     save_dict['label_refine_weak_fim'] = save_dict.pop('label_refine_fim')
            
        os.makedirs(fp_data, exist_ok=True)
        df = pd.DataFrame()
        fip_df = os.path.join(fp_data, 'sl.pkl')
        
        
        # if save_dict['concatIfExist'] and os.path.isfile(fip_df):
        #     df = pd.read_pickle(fip_df)
        #     df = df['ID'].max() + 1
        # else:
        #     pass
        #     #ID = 0
        
        #filepath_fisher = os.path.join(fp_data, 'fisher.hdf5')
        #fisherdict = defaultdict(None, {'ID': [], 'FISHER': []})
        #da = DataAccess()
        
        pbar = tqdm(total=len(sl))
        pbar.set_description("Saveing data " + dataset_name)
        for s in sl:
            pbar.update()
            s.name = "{:07n}".format(s.ID)
            s.save_dict.update(save_dict)
            s.cpu()
            folderpath = os.path.join(fp_data, str(s.name))
            if s.save_any():
                os.makedirs(folderpath, exist_ok=True)
            
            if s.save_dict['image']:
                if s.image is not None:
                    image = CTImage()
                    image.name = s.name
                    image.setImage(s.image[0,:,:,:])
                    image.save(folderpath)
                
            if s.save_dict['mask']:
                if s.mask is not None:
                    mask = CTRef()
                    mask.name = s.name + '_mask'
                    mask.setRef(s.mask[0,:,:,:])
                    mask.save(folderpath, image_format='nrrd')

            if s.save_dict['pred']:
                if s.pred is not None:
                    pred = mask()
                    pred.name = s.name + '_mask'
                    pred.setRef(s.pred[0,:,:,:])
                    pred.save(folderpath, image_format='nrrd')
                    
            if s.save_dict['weight']:
                if s.weight is not None:
                    weight = CTRef()
                    weight.name = s.name + '_mask'
                    weight.setRef(s.weight[0,:,:,:])
                    weight.save(folderpath, image_format='nrrd')
                    
            if s.save_dict['region']:
                if s.region is not None:
                    region = CTRef()
                    region.name = s.name + '_mask'
                    region.setRef(s.region[0,:,:,:])
                    region.save(folderpath, image_format='nrrd')
                    
            if s.save_dict['pseudo']:
                if s.pseudo is not None:
                    pseudo = CTRef()
                    pseudo.name = s.name + '_mask'
                    pseudo.setRef(s.pseudo[0,:,:,:])
                    pseudo.save(folderpath, image_format='nrrd')
                    
            if s.save_dict['refine']:
                if s.refine is not None:
                    refine = CTRef()
                    refine.name = s.name + '_mask'
                    refine.setRef(s.refine[0,:,:,:])
                    refine.save(folderpath, image_format='nrrd')
            
            features = s.features.clone()
            # features = defaultdict(lambda: None)
            # features['ID'] = s.ID
            # features['name'] = s.name
            # features['slice'] = s.slice
            # features['msg'] = s.msg
            # features['status'] = s.status
            # features['fip_image'] = s.fip_image
            # features['fip_mask'] = s.fip_mask
            # features['fip_pred'] = s.fip_pred
            # features['command'] = s.command
            if s.save_dict['info']:
                features['info'] = s.info
            else:
                features['info'] = dict()

            # Add patch to pacthlist
            df = df.append(features, ignore_index=True)
            df = df.reset_index(drop = True)

        pbar.close() 
        df.to_pickle(fip_df)


    @classmethod
    def save_hdf5(cls, sl, fp_data, save_dict, dataset_name='train', batchsize=100):
        
        # Define parameter
        
        os.makedirs(fp_data, exist_ok=True)
        fip_hdf5 = os.path.join(fp_data, 'sl.hdf5')
        dataaccess = DataAccess()
        num = math.ceil(len(sl)/batchsize)
        save_dict = defaultdict(lambda: None, save_dict)
        
        # Iterate over sample batches
        pbar = tqdm(total=num)
        pbar.set_description("Saveing data " + dataset_name)
        for b in range(num):
            pbar.update()
            #print('Batch: ' + str(b) + ' / ' + str(num))
            batch = sl[b*batchsize:(b+1)*batchsize]
            
            # Init datadict
            datadict = dict()
            for k in cls.da:
                datadict[k] = dict()
                D = getattr(sl[0], k, None)
                for key in D:
                    if D[key] is not None and (save_dict[key] or save_dict[k]): 
                        datadict[k][key]=[]

            # Fill datadict
            df_features = pd.DataFrame()
            for i,s in enumerate(batch):
                s.name = "{:07n}".format(s.ID)
                s.save_dict.update(save_dict)
                s.cpu()
                for k in cls.da:
                    D = getattr(s, k, None)
                    for key in D:
                        if D[key] is not None and (save_dict[key] or save_dict[k]): 
                            datadict[k][key].append(D[key].numpy())

                #features = s.features.copy()
                features = s.F.copy()
                features['ID'] = int(s.ID)
                features['name'] = s.name
                features['msg'] = s.msg
                features['status'] = s.status
                features['command'] = s.command
                features['dataset'] = dataset_name
                

                # Add features to dataframe
                #df_features = df_features.append(features, ignore_index=True)
                df_features = pd.concat([df_features, pd.DataFrame.from_records([features])])
                df_features = df_features.reset_index(drop = True)

            for k in cls.da:
                D = datadict[k]
                for key in D:
                    datadict[k][key] = np.vstack(datadict[k][key])


            datadict['F'] = dataaccess.dataFrameToDict(df_features)
            datadict['ID'] = np.array(df_features['ID'])
            dataaccess.save_dict(fip_hdf5, datadict)            
        pbar.close()


    @classmethod
    def load(cls, fp_data, load_dict, load_class, dataset_name='', hdf5=False):
        if hdf5:
            return cls.load_hdf5(fp_data=fp_data, load_dict=load_dict, load_class=load_class, dataset_name=dataset_name)
        else:
            return cls.load_pkl(fp_data=fp_data, load_dict=load_dict, load_class=load_class, dataset_name=dataset_name)


    @classmethod
    def load_pkl(fp_save, load_dict={}, dataset_name =''):
        

        fip_df = os.path.join(fp_save, 'sl.pkl')
        df = pd.read_pickle(fip_df)

        sl=[]
        pbar = tqdm(total=len(df))
        pbar.set_description("Loading data " + dataset_name)
        for i,row in df.iterrows():
            pbar.update()
            name = str(row['name'])
            s = ALSample()
            s.load_dict.update(load_dict)
            if s.load_dict['image'] and os.path.isfile(os.path.join(fp_save, name, name + '.mhd')):
                image = CTImage()
                image.name = name
                image.load(os.path.join(fp_save, image.name, image.name + '.mhd'))
                s.image = torch.unsqueeze(torch.FloatTensor(image.image().copy()).to('cpu'), 0)
                del image
            
            if s.load_dict['mask'] and os.path.isfile(os.path.join(fp_save, name, name + '_mask.nrrd')):
                mask = CTImage()
                mask.name = name + '_mask'
                mask.load(os.path.join(fp_save, name, name + '_mask.nrrd'))
                s.mask = torch.unsqueeze(torch.from_numpy(mask.image().astype(np.float32).copy()), 0)
                del mask
                
            if s.load_dict['pred'] and os.path.isfile(os.path.join(fp_save, name, name + '_pred.nrrd')):
                pred = CTImage()
                pred.name = name + '_pred'
                pred.load(os.path.join(fp_save, name, name + '_pred.nrrd'))
                s.pred = torch.unsqueeze(torch.from_numpy(pred.image().astype(np.float32).copy()), 0)
                del pred

            if s.load_dict['weight'] and os.path.isfile(os.path.join(fp_save, name, name + '_weight.nrrd')):
                weight = CTImage()
                weight.name = name + '_weight'
                weight.load(os.path.join(fp_save, name, name + '_weight.nrrd'))
                s.weight = torch.unsqueeze(torch.from_numpy(weight.image().astype(np.float32).copy()), 0)
                del weight

            if s.load_dict['region'] and os.path.isfile(os.path.join(fp_save, name, name + '_region.nrrd')):
                region = CTImage()
                region.name = name + 'region'
                region.load(os.path.join(fp_save, name, name + '_weight.nrrd'))
                s.region = torch.unsqueeze(torch.from_numpy(region.image().astype(np.float32).copy()), 0)
                del region

            if s.load_dict['pseudo'] and os.path.isfile(os.path.join(fp_save, name, name + '_pseudo.nrrd')):
                pseudo = CTImage()
                pseudo.name = name + 'pseudo'
                pseudo.load(os.path.join(fp_save, name, name + '_pseudo.nrrd'))
                s.pseudo = torch.unsqueeze(torch.from_numpy(pseudo.image().astype(np.float32).copy()), 0)
                del pseudo

            if s.load_dict['refine'] and os.path.isfile(os.path.join(fp_save, name, name + '_refine.nrrd')):
                refine = CTImage()
                refine.name = name + '_refine'
                refine.load(os.path.join(fp_save, name, name + '_refine.nrrd'))
                s.refine = torch.unsqueeze(torch.from_numpy(refine.image().astype(np.float32).copy()), 0)
                del refine
            
            s.ID = row.ID
            s.name = row.name
            s.slice = row.slice
            s.fip_image = row.fip_image
            s.fip_mask = row.fip_mask
            s.fip_pred = row.fip_pred
            if s.load_dict['info']:
                s.info = row.info
            else:
                s.info = dict()
            sl.append(s)
        pbar.close()
        return sl


    @classmethod
    def load_hdf5(cls, fp_data, load_dict={}, load_class=None, dataset_name=''):

        fip_hdf5 = os.path.join(fp_data, 'sl.hdf5')
        print('fip_hdf5', fip_hdf5)
        if not os.path.isfile(fip_hdf5):
            return []

        dataaccess = DataAccess()
        #datadict = defaultdict(None,{'XID': [], 'XImage': [], 'XMask': [], 'F': [], 'XFeatureLabels': []})
        keys_select = ['ID', 'F'] + [key for key in load_dict if load_dict[key]]
        datadict = dataaccess.read_dict(fip_hdf5, keys_select=keys_select)
        keys = list(datadict.keys())
        #XFeatureLabels = [x.decode('UTF-8') for x in list(datadict['XFeatureLabels'][0])]
        sl=[]
        IDList = list(datadict['F']['ID'])
        pbar = tqdm(total=len(datadict['ID']))
        pbar.set_description("Loading data " + dataset_name)
        for i,ID in enumerate(list(IDList)):
            pbar.update()
            s = load_class()
            s.Y=dict()
            s.Y=dict()
            s.F=dict()
            s.ID = ID
            s.name = str(ID)
            #s.load_dict.update(load_dict)
            if 'X' in keys:
                for key in datadict['X']:
                    if key in keys_select:
                        s.X[key]=torch.from_numpy(datadict['X'][key][i:i+1])
            if 'Y' in keys:
                for key in datadict['Y']:
                    if key in keys_select:
                        s.Y[key]=torch.from_numpy(datadict['Y'][key][i:i+1])
            if 'F' in keys:
                for key in datadict['F']:
                    if key in keys_select:
                        # !!!
                        # s.F[key]=datadict['F'][key][i]
                        if i<len(datadict['F'][key]):
                            s.F[key]=datadict['F'][key][i]

            sl.append(s)
        pbar.close()
        
        return sl
    
    @classmethod
    def sortByScan(cls, data):
        imagenames = list(np.unique([s.F['imagename'] for s in data]))
        data_sort = [[] for i in range(len(imagenames))]
        for s in data:
            idx = imagenames.index(s.F['imagename'])
            data_sort[idx].append(s)    
        return data_sort, imagenames

    # @classmethod
    # def savePseudo(cls, data, folderDict, fp_pseudo):
    #     soft = torch.nn.Softmax(dim=1)
    #     data_sort, imagenames = cls.sortByScan(data)
    #     data_pseudo=[]
    #     for batch, imagename in zip(data_sort, imagenames):
    #         fip_image = os.path.join(folderDict['fp_images'], batch[0].F['imagename'][0])
    #         if os.path.isfile(fip_image):
    #             image = CTImage(fip_image)
    #             shape = image.image().shape
    #             seg_arr = np.zeros(shape)
    #             for s in batch:
    #                 for key in s.P: s.P[key] = soft(s.P[key])
    #                 #s.Y['XRegionPred'] = soft(s.Y['XRegionPred'])
    #                 #s.Y['XMainPred'] = soft(s.Y['XMainPred'])
    #                 #s.Y['XSegmentPred'] = soft(s.Y['XSegmentPred'])
    #                 #s.pred = [soft(x) for x in s.pred]
    #                 arr = compute_one_hot_np(s.P['XRegionPred'].cpu().numpy(),1)
    #                 arr_num = np.zeros((arr.shape[0],arr.shape[2], arr.shape[3]), dtype=np.int16)
    #                 for i in range(arr.shape[1]):
    #                     arr_num = arr_num + arr[:,i,:,:]*(i+1)
    #                 seg_arr[int(s.F['slice'][0])]=arr_num[0]
    #             seg = CTRef()
    #             seg.setRef(seg_arr)
    #             seg.copyInformationFrom(image)
    #             #fip_pseudo = os.path.join(fp_pseudo, splitFilePath(batch[0].fip_mask[0])[1]+'.nrrd')
    #             fip_pseudo = os.path.join(fp_pseudo, batch[0].F['maskname'][0][0])
    #             seg.save(fip_pseudo)  
    #             data_pseudo=data_pseudo+batch
    #     return data_pseudo
    
        #     if load_dict['image']:
        #         image = CTImage()
        #         image.name = name
        #         image.load(os.path.join(fp_save, image.name, image.name + '.mhd'))
        #         s.image = torch.unsqueeze(torch.FloatTensor(image.image().copy()).to('cpu'), 0)
        #         del image
            
        #     filepath = os.path.join(fp_save, name, name + '_mask_weak.nrrd')
        #     if os.path.isfile(filepath) and load_dict['load_mask']:
        #         mask_weak = CTImage()
        #         mask_weak.name = name + '_mask_weak'
        #         mask_weak.load(os.path.join(fp_save, name, name + '_mask_weak.nrrd'))
        #         s.mask_weak = torch.unsqueeze(torch.from_numpy(mask_weak.image().astype(np.float32).copy()), 0)
        #         del mask_weak
                
        #     filepath = os.path.join(fp_save, name, name + '_mask_strong.nrrd')
        #     if os.path.isfile(filepath) and load_dict['load_mask']: 
        #         mask_strong = CTImage()
        #         mask_strong.name = name + '_mask_strongn'
        #         mask_strong.load(filepath)
        #         s.mask_strong = torch.unsqueeze(torch.from_numpy(mask_strong.image().astype(np.float32).copy()), 0)
        #         del mask_strong

        #     filepath = os.path.join(fp_save, name, name + '_pred_weak.mhd')
        #     if os.path.isfile(filepath) and load_dict['load_prediction']: 
        #         pred_weak = CTImage()
        #         pred_weak.name = name + '_pred_weak'
        #         pred_weak.load(filepath)
        #         s.pred_weak = torch.unsqueeze(torch.from_numpy(pred_weak.image().astype(np.float32).copy()), 0)
        #         del pred_weak

            
        #     filepath = os.path.join(fp_save, name, name + '_pred_strong.mhd')
        #     if os.path.isfile(filepath) and load_dict['load_prediction']: 
        #         pred_strong = CTImage()
        #         pred_strong.name = name + '_pred_strong'
        #         pred_strong.load(filepath)
        #         s.pred_strong = torch.unsqueeze(torch.from_numpy(pred_strong.image().astype(np.float32).copy()), 0)
        #         del pred_strong

        #     filepath = os.path.join(fp_save, name, name + '_weight_weak.mhd')
        #     if os.path.isfile(filepath) and load_dict['load_weight']: 
        #         weight_weak = CTImage()
        #         weight_weak.name = name + '_weight_weak'
        #         weight_weak.load(filepath)
        #         s.weight_weak = torch.unsqueeze(torch.from_numpy(weight_weak.image().astype(np.float32).copy()), 0)
        #         del weight_weak
        #     else:
        #         if s.image is not None:
        #             s.weight_weak=torch.ones(s.image.shape, dtype=torch.float32)

        #     filepath = os.path.join(fp_save, name, name + '_weight_strong.mhd')
        #     if os.path.isfile(filepath) and load_dict['load_weight']: 
        #         weight_strong = CTImage()
        #         weight_strong.name = name + '_weight_strong'
        #         weight_strong.load(filepath)
        #         s.weight_strong = torch.unsqueeze(torch.from_numpy(weight_strong.image().astype(np.float32).copy()), 0)
        #         del weight_strong
        #     else:
        #         if s.image is not None:
        #             s.weight_strong=torch.ones(s.image.shape, dtype=torch.float32)

        #     filepath = os.path.join(fp_save, name, name + '_label_mask_weak_fim.mhd')
        #     if os.path.isfile(filepath) and load_dict['label_mask_weak_fim']: 
        #         label_mask_weak_fim = CTImage()
        #         label_mask_weak_fim.name = name + '_label_mask_weak_fim'
        #         label_mask_weak_fim.load(filepath)
        #         s.label_mask_weak_fim = torch.unsqueeze(torch.from_numpy(label_mask_weak_fim.image().astype(np.float32).copy()), 0)
        #         del label_mask_weak_fim

        #     filepath = os.path.join(fp_save, name, name + '_label_mask_strong_fim.mhd')
        #     if os.path.isfile(filepath) and load_dict['label_mask_strong_fim']: 
        #         label_mask_strong_fim = CTImage()
        #         label_mask_strong_fim.name = name + '_label_mask_strong_fim'
        #         label_mask_strong_fim.load(filepath)
        #         s.label_mask_strong_fim = torch.unsqueeze(torch.from_numpy(label_mask_strong_fim.image().astype(np.float32).copy()), 0)
        #         del label_mask_strong_fim
                
        #     filepath = os.path.join(fp_save, name, name + '_label_mask_weak_var.mhd')
        #     if os.path.isfile(filepath) and load_dict['label_mask_weak_var']: 
        #         label_mask_weak_var = CTImage()
        #         label_mask_weak_var.name = name + '_label_mask_weak_var'
        #         label_mask_weak_var.load(filepath)
        #         s.label_mask_weak_var = torch.unsqueeze(torch.from_numpy(label_mask_weak_var.image().astype(np.float32).copy()), 0)
        #         del label_mask_weak_var
   
        #     filepath = os.path.join(fp_save, name, name + '_label_mask_strong_var.mhd')
        #     if os.path.isfile(filepath) and load_dict['label_mask_strong_var']: 
        #         label_mask_strong_var = CTImage()
        #         label_mask_strong_var.name = name + '_label_mask_strong_var'
        #         label_mask_strong_var.load(filepath)
        #         s.label_mask_strong_var = torch.unsqueeze(torch.from_numpy(label_mask_strong_var.image().astype(np.float32).copy()), 0)
        #         del label_mask_strong_var


        #     filepath = os.path.join(fp_save, name, name + '_pseudo_mask_weak.mhd')
        #     if os.path.isfile(filepath) and load_dict['pseudo_mask_weak']: 
        #         pseudo_mask_weak = CTImage()
        #         pseudo_mask_weak.name = name + '_pseudo_mask_weak'
        #         pseudo_mask_weak.load(filepath)
        #         s.pseudo_mask_weak = torch.unsqueeze(torch.from_numpy(pseudo_mask_weak.image().astype(np.float32).copy()), 0)
        #         del pseudo_mask_weak

        #     filepath = os.path.join(fp_save, name, name + '_pseudo_mask_strong.mhd')
        #     if os.path.isfile(filepath) and load_dict['pseudo_mask_strong']: 
        #         pseudo_mask_strong = CTImage()
        #         pseudo_mask_strong.name = name + '_pseudo_mask_strong'
        #         pseudo_mask_strong.load(filepath)
        #         s.pseudo_mask_strong = torch.unsqueeze(torch.from_numpy(pseudo_mask_strong.image().astype(np.float32).copy()), 0)
        #         del pseudo_mask_strong

        #     filepath = os.path.join(fp_save, name, name + '_label_refine_weak_fim.mhd')
        #     if os.path.isfile(filepath) and load_dict['label_refine_weak_fim']: 
        #         label_refine_weak_fim = CTImage()
        #         label_refine_weak_fim.name = name + '_label_refine_weak_fim'
        #         label_refine_weak_fim.load(filepath)
        #         s.label_refine_weak_fim = torch.unsqueeze(torch.from_numpy(label_refine_weak_fim.image().astype(np.float32).copy()), 0)
        #         del label_refine_weak_fim

        #     filepath = os.path.join(fp_save, name, name + '_label_refine_strong_fim.mhd')
        #     if os.path.isfile(filepath) and load_dict['label_refine_strong_fim']: 
        #         label_refine_strong_fim = CTImage()
        #         label_refine_strong_fim.name = name + '_label_refine_strong_fim'
        #         label_refine_strong_fim.load(filepath)
        #         s.label_refine_strong_fim = torch.unsqueeze(torch.from_numpy(label_refine_strong_fim.image().astype(np.float32).copy()), 0)
        #         del label_refine_strong_fim
            
        #     s.ID = row.ID
        #     s.name = row.name
        #     s.slice = row.slice
        #     s.imagename = row.imagename
        #     s.maskname = row.maskname
        #     #s.info = row.info
        #     if load_dict['info']:
        #         s.info = row.info
        #     else:
        #         s.info = dict()
        #     s_list.append(s)
            
        # pbar.close()
        
        # return s_list
    
    # @staticmethod
    # def loadFromScanList(data, fp_images, fp_references, load_dict={}):
    #     pbar = tqdm(total=len(data))
    #     pbar.set_description("Loading image")
    #     for s in data:
    #         pbar.update()
    #         s.loadFromScan(fp_images, fp_references, load_dict=load_dict)
    #     pbar.close()
    
    # def loadFromScan(self, fp_images, fp_references, load_dict={}):
    #     if load_dict['load_image']:
    #         fip_image = os.path.join(fp_images, self.imagename)
    #         image = CTImage(fip_image)
    #         self.image = torch.unsqueeze(torch.FloatTensor(image.image()[int(self.slice):int(self.slice)+1,:,:]), 0)
            
    #     fip_mask = os.path.join(fp_references, self.maskname)
    #     if os.path.isfile(fip_mask) and load_dict['load_mask']: 
    #         mask = CTImage(fip_mask)
    #         self.mask_weak = torch.zeros((1,2,512,512))
    #         self.mask_weak[0,1,:,:] = torch.FloatTensor(mask.image()[int(self.slice),:,:]==1)
    #         self.mask_weak[0,0,:,:]=1-self.mask_weak[0,1,:,:]
    #         self.mask_strong = torch.zeros((1,2,512,512))
    #         self.mask_strong[0,1,:,:] = torch.FloatTensor(mask.image()[int(self.slice),:,:]>1)
    #         self.mask_strong[0,0,:,:]=1-self.mask_strong[0,1,:,:]
        
    #     # !!!
    #     self.weight_weak = torch.ones((1,1,512,512))
    #     self.weight_strong = torch.ones((1,1,512,512))

        # fip_mask = os.path.join(fp_data, 'references', self.maskname)
        # filepath = os.path.join(fp_save, name, name + '_mask_strong.mhd')
        # if os.path.isfile(filepath) and load_dict['load_mask']: 
        #     mask_strong = CTImage()
        #     mask_strong.name = name + '_mask_strongn'
        #     mask_strong.load(filepath)
        #     mask_strong.convert(sitk.sitkFloat32)
        #     s.mask_strong = torch.unsqueeze(torch.FloatTensor(mask_strong.image()).to('cpu'), 0).clone()

        # filepath = os.path.join(fp_save, name, name + '_pred_weak.mhd')
        # if os.path.isfile(filepath) and load_dict['load_prediction']: 
        #     pred_weak = CTImage()
        #     pred_weak.name = name + '_pred_weak'
        #     pred_weak.load(filepath)
        #     s.pred_weak = torch.unsqueeze(torch.FloatTensor(pred_weak.image()).to('cpu'), 0).clone()
        
        # filepath = os.path.join(fp_save, name, name + '_pred_strong.mhd')
        # if os.path.isfile(filepath) and load_dict['load_prediction']: 
        #     pred_strong = CTImage()
        #     pred_strong.name = name + '_pred_strong'
        #     pred_strong.load(filepath)
        #     s.pred_strong = torch.unsqueeze(torch.FloatTensor(pred_strong.image()).to('cpu'), 0).clone()
        
        # filepath = os.path.join(fp_save, name, name + '_weight_weak.mhd')
        # if os.path.isfile(filepath) and load_dict['load_weight']: 
        #     weight_weak = CTImage()
        #     weight_weak.name = name + '_weight_weak'
        #     weight_weak.load(filepath)
        #     s.weight_weak = torch.unsqueeze(torch.FloatTensor(weight_weak.image()).to('cpu'), 0).clone()

        # filepath = os.path.join(fp_save, name, name + '_weight_weak.mhd')
        # if os.path.isfile(filepath) and load_dict['load_weight']: 
        #     weight_strong = CTImage()
        #     weight_strong.name = name + '_weight_strong'
        #     weight_strong.load(filepath)
        #     s.weight_strong = torch.unsqueeze(torch.FloatTensor(weight_strong.image()).to('cpu'), 0).clone()
            
            

class ALSampleMulti(ALSample):
    
    """
    Active learning sample multi task
    """

    def __init__(self):
        super().__init__()
        self.fip_mask = []

    # def cpu(self):
    #     self.__to_device('cpu')
    
    def cuda(self):
        self.__to_device('cuda')
        
    def __to_device(self, device='cpu'):
        members = ['image', 'mask', 'pred', 'weight', 'region', 'pseudo', 'refine']
        for member in members:
            att = getattr(self, member, None)
            if att is not None: 
                if isinstance(att, dict):
                    if att is not None: 
                        for key in att: att[key].to(device)
                else:
                    att.to(device)
                
        # if self.mask is not None: for key in self.mask: self.mask[key].to(device)
        # if self.pred is not None: self.pred =  for key in self.pred: self.pred[key].to(device)
        # if self.weight is not None: self.weight = for key in self.weight: self.weight[key].to(device)
        # if self.region is not None: self.region = for key in self.region: self.region[key].to(device)
        # if self.pseudo is not None: self.pseudo = for key in self.pseudo: self.pseudo[key].to(device)
        # if self.refine is not None: self.refine = for key in self.refine: self.refine[key].to(device)


        

    @classmethod
    def save_pkl(cls, sl, fp_data, save_dict, dataset_name=''):
        
        saveprops=dict({'image': (CTImage, 'mhd'),
                        'mask': (CTRef, 'nrrd'),
                        'pred': (CTRef, 'nrrd'),
                        'weight': (CTRef, 'nrrd'),
                        'region': (CTRef, 'nrrd'),
                        'pseudo': (CTRef, 'nrrd'),
                        'refine': (CTRef, 'nrrd')})
        os.makedirs(fp_data, exist_ok=True)
        df = pd.DataFrame()
        fip_df = os.path.join(fp_data, 'sl.pkl')
        pbar = tqdm(total=len(sl))
        pbar.set_description("Saveing data " + dataset_name)
        for s in sl:
            pbar.update()
            s.name = "{:07n}".format(s.ID)
            s.save_dict.update(save_dict)
            s.cpu()
            folderpath = os.path.join(fp_data, str(s.name))
            if s.save_any():
                os.makedirs(folderpath, exist_ok=True)
                
            for key in saveprops.keys():
                if s.save_dict[key]:
                    arr = getattr(s, key, None)
                    if arr is not None:
                        if isinstance(arr, list):
                            for i,x in enumerate(arr):
                                image = saveprops[key][0]()
                                image.name = s.name + '_' + key + '_' + str(i)
                                #if image==CTImage:
                                if isinstance(image, CTImage):
                                    image.setImage(x[0,:,:,:])
                                else:
                                    image.setRef(x[0,:,:,:])
                                image.save(folderpath, image_format=saveprops[key][1])
                        else:
                            image = saveprops[key][0]()
                            image.name = s.name
                            if isinstance(image, CTImage):
                                image.setImage(arr[0,:,:,:])
                            else:
                                image.setRef(arr[0,:,:,:])
                            image.save(folderpath, image_format=saveprops[key][1])
                    

                    
            features = defaultdict(lambda: None)
            features['ID'] = s.ID
            features['name'] = s.name
            features['slice'] = s.slice
            features['msg'] = s.msg
            features['status'] = s.status
            features['command'] = s.command
            features['fip_image'] = s.fip_image
            features['fip_mask'] = s.fip_mask
            features['fip_pred'] = s.fip_pred
            if s.save_dict['info']:
                features['info'] = s.info
            else:
                features['info'] = dict()

            # Add patch to pacthlist
            df = df.append(features, ignore_index=True)
            df = df.reset_index(drop = True)

        pbar.close() 
        df.to_pickle(fip_df)

    @classmethod
    def save_hdf5(cls, sl, fp_data, save_dict, dataset_name='train', batchsize=100):
        
        # Define parameter
        
        os.makedirs(fp_data, exist_ok=True)
        fip_hdf5 = os.path.join(fp_data, 'sl.hdf5')
        dataaccess = DataAccess()
        num = math.ceil(len(sl)/batchsize)
        
        #print('fip_hdf5', fip_hdf5)
        
        #saveX = cls.saveX
        #saveY = cls.saveY
        #if 'X' in save_dict: saveX=saveX+list(sl[0].X.keys())
        #if 'Y' in save_dict: saveY=saveY+list(sl[0].Y.keys())
        
        save_dict = defaultdict(lambda: None, save_dict)
        
        # Iterate over sample batches
        pbar = tqdm(total=num)
        pbar.set_description("Saveing data " + dataset_name)
        for b in range(num):
            pbar.update()
            #print('Batch: ' + str(b) + ' / ' + str(num))
            batch = sl[b*batchsize:(b+1)*batchsize]
            
            # Init datadict
            datadict = dict()
            #datadict['X'] = dict()
            for k in cls.da:
                datadict[k] = dict()
                D = getattr(sl[0], k, None)
                for key in D:
                    if D[key] is not None and (save_dict[key] or save_dict[k]): 
                        datadict[k][key]=[]
            
            # for key in sl[0].X: 
            #     if key[save_dict]:
            #         datadict['X'][key]={}
                    
            # #for key in cls.saveX: datadict['X'][key]=[]
            # datadict['Y'] = dict()
            # #for key in saveY: datadict['Y'][key]=[]
            # datadict['F'] = dict()
            
            # Fill datadict
            df_features = pd.DataFrame()
            for i,s in enumerate(batch):
                s.name = "{:07n}".format(s.ID)
                s.save_dict.update(save_dict)
                s.cpu()
                for k in cls.da:
                    D = getattr(s, k, None)
                    for key in D:
                        if D[key] is not None and (save_dict[key] or save_dict[k]): 
                            datadict[k][key].append(D[key].numpy())
                        
                # for key in saveX:
                #     if key in s.X:
                #         arr = s.X[key] 
                #         if arr is not None:
                #             datadict['X'][key].append(arr.numpy())
                # for key in saveY:
                #     if key in s.Y :
                #         arr = s.Y[key] 
                #         if arr is not None:
                #             datadict['Y'][key].append(arr.numpy())

                #features = s.features.copy()
                features = s.F.copy()
                features['ID'] = int(s.ID)
                features['name'] = s.name
                features['msg'] = s.msg
                features['status'] = s.status
                features['command'] = s.command
                features['dataset'] = dataset_name
                

                # Add features to dataframe
                df_features = df_features.append(features, ignore_index=True)
                df_features = df_features.reset_index(drop = True)

            for k in cls.da:
                D = datadict[k]
                for key in D:
                    datadict[k][key] = np.vstack(datadict[k][key])
                            
            # for key in list(datadict['X'].keys()):
            #     if datadict['X'][key]: 
            #         datadict['X'][key] = np.vstack(datadict['X'][key])
            #     else:
            #         del datadict['X'][key]
                    
            # for key in list(datadict['Y'].keys()):
            #     if datadict['Y'][key]: 
            #         datadict['Y'][key] = np.vstack(datadict['Y'][key])
            #     else:
            #         del datadict['Y'][key]

            datadict['F'] = dataaccess.dataFrameToDict(df_features)
            datadict['ID'] = np.array(df_features['ID'])
            dataaccess.save_dict(fip_hdf5, datadict)            
        pbar.close()


    # isinstance(value, np.ndarray)

        
    @classmethod
    def load_pkl(cls, fp_data, load_dict={}, dataset_name='train'):

        loadprops=dict({'image': (CTImage, 'mhd', 'single'),
                        'mask': (CTRef, 'nrrd', 'list'),
                        'pred': (CTRef, 'nrrd', 'list'),
                        'weight': (CTRef, 'nrrd', 'list'),
                        'region': (CTRef, 'nrrd', 'list'),
                        'pseudo': (CTRef, 'nrrd', 'list'),
                        'refine': (CTRef, 'nrrd', 'list')})
        
        fip_df = os.path.join(fp_data, 'sl.pkl')
        df = pd.read_pickle(fip_df)
        
        sl=[]
        pbar = tqdm(total=len(df))
        pbar.set_description("Loading data " + dataset_name)
        for i,row in df.iterrows():
            pbar.update()
            name = str(row['name'])
            s = ALSampleMulti()
            s.load_dict.update(load_dict)
            
            for key in loadprops.keys():
                if s.load_dict[key]:
                    image = loadprops[key][0]()
                    image.name = name
                    if loadprops[key][2]=='list':
                        st = os.path.join(fp_data, image.name, image.name + '_' + key + '*')
                        files = glob(st)
                        arr=[]
                        for f in files:
                            image.load(f)
                            if isinstance(image, CTImage):
                                arr=torch.unsqueeze(torch.FloatTensor(image.image().copy()),0)
                            else:
                                arr.append(torch.unsqueeze(torch.FloatTensor(image.ref().copy()),0))
                        setattr(s, key, arr)
                        del image
                    else:
                        image.load(os.path.join(fp_data, image.name, image.name + '.' + loadprops[key][1]))
                        s.image = torch.unsqueeze(torch.FloatTensor(image.image().copy()), 0)
                        del image
            
            s.ID = row.ID
            s.name = str(row.name)
            s.slice = row.slice
            s.fip_image = row.fip_image
            s.fip_mask = row.fip_mask
            s.fip_pred = row.fip_pred
            if s.load_dict['info']:
                s.info = row.info
            else:
                s.info = dict()
            sl.append(s)
        pbar.close()
        return sl

    @classmethod
    def load_hdf5(cls, fp_data, load_dict={}, load_class=ALSample, dataset_name=''):

        fip_hdf5 = os.path.join(fp_data, 'sl.hdf5')
        print('fip_hdf5', fip_hdf5)
        if not os.path.isfile(fip_hdf5):
            return []
            pass

        # loadprops=dict({'XImage': 'image',
        #                 'XMask': 'mask',
        #                 'XPred': 'pred',
        #                 'XWeight': 'weight',
        #                 'XRegion': 'region',
        #                 'XPseudo': 'pseudo',
        #                 'Xrefine': 'refine'})
        
        dataaccess = DataAccess()
        #datadict = defaultdict(None,{'XID': [], 'XImage': [], 'XMask': [], 'F': [], 'XFeatureLabels': []})
        keys_select = ['ID', 'F'] + [key for key in load_dict if load_dict[key]]
        datadict = dataaccess.read_dict(fip_hdf5, keys_select=keys_select)
        keys = list(datadict.keys())
        #XFeatureLabels = [x.decode('UTF-8') for x in list(datadict['XFeatureLabels'][0])]
        sl=[]
        IDList = list(datadict['F']['ID'])
        pbar = tqdm(total=len(datadict['ID']))
        pbar.set_description("Loading data " + dataset_name)
        for i,ID in enumerate(list(IDList)):
            pbar.update()
            s = load_class()
            s.Y=dict()
            s.Y=dict()
            s.F=dict()
            s.ID = ID
            s.name = str(ID)
            #s.load_dict.update(load_dict)
            if 'X' in keys:
                for key in datadict['X']:
                    if key in keys_select:
                        s.X[key]=torch.from_numpy(datadict['X'][key][i:i+1])
            if 'Y' in keys:
                for key in datadict['Y']:
                    if key in keys_select:
                        s.Y[key]=torch.from_numpy(datadict['Y'][key][i:i+1])
            if 'F' in keys:
                for key in datadict['F']:
                    if key in keys_select:
                        s.F[key]=datadict['F'][key][i]
                    
            # for key in loadprops.keys():
            #     if key in list(datadict.keys()):
            #         if key=='XImage':
            #             arr = torch.from_numpy(datadict['XImage'][i:i+1])
            #             setattr(s, loadprops[key], arr)
            #         else:
            #             arr=dict()
            #             for k in datadict[key]:
            #                 arr[k]=torch.from_numpy(datadict[key][k][i:i+1])
            #             setattr(s, loadprops[key], arr)
            sl.append(s)
        pbar.close()
        
        return sl

    def __plot_image(self, image, name, title, color=False, save=False, filepath=None, format_im=None, dpi=300):
        if color:
            #print('image1234', image.shape)
            im = np.zeros((image.shape[2], image.shape[3]))
            for c in range(image.shape[1]):
                im = im + (c+1) * image[0,c,:,:]
            plt.imshow(im)
            plt.imshow(im, cmap='Accent', interpolation='nearest')
            if title: plt.title(name) 
            if save: plt.savefig(filepath + name + format_im, format=format_im, dpi=dpi)
            plt.show()
        else:
            if title: plt.title(name) 
            if save: plt.savefig(filepath + name + format_im, format=format_im, dpi=dpi)
            plt.imshow(image[0,0,:,:], cmap='gray')
            plt.show()

                        
    def plotSample(self, plotlist=['XImage', 'XMask', 'XPred', 'XWeight', 'XRegion', 'XPseudo', 'XRefine'], save=False, fp='', name='', color=False, title=True, format_im='svg', dpi=300):
        self.cpu()
        filepath = os.path.join(fp, name)

        for d in self.da:
            if d in plotlist:
                plotlist = plotlist + list(getattr(self, d, None).keys())

        for pl in plotlist:
            name = self.name + '_' + pl  + '_' + str(int(self.ID))
            im = self.getXY(pl)
            if im is not None:
                image = im.data.numpy()
                if pl=='XImage' and image.shape[1]>1:
                    idx = int((image.shape[1]-1)/2)
                    image = image[:,idx:idx+1]
                    self.__plot_image(image, name, title, color=False, filepath=filepath, save=save)
                else:
                    self.__plot_image(image, name, title, color=color, filepath=filepath, save=save)
                

                            
                            