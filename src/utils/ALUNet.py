#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:26:04 2023

@author: bernifoellmer
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 09:55:14 2023

@author: bernifoellmer
"""

import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from glob import glob
import torch
import shutil
import math
from utils.helper import compute_one_hot_np, compute_one_hot_np_3
import subprocess
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
from utils.ct import CTImage, CTRef
from utils.DataAccess import DataAccess
from utils.ALManager import ALManager, SALDataset
from utils.ALDataset import ALDataset, ALDatasetMulti
from utils.ALAction import ALAction
from utils.ALSample import ALSample, ALSampleMulti
from utils.UNetSeg import UNetSeg
from utils.helper import splitFilePath, splitFolderPath, compute_one_hot_torch
import random
from utils.DataframeBaseModel import DataframeBaseModel, defaultdict
from utils.DatasetBaseModel import DatasetBaseModel, YAML, YAML_MODE, defaultdict
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.paths import nnUNet_preprocessed

class UNetSample(ALSample):
    """
    UNetSample
    """
    
    # def __init__(self, fp_dataset):
    #     ALSampleMulti.__init__(self, fp_dataset)

    def __init__(self):
        ALSample.__init__(self)
        self.Xlabel = ['XImage']
        self.Ylabel = ['XMask']
        self.Plabel = ['XMaskPred']
        self.patches = []

    @staticmethod
    def getSampleByIDAndIDP(sl, ID, IDP, return_idx=False):
        for idx,s in enumerate(sl):
            if s.F['ID']==ID and s.F['IDP']==IDP:
                if return_idx:
                    return idx
                else:
                    return s
        return None

    @staticmethod
    def loss_fisher(net, idx_output=0):
        """ Fisher loss
        """

        def func(Xinput):
            loss=[]
            pred = net.model['unet'].network(Xinput)
            pred_weak_bin_log = torch.log_softmax(pred[idx_output], dim=1)
            pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
            print('pred_weak_bin_log', pred_weak_bin_log.shape)
            for c in range(pred_weak_bin_prop.shape[1]):
                loss.append(torch.mean(pred_weak_bin_log[:,c] * pred_weak_bin_prop[:,c]))
            loss = torch.unsqueeze(torch.hstack(loss), 0)
            return loss
        return func
    
    @staticmethod
    def samplesTopatches(man, opts, folderDict, data, tile_step_size=0.5):
        # shape = (564,564)
        # patch_size = (512,512)
        # classLocations = np.array([[0,308,256]])
        # selected_class_or_region=(0,1)
        # class_locations=(0,1)
        # data = action_round
        
        #import pandas as pd
        #df = pd.read_pickle('/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset105_ALUNet/nnUNetPlans_2d/volume-0.pkl')
        
        from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window
        net = man.load_model(opts, folderDict, previous=True, load_weights=False)
        dataloader_train = net.model['unet'].get_dataloaders_BF(data, batch_size=None)
        patch_size = dataloader_train.data_loader.patch_size
        selected_class_or_region = dataloader_train.data_loader.annotated_classes_key
        #dim=2
        dim =opts.dim
        #shapeOrg = (512,512)
        
        def convert_bbox(bboxLbs, bboxUbs, imn, d_shape, dim):
            spacingPre = d_shape[imn]['spacingPre'][1:]
            spacingOrg = d_shape[imn]['spacingOrg'][1:]
            shapeOrg = d_shape[imn]['shapeOrg']
            bboxLbsOrg = np.array([int(round(i / j * k)) for i, j, k in zip(spacingPre, spacingOrg, bboxLbs)])
            bboxUbsOrg = np.array([int(round(i / j * k)) for i, j, k in zip(spacingPre, spacingOrg, bboxUbs)])
            bboxLbsOrg = [max(0, bboxLbsOrg[i]) for i in range(dim)]
            bboxUbsOrg = [min(shapeOrg[i], bboxUbsOrg[i]) for i in range(dim)]
            return bboxLbsOrg, bboxUbsOrg
            
        
        #plans_file = '/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset105_ALUNet/nnUNetPlans.json'
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset_name
        plans_file = os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans.json')
        with open(plans_file, 'r') as f:
            plans = json.load(f)
        #spacingPre = [-1] + plans['configurations'][opts.configuration]['spacing']
        spacingPre = plans['configurations'][opts.configuration]['spacing']
        
        # Extract image shapes
        imagenames = list(np.unique([s.F['imagename'] for s in data]))
        d_shape={}
        for im in imagenames:
            imn = im.split('.')[0]
            for s in data:
                if s.F['imagename']==im:
                    shapeOrg = (s.F['depth'], s.F['width'], s.F['height'])
                    break
                    
            #fip = os.path.join('/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset105_ALUNet/nnUNetPlans_2d', imn + '.npz')
            fip = os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans_' + opts.configuration, imn + '.npz')
            spacingOrg = pd.read_pickle(os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans_'+opts.configuration , imn + '.pkl'))['spacing']
            d_shape[imn] = {'shapeOrg': shapeOrg,
                            'shapePre': np.load(fip)['seg'].shape[4-dim:], 
                            'spacingOrg': spacingOrg,
                            'spacingPre': spacingPre}
        
        # Create patches from image slice
        datap=[]
        for s in data:
            #shape = d_shape[s.F['imagename'].split('.')[0]][2:]
            imn = s.F['imagename'].split('.')[0]
            shapePre = d_shape[imn]['shapePre']
            
            # old_spacing = d_shape[imn]['spacing_pre'][1:]
            # new_spacing = d_shape[imn]['spacing_org'][1:]
            # old_shape = d_shape[imn]['shape_pre'][2:]
            # shapeOrg = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
            
            steps = compute_steps_for_sliding_window(shapePre, patch_size, tile_step_size)
            selected_slice = int(data[0].F['slice'])
            IDP=0
            for sx in list(steps[0]):
                for sy in list(steps[1]):
                    
                    # if s.ID==28217:
                    #     if IDP==3:
                    #         sys.exit()
                            
                    coord = np.array([[0,selected_slice,int(sx+patch_size[0]/2),int(sy+patch_size[1]/2)]])
                    class_locations = {selected_class_or_region: coord}
                    bbox_lbs, bbox_ubs = dataloader_train.data_loader.get_bbox(shapePre, None, class_locations, overwrite_class=selected_class_or_region)
                    valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                    valid_bbox_ubs = [min(shapePre[i], bbox_ubs[i]) for i in range(dim)]
                    patch=opts.CLPatch()
                    patch.ID=s.ID
                    patch.IDP=IDP
                    patch.F = s.F.copy()
                    patch.F['IDP']=IDP
                    patch.F['classLocations']=coord
                    patch.F['bboxLbs']=valid_bbox_lbs
                    patch.F['bboxUbs']=valid_bbox_ubs
                    patch.F['shapePre']=d_shape[imn]['shapePre']
                    patch.F['shapeOrg']=d_shape[imn]['shapeOrg']
                    patch.F['spacingOrg']=d_shape[imn]['spacingOrg']
                    patch.F['spacingPre']=d_shape[imn]['spacingPre']
                    bboxLbsOrg, bboxUbsOrg = convert_bbox(patch.F['bboxLbs'], patch.F['bboxUbs'], imn, d_shape, dim)
                    patch.F['bboxLbsOrg']=bboxLbsOrg
                    patch.F['bboxUbsOrg']=bboxUbsOrg
                    datap.append(patch)
                    IDP=IDP+1

                    
        return datap


    @staticmethod
    def samplesTopatches3D(man, opts, folderDict, data, tile_step_size=0.5):
        # shape = (564,564)
        # patch_size = (512,512)
        # classLocations = np.array([[0,308,256]])
        # selected_class_or_region=(0,1)
        # class_locations=(0,1)
        # data = action_round
        # tile_step_size=0.5
        
        #import pandas as pd
        #df = pd.read_pickle('/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset105_ALUNet/nnUNetPlans_2d/volume-0.pkl')
        
        from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window
        net = man.load_model(opts, folderDict, previous=True, load_weights=False)
        dataloader_train = net.model['unet'].get_dataloaders_BF(data, batch_size=None)
        patch_size = dataloader_train.data_loader.patch_size
        selected_class_or_region = dataloader_train.data_loader.annotated_classes_key
        #dim=2
        dim = opts.dim
        #shapeOrg = (512,512)

        def convert_bbox(bboxLbs, bboxUbs, imn, d_shape, dim):
            # bboxLbs=patch.F['bboxLbs']
            # bboxUbs=patch.F['bboxUbs']
            bboxLbsOrg=[]
            bboxUbsOrg=[]
            for d in range(dim):
                xl=bboxLbs[d]
                xu=bboxUbs[d]
                spacingPre = d_shape[imn]['spacingPre'][d]
                spacingOrg = d_shape[imn]['spacingOrg'][d]
                shapePre = d_shape[imn]['shapePre'][d]
                shapeOrg = d_shape[imn]['shapeOrg'][d]
                ratio = spacingPre/spacingOrg
                #bboxLbsOrg.append(round(ratio*xl))
                #bboxUbsOrg.append(round(ratio*xu))
                #ratio = spacingPre/spacingOrg
                # if ratio>1.0:
                #     pad=0
                # else:
                #     #pad=(shapeOrg-shapePre)/2
                #     pad=0
                #     #sys.exit()
                #bboxLbsOrg.append(round(ratio*(xl-pad)))
                #bboxUbsOrg.append(round(ratio*(xu-pad)))
                bboxLbsOrg.append(round(ratio*xl))
                bboxUbsOrg.append(round(ratio*xu))
            return bboxLbsOrg, bboxUbsOrg
        
        # def convert_bbox(bboxLbs, bboxUbs, imn, d_shape, dim):
        #     spacingPre = d_shape[imn]['spacingPre']
        #     spacingOrg = d_shape[imn]['spacingOrg']
        #     shapeOrg = d_shape[imn]['shapeOrg']
        #     bboxLbsOrg = np.array([int(round(i / j * k)) for i, j, k in zip(spacingPre, spacingOrg, bboxLbs)])
        #     bboxUbsOrg = np.array([int(round(i / j * k)) for i, j, k in zip(spacingPre, spacingOrg, bboxUbs)])
        #     bboxLbsOrg = [max(0, bboxLbsOrg[i]) for i in range(dim)]
        #     bboxUbsOrg = [min(shapeOrg[i], bboxUbsOrg[i]) for i in range(dim)]
        #     return bboxLbsOrg, bboxUbsOrg
            
        
        #plans_file = '/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset105_ALUNet/nnUNetPlans.json'
        #dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset_name
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        plans_file = os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans.json')
        with open(plans_file, 'r') as f:
            plans = json.load(f)
        #spacingPre = [-1] + plans['configurations'][opts.configuration]['spacing']
        spacingPre = plans['configurations'][opts.configuration]['spacing']
        
        # Extract image shapes
        imagenames = list(np.unique([s.F['imagename'] for s in data]))
        d_shape={}
        for im in imagenames:
            imn = im.split('.')[0]

            #fip = os.path.join('/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset105_ALUNet/nnUNetPlans_2d', imn + '.npz')
            fip = os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans_' + opts.configuration, imn + '.npz')
            #shape_pre = np.load(fip)['data'].shape[4-dim:]
            shape_pre = np.load(fip)['data'].shape[4-dim:]
            spacingOrg = pd.read_pickle(os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans_'+opts.configuration , imn + '.pkl'))['spacing'][-dim:]
            shapeRaw = pd.read_pickle(os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans_'+opts.configuration , imn + '.pkl'))['shape_before_cropping'][-dim:]
            shapeOrg = pd.read_pickle(os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans_'+opts.configuration , imn + '.pkl'))['shape_after_cropping_and_before_resampling'][-dim:]
            bbox_used_for_cropping = pd.read_pickle(os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans_'+opts.configuration , imn + '.pkl'))['bbox_used_for_cropping'][-dim:]
            bbox_used_for_cropping_depth = pd.read_pickle(os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans_'+opts.configuration , imn + '.pkl'))['bbox_used_for_cropping'][0]
            d_shape[imn] = {'shapeRaw': shapeRaw,
                            'shapeOrg': shapeOrg,
                            'shapePre': shape_pre, 
                            'spacingOrg': spacingOrg,
                            'spacingPre': spacingPre,
                            'bbox_used_for_cropping': bbox_used_for_cropping,
                            'bbox_used_for_cropping_depth': bbox_used_for_cropping_depth}
            
        #q = pd.read_pickle('/mnt/HHD/data/UNetAL/nnunet/nnUNet_preprocessed/Dataset110_BrainTumour/nnUNetPlans_2d/BRATS_001.pkl')
        #q=np.load('/mnt/HHD/data/UNetAL/nnunet/nnUNet_preprocessed/Dataset110_BrainTumour/nnUNetPlans_2d/BRATS_001.npy')
        # Create patches from image slice
        datap=[]
        for s in data:
            imn = s.F['imagename'].split('.')[0]
            bbox_used_for_cropping_depth = d_shape[imn]['bbox_used_for_cropping_depth']
            if (dim==3 or (s.F['slice']>=bbox_used_for_cropping_depth[0] and s.F['slice']<bbox_used_for_cropping_depth[1])):
                #if s.ID==1174:
                #    sys.exit()
                #shape = d_shape[s.F['imagename'].split('.')[0]][2:]
                
                shapePre = d_shape[imn]['shapePre']
                shapeOrg = d_shape[imn]['shapeOrg']
                
                steps = compute_steps_for_sliding_window(shapePre, patch_size, tile_step_size)
                #selected_slice = int(data[0].F['slice'])
                if dim==2:
                    IDP=0
                    selected_slice = int(data[0].F['slice'])
                    for sx in list(steps[0]):
                        for sy in list(steps[1]):
                            coord = np.array([[0, selected_slice, int(sx+patch_size[0]/2),int(sy+patch_size[1]/2)]])
                            class_locations = {selected_class_or_region: coord}
                            bbox_lbs, bbox_ubs = dataloader_train.data_loader.get_bbox(shapePre, None, class_locations, overwrite_class=selected_class_or_region)
                            #print('pad123', dataloader_train.data_loader.need_to_pad)
                            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                            valid_bbox_ubs = [min(shapePre[i], bbox_ubs[i]) for i in range(dim)]
                            patch=opts.CLPatch()
                            patch.ID=s.ID
                            patch.IDP=IDP
                            patch.F = s.F.copy()
                            # Update slice
                            #sys.exit()
                            
                            patch.F['sliceOrg']=patch.F['slice']
                            patch.F['slice']=patch.F['slice']-bbox_used_for_cropping_depth[0]
                            patch.F['IDP']=IDP
                            patch.F['classLocations']=coord
                            patch.F['bboxLbs']=valid_bbox_lbs
                            patch.F['bboxUbs']=valid_bbox_ubs
                            patch.F['shapePre']=d_shape[imn]['shapePre']
                            patch.F['shapeOrg']=d_shape[imn]['shapeOrg']
                            patch.F['shapeRaw']=d_shape[imn]['shapeRaw']
                            patch.F['spacingOrg']=d_shape[imn]['spacingOrg']
                            patch.F['spacingPre']=d_shape[imn]['spacingPre']
                            patch.F['bboxUsedForCropping']=d_shape[imn]['bbox_used_for_cropping']
                            patch.F['bboxUsedForCroppingDepth']=d_shape[imn]['bbox_used_for_cropping_depth']
                            bboxLbsOrg, bboxUbsOrg = convert_bbox(patch.F['bboxLbs'], patch.F['bboxUbs'], imn, d_shape, dim)
                            valid_bbox_lbs = [max(0, bboxLbsOrg[i]) for i in range(dim)]
                            valid_bbox_ubs = [min(shapeOrg[i], bboxUbsOrg[i]) for i in range(dim)]
                            patch.F['bboxLbsOrg']=valid_bbox_lbs
                            patch.F['bboxUbsOrg']=valid_bbox_ubs
                            datap.append(patch)
                            IDP=IDP+1
                else:
                    IDP=0
                    for sx in list(steps[0]):
                        for sy in list(steps[1]):
                            for sz in list(steps[2]):
                                coord = np.array([[0,int(sx+patch_size[0]/2),int(sy+patch_size[1]/2),int(sz+patch_size[2]/2)]])
                                class_locations = {selected_class_or_region: coord}
                                bbox_lbs, bbox_ubs = dataloader_train.data_loader.get_bbox(shapePre, None, class_locations, overwrite_class=selected_class_or_region)
                                #print('pad123', dataloader_train.data_loader.need_to_pad)
                                valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                                valid_bbox_ubs = [min(shapePre[i], bbox_ubs[i]) for i in range(dim)]
                                patch=opts.CLPatch()
                                patch.ID=s.ID
                                patch.IDP=IDP
                                patch.F = s.F.copy()
                                patch.F['IDP']=IDP
                                patch.F['classLocations']=coord
                                patch.F['bboxLbs']=valid_bbox_lbs
                                patch.F['bboxUbs']=valid_bbox_ubs
                                patch.F['shapePre']=d_shape[imn]['shapePre']
                                patch.F['shapeOrg']=d_shape[imn]['shapeOrg']
                                patch.F['shapeRaw']=d_shape[imn]['shapeRaw']
                                patch.F['spacingOrg']=d_shape[imn]['spacingOrg']
                                patch.F['spacingPre']=d_shape[imn]['spacingPre']
                                patch.F['bboxUsedForCropping']=d_shape[imn]['bbox_used_for_cropping']
                                patch.F['bboxUsedForCroppingDepth']=d_shape[imn]['bbox_used_for_cropping_depth']
                                bboxLbsOrg, bboxUbsOrg = convert_bbox(patch.F['bboxLbs'], patch.F['bboxUbs'], imn, d_shape, dim)
                                valid_bbox_lbs = [max(0, bboxLbsOrg[i]) for i in range(dim)]
                                valid_bbox_ubs = [min(shapeOrg[i], bboxUbsOrg[i]) for i in range(dim)]
                                patch.F['bboxLbsOrg']=valid_bbox_lbs
                                patch.F['bboxUbsOrg']=valid_bbox_ubs
                                datap.append(patch)
                                IDP=IDP+1
                            
                            #ratio0 = patch.F['spacingPre'][0]/patch.F['spacingOrg'][0]
                            #if ratio0<1.0:
                            #    sys.exit()
                            
                            #im = CTImage('/mnt/HHD/data/UNetAL/nnunet/nnUNet_raw/Dataset110_BrainTumour/labelsTr/BRATS_001.nii.gz')

                    
        return datap
    
    # action_round=[patch]
    # dataset = opts.CLDataset()
    # #dataset.load_dataset_hdf53D(man, opts, folderDict, action_round)
    # dataset.load_dataset_hdf53D(man, opts, folderDict, action_round)
    # net = man.load_model(opts, folderDict, previous=False)
    # opts.CLSample.predict(action_round, opts, net)
    
    @staticmethod
    def load_dataset_nnunet(sl, opts, net, NumChannels=2):

        # net = man.load_model(opts, folderDict, previous=False)
        # sl = man.getRandom(dataset='query', NumSamples=20, remove=True)
        #dataset = opts.CLDataset()
        #net.model['unet'].network.eval()
        #loss_func = net.model['unet'].loss
        #soft1 = torch.nn.Softmax(dim=1)
        
        # Order samples by image
        imagenames = list(np.unique([s.F['imagename'] for s in sl]))
        data_load = pd.DataFrame()
        for imn in imagenames:
            data_im = [{'imagename': s.F['imagename'], 'slice': int(s.F['slice']), 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in sl if s.F['imagename']==imn]
            data_load = pd.concat([data_load, pd.DataFrame(data_im)])
        data_load.reset_index(inplace=True)
        
        # Create dataloader
        dataloader_train = net.model['unet'].get_dataloaders_BF(data_load, batch_size=8)
        NumBatches = math.ceil(len(data_load)/dataloader_train.data_loader.batch_size)
        device='cuda'
        pbar = tqdm(total=NumBatches)
        pbar.set_description("Load patch")
        for b in range(NumBatches):
            pbar.update()
            batch = next(dataloader_train)
            data = batch['data']
            target = batch['target']
            props = batch['properties']
            data = data.to(device, non_blocking=True)
            #out = net.model['unet'].network(data)
            #for i in range(len(out)): out[i] = out[i].detach_().cpu()
            #pred=soft1(out[0])
            
            # Create idxs to sort samples
            idxs=[]
            for p in props:
                for i,s in enumerate(sl):
                    sID = str(s.F['ID']) + '_' + str(s.F['IDP'])
                    propID = str(p['ID']) + '_' + str(p['IDP'])
                    if sID==propID:
                        idxs.append(i)
                        continue
            # Set prediction      
            for i,idx in enumerate(idxs):
                sl[idx].X['XImage'] = data[i:i+1]
                arr = target[0]
                shapet = arr.shape
                labels = list(np.unique(target[0][i:i+1]))
                #print('labels', labels)
                if 2 not in labels:
                    #labels = [x for x in labels if x not in [-1, 2]]
                    #NumChannels = int(np.max(labels)+1)
                    #print('labels123', labels)
                    #print('NumChannels123', NumChannels)
                    arr_new = np.zeros((1, NumChannels, shapet[2], shapet[3]))
                    for c in range(NumChannels):
                        arr_new[0,c:c+1,:,:] = (arr[i]==c)*1
                    sl[idx].Y['XMask'] = torch.tensor(arr_new)
        pbar.close()
        
    @staticmethod
    def predict(sl, opts, net, reshapeOrg=False):

        # sl=action_round
        dataset = opts.CLDataset()
        net.model['unet'].network.eval()
        loss_func = net.model['unet'].loss
        soft1 = torch.nn.Softmax(dim=1)
        
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        fp_nnUNet_preprocessed = os.path.join(opts.fp_nnunet, 'nnUNet_preprocessed')
        plans = load_json(os.path.join(fp_nnUNet_preprocessed, dname, 'nnUNetPlans.json'))
        plans_manager = PlansManager(plans)
        configuration_name = '2d'
        configuration_manager = plans_manager.get_configuration(configuration_name)

        # Order samples by image
        if opts.dim==2:
            imagenames = list(np.unique([s.F['imagename'] for s in sl]))
            data_load = pd.DataFrame()
            for imn in imagenames:
                data_im = [{'imagename': s.F['imagename'], 'slice': int(s.F['slice']), 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in sl if s.F['imagename']==imn]
                data_load = pd.concat([data_load, pd.DataFrame(data_im)])
            data_load.reset_index(inplace=True)
        else:
            imagenames = list(np.unique([s.F['imagename'] for s in sl]))
            data_load = pd.DataFrame()
            for imn in imagenames:
                data_im = [{'imagename': s.F['imagename'], 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in sl if s.F['imagename']==imn]
                data_load = pd.concat([data_load, pd.DataFrame(data_im)])
            data_load.reset_index(inplace=True)
        
        # Create dataloader
        dataloader_train = net.model['unet'].get_dataloaders_BF(data_load, batch_size=None)
        NumBatches = math.ceil(len(data_load)/dataloader_train.data_loader.batch_size)
        patch_size = dataloader_train.data_loader.patch_size
        device='cuda'
        pbar = tqdm(total=NumBatches)
        pbar.set_description("Predict patch")
        for b in range(NumBatches):
            pbar.update()
            batch = next(dataloader_train)
            data = batch['data']
            #target = batch['target']
            props = batch['properties']
            data = data.to(device, non_blocking=True)
            out = net.model['unet'].network(data)
            for i in range(len(out)): out[i] = out[i].detach_().cpu()
            pred=soft1(out[0])
            
            # Create idxs to sort samples
            idxs=[]
            for p in props:
                for i,s in enumerate(sl):
                    sID = str(s.F['ID']) + '_' + str(s.F['IDP'])
                    propID = str(p['ID']) + '_' + str(p['IDP'])
                    if sID==propID:
                        idxs.append(i)
                        continue

            # Set prediction      
            for i,idx in enumerate(idxs):
                #print(i)
                s=sl[idx]
                s.P['XMaskPred'] = pred[i:i+1]
                s.X['XImage'] = data[i:i+1]
                
                if reshapeOrg:
                    if opts.dim==2:
                        # Crop if padding performed
                        s.X['XImage'] = s.X['XImage'][:,:,0:s.F['shapePre'][0],0:s.F['shapePre'][1]]
                        s.P['XMaskPred'] = s.P['XMaskPred'][:,:,0:s.F['shapePre'][0],0:s.F['shapePre'][1]]
                        
                        # 2) Reshape to target resolution
                        original_spacing = np.array([1, s.F['spacingOrg'][0], s.F['spacingOrg'][1]])
                        target_spacing = np.array([1, s.F['spacingPre'][0], s.F['spacingPre'][1]])
                        shape_tar_seg = np.array([s.P['XMaskPred'].shape[1],s.F['bboxUbsOrg'][0]-s.F['bboxLbsOrg'][0], s.F['bboxUbsOrg'][1]-s.F['bboxLbsOrg'][1]])
                        s.P['XMaskPred'] = torch.from_numpy(configuration_manager.resampling_fn_data(s.P['XMaskPred'], shape_tar_seg, original_spacing, target_spacing))
      
                        original_spacing = np.array([1, s.F['spacingOrg'][0], s.F['spacingOrg'][1]])
                        target_spacing = np.array([1, s.F['spacingPre'][0], s.F['spacingPre'][1]])
                        shape_tar_seg = np.array([s.X['XImage'].shape[0],s.F['bboxUbsOrg'][0]-s.F['bboxLbsOrg'][0], s.F['bboxUbsOrg'][1]-s.F['bboxLbsOrg'][1]])
                        s.X['XImage'] = torch.from_numpy(configuration_manager.resampling_fn_data(s.X['XImage'], shape_tar_seg, original_spacing, target_spacing))
                    else:
                        # Crop if padding performed
                        s.X['XImage'] = s.X['XImage'][:,:,0:s.F['shapePre'][0],0:s.F['shapePre'][1]]
                        s.P['XMaskPred'] = s.P['XMaskPred'][:,:,0:s.F['shapePre'][0],0:s.F['shapePre'][1]]
                        
                        # 2) Reshape to target resolution
                        original_spacing = np.array([1, s.F['spacingOrg'][0], s.F['spacingOrg'][1], s.F['spacingOrg'][2]])
                        target_spacing = np.array([1, s.F['spacingPre'][0], s.F['spacingPre'][1], s.F['spacingPre'][2]])
                        shape_tar_seg = np.array([s.F['bboxUbsOrg'][0]-s.F['bboxLbsOrg'][0], s.F['bboxUbsOrg'][1]-s.F['bboxLbsOrg'][1], s.F['bboxUbsOrg'][2]-s.F['bboxLbsOrg'][2]])
                        s.P['XMaskPred'] = torch.from_numpy(configuration_manager.resampling_fn_data(s.P['XMaskPred'][0], shape_tar_seg, original_spacing, target_spacing))
                        s.P['XMaskPred'] = torch.unsqueeze(s.P['XMaskPred'], dim=0)
                        
                        original_spacing = np.array([1, s.F['spacingOrg'][0], s.F['spacingOrg'][1], s.F['spacingOrg'][2]])
                        target_spacing = np.array([1, s.F['spacingPre'][0], s.F['spacingPre'][1], s.F['spacingPre'][2]])
                        shape_tar_seg = np.array([s.F['bboxUbsOrg'][0]-s.F['bboxLbsOrg'][0], s.F['bboxUbsOrg'][1]-s.F['bboxLbsOrg'][1], s.F['bboxUbsOrg'][2]-s.F['bboxLbsOrg'][2]])
                        s.X['XImage'] = torch.from_numpy(configuration_manager.resampling_fn_data(s.X['XImage'][0], shape_tar_seg, original_spacing, target_spacing))
                        s.X['XImage'] = torch.unsqueeze(s.X['XImage'], dim=0)
                           
                #print('t', target[0].shape)
                #sl[idx].Y['XMask'] = torch.concat((1-target[0][i:i+1], target[0][i:i+1]), dim=1)
        pbar.close()


    def __plot_image(self, image, name, title, color=False, save=False, filepath=None, format_im=None, dpi=300):
        if color:
            #print('image1234', image.shape)
            im = np.zeros((image.shape[2], image.shape[3]))
            for c in range(image.shape[1]):
                im = im + (c+1) * image[0,c,:,:]
            plt.imshow(im)
            #plt.imshow(im, cmap='Accent', interpolation='nearest')
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
            #print('pl', pl)
            #print('name', name)
            
            if im is not None:
                print('im', im.shape)
                image = im.data.numpy()
                if pl=='XImage' and len(image.shape)==4:
                    idx = int((image.shape[1]-1)/2)
                    image = image[:,idx:idx+1]
                    self.__plot_image(image, name, title, color=False, filepath=filepath, save=save)
                elif pl=='XImage' and len(image.shape)==5:
                    idx = int((image.shape[2]-1)/2)
                    image = image[:,0,idx:idx+1]
                    self.__plot_image(image, name, title, color=False, filepath=filepath, save=save)
                elif len(image.shape)==5:
                    #print('XMaskPre123')
                    idx = int((image.shape[2]-1)/2)
                    image = image[:,1:2, idx]
                    #print('XMaskPre12356', image.shape)
                    self.__plot_image(image, name, title, color=False, filepath=filepath, save=save)
                else:
                    #image = image[:,1:2]
                    self.__plot_image(image, name, title, color=color, filepath=filepath, save=save)

    @classmethod
    def savePseudo(cls, opts, data, folderDict, fp_pseudo, region='XMaskPred'):
        # cls=ALSegmentCACSSample
        # data=data_samples
        
        soft = torch.nn.Softmax(dim=1)
        
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        fp_nnunet = opts.fp_nnunet
        fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
        fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')
        fp_nnUNet_results = os.path.join(fp_nnunet, 'nnUNet_results')
        fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')
        fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
        
        imagenames = sorted(list(np.unique([s.F['imagename'] for s in data])))
        pbar = tqdm(total=len(imagenames))
        pbar.set_description("Creating labels")
        for i,imn in enumerate(imagenames):
            pbar.update()
            fip = os.path.join(fp_labelsTr, imn)
            ref = CTRef(fip)
            arr = ref.ref()
            arr=arr*0
            #sys.exit()
            for s in data:
                if s.F['imagename']==imn:
                    #if s.F['IDP']==3:
                    #    sys.exit()
                    #if s.F['sliceOrg']==51:
                    #    sys.exit()
                    #print(s.F['slice'])
                    #if s.F['slice']==25:
                    #    print('102')
                    #    sys.exit()
                    # if s.F['slice']==74:
                    #     sys.exit()
                    #sys.exit()
                    for key in s.P: s.P[key] = soft(s.P[key])
                    segP = s.P[region]
                    name_pseudo = s.F['maskname'][0:-7] +'.nrrd' 
                    bboxLbs = s.F['bboxLbsOrg'].copy()
                    bboxUbs = s.F['bboxUbsOrg'].copy()
                    bbC = s.F['bboxUsedForCropping'].copy()
                    # Convert binary to num
                    if opts.dim==2:
                        segP = compute_one_hot_np(segP.cpu().numpy(),1)[0]
                        segP_num = np.zeros((segP.shape[1], segP.shape[2]), dtype=np.int16)
                        for i in range(0, segP.shape[0]):
                            segP_num = segP_num + (segP[i,:,:]*(i))
                    else:
                        segP = compute_one_hot_np_3(segP.cpu().numpy(),1)[0]
                        segP_num = np.zeros((segP.shape[1], segP.shape[2], segP.shape[3]), dtype=np.int16)
                        for i in range(0, segP.shape[0]):
                            segP_num = segP_num + (segP[i,:,:,:]*(i))

                    if opts.dim==2:
                        # !!! Please correct this for bboxLbsOrg and bboxUbsOrg
                        if s.F['IDP']==1:
                            dx = s.F['shapeOrg'][0]-segP_num.shape[0]
                            bboxLbs[0]=bboxLbs[0]+dx
                            bboxUbs[0]=bboxUbs[0]+dx
                        # if s.F['IDP']==2:
                        #     dy = s.F['shapeOrg'][1]-segP_num.shape[1]
                        #     bboxLbs[1]=bboxLbs[1]+dy
                        #     bboxUbs[1]=bboxUbs[1]+dy
                        if s.F['IDP']==3:
                            dx = s.F['shapeOrg'][0]-segP_num.shape[0]
                            #dy = s.F['shapeOrg'][1]-segP_num.shape[1]
                            bboxLbs[0]=bboxLbs[0]+dx
                            bboxUbs[0]=bboxUbs[0]+dx
                            # bboxLbs[1]=bboxLbs[1]+dy
                            # bboxUbs[1]=bboxUbs[1]+dy
                            
                        sl = s.F['slice']
                        arrC = arr[sl,bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1]]
                        #arrC[bboxLbs[0]:bboxUbs[0], bboxLbs[1]:bboxUbs[1]] = segP_num
                        arrC[bboxLbs[1]:bboxUbs[1], bboxLbs[0]:bboxUbs[0]] = segP_num
                        arr[sl, bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1]] = arrC
                    else:
                        arrC = arr[bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1], bbC[2][0]:bbC[2][1]]
                        arrC[bboxLbs[0]:bboxUbs[0], bboxLbs[1]:bboxUbs[1], bboxLbs[2]:bboxUbs[2]] = segP_num
                        arr[bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1], bbC[2][0]:bbC[2][1]] = arrC
                        
            ref.setRef(arr)
            fip_pseudo = os.path.join(fp_pseudo, name_pseudo)
            ref.save(fip_pseudo)
        pbar.close()


                    
class UNetPatch(UNetSample):
    """
    UNetPatch
    """

    def __init__(self):
        UNetSample.__init__(self)
        self.IDP=None
        self.F['props']=None
    
    @staticmethod
    def getPatchbyID(sl, ID, IDP):
        for s in sl:
            if s.F['ID']==ID and s.F['IDP']==IDP:
                return s
    @staticmethod
    def loss_fisher(net, idx_output=None, idx_class=None):
        """ Fisher loss
        """

        def func(Xinput):
            loss=[]
            if idx_output is None:
                pred = net.model['unet'].network(Xinput)
                for idxOut in range(0,len(pred)):
                    pred_weak_bin_log = torch.log_softmax(pred[idxOut], dim=1)
                    pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
                    n_output = pred_weak_bin_prop.shape[1]
                    pseudo_bin = torch.nn.functional.one_hot(torch.argmax(pred_weak_bin_prop, dim=1, keepdims=False),n_output).permute(0, 3, 1, 2)
                    #print('pseudo_bin123', pseudo_bin.shape)
                    #sys.exit()
                    if idx_class is None:
                        for c in range(pred_weak_bin_prop.shape[1]):
                            loss.append(torch.mean(pred_weak_bin_log[:,c] * pseudo_bin[:,c]))
                    else:
                        loss.append(torch.mean(pred_weak_bin_log[:,idx_class] * pseudo_bin[:,idx_class]))
                
                loss = torch.unsqueeze(torch.hstack(loss), 0)
            else:
                pred = net.model['unet'].network(Xinput)
                pred_weak_bin_log = torch.log_softmax(pred[idx_output], dim=1)
                pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
                for c in range(pred_weak_bin_prop.shape[1]):
                    loss.append(torch.mean(pred_weak_bin_log[:,c] * pred_weak_bin_prop[:,c]))
                loss = torch.unsqueeze(torch.hstack(loss), 0)
            return loss
        return func
    
    @staticmethod
    def samplesToloader(data, opts, net, reshapeOrg=False, dtype=torch.FloatTensor, batch_size=1, verbose=False):
        
        #dtype=torch.DoubleTensor
        
        # sl=action_round
        #dataset = opts.CLDataset()
        net.model['unet'].network.eval()
        #loss_func = net.model['unet'].loss
        soft1 = torch.nn.Softmax(dim=1)
        
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
        dataloader_train = net.model['unet'].get_dataloaders_BF(data_load, batch_size=None)
        NumBatches = math.ceil(len(data_load)/dataloader_train.data_loader.batch_size)
        #patch_size = dataloader_train.data_loader.patch_size
        #batch_size = dataloader_train.data_loader.batch_size
        
        device='cuda'
        if verbose:
            pbar = tqdm(total=NumBatches)
            pbar.set_description("Create data loader")
        for b in range(NumBatches):
            if verbose: pbar.update()
            batch = next(dataloader_train)
            datab = batch['data']
            #target = batch['target'][0]
            props = batch['properties']
            datab = datab.to(device, non_blocking=True)
            out = net.model['unet'].network(datab)
            for i in range(len(out)): out[i] = out[i].detach_().cpu()
            pred=soft1(out[0])
            target = torch.argmax(pred, dim=1, keepdim=True)
            
            #target[target==label_ignore]
            if opts.dim==2:
                tar = torch.zeros((target.shape[0], pred.shape[1], target.shape[2], target.shape[3]))
            else:
                tar = torch.zeros((target.shape[0], pred.shape[1], target.shape[2], target.shape[3], target.shape[4]))
            for c in range(0,tar.shape[1]):
                tar[:,c]=target[:,0]==c
                
            # Create idxs to sort samples
            idxs=[]
            for p in props:
                for i,s in enumerate(data):
                    sID = str(s.F['ID']) + '_' + str(s.F['IDP'])
                    propID = str(p['ID']) + '_' + str(p['IDP'])
                    if sID==propID:
                        idxs.append(i)
                        continue
                    
            # for s in data:
            #     print(s.F['ID'])

            # Set prediction      
            for i,idx in enumerate(idxs):
                #print(i)
                s=data[idx]
                s.Y['XMask'] = tar[i:i+1]
                s.P['XMaskPred'] = pred[i:i+1]
                s.X['XImage'] = datab[i:i+1]
                
        data_leader = []
        for s in data:
            #image = s.X['XImage']
            data_leader.append([s.X['XImage'][0].type(dtype).cuda(), s.Y['XMask'][0].type(dtype).cuda()])
        loader= torch.utils.data.DataLoader(dataset=data_leader, batch_size=batch_size, shuffle=False, pin_memory=False)
        
        return loader
        
        # data_leader = []
        # for s in data:
        #     image = s.image_norm()
        #     data_leader.append([image[0].type(dtype), s.mask.type(dtype), s.weight.type(dtype)])
        # loader= torch.utils.data.DataLoader(dataset=data_leader, batch_size=batch_size, shuffle=False, pin_memory=False)
        # return loader

class UNetALDataset(SALDataset):
    """
    UNetALDataset
    """
    
    name = None
    data = []
    loadDataFlag = False

    def __init__(self, name):
        SALDataset.__init__(self, name)

    def delete(self, data):
        ID_list = [(int(s.ID), int(s.F['IDP'])) for s in data]
        data_new=[]
        for s in self.data:
            if (int(s.ID), int(s.F['IDP'])) not in ID_list:
                data_new.append(s)
        self.data = data_new
        

        
        
class UNetManager(ALManager):
    
    def __init__(self, fp_dataset):
        ALManager.__init__(self, fp_dataset)
        self.fp_dataset = fp_dataset
        for key in self.datasets.keys():
            self.datasets[key] = UNetALDataset(key)
        self.datasets['action']=UNetALDataset('action')
        self.datasets['action_round']=UNetALDataset('action_round')

    def init_datasets(self, opts, folderDict, exclude=['unlabeled']):
        
        # self = man
        #hdf5filepath = os.path.join(opts.fp_active, 'hdf5_all.hdf5')
        hdf5filepath = os.path.join(opts.fp_active, 'hdf5_all_'+str(opts.dim)+'D.hdf5')
        
        if opts.dim==2:
            # Create DataAccess
            dataaccess = DataAccess()
            data=dataaccess.read_dict(hdf5filepath, keys_select=['ID', 'imagename', 'maskname','slice', 'TRAIN', 'VALID', 'TEST'])
    
            imagenames = list(data['F']['imagename'])
            self.datasets['train'].data=[]
            self.datasets['query'].data=[]
            self.datasets['valid'].data=[]
            self.datasets['test'].data=[]
            
            # Init dataset
            for i in range(len(imagenames)):
                s = UNetSample()
                s.name = str(data['ID'][i])
                s.ID = int(data['ID'][i])
                s.imagename = data['F']['imagename'][i]
                s.maskname = data['F']['maskname'][i]
                s.F = dict({k:data['F'][k][i] for k in data['F']})
                if s.F['TRAIN']:
                    self.datasets['query'].data.append(s)
                elif s.F['VALID']:
                    #self.datasets['valid'].data.append(s)
                    # !!! Validations et is split by nnUNet engine
                    self.datasets['query'].data.append(s)
                elif s.F['TEST']:
                    self.datasets['test'].data.append(s)
        else:
            # Create DataAccess
            dataaccess = DataAccess()
            data=dataaccess.read_dict(hdf5filepath, keys_select=['ID', 'imagename', 'maskname', 'TRAIN', 'VALID', 'TEST',
                                                                  'height', 'width', 'depth'])

            #data=dataaccess.read_dict(hdf5filepath, keys_select=['ID', 'imagename', 'maskname','height', 'width', 'depth'])
            
            imagenames = list(data['F']['imagename'])
            self.datasets['train'].data=[]
            self.datasets['query'].data=[]
            self.datasets['valid'].data=[]
            self.datasets['test'].data=[]
            
            # Init dataset
            for i in range(len(imagenames)):
                s = UNetSample()
                s.name = str(data['ID'][i])
                s.ID = int(data['ID'][i])
                s.imagename = data['F']['imagename'][i]
                s.maskname = data['F']['maskname'][i]
                s.F = dict({k:data['F'][k][i] for k in data['F']})
                if s.F['TRAIN']:
                    self.datasets['query'].data.append(s)
                elif s.F['VALID']:
                    #self.datasets['valid'].data.append(s)
                    # !!! Validations et is split by nnUNet engine
                    self.datasets['query'].data.append(s)
                elif s.F['TEST']:
                    self.datasets['test'].data.append(s)

    def init_dataset_mds(self, opts, man, folderDict):
    
        NumNewSamples = opts.AL_steps[0]
        dataset = opts.CLDataset()
        
        # Create raw dataset
        #datasetID = opts.dataset_dict[opts.dataset]
        #strategyID = list(opts.strategy_dict.keys()).index('INIT')
        #ID=100+10*datasetID+strategyID
        opts.dataset_name_or_id = opts.dataset_name_or_id[0:-1]+'0'
        name = 'Task' + str(opts.dataset_dict[opts.dataset]).zfill(2) + '_' + opts.dataset
        cmd = 'nnUNetv2_convert_MSD_dataset -i ' + opts.fp_mds +'/'+ name+' -overwrite_id ' + opts.dataset_name_or_id
        print('cmd123', cmd)
        returned_value = subprocess.call(cmd, shell=True)
    
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        fp_nnunet = opts.fp_nnunet
        fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')
        fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
        fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
        fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
        imagenames_train = list(np.unique([s.F['imagename'] for s in man.datasets['train'].data]))
        imagenames_valid = list(np.unique([s.F['imagename'] for s in man.datasets['valid'].data]))
        imagenames_query = list(np.unique([s.F['imagename'] for s in man.datasets['query'].data]))
        imagenames_action_round = list(np.unique([s.F['imagename'] for s in man.datasets['action_round'].data]))
        imagenames_test = list(np.unique([s.F['imagename'] for s in man.datasets['test'].data]))
        imagenames = sorted(list(np.unique(imagenames_train + imagenames_valid + imagenames_query + imagenames_action_round)))
    
        # Add ignore label
        dataset_json = load_json(os.path.join(fp_nnUNet_raw, dname, 'dataset.json'))
        if 'ignore' not in dataset_json['labels']:
            dataset_json['labels']['ignore']=int(np.max(list(dataset_json['labels'].values()))+1)
            save_json(dataset_json, os.path.join(fp_nnUNet_raw, dname, 'dataset.json'))
    
        # Create empty labels train
        label_ignore = load_json(os.path.join(fp_nnUNet_raw, dname, 'dataset.json'))['labels']['ignore']
        name = 'Task' + str(opts.dataset_dict[opts.dataset]).zfill(2) + '_' + opts.dataset
        fp_ref = os.path.join(opts.fp_mds, name, 'labelsTr')
        pbar = tqdm(total=len(imagenames))
        pbar.set_description("Creating labels train")
        for i,imn in enumerate(imagenames):
            pbar.update(1)
            fip = os.path.join(fp_ref, imn)
            fipout = os.path.join(fp_labelsTr, imn)
            image = CTImage(fip)
            shape = image.image().shape
            arr = np.ones(shape)*label_ignore
            ref = CTRef()
            ref.setRef(arr)
            ref.copyInformationFrom(image)
            ref.save(fipout)
        pbar.close()
    
        # Preprocessing raw data and split data
        cmd = "nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean"
        returned_value = subprocess.call(cmd, shell=True)
    
        # Convert samples to patches
        keys = ['train', 'valid', 'query', 'action', 'action_round']
        for key in keys:
            data = man.datasets[key].data
            if opts.dim==2:
               # data = opts.CLSample.samplesTopatches(man, opts, folderDict, data)
               data = opts.CLSample.samplesTopatches3D(man, opts, folderDict, data)
            else:
                data = opts.CLSample.samplesTopatches3D(man, opts, folderDict, data)
            man.datasets[key].data = data
        man.save(save_dict=dict(), save_class=opts.CLPatch, hdf5=True)
        
        # Copy split file
        fip_split_nnunet = os.path.join(fp_nnUNet_preprocessed, dname, 'splits_final.json')
        shutil.copyfile(fip_split_nnunet, opts.fip_split)
        
        # Split validation data
        data_split = json.load(open(opts.fip_split))
        images_valid = data_split[opts.fold]['val']
        data_valid=[]
        data_query=[]
        for s in man.datasets['query'].data:
            if s.F['imagename'].split('.')[0] in images_valid:
                data_valid.append(s)
            else:
                data_query.append(s)
        man.datasets['valid'].data = data_valid
        man.datasets['query'].data = data_query
        
      
        # Create initial training set from labeling
        action_round = man.getRandom(dataset='query', NumSamples=NumNewSamples, remove=True)
    
        ################################
        ###### Automatic LABELING ######
        ################################
    
        # Fill valid label
        name = 'Task' + str(opts.dataset_dict[opts.dataset]).zfill(2) + '_' + opts.dataset
        fp_ref = os.path.join(opts.fp_mds, name, 'labelsTr')
        pbar = tqdm(total=len(imagenames))
        pbar.set_description("Creating labels train")
        for i,imn in enumerate(imagenames):
            pbar.update(1)
            if imn.split('.')[0] in images_valid:
                fip = os.path.join(fp_ref, imn)
                fipout = os.path.join(fp_labelsTr, imn)
                ref = CTRef(fip)
                ref.save(fipout)
        pbar.close()
    
        # Fill action_round
        imagenames = sorted(list(np.unique([s.F['imagename'] for s in action_round])))
        if opts.dim==2:
            #dataset.load_dataset_hdf5(opts, folderDict, action_round)
            dataset.load_dataset_mds3D(man, opts, folderDict, action_round)
        else:
            #dataset.load_dataset_hdf53D(man, opts, folderDict, action_round)
            dataset.load_dataset_mds3D(man, opts, folderDict, action_round)
            
        pbar = tqdm(total=len(imagenames))
        pbar.set_description("Creating labels")
        for i,imn in enumerate(imagenames):
            pbar.update()
            #fip = os.path.join(fp_labelsTr, imn.split('.')[0]) + '.nrrd'
            fip = os.path.join(fp_labelsTr, imn)
            ref = CTRef(fip)
            arr = ref.ref()
            #sys.exit()
            for s in action_round:
                if s.F['imagename']==imn:
                    #sys.exit()
                    #print('bboxLbs', s.F['bboxLbs'])
                    #print('bboxUbs', s.F['bboxUbs'])
                    segP = s.Y['XMask'][0]
                    # Convert binary to num
                    if opts.dim==2:
                        segP_num = np.zeros((segP.shape[1], segP.shape[2]), dtype=np.int16)
                        for i in range(0, segP.shape[0]):
                            segP_num = segP_num + (segP[i,:,:]*(i)).numpy()
                        bboxLbs = s.F['bboxLbsOrg']
                        bboxUbs = s.F['bboxUbsOrg']
                    else:
                        segP_num = np.zeros((segP.shape[1], segP.shape[2], segP.shape[3]), dtype=np.int16)
                        for i in range(0, segP.shape[0]):
                            segP_num = segP_num + (segP[i,:,:,:]*(i)).numpy()
                        bboxLbs = s.F['bboxLbsOrg']
                        bboxUbs = s.F['bboxUbsOrg']
                    
                    bbC = s.F['bboxUsedForCropping']
                    if opts.dim==2:
                        sl = s.F['sliceOrg']
                        arrC = arr[sl,bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1]]
                        arrC[bboxLbs[0]:bboxUbs[0], bboxLbs[1]:bboxUbs[1]] = segP_num
                        arr[sl, bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1]] = arrC
                    else:
                        arrC = arr[bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1], bbC[2][0]:bbC[2][1]]
                        arrC[bboxLbs[0]:bboxUbs[0], bboxLbs[1]:bboxUbs[1], bboxLbs[2]:bboxUbs[2]] = segP_num
                        arr[bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1], bbC[2][0]:bbC[2][1]] = arrC
                    #arr[bboxLbs[0]:bboxUbs[0], bboxLbs[1]:bboxUbs[1], bboxLbs[2]:bboxUbs[2]] = segP_num
            ref.setRef(arr)
            ref.save(fip)
        pbar.close()
        
        # Update datasets
        man.datasets['action'].data = man.datasets['action'].data + action_round
        man.datasets['train'].data = man.datasets['train'].data + action_round
        man.save(include=['action'], save_dict={'X': False, 'Y': False, 'F': True}, save_class=opts.CLPatch, hdf5=True)
        man.save(include=['train', 'valid', 'query', 'test'], save_dict={}, save_class=opts.CLPatch, hdf5=True)


    def createALFolderpath(self, opts, fip_split=None, method=None, NewVersion=False, VersionUse=None, copy_prev_labelsTr=False):
        fp_active=opts.fp_active
        fp_images=opts.fp_images
        fp_references_org=opts.fp_references_org
        folderDict = super().createALFolderpath(fp_active=fp_active, fip_split=fip_split, fp_images=fp_images, fp_references_org=fp_references_org, method=method, NewVersion=NewVersion, VersionUse=None)
        folderDict['fip_hdf5_all'] = os.path.join(opts.fp_active, 'hdf5_all_'+str(opts.dim)+'D.hdf5')
        
        print('copy_init123', folderDict['copy_init'])
        
        # Copy nnunet dataset
        if folderDict['copy_init']:
            dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
            fp_nnunet = opts.fp_nnunet
            fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
            fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')
            fp_nnUNet_results = os.path.join(fp_nnunet, 'nnUNet_results')
            dnameInit = 'Dataset' + '1'+opts.dataset_name_or_id[1:2]+'0' + '_' + opts.dataset
            copy_tree(os.path.join(fp_nnUNet_raw, dnameInit), os.path.join(fp_nnUNet_raw, dname))
            copy_tree(os.path.join(fp_nnUNet_preprocessed, dnameInit), os.path.join(fp_nnUNet_preprocessed, dname))
            copy_tree(os.path.join(fp_nnUNet_results, dnameInit), os.path.join(fp_nnUNet_results, dname))
        
            # Change dataset_name in nnUUnetPlans.json
            plans_file = os.path.join(nnUNet_preprocessed, dname, 'nnUNetPlans.json')
            plans = load_json(plans_file)
            plans['dataset_name'] = dname
            save_json(plans, plans_file)
            
        # Reset labelsTr from prvious round
        if folderDict['version']>2 and copy_prev_labelsTr:
            dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
            fp_nnunet = opts.fp_nnunet
            fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
            fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
            fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
            fp_labelsTr_prev = os.path.join(folderDict['modelpath_prev'], 'labelsTr')
            if os.path.exists(fp_labelsTr):
                shutil.rmtree(fp_labelsTr)
            shutil.copytree(fp_labelsTr_prev, fp_labelsTr)
            
        # Update path for nnUNet
        #ALFolderpathMethod = os.path.join(fp_active, method)
        #if not os.path.isdir(ALFolderpathMethod) and not method=='INIT':
        #folderDict['modelpath'] = 
        return folderDict
        
    def load_model(self, opts, folderDict, previous=False, const_dropout=False, load_weights=True):
        # self=man
        
        #dataset_name_or_id = '105'
        dataset_name_or_id = opts.dataset_name_or_id
        #configuration='2d'
        configuration = opts.configuration
        #fold=0
        #tr='nnUNetTrainer'
        tr=opts.nnUNetTrainer
        #p='nnUNetPlans'
        p=opts.plans_identifier
        use_compressed='False'
        from nnunet.nnunetv2.run.run_training import get_trainer_from_args
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, opts.fold, tr, p, use_compressed)
        

        #fip_checkpoint='/mnt/HHD/data/UNetAL/LITS/AL/RANDOM/RANDOM_V01/model/fold_0/checkpoint_best.pth'
        #nnunet_trainer.load_checkpoint(fip_checkpoint)

        settingsfilepath_model = os.path.join(folderDict['modelpath'], 'LITS.yml')
        net = UNetSegAL(settingsfilepath_model, overwrite=True)
        net.model = {'unet': nnunet_trainer}
        

        
        #net.props['fip_checkpoint']='/mnt/HHD/data/UNetAL/LITS/AL/RANDOM/RANDOM_V01/model'
        if previous:
            net.props['fip_checkpoint']=folderDict['modelpath_prev']
        else:
            net.props['fip_checkpoint']=folderDict['modelpath']
            
        #print('path123', os.path.join(net.props['fip_checkpoint'], opts.nnUNetResults, 'fold_0', 'checkpoint_best.pth'))
            
        if load_weights:
            net.model['unet'].load_checkpoint(os.path.join(net.props['fip_checkpoint'], opts.nnUNetResults, 'fold_'+str(opts.fold), 'checkpoint_best.pth'))
        else:
            net.model['unet'].initialize()

        return net
        
        
class UNetDataframe(DataframeBaseModel):
    """ UNetDataframe for ct dataset"""
    
    def __init__(self, settingsfilepath, overwrite=False, folderpathData=None, dataname=''):

        self.generator = {}
        self.Xlabel = dict({'image': 0})
        self.props = defaultdict(lambda: None,
            Input_size = (1, 512, 512),
            NumChannelsIn = 1,
            NumChannelsOut = 2,
            batch_size = 8,
            oversampling = True,
            dataorder = {'WIDTH': 2, 'HEIGHT': 3, 'CHANNEL': 1, 'SAMPLES': 0},
            Xmin = -2000,
            Xmax = 1300,
            SLL = False
        )
        DataframeBaseModel.__init__(self, self.props, settingsfilepath, overwrite)
        
    def ttaug(self, Xaug, Yaug, X_idx=['XImage'], Y_idx=['XMask'], aug=None):
        
        for idx in X_idx:
            Xaug[idx] = torch.tensor(Xaug[idx])
        for idx in Y_idx:
            Yaug[idx] = torch.tensor(Yaug[idx])
            
        batch_size = Xaug[X_idx[0]].shape[0]
        if aug is None:
            aug = np.random.choice(8, batch_size)
        for i in range(batch_size):
            if aug[i]==0:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 0, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 0, dims=(1,2)) 
            if aug[i]==1:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 1, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 1, dims=(1,2)) 
            if aug[i]==2:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 2, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 2, dims=(1,2)) 
            if aug[i]==3:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 3, dims=(1,2))  
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 3, dims=(1,2)) 
            if aug[i]==4:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.flip(Xaug[idx][i], dims=(2,))
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 0, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.flip(Yaug[idx][i], dims=(2,))
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 0, dims=(1,2)) 
            if aug[i]==5:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.flip(Xaug[idx][i], dims=(2,))
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 1, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.flip(Yaug[idx][i], dims=(2,))
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 1, dims=(1,2)) 
            if aug[i]==6:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.flip(Xaug[idx][i], dims=(2,))
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 2, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.flip(Yaug[idx][i], dims=(2,))
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 2, dims=(1,2)) 
            if aug[i]==7:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.flip(Xaug[idx][i], dims=(2,))
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 3, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.flip(Yaug[idx][i], dims=(2,))
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 3, dims=(1,2)) 

        for idx in X_idx:
            Xaug[idx] = Xaug[idx].numpy()
        for idx in Y_idx:
            Yaug[idx] = Yaug[idx].numpy()
            
        Yaug['aug'] = aug    
        return  Xaug, Yaug
        
    def ttaugInv(self, Xaug, Yaug, X_idx=['XImage'], Y_idx=['XMask']):
        batch_size = Xaug[X_idx[0]].shape[0]
        aug = Yaug['aug']
        
        for idx in X_idx:
            Xaug[idx] = torch.tensor(Xaug[idx])
        for idx in Y_idx:
            Yaug[idx] = torch.tensor(Yaug[idx])
            
        for i in range(batch_size):
            if aug[i]==0:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 0, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 0, dims=(1,2)) 
            if aug[i]==1:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], -1, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], -1, dims=(1,2))  
            if aug[i]==2:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], -2, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], -2, dims=(1,2))  
            if aug[i]==3:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], -3, dims=(1,2)) 
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], -3, dims=(1,2)) 
            if aug[i]==4:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], 0, dims=(1,2))  
                    Xaug[idx][i] = torch.flip(Xaug[idx][i], dims=(2,))
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], 0, dims=(1,2)) 
                    Yaug[idx][i] = torch.flip(Yaug[idx][i], dims=(2,))
            if aug[i]==5:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], -1, dims=(1,2))  
                    Xaug[idx][i] = torch.flip(Xaug[idx][i], dims=(2,))
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], -1, dims=(1,2))  
                    Yaug[idx][i] = torch.flip(Yaug[idx][i], dims=(2,))
            if aug[i]==6:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], -2, dims=(1,2))  
                    Xaug[idx][i] = torch.flip(Xaug[idx][i], dims=(2,))
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], -2, dims=(1,2)) 
                    Yaug[idx][i] = torch.flip(Yaug[idx][i], dims=(2,))
            if aug[i]==7:
                for idx in X_idx: 
                    Xaug[idx][i] = torch.rot90(Xaug[idx][i], -3, dims=(1,2)) 
                    Xaug[idx][i] = torch.flip(Xaug[idx][i], dims=(2,))
                for idx in Y_idx: 
                    Yaug[idx][i] = torch.rot90(Yaug[idx][i], -3, dims=(1,2)) 
                    Yaug[idx][i] = torch.flip(Yaug[idx][i], dims=(2,))
                    
        for idx in X_idx:
            Xaug[idx] = Xaug[idx].numpy()
        for idx in Y_idx:
            Yaug[idx] = Yaug[idx].numpy()
            
        return  Xaug, Yaug
    
    def createIterativeGeneratorTrain(self, generatorname, fip_hdf5_list, NumSamples=100, batch_size=None, shuffle=True):
        
        # self=dataframe
        
        #sampling = 'replay_sampling'
        sampling = 'random_sampling'
        
        # Reset generator
        self.generator['train'] = None
        dataaccess = DataAccess()

        # Read ID of trainig and query set
        ID = dataaccess.get_ID_dataset([fip_hdf5_list['train']], dataset=None)
        #IDaction = list(dataaccess.get_ID_dataset([fip_hdf5_list['action']], dataset=None))
        IDsel = list(ID)
        fip_data = [fip_hdf5_list['action']] + fip_hdf5_list['fip_action_previous'] + [fip_hdf5_list['train']] + [fip_hdf5_list['hdf5_all']]

        # Read data
        #data = dataaccess.read_dict_list(fip_data, ID=IDsel, keys_select=['ID', 'XImage', 'XCandidate', 'XRegion', 'XMain', 'XSegment', 'MAINEXIST', 'REGIONEXIST'], dtype=np.ndarray)
        data = dataaccess.read_dict_list(fip_data, ID=IDsel, keys_select=['ID', 'XImage', 'XMask'], dtype=np.ndarray)
        
        
        if self.props['SSL']:
            print('SSL')
            
            #self.model['unet'].eval()
            self.teacher.eval()
            NumQ = int(len(IDsel)*0.25)
            soft = torch.nn.Softmax(dim=1)
            Xmin=self.props['Xmin']
            Xmax=self.props['Xmax']
            IDq = dataaccess.get_ID_dataset([fip_hdf5_list['query']], dataset=None)
            random.shuffle(IDq)
            IDqsel = np.random.choice(IDq, size=NumQ, replace=False)
            dataq = dataaccess.read_dict(fip_hdf5_list['hdf5_all'], ID=list(IDqsel), keys_select=['ID', 'XImage', 'XMask'], dtype=np.ndarray)
            Ximageq = dataq['X']['XImage']
            Ximageq = (Ximageq - Xmin) / (Xmax - Xmin)
            for j in range(len(Ximageq)):
                Xim = torch.FloatTensor(Ximageq[j:j+1]).to('cuda')
                #Xout = self.model['unet'](Xim)  
                Xout = self.teacher(Xim)  
                pred = soft(Xout).cpu()
                pred_mask = (torch.max(pred, dim=1, keepdim=True)[0]>0.95)*1
                pred_mask = pred_mask.repeat(1, 2, 1, 1)
                mask = (pred[:,1:,:,:]>0.5)*1
                mask = torch.concat((1-mask, mask), dim=1)
                mask = mask * pred_mask
                dataq['Y']['XMask'][j] = mask
                
            # Append data
            data['F']['ID'] = np.concatenate((data['F']['ID'], dataq['F']['ID']))
            data['X']['XImage'] = np.concatenate((data['X']['XImage'], dataq['X']['XImage']))
            data['Y']['XMask'] = np.concatenate((data['Y']['XMask'], dataq['Y']['XMask']))
            self.teacher.train()
            #self.model['unet'].train()
            
                
        # Update data dict for further processing
        data['X']['ID'] = data['F']['ID']
        self.NumSamplesTrain = data['X']['XImage'].shape[0]

        # Append data augmentation
        self.augmenter_train.order_data=[]
        #self.augmenter_train.order_data.append(self.augmenter_train.shuffle_aug())
        self.augmenter_train.order_data.append(self.augmenter_train.normalize_minmax_aug(Xmin=self.props['Xmin'], Xmax=self.props['Xmax'], X_idx=['XImage'], Y_idx=[], clip=True))
        #self.augmenter.order_data.append(self.augmenter.noise_aug(mean=0.0, std=0.01, X_idx=[Xlabel['image']], mode='normal'))

        # Append batch 
        X_border_image = [0.0 for k in range(self.props['Input_size'][2])]
        Y_border_out = [1.0] + [0.0 for k in range(self.props['NumChannelsOut']-1)]
        self.augmenter_train.order_batch=[]
        self.augmenter_train.order_batch.append(self.augmenter_train.translate_aug(X_idx=['XImage'], 
                                                                        Y_idx=['XMask'], 
                                                                        X_border=[X_border_image], 
                                                                        Y_border=[Y_border_out], 
                                                                        x_std=10))
        
        # Apply augmentation
        #aug = np.zeros((8))
        aug = None
        params={}
        #params['prop']=prop
        X, Y = self.augmenter_train.apply_data(data['X'], data['Y'])
        #Xq, Yq = self.augmenter_train.apply_data(dataq['X'], dataq['Y'])
        #print('X123', X)
        def generator_train():
            while True:
                Xaug, Yaug = self.augmenter_train.selectBatch(X, Y, self.props['batch_size'], method=sampling, params=params)
                #Xaugq, Yaugq = self.augmenter_train.selectBatch(Xq, Yq, self.props['batch_size'], method=sampling, params=params)
                #for k in Xaug: Xaug[k] = torch.concat((Xaug[k], Xaugq[k]))
                #for k in Yaug: Yaug[k] = torch.concat((Yaug[k], Yaugq[k]))
                    
                #Xaug, Yaug = self.augmenter_train.selectBatch(X, Y, self.props['batch_size'], method='random_sampling', params=params)
                Xaug, Yaug = self.augmenter_train.apply_batch(Xaug, Yaug)
                #Xaug, Yaug = self.ttaug(Xaug, Yaug, X_idx=['XImage'], Y_idx=['XMask'], aug=aug)
                #Xaug['XImageQ'] = Xaugq['XImage']
                #Yaug['XMaskQ'] = Yaugq['XMask']
                yield (Xaug, Yaug)
                
        self.generator['train'] = generator_train()
            


    def createIterativeGeneratorValid(self, generatorname, fip_hdf5_list, NumSamples=100, batch_size=None, shuffle=True):

        # Reset generator
        self.generator['valid'] = None
        
        dataaccess = DataAccess()
        ID = list(dataaccess.get_ID_dataset([fip_hdf5_list['valid']], dataset='valid'))
        #ID = list(dataaccess.get_ID_dataset([fip_hdf5_list['train']], dataset='train'))

        # Select random subset
        NumSamples = min(NumSamples, len(ID))
        random.shuffle(ID)
        IDsel = ID[0:NumSamples]
        data = dataaccess.read_dict(fip_hdf5_list['hdf5_all'], ID=IDsel, keys_select=['ID', 'XImage', 'XMask'])
        data['X']['ID'] = data['F']['ID']      
        self.NumSamplesValid = data['X']['XImage'].shape[0]

        # Append data augmentation
        self.augmenter_valid.order_data=[]
        self.augmenter_valid.order_data.append(self.augmenter_valid.shuffle_aug())
        self.augmenter_valid.order_data.append(self.augmenter_valid.normalize_minmax_aug(Xmin=self.props['Xmin'], Xmax=self.props['Xmax'], X_idx=['XImage'], Y_idx=[], clip=True))
        #self.augmenter.order_data.append(self.augmenter.noise_aug(mean=0.0, std=0.01, X_idx=[Xlabel['image']], mode='normal'))
        
        # Append batch 
        self.augmenter_valid.order_batch=[]
       
        # Apply augmentation
        params={}
        X, Y = self.augmenter_valid.apply_data(data['X'], data['Y'])
        
        def generator_valid():
            while True:
                Xaug, Yaug = self.augmenter_valid.selectBatch(X, Y, self.props['batch_size'], method='random_sampling', params=params)
                Xaug, Yaug  = self.augmenter_valid.apply_batch(Xaug, Yaug )
                yield (Xaug, Yaug)
                
        self.generator['valid'] = generator_valid()
            

    def createIterativeGeneratorTest(self, generatorname, fip_hdf5_list, NumSamples=100, batch_size=None, shuffle=True):

        # Reset generator
        self.generator['test'] = None
        
        dataaccess = DataAccess()
        ID = list(dataaccess.get_ID_dataset([fip_hdf5_list['test']], dataset='test'))

        # Select random subset
        NumSamples = min(NumSamples, len(ID))
        random.shuffle(ID)
        IDsel = ID[0:NumSamples]
        data = dataaccess.read_dict(fip_hdf5_list['hdf5_all'], ID=IDsel, keys_select=['ID', 'XImage', 'XMask'])
        data['X']['ID'] = data['F']['ID']

        self.NumSamplesValid = data['X']['XImage'].shape[0]

        
        # Append data augmentation
        self.augmenter_test.order_data=[]
        self.augmenter_test.order_data.append(self.augmenter_valid.shuffle_aug())
        self.augmenter_test.order_data.append(self.augmenter_valid.normalize_minmax_aug(Xmin=self.props['Xmin'], Xmax=self.props['Xmax'], X_idx=['XImage'], Y_idx=[], clip=True))
        #self.augmenter.order_data.append(self.augmenter.noise_aug(mean=0.0, std=0.01, X_idx=[Xlabel['image']], mode='normal'))
        
        # Append batch 
        self.augmenter_test.order_batch=[]
       
        # Apply augmentation
        params={}
        X, Y = self.augmenter_test.apply_data(data['X'], data['Y'])
        
        def generator_test():
            while True:
                Xaug, Yaug = self.augmenter_test.selectBatch(X, Y, self.props['batch_size'], method='random_sampling', params=params)
                Xaug, Yaug  = self.augmenter_test.apply_batch(Xaug, Yaug )
                yield (Xaug, Yaug)
                
        self.generator['valid'] = generator_test()
     

class UNetDataset(ALDataset):
    def __init__(self, name=''):
        ALDataset.__init__(self, name, NumChannelsIn=1, NumChannelsOut=2, Tasknames='XMask')

    def update_samples_from_action(self, opts, man, fp_manual, data_samples):
        # self=dataset
        # data_samples=action_round
        
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
        actionlist = ALAction.load(fip_actionlist) 

        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        fp_nnunet = opts.fp_nnunet
        fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
        fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
        fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')

        #fip_refine = None
        pbar = tqdm(total=len(data_samples))
        pbar.set_description("Loading image " )
        for a in actionlist:
            pbar.update()
            if a.status=='solved':
                s = opts.CLSample.getSampleByIDAndIDP(data_samples, a.id, a.idp)
                ref_refine = CTRef(a.fip_refine)
                arrR = ref_refine.ref()
                arrR[arrR==0]=1
                arrR=arrR-1
                fip_label = os.path.join(fp_labelsTr, s.F['imagename'])
                ref_label = CTRef(fip_label)
                arrL = ref_label.ref()
                #arrL[s.F['slice'], s.F['bboxLbsOrg'][0]:s.F['bboxUbsOrg'][0], s.F['bboxLbsOrg'][1]:s.F['bboxUbsOrg'][1]] = arrR[s.F['slice'], s.F['bboxLbsOrg'][0]:s.F['bboxUbsOrg'][0], s.F['bboxLbsOrg'][1]:s.F['bboxUbsOrg'][1]]
                arrL[s.F['sliceOrg'], s.F['bboxLbsOrg'][0]:s.F['bboxUbsOrg'][0], s.F['bboxLbsOrg'][1]:s.F['bboxUbsOrg'][1]] = arrR[s.F['sliceOrg'], s.F['bboxLbsOrg'][0]:s.F['bboxUbsOrg'][0], s.F['bboxLbsOrg'][1]:s.F['bboxUbsOrg'][1]]
                ref_label.setRef(arrL)
                ref_label.save(fip_label)
        pbar.close()
        
    def load_dataset_hdf53D(self, man, opts, folderDict, data, debug=True):
        # self = dataset
        # data=data_update
        # debug=True

        # Create DataAccess
        dataaccess = DataAccess()
        hdf5filepath = folderDict['fip_hdf5_all']
        # Get index
        #IDs = [s.ID for s in data]
        
        #fp_nnunet = os.path.join(splitFolderPath(folderDict['fip_hdf5_all'])[0], 'nnunet')
        fp_nnunet = opts.fp_nnunet
        fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')

        #dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset_name
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        plans = load_json(os.path.join(fp_nnUNet_preprocessed, dname, 'nnUNetPlans.json'))
        plans_manager = PlansManager(plans)
        #configuration_name = '2d'
        configuration_manager = plans_manager.get_configuration(opts.configuration)
        
        # datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, keys_select=['XImage'], debug=debug)
        # datadict = dataaccess.read_dict(hdf5filepath, keys_select=['F'], debug=debug)
        # IDs=[2,4]
        # datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, keys_select=['F'], debug=debug)
        
        datadict = dataaccess.read_dict(hdf5filepath, keys_select=['F'], debug=debug)
        depthL = datadict['F']['depth']
        
        net = man.load_model(opts, folderDict, previous=False, load_weights=False)
        dataloader_train = net.model['unet'].get_dataloaders_BF(data, batch_size=None)
        patch_size = dataloader_train.data_loader.patch_size
        
        imagenames = list(np.unique([s.F['imagename'] for s in data]))
        
        q=net.model['unet']
        q.get_dataloaders()
        
        # Load image by index
        #datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, debug=debug)
        IDT=-1
        pbar = tqdm(total=len(imagenames))
        pbar.set_description("Loading image")
        for imn in imagenames:
            pbar.update(1)
            
            # Extract ID
            for s in data:
                if s.F['imagename']==imn:
                    ID = s.F['ID']
                    break
            
            idx0 = depthL[0:ID].sum()
            idx1 = depthL[0:ID].sum()+depthL[ID]
            idx = [x for x in range(idx0,idx1)]
            datadictIm = dataaccess.read_dict(hdf5filepath, idx=idx, keys_select=['X', 'Y'], debug=False)
            X = {'XImage': torch.from_numpy(datadictIm['X']['XImage'])}
            Y = {'XMask': torch.from_numpy(datadictIm['Y'][k]) for k in datadictIm['Y']}
            X['XImage'] = np.swapaxes(X['XImage'], 0, 2)
            Y['XMask'] = np.swapaxes(Y['XMask'], 0, 2)
            for i,s in enumerate(data):
                if s.F['imagename']==imn:
                    #ID = s.F['ID']
                    bboxLbs = s.F['bboxLbsOrg']
                    bboxUbs = s.F['bboxUbsOrg']
                    
                    dataPatch = X['XImage'][:,:,bboxLbs[0]:bboxUbs[0],bboxLbs[1]:bboxUbs[1],bboxLbs[2]:bboxUbs[2]]
                    segPatch = Y['XMask'][:,:,bboxLbs[0]:bboxUbs[0],bboxLbs[1]:bboxUbs[1],bboxLbs[2]:bboxUbs[2]]
                    
                    s.Y['XMask'] = segPatch.clone()
                    s.X['XImage'] = dataPatch.clone()
        pbar.close()
        del datadict
        
    def load_dataset_mds3D(self, man, opts, folderDict, data, debug=True):
        # self = dataset
        # data=action_round
        # debug=True

        # Create DataAccess
        dataaccess = DataAccess()
        hdf5filepath = folderDict['fip_hdf5_all']
        # Get index
        #IDs = [s.ID for s in data]
        
        #fp_nnunet = os.path.join(splitFolderPath(folderDict['fip_hdf5_all'])[0], 'nnunet')
        fp_nnunet = opts.fp_nnunet
        fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
        fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')

        #dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset_name
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        plans = load_json(os.path.join(fp_nnUNet_preprocessed, dname, 'nnUNetPlans.json'))
        plans_manager = PlansManager(plans)
        #configuration_name = '2d'
        configuration_manager = plans_manager.get_configuration(opts.configuration)
        NumChannels = len(load_json(os.path.join(fp_nnUNet_raw, dname, 'dataset.json'))['labels'])-1
        # datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, keys_select=['XImage'], debug=debug)
        # datadict = dataaccess.read_dict(hdf5filepath, keys_select=['F'], debug=debug)
        # IDs=[2,4]
        # datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, keys_select=['F'], debug=debug)
        
        datadict = dataaccess.read_dict(hdf5filepath, keys_select=['F'], debug=debug)
        #depthL = datadict['F']['depth']
        
        net = man.load_model(opts, folderDict, previous=False, load_weights=False)
        dataloader_train = net.model['unet'].get_dataloaders_BF(data, batch_size=None)
        patch_size = dataloader_train.data_loader.patch_size
        
        imagenames = list(np.unique([s.F['imagename'] for s in data]))
        
        # Load image by index
        #datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, debug=debug)
        IDT=-1
        pbar = tqdm(total=len(imagenames))
        pbar.set_description("Loading image")
        for imn in imagenames:
            pbar.update(1)
            
            # Extract ID
            for s in data:
                if s.F['imagename']==imn:
                    ID = s.F['ID']
                    break

            name = 'Task' + str(opts.dataset_dict[opts.dataset]).zfill(2) + '_' + opts.dataset
            fp_images = os.path.join(opts.fp_mds, name, 'imagesTr')
            fp_ref = os.path.join(opts.fp_mds, name, 'labelsTr')
            XImage = CTImage(os.path.join(fp_images, imn)).image()
            XImage = np.expand_dims(XImage, axis=0)
            if XImage.ndim==4:
                XImage = np.expand_dims(XImage, axis=0)
            ref = CTRef(os.path.join(fp_ref, imn))
            ref = ref.numTobin(NumChannels, offset=0)
            XMask = ref.ref()
            XMask = np.swapaxes(XMask, 0, 1)
            XMask = np.expand_dims(XMask, axis=0)
            
            # Crop image
            bbC = s.F['bboxUsedForCropping']
            if opts.dim==2:
                sl = s.F['slice']
                XImage = XImage[:,:,:,bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1]]
                XMask = XMask[:,:,:,bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1]]
            else:
                XImage = XImage[:,:,bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1], bbC[2][0]:bbC[2][1]]
                XMask = XMask[:,:,bbC[0][0]:bbC[0][1], bbC[1][0]:bbC[1][1], bbC[2][0]:bbC[2][1]]
                        
            X = {'XImage': torch.from_numpy(XImage)}
            Y = {'XMask': torch.from_numpy(XMask)}
            
            #X['XImage'] = np.swapaxes(X['XImage'], 0, 2)
            #Y['XMask'] = np.swapaxes(Y['XMask'], 0, 2)
            for i,s in enumerate(data):
                if s.F['imagename']==imn:
                    #ID = s.F['ID']
                    bboxLbs = s.F['bboxLbsOrg']
                    bboxUbs = s.F['bboxUbsOrg']
                    
                    #sys.exit()
                    
                    if opts.dim==2:
                        sl = s.F['slice']
                        dataPatch = X['XImage'][:,:,sl,bboxLbs[0]:bboxUbs[0],bboxLbs[1]:bboxUbs[1]]
                        segPatch = Y['XMask'][:,:,sl,bboxLbs[0]:bboxUbs[0],bboxLbs[1]:bboxUbs[1]]
                    else:
                        dataPatch = X['XImage'][:,:,bboxLbs[0]:bboxUbs[0],bboxLbs[1]:bboxUbs[1],bboxLbs[2]:bboxUbs[2]]
                        segPatch = Y['XMask'][:,:,bboxLbs[0]:bboxUbs[0],bboxLbs[1]:bboxUbs[1],bboxLbs[2]:bboxUbs[2]]
                        
                    s.Y['XMask'] = segPatch.clone()
                    s.X['XImage'] = dataPatch.clone()
        pbar.close()
        del datadict
        

        
    def load_dataset_hdf5(self, man, opts, folderDict, data, debug=True):
        # self = dataset
        # data=man.datasets['action_round'].data
        # debug=True

        # Create DataAccess
        dataaccess = DataAccess()
        hdf5filepath = folderDict['fip_hdf5_all']
        # Get index
        IDs = [s.ID for s in data]
        
        fp_nnunet = os.path.join(splitFolderPath(folderDict['fip_hdf5_all'])[0], 'nnunet')
        fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')

        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset_name
        plans = load_json(os.path.join(fp_nnUNet_preprocessed, dname, 'nnUNetPlans.json'))
        plans_manager = PlansManager(plans)
        configuration_name = '2d'
        configuration_manager = plans_manager.get_configuration(configuration_name)
        
        # Load image by index
        datadict = dataaccess.read_dict(hdf5filepath, ID=IDs, debug=debug)
        pbar = tqdm(total=len(data))
        pbar.set_description("Loading image")
        for i,s in enumerate(data):
            pbar.update(1)
            # 1) Read image
            s.X = {'XImage': torch.from_numpy(datadict['X']['XImage'][i:i+1,:])}
            s.Y = {'XMask': torch.from_numpy(datadict['Y'][k][i:i+1]) for k in datadict['Y']}
            s.features = {k: datadict['F'][k][i:i+1] for k in datadict['F']}
            
            # 2) Reshape to target resolution
            shape_org_seg = s.Y['XMask'].shape
            original_spacing = s.F['spacingOrg']
            target_spacing = np.array([original_spacing[0], s.F['spacingPre'][1], s.F['spacingPre'][2]])
            shape_tar_seg = compute_new_shape(shape_org_seg[1:], original_spacing, target_spacing)
            segRes = configuration_manager.resampling_fn_seg(s.Y['XMask'], shape_tar_seg, original_spacing, target_spacing)

            shape_org_seg = s.X['XImage'].shape
            original_spacing = s.F['spacingOrg']
            target_spacing = np.array([original_spacing[0], s.F['spacingPre'][1], s.F['spacingPre'][2]])
            shape_tar_seg = compute_new_shape(shape_org_seg[1:], original_spacing, target_spacing)
            dataRes = configuration_manager.resampling_fn_data(s.X['XImage'], shape_tar_seg, original_spacing, target_spacing)
                        
            # 3) Padding if needed
            from acvl_utils.cropping_and_padding.padding import pad_nd_image
            segPad, slicer_revert_padding = pad_nd_image(torch.tensor(segRes), configuration_manager.patch_size,'constant', {'value': 0}, True,None)
            dataPad, slicer_revert_padding = pad_nd_image(torch.tensor(dataRes), configuration_manager.patch_size,'constant', {'value': 0}, True,None)
    
            # 4) Extract patch
            bboxLbs = s.F['bboxLbs']
            bboxUbs = s.F['bboxUbs']
            segPatch = segPad[:,:,bboxLbs[1]:bboxUbs[1],bboxLbs[0]:bboxUbs[0]]
            dataPatch = dataPad[:,:,bboxLbs[1]:bboxUbs[1],bboxLbs[0]:bboxUbs[0]]
            
            if list(segPad.shape[2:])==configuration_manager.patch_size:
                #segPadN = torch.zeros((segPad.shape[0], self.NumChannelsOut, segPad.shape[2], segPad.shape[3]))
                #for i in range(segPadN.shape[1]):
                #    segPadN[:,i,:,:] = segPad==(i+1)
                #s.Y['XMask'] = segPadN
                s.Y['XMask'] = segPad
                s.X['XImage'] = dataPad
            else:
                # segPatchN = torch.zeros((segPatch.shape[0], self.NumChannelsOut, segPatch.shape[2], segPatch.shape[3]))
                # for i in range(segPatchN.shape[1]):
                #     segPatchN[:,i,:,:] = segPatch==(i+1)
                # s.Y['XMask'] = segPatchN 
                s.Y['XMask'] = segPatch
                s.X['XImage'] = dataPatch
            s.F['slicer_revert_padding'] = slicer_revert_padding
            #     sys.exit()
        pbar.close()
        del datadict
        
              
        
    def create_dataset_hdf5_seg(self, opts, hdf5filepath, NumSamples=10, test_set=False):
        
        # self = dataset
        
        # Create folderpath
        fp_data = os.path.join(opts.fp_modules, opts.dataset, 'data')
        fp_active_data = os.path.join(opts.fp_active, 'data')
        os.makedirs(fp_active_data, exist_ok=True)
        fp_images = opts.fp_images
        fp_references = opts.fp_references
        #if fp_images is None: fp_images = os.path.join(fp_data, 'images')
        #if fp_references is None: fp_references = os.path.join(fp_active_data, 'references')
        
        #fp_lits = '/mnt/HHD/data/LITS17'
        fip_images = glob(fp_images + '/vol*.nii')
        random.shuffle(fip_images)
        
        split_ratio=[0.6, 0.1, 0.3]
        NumAll = len(fip_images)
        NumTrain = int(np.round(NumAll*split_ratio[0]))
        NumValid = int(np.round(NumAll*split_ratio[1]))
        NumTest = NumAll-NumTrain-NumValid
        
        fip_split = os.path.join(fp_active_data, 'split.xlsx')  
        df_split = pd.DataFrame(columns=['filename', 'filename_label', 'train', 'test','valid'])
        
        # Select train samples
        for fip_image in fip_images[0:NumTrain]:
            fip_ref = os.path.join(fp_references, 'segmentation-' + fip_image.split('-')[1])
            row = dict({'filename': os.path.basename(fip_image),
                        'filename_label': os.path.basename(fip_ref),
                        'train': True,'valid': False,'test': False})
            df_split = df_split.append(row, ignore_index=True)

        # Select valid samples
        for fip_image in fip_images[NumTrain:NumTrain+NumValid]:
            fip_ref = os.path.join(fp_references, 'segmentation-' + fip_image.split('-')[1])
            row = dict({'filename': os.path.basename(fip_image),
                        'filename_label': os.path.basename(fip_ref),
                        'train': False,'valid': True,'test': False})
            df_split = df_split.append(row, ignore_index=True)
        
        # Select test samples
        for fip_image in fip_images[NumTrain+NumValid:]:
            fip_ref = os.path.join(fp_references, 'segmentation-' + fip_image.split('-')[1])
            row = dict({'filename': os.path.basename(fip_image),
                        'filename_label': os.path.basename(fip_ref),
                        'train': False,'valid': False,'test': True})
            df_split = df_split.append(row, ignore_index=True)

        # Save df_split
        df_split.to_excel(fip_split, index=False)
        df_split = df_split.sample(frac=1)
        

        if NumSamples is not None:
            df_split = df_split[0:NumSamples]
            
        dataaccess = DataAccess()
        #maskRegion_list, maskSegment_list = self.extract_coding_seg()
        
        if opts.dim==2:
            patch_id = 0
            pbar = tqdm(total=len(df_split))
            pbar.set_description("Creating patches from image")
            for i, row in df_split.iterrows():
                pbar.update(1)
                #sys.exit()
                fip_image = os.path.join(fp_images, row['filename'])
                fip_label = os.path.join(fp_references, row['filename_label'])
                #fip_lesion = os.path.join(fp_references, row['filename_lesion'])
                
                # Load image
                if os.path.exists(fip_image):
                    # Init feature dataframe
                    df_features = pd.DataFrame()
                    
                    image = CTImage(fip_image)
                    height, width, depth = image.image(axis='WHC').shape
                    
    
                    # Load region
                    if os.path.exists(fip_label):
                        mask = CTRef(fip_label)
                                  
                    for s in range(0, mask.ref().shape[0]):
                        patch_id_str = "{:07n}".format(patch_id)
                             
                        # Save feature
                        features = defaultdict(lambda: None)
                        features['ID'] = patch_id
                        features['slice'] = s
                        features['TRAIN'] = row['train']
                        features['VALID'] = row['valid']
                        features['TEST'] = row['test']
                        features['imagename'] = row['filename']
                        features['maskname'] = row['filename_label']
    
                        # Add patch to pacthlist
                        df_features = df_features.append(features, ignore_index=True)
                        df_features = df_features.reset_index(drop = True)
        
                        # Update patch_id
                        patch_id = patch_id + 1
                    
                    # Update data of the image
                    # XIDTmp = np.array(df_features['ID']).astype(np.int32)
                    # XCandidateTmp = XCandidateTmp.astype(np.uint8)
                    # XRegionTmp = XRegionTmp.astype(np.uint8)
                    # XSegmentTmp = XSegmentTmp.astype(np.uint8)
    
                    XImageTmp = np.expand_dims(image.image(), axis=1)
                    XMaskTmp = np.zeros((depth,2,height, width))
                    XMaskTmp[:,1,:,:] = (mask.ref()>0)*1
                    XMaskTmp[:,0,:,:] = 1-XMaskTmp[:,1,:,:]
                    XMaskTmp = XMaskTmp.astype(np.uint8)
                          
                    XFeatures = dataaccess.dataFrameToDict(df_features)
                    XID = np.array(df_features['ID']).astype(np.int32)
                    XMask=dict({'XMask': XMaskTmp})
                    XImage=dict({'XImage': XImageTmp})
                    datadict = dict({'ID': XID, 'X': XImage, 'Y': XMask, 'P': [], 'F': XFeatures})
                    dataaccess.save_dict(hdf5filepath, datadict)
                    
            pbar.close()
        else:
            patch_id = 0
            pbar = tqdm(total=len(df_split))
            pbar.set_description("Creating patches from image")
            for i, row in df_split.iterrows():
                pbar.update(1)

                fip_image = os.path.join(fp_images, row['filename'])
                fip_label = os.path.join(fp_references, row['filename_label'])
                #fip_lesion = os.path.join(fp_references, row['filename_lesion'])
                
                # Load image
                if os.path.exists(fip_image):
                    # Init feature dataframe
                    df_features = pd.DataFrame()
                    
                    image = CTImage(fip_image)
                    height, width, depth = image.image(axis='WHC').shape
                    

                    # Load region
                    if os.path.exists(fip_label):
                        mask = CTRef(fip_label)
                        
                    patch_id_str = "{:07n}".format(patch_id)
                             
                    # Save feature
                    features = defaultdict(lambda: None)
                    features['ID'] = patch_id
                    features['TRAIN'] = row['train']
                    features['VALID'] = row['valid']
                    features['TEST'] = row['test']
                    features['imagename'] = row['filename']
                    features['maskname'] = row['filename_label']
                    features['height'] = height
                    features['width'] = width
                    features['depth'] = depth

                    # Add patch to pacthlist
                    df_features = df_features.append(features, ignore_index=True)
                    df_features = df_features.reset_index(drop = True)
    
                    # Update patch_id
                    patch_id = patch_id + 1
                    
                    # Update data of the image
                    # XIDTmp = np.array(df_features['ID']).astype(np.int32)
                    # XCandidateTmp = XCandidateTmp.astype(np.uint8)
                    # XRegionTmp = XRegionTmp.astype(np.uint8)
                    # XSegmentTmp = XSegmentTmp.astype(np.uint8)

                    XImageTmp = np.expand_dims(image.image(), axis=0)
                    XMaskTmp = np.zeros((2,depth,height, width))
                    XMaskTmp[1,:,:,:] = (mask.ref()>0)*1
                    XMaskTmp[0,:,:,:] = 1-XMaskTmp[1,:,:,:]
                    XMaskTmp = XMaskTmp.astype(np.uint8)
                    
                    XImageTmp = np.expand_dims(XImageTmp, axis=0)
                    XMaskTmp = np.expand_dims(XMaskTmp, axis=0)
                    XImageTmp=XImageTmp.swapaxes(0,2)
                    XMaskTmp=XMaskTmp.swapaxes(0,2)
                          
                    XFeatures = dataaccess.dataFrameToDict(df_features)
                    XID = np.array(df_features['ID']).astype(np.int32)
                    XMask=dict({'XMask': XMaskTmp})
                    XImage=dict({'XImage': XImageTmp})
                    datadict = dict({'ID': XID, 'X': XImage, 'Y': XMask, 'P': [], 'F': XFeatures})
                    dataaccess.save_dict(hdf5filepath, datadict)
                    
            pbar.close()


    def create_dataset_mds(self, opts, hdf5filepath, NumSamples=10, test_set=False):
        
        # self = dataset
        
        # Create folderpath
        #fp_data = os.path.join(opts.fp_modules, opts.dataset, 'data')
        name = 'Task' + str(opts.dataset_dict[opts.dataset]).zfill(2) + '_' + opts.dataset
        fp_mds = opts.fp_mds
        fp_imagesTr = os.path.join(fp_mds, name, 'imagesTr')
        fp_labelsTr = os.path.join(fp_mds, name, 'labelsTr')
        #fp_imagesTs = os.path.join(fp_mds, name, 'imagesTs')
        #fp_labelsTs = os.path.join(fp_mds, name, 'labelsTs')
        
        fip_images = sorted(glob(fp_imagesTr+'/*.nii.gz'))
        fip_references = sorted(glob(fp_labelsTr+'/*.nii.gz'))
        
        dataaccess = DataAccess()
        
        if opts.dim==2:
            patch_id = 0
            pbar = tqdm(total=len(fip_images))
            pbar.set_description("Creating patches from image")
            for i in range(len(fip_images)):
                pbar.update()
                fip_image = fip_images[i]
                fip_ref = fip_references[i]
                #df_features = pd.DataFrame()
                df_features = []
                image = CTImage(fip_image)
                if image.image().ndim==3:
                    depth, width, height = image.image().shape
                    nmod=1
                else:
                    nmod, depth, width, height = image.image().shape
                mask = CTRef(fip_ref)     
                for s in range(0, depth):
                    patch_id_str = "{:07n}".format(patch_id)
                         
                    # Save feature
                    features = defaultdict(lambda: None)
                    features['ID'] = patch_id
                    features['slice'] = s
                    features['imagename'] = os.path.basename(fip_image)
                    features['maskname'] = os.path.basename(fip_ref)
                    features['TRAIN'] = True
                    features['VALID'] = False
                    features['TEST'] = False
                    
                    # Add patch to pacthlist
                    #df_features = df_features.append(features, ignore_index=True)
                    #df_features = df_features.reset_index(drop = True)
                    df_features.append(features)
            
                    # Update patch_id
                    patch_id = patch_id + 1
                    
                    # XFeatures = dataaccess.dataFrameToDict(df_features)
                    # XID = np.array(df_features['ID']).astype(np.int32)
                    # datadict = dict({'ID': XID, 'P': [], 'F': XFeatures})
                    # dataaccess.save_dict(hdf5filepath, datadict)
                
                df_features = pd.DataFrame(df_features)
                XFeatures = dataaccess.dataFrameToDict(df_features)
                XID = np.array(df_features['ID']).astype(np.int32)
                datadict = dict({'ID': XID, 'P': [], 'F': XFeatures})
                dataaccess.save_dict(hdf5filepath, datadict)
            pbar.close()
        else:
            #df_features = pd.DataFrame()
            #df_features=[]
            patch_id = 0
            pbar = tqdm(total=len(fip_images))
            pbar.set_description("Creating patches from image")
            for i in range(len(fip_images)):
                pbar.update()
                fip_image = fip_images[i]
                fip_ref = fip_references[i]
                #df_features = pd.DataFrame()
                image = CTImage(fip_image)
                #height, width, depth = image.image(axis='WHC').shape
                if image.image().ndim==3:
                    depth, width, height = image.image().shape
                    nmod=1
                else:
                    nmod, depth, width, height = image.image().shape
                mask = CTRef(fip_ref)  
                #mask = CTRef(fip_ref) 
                
                # Save feature
                features = defaultdict(lambda: None)
                features['ID'] = patch_id
                features['imagename'] = os.path.basename(fip_image)
                features['maskname'] = os.path.basename(fip_ref)
                features['height'] = height
                features['width'] = width
                features['depth'] = depth
                features['nmod'] = nmod
                features['TRAIN'] = True
                features['VALID'] = False
                features['TEST'] = False
                    
                # Add patch to pacthlist
                #df_features = df_features.append(features, ignore_index=True)
                #df_features = df_features.reset_index(drop = True)
                #df_features.append(features)
                df_features=pd.DataFrame(features, index=[0])
        
                # Update patch_id
                patch_id = patch_id + 1
                
                XFeatures = dataaccess.dataFrameToDict(df_features)
                XID = np.array(df_features['ID']).astype(np.int32)
                datadict = dict({'ID': XID, 'P': [], 'F': XFeatures})
                dataaccess.save_dict(hdf5filepath, datadict)
                #print('patch_id123', patch_id)

            pbar.close()
                 
    def create_action(self, opts, man, fp_manual, data_samples, init_phase=False):
        # action_round=man.datasets['action_round'].data
        # data_samples=action_round
        # region='XMaskPred'
        # fp_manual='/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XALabeler/XALabeler/data_manual'
        
        folderDict = man.folderDict
        
        # Delete fp_manual
        shutil.rmtree(fp_manual)
        os.makedirs(fp_manual, exist_ok=True)
       
        # Create folder structure
        fp_images = os.path.join(fp_manual, 'images')
        fp_pseudo = os.path.join(fp_manual, 'pseudo')
        fp_refine = os.path.join(fp_manual, 'refine')
        fp_mask = os.path.join(fp_manual, 'mask')
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
        fip_color = os.path.join(fp_manual, 'XALabelerLUT.ctbl')
        colors_raw = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], 
                      [0,1,1], [1,0,1], [0.75,0.75,0.75],[0.5,0.5,0.5],
                      [0.5,0,0], [0.5,0.5,0], [0,0.5,0], [0.5,0,0.5],
                      [0,0.5,0.5], [0,0,0.5], [1,1,1]]
        
        # colors = [[0,'OTHER', [0,0,0]], 
        #           [1,'BG', [1,0,0]], 
        #           [2,'LM', [0,1,0]], 
        #           [3,'LAD-PROXIMAL', [0,0,1]], 
        #           [4,'LAD-MID', [1,1,0]], 
        #           [5,'LAD-DISTAL', [0,1,1]],
        #           [6,'LAD-SIDE', [1,0,1]],
        #           [7,'LCX-PROXIMAL', [0.75,0.75,0.75]],
        #           [8,'LCX-MID', [0.5,0.5,0.5]],
        #           [9,'LCX-DISTAL', [0.5,0,0]],
        #           [10,'LCX-SIDE', [0.5,0.5,0]],
        #           [11,'RCA-PROXIMAL', [0,0.5,0]],
        #           [12,'RCA-MID', [0.5,0,0.5]],
        #           [13,'RCA-DISTAL', [0,0.5,0.5]],
        #           [14,'RCA-SIDE', [0,0,0.5]],
        #           [15,'ART', [1,1,1]]]
        
        
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        fp_nnUNet_preprocessed = os.path.join(opts.fp_nnunet, 'nnUNet_preprocessed')
        labels = load_json(os.path.join(fp_nnUNet_preprocessed, dname, 'dataset.json'))['labels']
        colors=[]
        for i in range(len(labels)-1):
            key = list(labels.keys())[list(labels.values()).index(i)]
            colors.append([i, key, colors_raw[i]])

        os.makedirs(fp_pseudo, exist_ok=True)
        os.makedirs(fp_refine, exist_ok=True)
        os.makedirs(fp_mask, exist_ok=True)
       
        # Create settings file
        settings={'method': 'xalabeler',
                  'fp_images': folderDict['fp_images'],
                  'fp_pseudo': fp_pseudo,
                  'fp_refine': fp_refine,
                  'fp_mask': '',
                  'fip_actionlist': fip_actionlist,
                  'foregroundOpacity': 0.3}
       
        fip_settings = os.path.join(fp_manual, 'settings_XALabeler.json')
        with open(fip_settings, 'w') as file:
            file.write(json.dumps(settings, indent=4))
       
        # Create pseudo label
        if folderDict['version']>1:
            #net = manager.load_model(opts, folderDict, previous=True)
            net = man.load_model(opts, folderDict, previous=True)
        else:
            net = man.load_model(opts, folderDict, load_weights=False)
            #nnunet_trainer.initialize()
            #from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
            # initialize nnunet trainer
            #preprocessed_dataset_folder_base = os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
            # preprocessed_dataset_folder_base = os.path.join(fp_nnUNet_preprocessed, dname)
            # plans_file = os.path.join(preprocessed_dataset_folder_base, opts.plans_identifier + '.json')
            # plans = load_json(plans_file)
            # dataset_json = load_json(os.path.join(preprocessed_dataset_folder_base, 'dataset.json'))
            # nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
            #                                 dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device)
            # net = man.load_model(opts, folderDict, previous=False, load_weights=False)
            # settingsfilepath_model = os.path.join(folderDict['modelpath'], 'SegmentCACS.yml')
            # net = SegmentCACS(settingsfilepath_model, overwrite=True)
            # params={'batch_size': 8, 'hdf5filepath': '', 'device': 'cuda',
            #     'lr': 1e-04, 'gamma': 0.98, 'loss_method': 'uncertainty', 'step_size': 5, 
            #     'epochs': 5000, 'epochs_step': 300}
            # net.props['Input_size'] = (512, 512, 5)
            # #dataframe.props['Input_size']=(512, 512, 5)
            # net.create25Dseg(params=params)
            #opts.CLSample.predict(data_samples, net=net)
        
        dataset = opts.CLDataset()
        dataset.load_dataset_mds3D(man, opts, man.folderDict, data_samples)
        for s in data_samples:
            s.Y['XMask']=None
            s.P['XMaskPred']=None
        
        opts.CLSample.predict(data_samples, opts, net, reshapeOrg=True)
        
        # Reset pseudo label in inti phase
        if init_phase:
            for s in data_samples:
                s.P['XMaskPred'][:,0]=0.9
                s.P['XMaskPred'][:,1:]=0.1
            
            
        opts.CLSample.savePseudo(opts, data_samples, folderDict, fp_pseudo, region='XMaskPred')
       
        dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        fp_nnunet = opts.fp_nnunet
        fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
        fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')

        # Create actionlist
        actionlist=[]
        for s in data_samples:
            action=ALAction()
            action.name='action'
            action.status='open'
            action.id=int(s.F['ID'])
            action.idp=int(s.F['IDP'])
            if opts.dim==2:
                action.slice=int(s.F['slice'])
            else:
                action.slice=None
            action.bboxLbsOrg=[int(s.F['slice'])] + [int(x) for x in list(s.F['bboxLbsOrg'])]
            action.bboxUbsOrg=[int(s.F['slice']+1)] + [int(x) for x in list(s.F['bboxUbsOrg'])]
            action.dim=opts.dim
            action.imagename=s.F['imagename']
            action.pseudoname=action.imagename+'-pseudo'
            action.refinename=action.imagename+'-refine'
            action.maskname=None
            action.label=colors
            action.fip_image=os.path.join(fp_imagesTr, s.F['imagename'][:-7]+'_0000.nii.gz')
            action.fip_pseudo=os.path.join(fp_pseudo, s.F['maskname'][:-7]+'.nrrd')
            action.fip_refine=os.path.join(fp_refine, s.F['maskname'][:-7]+'.nrrd')
            action.fip_mask=os.path.join(fp_mask, s.F['maskname'][:-7]+'.nrrd')
            actionlist.append(action)
            # Update bboxLbsOrgW and bboxUbsOrgW
            if opts.dim==2:
                ref = CTRef(action.fip_pseudo)
                action.bboxLbsOrgW = ref.ref_sitk.TransformIndexToPhysicalPoint(tuple([int(s.F['slice'])]+[int(x) for x in list(s.F['bboxLbsOrg'])]))
                action.bboxUbsOrgW = ref.ref_sitk.TransformIndexToPhysicalPoint(tuple([int(s.F['slice']+1)]+[int(x) for x in list(s.F['bboxUbsOrg'])]))
                
            else:
                ref = CTRef(action.fip_pseudo)
                action.bboxLbsOrgW = ref.ref_sitk.TransformIndexToPhysicalPoint(tuple([int(x) for x in list(s.F['bboxLbsOrg'])]))
                action.bboxUbsOrgW = ref.ref_sitk.TransformIndexToPhysicalPoint(tuple([int(x) for x in list(s.F['bboxUbsOrg'])]))
                
        # Sort actionlist by scan
        actionlist_sort=[]
        imagenames = np.unique([action.imagename for action in actionlist])
        for im in imagenames:
            for action in actionlist:
                if action.imagename==im:
                    actionlist_sort.append(action)
            
        
        ALAction.save(fip_actionlist, actionlist_sort)   

        lines = ['# Color']
        for i in range(len(colors)):
            if i==0:
                trans = '0'
            else:
                trans = '255'
            col = str(colors[i][0]) + ' ' + str(colors[i][1]) + ' ' + str(colors[i][2][0]*255) + ' ' + str(colors[i][2][1]*255) + ' ' + str(colors[i][2][2]*255) + ' ' + trans
            lines = lines + [col]
       
        with open(fip_color, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n') 
                
                
                
class UNetSegAL(UNetSeg):
    
    """
    UNetSegAL model
    """

    def __init__(self, settingsfilepath, overwrite=False):
        UNetSeg.__init__(self, settingsfilepath=settingsfilepath, overwrite=overwrite)
        
        

