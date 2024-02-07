import sys
import numpy as np
from typing import Union, Tuple, List
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import time

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
import os

class nnUNetDataLoader2DBF(nnUNetDataLoaderBase):
    
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 infinite: bool = False,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 data_load: list = [],
                 unlabeled: bool = False):
        self.data_load=data_load
        self.unlabeled=unlabeled
        #print('data123', data)
        #print('patch_size123', patch_size)
        #print('final_patch_size123', final_patch_size)
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent, 
                         infinite, sampling_probabilities, pad_sides, probabilistic_oversampling)
        self.IDS=0
        self.current_key=None
        self.data_org=None
        self.seg_org=None
        self.properties=None
        self.shape_org=None
        #indices = [s.F['imagename'].split('.')[0] for s in data_load]
        #print('indices123456', indices)
        #sys.exit()
        #self.indices = indices

    def generate_train_batch(self):
        #print('generate_train_batch123')
        
        self.data_load['props']=None
        j=0
        data_all=[]
        seg_all=[]
        case_properties = []
        
        selected_keys=[]
        end = min(self.IDS+self.batch_size, len(self.data_load))
        for i in range(self.IDS, end):
            
            start = time.time()
            
            row = self.data_load.iloc[i]
            key = row['imagename'].split('.')[0]
            #print('kk:', self.current_key, key)
            if self.current_key != key:
                self.current_key = key
                #print('current_key123', self.current_key)
                self.data_org, self.seg_org, self.properties = self._data.load_case(self.current_key)
                self.shape_org = self.data_org.shape
                #print('current_key', self.current_key)
                #label_exist = np.any([x in seg for x in list(self.annotated_classes_key)])
                
            #print('self.data_org123', self.data_org.shape)
            
            
            force_fg = self.get_do_oversample(j)
            # select a class/region first, then a slice where this class is present, then crop to that area
            if not force_fg:
                if self.has_ignore:
                    selected_class_or_region = self.annotated_classes_key
                else:
                    selected_class_or_region = None
            else:
                # filter out all classes that are not present here
                eligible_classes_or_regions = [i for i in self.properties['class_locations'].keys() if len(self.properties['class_locations'][i]) > 0]
            
                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])
            
                selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    len(eligible_classes_or_regions) > 0 else None
            if selected_class_or_region is not None:
                #selected_slice = int(self.data_load[j].F['slice'])
                selected_slice = row['slice']
            else:
                selected_slice = np.random.choice(len(self.data_org[0]))
                
            #print('time1', time.time()-start)
            
            
            #print('selected_slice123', selected_slice)
            #print('self.current_key123', self.current_key)
            #print('data123', self.data_org.shape)
            #print('row123', row)
            data_sel = self.data_org[:, selected_slice]
            seg_sel = self.seg_org[:, selected_slice]
            #print('data_sel123', data_sel.shape)
            
            
            tile_step_size=0.5
            steps = compute_steps_for_sliding_window(self.shape_org[2:], self.patch_size, tile_step_size)
            if selected_class_or_region is None:
                selected_class_or_region = (0,1)
                
            shape = data_sel.shape[1:]
            dim = len(shape)
            
            #data_org = data.copy()
            #seg_org = seg.copy()
            
            #print('time2', time.time()-start)
            
            #print('sample', i, row['ID'], row['IDP'])
            
            IDP=0
            properties_list=[]
            for sx in list(steps[0]):
                for sy in list(steps[1]):
                    coord = np.array([[0,selected_slice,int(sx+self.patch_size[0]/2),int(sy+self.patch_size[1]/2)]])
                    class_locations = {selected_class_or_region: coord}
                    if IDP==row['IDP']:
                        #pr = self.properties.copy()
                        #pr['class_locations'] = class_locations
                        pr={'class_locations': class_locations, 'ID': row['ID'], 'IDP': row['IDP'], 'imagename': row['imagename'], 'slice': row['slice']}
                        properties_list.append(pr)
                        self.data_load.at[i,'props']=pr.copy()
                        
                        class_locations = {
                            selected_class_or_region: pr['class_locations'][selected_class_or_region][pr['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
                        } if (selected_class_or_region is not None) else None
                        

                        
                        #print('class_locations1234', pr['class_locations'])
                        
                        # print('shape123', shape)
                        # print('selected_class_or_region', selected_class_or_region)
                        # print('class_locations', pr['class_locations'])
                        # print('annotated_classes_key', self.annotated_classes_key)
                        # print('current_key1243', self.current_key)
                        
                        # bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                        #                                    pr['class_locations'], overwrite_class=selected_class_or_region)

                        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                                                           class_locations, overwrite_class=selected_class_or_region)
       
                
                        # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
                        # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
                        # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
                        # later
                        valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                        valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
        
                        # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
                        # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
                        # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
                        # remove label -1 in the data augmentation but this way it is less error prone)
                        this_slice = tuple([slice(0, data_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        
                        #print('this_slice123', this_slice)
                        
                        #data = data_sel[this_slice].copy()
                        data = data_sel[this_slice]
                        this_slice = tuple([slice(0, seg_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                        #seg = seg_sel[this_slice].copy()
                        seg = seg_sel[this_slice]
                        padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
                        
                        #print('data123', data.shape)
                        #print('padding123', padding)
                        
                        #tmp = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
                        #print('tmp123', tmp.shape)
                        
                        #data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
                        datap = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
                        datap = np.expand_dims(datap, 0)
                        data_all.append(datap)
                        #seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
                        #seg_all.append(np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1))
                        
                        segp = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
                        segp = np.expand_dims(segp, 0)
                        seg_all.append(segp)
                        case_properties = case_properties + [pr]
                        selected_keys.append(self.current_key)
                    IDP=IDP+1
            j*j+1
            
        #print('data_all12345', data_all[0].shape)
        #print('self.batch_size123', self.batch_size)
        
        self.IDS = self.IDS + self.batch_size
        data_all = np.vstack(data_all)
        seg_all = np.vstack(seg_all)
        #data_all = np.expand_dims(data_all, 1)
        #seg_all = np.expand_dims(seg_all, 1)
        
        #print('data_all123', data_all.shape)
        #sys.exit()
        
        #print('seg_all123', seg_all.shape)
        #print('data_all123', data_all.shape)
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}

        
        
        
        # selected_keys = self.get_indices()
        # # preallocate memory for data and seg
        # #data_all = np.zeros(self.data_shape, dtype=np.float32)
        # data_all=[]
        # #seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        # seg_all=[]
        # case_properties = []
        
        # #print('self.dataset123', list(self._data.dataset.keys()))
        
        # # if self.unlabeled:
        # #     checkBG = False
        # # else:
        # #     checkBG = True

        # for j, current_key in enumerate(selected_keys):
        #     #print('nnUNetDataLoader2DBF')
        #     #print('current_key123', current_key)
            
        #     start = time.time()
            
        #     # oversampling foreground will improve stability of model training, especially if many patches are empty
        #     # (Lung for example)
        #     force_fg = self.get_do_oversample(j)
        #     data, seg, properties = self._data.load_case(current_key)
        #     #case_properties.append(properties)
            
        #     shape_org = data.shape
            
        #     #print('time0', time.time()-start)
        #     #a=np.unique(seg)
        #     #print('time000', time.time()-start)
            
        #     #print('seg123', np.unique(seg))
        #     #if list(np.unique(seg))!=[2]:
        #     label_exist = np.any([x in seg for x in list(self.annotated_classes_key)])
        #     #print('unlabeled123', self.unlabeled)
        #     #print('label_exist123', label_exist)
        #     if (not self.unlabeled and label_exist) or self.unlabeled:
        #         #print('current_key1234', current_key)
        #         # print('seg123', np.unique(seg))
        #         #print('force_fg123', force_fg)
        #         # print('data123', data.shape)
        #         # print('self.has_ignore', self.has_ignore)
        #         #print('self.annotated_classes_key', self.annotated_classes_key)
    
    
        #         # select a class/region first, then a slice where this class is present, then crop to that area
        #         if not force_fg:
        #             if self.has_ignore:
        #                 #print('time01', time.time()-start)
        #                 selected_class_or_region = self.annotated_classes_key
        #                 #print('time02', time.time()-start)
        #             else:
        #                 selected_class_or_region = None
        #         else:
        #             # filter out all classes that are not present here
        #             eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]
    
        #             # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
        #             # strange formulation needed to circumvent
        #             # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        #             tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
        #             if any(tmp):
        #                 if len(eligible_classes_or_regions) > 1:
        #                     eligible_classes_or_regions.pop(np.where(tmp)[0][0])
    
        #             selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
        #                 len(eligible_classes_or_regions) > 0 else None
        #         if selected_class_or_region is not None:
        #             #print('time1', time.time()-start)
        #             #print('properties123', properties['class_locations'])
        #             #print('selected_class_or_region123', selected_class_or_region)
        #             # print('selected_class_or_region123', selected_class_or_region)
        #             #print('propertiesclass_locations123', np.unique(properties['class_locations'][selected_class_or_region][:, 1]))
        #             #selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
        #             #selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
        #             #idx_s = self.current_position - self.batch_size + j
        #             #print('idx_s213', idx_s)
        #             #selected_slice = int(self.data_load[idx_s].F['slice'])
        #             selected_slice = int(self.data_load[j].F['slice'])
        #             #print('selected_slice213', selected_slice)
        #         else:
                    
        #             selected_slice = np.random.choice(len(data[0]))
    
        #         data = data[:, selected_slice]
        #         seg = seg[:, selected_slice]
                
        #         # !!!
        #         #print('shape_org123', shape_org)
        #         #print('selected_class_or_region123', selected_class_or_region)
        #         #selected_class_or_region = None
        #         #self.has_ignore = False
        #         # selected_slice=10
        #         # properties['class_locations'][(0,1)]=np.array([[0,selected_slice,256,256],[0,selected_slice,257,257]])
                
                
        #         from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        #         from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window
        #         from nnunetv2.paths import nnUNet_results, nnUNet_raw
        #         import torch
        #         import os
        #         # predictor = nnUNetPredictor(
        #         #     tile_step_size=0.5,
        #         #     use_gaussian=False,
        #         #     use_mirroring=False,
        #         #     perform_everything_on_gpu=True,
        #         #     device=torch.device('cuda', 0),
        #         #     verbose=False,
        #         #     verbose_preprocessing=False,
        #         #     allow_tqdm=True
        #         # )
        #         # predictor.initialize_from_trained_model_folder(
        #         #     os.path.join(nnUNet_results, 'Dataset105_ALUNet/nnUNetTrainer__nnUNetPlans__2d'),
        #         #     use_folds=(0,),
        #         #     checkpoint_name='checkpoint_best.pth',
        #         # )
        #         #print('data123', data.shape)
        #         #print('selected_slice123', selected_slice)
        #         #slicers_tmp = predictor._internal_get_sliding_window_slicers(shape_org[1:])
        #         #print('slicers_tmp123', len(slicers_tmp))
                
                

        #         #steps = compute_steps_for_sliding_window(shape_org[2:], self.configuration_manager.patch_size,self.tile_step_size)
        #         tile_step_size=0.5
        #         steps = compute_steps_for_sliding_window(shape_org[2:], self.patch_size, tile_step_size)
                
        #         #selected_slice=10
        #         #['class_locations'][(0,1)]=np.array([[0,selected_slice,256,256],[0,selected_slice,257,257]])
        #         if selected_class_or_region is None:
        #             selected_class_or_region = (0,1)
                    
        #         # cloc=[]
        #         # for sx in list(steps[0]):
        #         #     for sy in list(steps[1]):
        #         #         cloc.append(np.array([0,selected_slice,int(sx+self.patch_size[0]/2),int(sy+self.patch_size[1]/2)]))
        #         # class_locations = np.vstack(cloc)
        #         # print('class_locations123', class_locations)
        #         # properties['class_locations'][selected_class_or_region] = class_locations

        #         properties_list=[]
        #         for sx in list(steps[0]):
        #             for sy in list(steps[1]):
        #                 coord = np.array([[0,selected_slice,int(sx+self.patch_size[0]/2),int(sy+self.patch_size[1]/2)]])
        #                 class_locations = {selected_class_or_region: coord}
                        

        #                 #pr={'class_locations': class_locations}
        #                 pr = properties.copy()
        #                 pr['class_locations'] = class_locations
                        
        #                 class_locations = {
        #                     selected_class_or_region: pr['class_locations'][selected_class_or_region][pr['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
        #                 } if (selected_class_or_region is not None) else None
                        
        #                 pr={'class_locations': class_locations}
            
        #                 properties_list.append(pr)

        #         #class_locations = np.vstack(cloc)
        #         #print('class_locations123', class_locations)
        #         #properties['class_locations'][selected_class_or_region] = class_locations
                
                
        #         #print('steps123', steps)
        #         #print('selected_class_or_region123', selected_class_or_region)
        #         #print('patch_size123', self.patch_size)
        #         #properties['class_locations'] = None
    
        #         # the line of death lol
        #         # this needs to be a separate variable because we could otherwise permanently overwrite
        #         # properties['class_locations']
        #         # selected_class_or_region is:
        #         # - None if we do not have an ignore label and force_fg is False OR if force_fg is True but there is no foreground in the image
        #         # - A tuple of all (non-ignore) labels if there is an ignore label and force_fg is False
        #         # - a class or region if force_fg is True
        #         # class_locations = {
        #         #     selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
        #         # } if (selected_class_or_region is not None) else None
    
        #         # print(properties)
        #         shape = data.shape[1:]
        #         dim = len(shape)
                
        #         data_org = data.copy()
        #         seg_org = seg.copy()
        #         #print('class_locations345', class_locations)
        #         for prop in properties_list:
        #             #print('prop789', prop)
        #             bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
        #                                                prop['class_locations'], overwrite_class=selected_class_or_region)
        
                    
        #             #print('bbox_lbs123', bbox_lbs)
        #             #print('bbox_ubs123', bbox_ubs)
        #             #print('shape123', shape)
                    
        #             # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        #             # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        #             # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        #             # later
        #             valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        #             valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
                    
        #             #print('valid_bbox_lbs123', valid_bbox_lbs)
        #             #print('valid_bbox_ubs123', valid_bbox_ubs)
        
        #             # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        #             # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        #             # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        #             # remove label -1 in the data augmentation but this way it is less error prone)
        #             this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                    
        #             #print('data456', data_org.shape)
        #             #data = data[this_slice]
        #             data = data_org[this_slice]
        #             #print('data4567', data.shape)
                    
        #             #print('this_slice123', this_slice)
        
        #             this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        #             seg = seg_org[this_slice]
        
        #             padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
                    
        #             #print('padding123', padding)
                    
        #             #data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
        #             data_all.append(np.pad(data, ((0, 0), *padding), 'constant', constant_values=0))
        #             #seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
        #             seg_all.append(np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1))
        #             case_properties = case_properties + [prop]
                
                
        #         #print('time2', time.time()-start)
                
        #         # bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
        #         #                                    class_locations, overwrite_class=selected_class_or_region)
    
        #         # print('class_locations345', class_locations)
        #         # print('bbox_lbs123', bbox_lbs)
        #         # print('bbox_ubs123', bbox_ubs)
        #         # print('shape123', shape)
                
        #         # # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        #         # # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        #         # # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        #         # # later
        #         # valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        #         # valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
    
        #         # # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        #         # # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        #         # # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        #         # # remove label -1 in the data augmentation but this way it is less error prone)
        #         # this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        #         # data = data[this_slice]
    
        #         # this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        #         # seg = seg[this_slice]
    
        #         # padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
        #         # data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
        #         # seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
        #         # #print('time2', time.time()-start)
        # data_all = np.vstack(data_all)
        # seg_all = np.vstack(seg_all)
        # data_all = np.expand_dims(data_all, 1)
        # seg_all = np.expand_dims(seg_all, 1)
        # #print('seg_all123', seg_all.shape)
        # #print('data_all123', data_all.shape)
        # return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}

        
    # def generate_train_batch(self):
    #     #print('generate_train_batch123')
    #     selected_keys = self.get_indices()
    #     # preallocate memory for data and seg
    #     #data_all = np.zeros(self.data_shape, dtype=np.float32)
    #     data_all=[]
    #     #seg_all = np.zeros(self.seg_shape, dtype=np.int16)
    #     seg_all=[]
    #     case_properties = []
        
    #     #print('self.dataset123', list(self._data.dataset.keys()))
        
    #     # if self.unlabeled:
    #     #     checkBG = False
    #     # else:
    #     #     checkBG = True

    #     for j, current_key in enumerate(selected_keys):
    #         #print('nnUNetDataLoader2DBF')
    #         #print('current_key123', current_key)
            
    #         start = time.time()
            
    #         # oversampling foreground will improve stability of model training, especially if many patches are empty
    #         # (Lung for example)
    #         force_fg = self.get_do_oversample(j)
    #         data, seg, properties = self._data.load_case(current_key)
    #         #case_properties.append(properties)
            
    #         shape_org = data.shape
            
    #         #print('time0', time.time()-start)
    #         #a=np.unique(seg)
    #         #print('time000', time.time()-start)
            
    #         #print('seg123', np.unique(seg))
    #         #if list(np.unique(seg))!=[2]:
    #         label_exist = np.any([x in seg for x in list(self.annotated_classes_key)])
    #         #print('unlabeled123', self.unlabeled)
    #         #print('label_exist123', label_exist)
    #         if (not self.unlabeled and label_exist) or self.unlabeled:
    #             #print('current_key1234', current_key)
    #             # print('seg123', np.unique(seg))
    #             #print('force_fg123', force_fg)
    #             # print('data123', data.shape)
    #             # print('self.has_ignore', self.has_ignore)
    #             #print('self.annotated_classes_key', self.annotated_classes_key)
    
    
    #             # select a class/region first, then a slice where this class is present, then crop to that area
    #             if not force_fg:
    #                 if self.has_ignore:
    #                     #print('time01', time.time()-start)
    #                     selected_class_or_region = self.annotated_classes_key
    #                     #print('time02', time.time()-start)
    #                 else:
    #                     selected_class_or_region = None
    #             else:
    #                 # filter out all classes that are not present here
    #                 eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]
    
    #                 # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
    #                 # strange formulation needed to circumvent
    #                 # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    #                 tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
    #                 if any(tmp):
    #                     if len(eligible_classes_or_regions) > 1:
    #                         eligible_classes_or_regions.pop(np.where(tmp)[0][0])
    
    #                 selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
    #                     len(eligible_classes_or_regions) > 0 else None
    #             if selected_class_or_region is not None:
    #                 #print('time1', time.time()-start)
    #                 #print('properties123', properties['class_locations'])
    #                 #print('selected_class_or_region123', selected_class_or_region)
    #                 # print('selected_class_or_region123', selected_class_or_region)
    #                 #print('propertiesclass_locations123', np.unique(properties['class_locations'][selected_class_or_region][:, 1]))
    #                 #selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
    #                 #selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
    #                 #idx_s = self.current_position - self.batch_size + j
    #                 #print('idx_s213', idx_s)
    #                 #selected_slice = int(self.data_load[idx_s].F['slice'])
    #                 selected_slice = int(self.data_load[j].F['slice'])
    #                 #print('selected_slice213', selected_slice)
    #             else:
                    
    #                 selected_slice = np.random.choice(len(data[0]))
    
    #             data = data[:, selected_slice]
    #             seg = seg[:, selected_slice]
                
    #             # !!!
    #             #print('shape_org123', shape_org)
    #             #print('selected_class_or_region123', selected_class_or_region)
    #             #selected_class_or_region = None
    #             #self.has_ignore = False
    #             # selected_slice=10
    #             # properties['class_locations'][(0,1)]=np.array([[0,selected_slice,256,256],[0,selected_slice,257,257]])
                
                
    #             from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    #             from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window
    #             from nnunetv2.paths import nnUNet_results, nnUNet_raw
    #             import torch
    #             import os
    #             # predictor = nnUNetPredictor(
    #             #     tile_step_size=0.5,
    #             #     use_gaussian=False,
    #             #     use_mirroring=False,
    #             #     perform_everything_on_gpu=True,
    #             #     device=torch.device('cuda', 0),
    #             #     verbose=False,
    #             #     verbose_preprocessing=False,
    #             #     allow_tqdm=True
    #             # )
    #             # predictor.initialize_from_trained_model_folder(
    #             #     os.path.join(nnUNet_results, 'Dataset105_ALUNet/nnUNetTrainer__nnUNetPlans__2d'),
    #             #     use_folds=(0,),
    #             #     checkpoint_name='checkpoint_best.pth',
    #             # )
    #             #print('data123', data.shape)
    #             #print('selected_slice123', selected_slice)
    #             #slicers_tmp = predictor._internal_get_sliding_window_slicers(shape_org[1:])
    #             #print('slicers_tmp123', len(slicers_tmp))
                
                

    #             #steps = compute_steps_for_sliding_window(shape_org[2:], self.configuration_manager.patch_size,self.tile_step_size)
    #             tile_step_size=0.5
    #             steps = compute_steps_for_sliding_window(shape_org[2:], self.patch_size, tile_step_size)
                
    #             #selected_slice=10
    #             #['class_locations'][(0,1)]=np.array([[0,selected_slice,256,256],[0,selected_slice,257,257]])
    #             if selected_class_or_region is None:
    #                 selected_class_or_region = (0,1)
                    
    #             # cloc=[]
    #             # for sx in list(steps[0]):
    #             #     for sy in list(steps[1]):
    #             #         cloc.append(np.array([0,selected_slice,int(sx+self.patch_size[0]/2),int(sy+self.patch_size[1]/2)]))
    #             # class_locations = np.vstack(cloc)
    #             # print('class_locations123', class_locations)
    #             # properties['class_locations'][selected_class_or_region] = class_locations

    #             properties_list=[]
    #             for sx in list(steps[0]):
    #                 for sy in list(steps[1]):
    #                     coord = np.array([[0,selected_slice,int(sx+self.patch_size[0]/2),int(sy+self.patch_size[1]/2)]])
    #                     class_locations = {selected_class_or_region: coord}
                        

    #                     #pr={'class_locations': class_locations}
    #                     pr = properties.copy()
    #                     pr['class_locations'] = class_locations
                        
    #                     class_locations = {
    #                         selected_class_or_region: pr['class_locations'][selected_class_or_region][pr['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
    #                     } if (selected_class_or_region is not None) else None
                        
    #                     pr={'class_locations': class_locations}
            
    #                     properties_list.append(pr)

    #             #class_locations = np.vstack(cloc)
    #             #print('class_locations123', class_locations)
    #             #properties['class_locations'][selected_class_or_region] = class_locations
                
                
    #             #print('steps123', steps)
    #             #print('selected_class_or_region123', selected_class_or_region)
    #             #print('patch_size123', self.patch_size)
    #             #properties['class_locations'] = None
    
    #             # the line of death lol
    #             # this needs to be a separate variable because we could otherwise permanently overwrite
    #             # properties['class_locations']
    #             # selected_class_or_region is:
    #             # - None if we do not have an ignore label and force_fg is False OR if force_fg is True but there is no foreground in the image
    #             # - A tuple of all (non-ignore) labels if there is an ignore label and force_fg is False
    #             # - a class or region if force_fg is True
    #             # class_locations = {
    #             #     selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
    #             # } if (selected_class_or_region is not None) else None
    
    #             # print(properties)
    #             shape = data.shape[1:]
    #             dim = len(shape)
                
    #             data_org = data.copy()
    #             seg_org = seg.copy()
    #             #print('class_locations345', class_locations)
    #             for prop in properties_list:
    #                 #print('prop789', prop)
    #                 bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
    #                                                    prop['class_locations'], overwrite_class=selected_class_or_region)
        
                    
    #                 #print('bbox_lbs123', bbox_lbs)
    #                 #print('bbox_ubs123', bbox_ubs)
    #                 #print('shape123', shape)
                    
    #                 # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
    #                 # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
    #                 # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
    #                 # later
    #                 valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
    #                 valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
                    
    #                 #print('valid_bbox_lbs123', valid_bbox_lbs)
    #                 #print('valid_bbox_ubs123', valid_bbox_ubs)
        
    #                 # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
    #                 # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
    #                 # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
    #                 # remove label -1 in the data augmentation but this way it is less error prone)
    #                 this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                    
    #                 #print('data456', data_org.shape)
    #                 #data = data[this_slice]
    #                 data = data_org[this_slice]
    #                 #print('data4567', data.shape)
                    
    #                 #print('this_slice123', this_slice)
        
    #                 this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
    #                 seg = seg_org[this_slice]
        
    #                 padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
                    
    #                 #print('padding123', padding)
                    
    #                 #data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
    #                 data_all.append(np.pad(data, ((0, 0), *padding), 'constant', constant_values=0))
    #                 #seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
    #                 seg_all.append(np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1))
    #                 case_properties = case_properties + [prop]
                
                
    #             #print('time2', time.time()-start)
                
    #             # bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
    #             #                                    class_locations, overwrite_class=selected_class_or_region)
    
    #             # print('class_locations345', class_locations)
    #             # print('bbox_lbs123', bbox_lbs)
    #             # print('bbox_ubs123', bbox_ubs)
    #             # print('shape123', shape)
                
    #             # # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
    #             # # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
    #             # # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
    #             # # later
    #             # valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
    #             # valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
    
    #             # # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
    #             # # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
    #             # # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
    #             # # remove label -1 in the data augmentation but this way it is less error prone)
    #             # this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
    #             # data = data[this_slice]
    
    #             # this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
    #             # seg = seg[this_slice]
    
    #             # padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
    #             # data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
    #             # seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
    #             # #print('time2', time.time()-start)
    #     data_all = np.vstack(data_all)
    #     seg_all = np.vstack(seg_all)
    #     data_all = np.expand_dims(data_all, 1)
    #     seg_all = np.expand_dims(seg_all, 1)
    #     #print('seg_all123', seg_all.shape)
    #     #print('data_all123', data_all.shape)
    #     return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


# if __name__ == '__main__':
#     folder = '/media/fabian/data/nnUNet_preprocessed/Dataset004_Hippocampus/2d'
#     ds = nnUNetDataset(folder, None, 1000)  # this should not load the properties!
#     dl = nnUNetDataLoader2DBF(ds, 366, (65, 65), (56, 40), 0.33, None, None)
#     a = next(dl)
