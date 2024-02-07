import numpy as np
from typing import Union, Tuple, List
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window

class nnUNetDataLoader3DBF(nnUNetDataLoaderBase):
    
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
        
        #print('data_load123', self.data_load)
        
        #import sys
        #sys.exit()
        
        self.data_load['props']=None
        j=0
        data_all=[]
        seg_all=[]
        case_properties = []
        
        #print('self.patch_size', self.patch_size)
        #sys.exit()
        
        selected_keys=[]
        end = min(self.IDS+self.batch_size, len(self.data_load))
        for i in range(self.IDS, end):
            row = self.data_load.iloc[i]
            key = row['imagename'].split('.')[0]
            if self.current_key != key:
                self.current_key = key
                self.data_org, self.seg_org, self.properties = self._data.load_case(self.current_key)
                self.shape_org = self.data_org.shape

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
            # if selected_class_or_region is not None:
            #     #selected_slice = int(self.data_load[j].F['slice'])
            #     print('self.data_org', self.data_org.shape)
            #     selected_slice = row['slice']
            # else:
            #     selected_slice = np.random.choice(len(self.data_org[0]))
            #data_sel = self.data_org[:, selected_slice]
            #seg_sel = self.seg_org[:, selected_slice]
            data_sel = self.data_org
            seg_sel = self.seg_org
            tile_step_size=0.5
            #print('self.shape_org123', self.shape_org)
            #print('self.patch_size123', self.patch_size)
            #print('tile_step_size123', tile_step_size)
            steps = compute_steps_for_sliding_window(self.shape_org[1:], self.patch_size, tile_step_size)
            if selected_class_or_region is None:
                selected_class_or_region = (0,1)
                
            #print('steps123', steps)
                
            shape = data_sel.shape[1:]
            dim = len(shape)

            IDP=0
            properties_list=[]
            for sx in list(steps[0]):
                for sy in list(steps[1]):
                    for sz in list(steps[2]):
                        coord = np.array([[0,int(sx+self.patch_size[0]/2),int(sy+self.patch_size[1]/2),int(sz+self.patch_size[2]/2)]])
                        class_locations = {selected_class_or_region: coord}
                        #print('class_locations123', class_locations)
                        if IDP==row['IDP']:
                            pr={'class_locations': class_locations, 'ID': row['ID'], 'IDP': row['IDP'], 'imagename': row['imagename']}
                            properties_list.append(pr)
                            self.data_load.at[i,'props']=pr.copy()
                            
                            class_locations = {
                                selected_class_or_region: pr['class_locations'][selected_class_or_region]
                            } if (selected_class_or_region is not None) else None
                            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                                                               class_locations, overwrite_class=selected_class_or_region)
           
                            #print('bbox_lbs123', bbox_lbs)
                            #print('bbox_ubs123', bbox_ubs)
                            # print('shape', shape)
                            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
                            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
                            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
                            # later
                            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
                            
                            # print('valid_bbox_lbs123', valid_bbox_lbs)
                            # print('valid_bbox_ubs123', valid_bbox_ubs)
            
                            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
                            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
                            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
                            # remove label -1 in the data augmentation but this way it is less error prone)
                            this_slice = tuple([slice(0, data_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                            #print('this_slice123', this_slice)
                            data = data_sel[this_slice]
                            #print('data123', data.shape)
                            #if data.shape[2]==129:
                            #    sys.exit()
                            this_slice = tuple([slice(0, seg_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                            #print('this_slice1234', this_slice)
                            seg = seg_sel[this_slice]
                            #print('seg123', seg.shape)
                            #print('seg', seg.shape)
                            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
                            #print('padding123', padding)
                            
                            datap = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
                            datap = np.expand_dims(datap, 0)
                            data_all.append(datap)
                            segp = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
                            segp = np.expand_dims(segp, 0)
                            seg_all.append(segp)
                            
                            #data_all.append(np.pad(data, ((0, 0), *padding), 'constant', constant_values=0))
                            #seg_all.append(np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1))
                            case_properties = case_properties + [pr]
                            selected_keys.append(self.current_key)
                        IDP=IDP+1
            j*j+1
        self.IDS = self.IDS + self.batch_size
        # for x in data_all:
        #     print('xshape', x.shape)
        data_all = np.vstack(data_all)
        seg_all = np.vstack(seg_all)
        #data_all = np.expand_dims(data_all, 1)
        #seg_all = np.expand_dims(seg_all, 1)
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}

        



    # def generate_train_batch(self):
    #     selected_keys = self.get_indices()
    #     # preallocate memory for data and seg
    #     data_all = np.zeros(self.data_shape, dtype=np.float32)
    #     seg_all = np.zeros(self.seg_shape, dtype=np.int16)
    #     case_properties = []

    #     for j, i in enumerate(selected_keys):
    #         # oversampling foreground will improve stability of model training, especially if many patches are empty
    #         # (Lung for example)
    #         force_fg = self.get_do_oversample(j)

    #         data, seg, properties = self._data.load_case(i)
    #         case_properties.append(properties)

    #         # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
    #         # self._data.load_case(i) (see nnUNetDataset.load_case)
    #         shape = data.shape[1:]
    #         dim = len(shape)
    #         bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

    #         # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
    #         # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
    #         # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
    #         # later
    #         valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
    #         valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

    #         # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
    #         # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
    #         # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
    #         # remove label -1 in the data augmentation but this way it is less error prone)
    #         this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
    #         data = data[this_slice]

    #         this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
    #         seg = seg[this_slice]

    #         padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
    #         data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
    #         seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

    #     return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
