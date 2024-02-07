# -*- coding: utf-8 -*-

import sys, os
import tables
from tables import *
from tables import Atom, Filters
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
import torch
from more_itertools import locate, iter_index

def find_idx(idx_src, idx_dst):

    idx_src = np.array(idx_src)
    idx_dst = np.array(idx_dst)

    idx1 = np.argsort(idx_src)
    l1s=sorted(idx_src)
    
    idx2 = np.argsort(idx_dst)
    l2s=sorted(idx_dst)
    
    k=0
    idx=np.ones((len(l1s)), dtype=np.int64)*-1
    for i,x in enumerate(l1s):
        for j,y in enumerate(l2s[k:]):
            if x==y:
                idx[idx1[i]] = idx2[k+j]
                #k=k+j+1
                k=k+j
                break
    idx=idx[idx>-1]
    return idx


# idx_src = [2, 4, 6]
# idx_dst = [5, 2, 7, 4, 6, 7, 8]

# idx_src=np.array([i for i in range(5)])
# np.random.shuffle(idx_src)

# idx_dst=np.array([i for i in range(5)])
# np.random.shuffle(idx_dst)
                
# idx = find_idx(idx_src, idx_dst)

class DataAccess:
    """
    DataAccess class
    """
    
    def __init__(self):
        pass
    
    def save(self, filepath, datadict=defaultdict(None, {})):
        
        # Close all open files
        tables.file._open_files.close_all()

        # Open file
        FILTERS = tables.Filters(complib='zlib', complevel=5)
        file = tables.open_file(filepath, mode='a', filters=FILTERS)
        # Extract file keys
        file_keys = [k.name for k in file.root]
        for key in datadict:
            if len(datadict[key])>0:
                datatype = type(datadict[key][0])
            else:
                datatype = np.ndarray
            
            # Check if dataype is list of strings
            if datatype==str:
                key_c = key
                
                d = datadict[key]
                # If key alread in file, append data
                if not key_c in file_keys:
                    atom = Atom.from_kind('string', 1000)
                    array_c = file.create_earray(file.root, key_c, atom, shape=(0,))
                    array_c.append(np.array(d))
            else:
                # Iterate over data in lists
                for i, c in enumerate(datadict[key]):
                    # If key alread in file, append data
                    key_c = key + '_' + str(i).zfill(2)
                    d = datadict[key][i]
                    if key_c in file_keys:
                        for idx in range(d.shape[0]):
                            file.root[key_c].append(d[idx:idx+1])
                    # If key not in file, create data array
                    else:
                        if len(d)>0:
                            atom = tables.Atom.from_dtype(np.dtype((d.dtype, ())))
                            shape=tuple([0] + list(d.shape[1:]))
                            filters = Filters(complevel=4)
                            array_c = file.create_earray(file.root, key_c, atom, shape, filters=filters)
                            for idx in range(d.shape[0]):
                                array_c.append(d[idx:idx+1])

        # Close file
        file.close()

    def read(self, hdf5filepath, datadict=defaultdict(None, {}), NumSamples=[None,None]):
        # self=dataaccess
        file = tables.open_file(hdf5filepath, mode='r')
        file_keys = [k.name for k in file.root]
        file_keys = sorted(file_keys)
        for key in file_keys:
            NumSamplesKey = NumSamples.copy()
            keysplit = key.split('_')
            if len(keysplit)==2:
                array = file.root[key]
                # Update NumSamplesKey
                if NumSamplesKey[0] is None:
                    NumSamplesKey[0]=0
                if NumSamplesKey[1] is None:
                    NumSamplesKey[1]=array.nrows
                if keysplit[0] in list(datadict.keys()):
                    datadict[keysplit[0]].append(array[NumSamplesKey[0]:NumSamplesKey[1]])
            else:
                if keysplit[0]  in list(datadict.keys()):
                    array = file.root[key]
                    datadict[keysplit[0]] = array[:]
        file.close()
        return datadict

    def read_size(self, filepath):
        file = tables.open_file(filepath, mode='r')
        file_keys = [k.name for k in file.root]
        file_keys = sorted(file_keys)
        datadict_size=dict()
        for key in file_keys:
            keysplit = key.split('_')
            if len(keysplit)==2:
                array = file.root[key]
                datadict_size[keysplit[0]] = len(array)
            else:
                datadict_size[keysplit[0]] = None
        file.close()
        return datadict_size
    
    def read_labels(self, filepath):
        file = tables.open_file(filepath, mode='r')
        Xlabel = list(file.root['Xlabel'])
        Xlabel = {x.decode("utf-8"): i  for i,x in enumerate(Xlabel)}
        Ylabel = list(file.root['Ylabel'])
        Ylabel = {x.decode("utf-8"): i  for i,x in enumerate(Ylabel)}
        return Xlabel, Ylabel
    
    def readRandomized(self, filepath, datadict=defaultdict(None, {}), NumSamples=100, sameIdx=['train', 'test', 'valid'], shuffle=True, return_idx=False):
        
        # self = dataaccess
        
        file = tables.open_file(filepath, mode='r')
        data_keys = list(datadict.keys())
        
        # Extract sorted file keys
        file_keys = [k.name for k in file.root]
        file_keys = sorted(file_keys, key=len)
        
        # Get keys and corresponding number of rows
        keyrows = defaultdict(None, {})
        for key in file_keys:
            keysplit = key.split('_')
            if keysplit[0] in data_keys or key in data_keys:
                if len(keysplit)==2:
                    array = file.root[key]
                    if NumSamples is None:
                        nrows = array.nrows
                    else:
                        nrows = min(NumSamples, array.nrows)
                    idxAll = [i for i in range(array.nrows)]
                    if shuffle:
                        random.shuffle(idxAll)
                    idx = idxAll[0:nrows]
                    keyrows[keysplit[0]] = idx
                else:
                    keyrows[keysplit[0]] = []
                    
        # Create same index for Xtrain and  Ytrain as well as Xtest, Ytest and Xvalid and Yvalid and update keyrows
        keyrows_same = defaultdict(None, {})
        for same in sameIdx:
            for k in keyrows.keys():
                if same in k:
                    keyrows_same[same] = keyrows[k]
                    
        for ksame in keyrows_same.keys():
            for k in keyrows.keys():
                if ksame in k:
                    keyrows[k] = keyrows_same[ksame]
                       
        # Extract data based in idx
        for key in file_keys:
            #print('key', key)
            keysplit = key.split('_')
            if keysplit[0] in data_keys or key in data_keys:
                if len(keysplit)==2:
                    idx_array = keyrows[keysplit[0]]
                    #array = np.vstack([np.expand_dims(file.root[key][i], axis=0) for i in idx_array])
                    array=[]
                    pbar = tqdm(total=len(idx_array))
                    pbar.set_description("Loading " + key)
                    for i in idx_array:
                        pbar.update()
                        array.append(np.expand_dims(file.root[key][i], axis=0))
                    pbar.close()
                    array = np.vstack(array)
                    
                    if len(array.shape)==1:
                        #datadict[keysplit[0]].append(array[:])
                        if key in data_keys:
                            datadict[key].append(array[:])
                        else:
                            datadict[keysplit[0]].append(array[:])
                    else:
                        #datadict[keysplit[0]].append(array)
                        if key in data_keys:
                            datadict[key].append(array)
                        else:
                            datadict[keysplit[0]].append(array)
                else:
                    array = file.root[key]
                    datadict[keysplit[0]] = array[:]
        file.close()
        
        if return_idx:
            return datadict, idx_array
        else:
            return datadict

    def readIdx(self, hdf5filepath, datadict=defaultdict(None, {}), idx=[]):
        # self=dataaccess
        
        file = tables.open_file(hdf5filepath, mode='r')
        data_keys = list(datadict.keys())
        
        # Extract sorted file keys
        file_keys = [k.name for k in file.root]
        file_keys = sorted(file_keys, key=len)
        
        # Read IDX
        XID_00 = list(np.expand_dims(file.root['XID_00'], axis=0)[0])
        IDX = [XID_00.index(i) for i in idx]
                               
        # Extract data based in idx
        for key in file_keys:
            #print('key', key)
            keysplit = key.split('_')
            if keysplit[0] in data_keys:
                if len(keysplit)==2:
                    #idx_array = keyrows[keysplit[0]]
                    #array = np.vstack([np.expand_dims(file.root[key][i], axis=0) for i in idx_array])
                    array=[]
                    pbar = tqdm(total=len(IDX))
                    pbar.set_description("Loading " + key)
                    for i in IDX:
                        pbar.update()
                        array.append(np.expand_dims(file.root[key][i], axis=0))
                    pbar.close()
                    array = np.vstack(array)
                    
                    if len(array.shape)==1:
                        datadict[keysplit[0]].append(array[:])
                    else:
                        datadict[keysplit[0]].append(array)
                else:
                    array = file.root[key]
                    datadict[keysplit[0]] = array[:]
        file.close()
        return datadict
    
    # def readIdx(self, hdf5filepath, datadict=defaultdict(None, {}), idx=[]):
    #     # self=dataaccess
        
    #     file = tables.open_file(hdf5filepath, mode='r')
    #     data_keys = list(datadict.keys())
        
    #     # Extract sorted file keys
    #     file_keys = [k.name for k in file.root]
    #     file_keys = sorted(file_keys, key=len)
                               
    #     # Extract data based in idx
    #     for key in file_keys:
    #         #print('key', key)
    #         keysplit = key.split('_')
    #         if keysplit[0] in data_keys:
    #             if len(keysplit)==2:
    #                 #idx_array = keyrows[keysplit[0]]
    #                 #array = np.vstack([np.expand_dims(file.root[key][i], axis=0) for i in idx_array])
    #                 array=[]
    #                 pbar = tqdm(total=len(idx))
    #                 pbar.set_description("Loading " + key)
    #                 for i in idx:
    #                     pbar.update()
    #                     array.append(np.expand_dims(file.root[key][i], axis=0))
    #                 pbar.close()
    #                 array = np.vstack(array)
                    
    #                 if len(array.shape)==1:
    #                     datadict[keysplit[0]].append(array[:])
    #                 else:
    #                     datadict[keysplit[0]].append(array)
    #             else:
    #                 array = file.root[key]
    #                 datadict[keysplit[0]] = array[:]
    #     file.close()
    #     return datadict

    def dataFrameToList(self, df_features):
        F=[]
        Ylabel=[]
        for col in df_features:
            a = np.array(list(df_features[col]))
            if a.dtype==np.dtype('<U5') or a.dtype==np.dtype('<U83') or a.dtype==np.dtype('<U65'):
                F.append(np.array(list(df_features[col]), dtype = np.string_))
                Ylabel.append(col)
            else:
                if not a.dtype is np.dtype('object'):
                    F.append(np.array(list(df_features[col])))
                    Ylabel.append(col)
        return F, Ylabel
    
    
    
    def save_dict(self, filepath, d=dict({})):
        
        # self=da
        
        # Close all open files
        tables.file._open_files.close_all()

        def save_array(file, key, value, key_arr):
            #print('key_arr1234', key, 'key_arr12345', key_arr)
            file_keys = [k.name for k in file.root]
            datatype = type(value)
            if datatype==str:
                #valued = datadict[k0]
                # If key alread in file, append data
                if key_arr in file_keys:
                    #for idx in range(value.shape[0]):
                    #    file.root[key_arr].append(value[idx:idx+1])
                    file.root[key_arr].append(value)
                else:
                    atom = Atom.from_kind(('string', 5000))
                    #array_c = file.create_earray(file.root, key_arr, atom, shape=(0,))
                    array_c = file.create_vlarray(file.root, key_arr, atom, shape=(0,))
                    array_c.append(np.array(value))
            else:
                if key_arr in file_keys:
                    file.root[key_arr].append(value)
                    #for idx in range(value.shape[0]):
                    #    file.root[key_arr].append(value[idx:idx+1])
                # If key not in file, create data array
                else:
                    #print('keylen', len(value))
                    if len(value)>0:
                        if not isinstance(value, np.ndarray):
                            raise ValueError('Value has to be a numpy array.')
                        #print('value123', value)
                        atom = tables.Atom.from_dtype(np.dtype((value.dtype, ())))
                        shape=tuple([0] + list(value.shape[1:]))
                        filters = Filters(complevel=4)
                        array_c = file.create_earray(file.root, key_arr, atom, shape, filters=filters)
                        array_c.append(value)
                        #for idx in range(value.shape[0]):
                        #    array_c.append(value[idx:idx+1])
                            
        def save_d(file, d, key_arr):
            for key in d:
                if len(key.split('_'))>1:
                    #print('Key:', key)
                    raise ValueError('Underscore not allowed in key name.')
                value = d[key]
                if isinstance(value, dict):
                    save_d(file, value, key_arr + '_' + key)
                else:
                    save_array(file, key, value, key_arr + '_' + key)

        # Open file
        file = tables.open_file(filepath, mode='a')
        key_arr='root'
        save_d(file, d, key_arr)
        
        file.close()
        
                
    def read_dict(self, filepath, idx=[], ID=[], keys_select=[], dtype=np.ndarray, ignoreNoID=False, debug=True):
        # self=dataaccess
        str_utf8 = np.vectorize(lambda x: x.decode('utf8'))
        
        d=dict({})
        def read_d(file, key, d, key_file, idx):
            #print('key', key, 'key_file', key_file)
            # key=key_file
            keysplit = key.split('_')
            key0 = keysplit[0]
            if len(keysplit)>1:
                key = '_'.join(keysplit[1:])
                if key0 in keys_select:
                    keys_select.append(key)
                    
                if key0 in d:
                    dn=d[key0]
                else:
                    dn = dict()
                _, _ =read_d(file, key, dn, key_file, idx)
                if dn:
                    d[keysplit[0]]=dn
                #print('keyX', keysplit[0])
            else:
                if key in keys_select or not keys_select:
                    if idx:  
                        if isinstance(idx, list):
                            array=[]
                            if debug:
                                pbar = tqdm(total=len(idx))
                                pbar.set_description("Loading " + key)
                            for i in idx:
                                if debug: pbar.update()
                                # test
                                #q = file.root[key_file][i:i+5]
                                #print('q123', q.shape)
                                #!!!
                                #array.append(np.expand_dims(file.root[key_file][i], axis=0))
                                #array.append(np.expand_dims(file.root[key_file][i], axis=0))
                                try:
                                    array.append(np.expand_dims(file.root[key_file][i], axis=0))
                                except:
                                    print('ERROR HDF5:', i)
                                #     array.append(np.expand_dims(file.root[key_file][i-1], axis=0))
                            if debug: pbar.close()
                            array = np.vstack(array)
                            d[key]=array[:]
                        elif isinstance(idx, tuple):
                            # array=[]
                            # if debug:
                            #     pbar = tqdm(total=len(idx))
                            #     pbar.set_description("Loading " + key)
                            # for i in idx:
                            #     if debug: pbar.update()
                            #     # test
                            #     #q = file.root[key_file][i:i+5]
                            #     #print('q123', q.shape)
                            #     array.append(np.expand_dims(file.root[key_file][i], axis=0))
                            # if debug: pbar.close()
                            array = file.root[key_file][idx[0]:idx[1]]
                            d[key]=array[:]
                    else:
                        array = file.root[key_file]
                        d[key]=array[:]
                    # Convert string into utf8
                    if '|S' in str(d[key].dtype):
                        d[key] = str_utf8(d[key])
                    else:
                        if dtype==torch.Tensor:
                            d[key] = torch.from_numpy(d[key])
                    
                    # if idx:  
                    #     array=[]
                    #     if debug:
                    #         pbar = tqdm(total=len(idx))
                    #         pbar.set_description("Loading " + key)
                    #     for i in idx:
                    #         if debug: pbar.update()
                    #         # test
                    #         #q = file.root[key_file][i:i+5]
                    #         #print('q123', q.shape)
                    #         array.append(np.expand_dims(file.root[key_file][i], axis=0))
                    #     if debug: pbar.close()
                    #     array = np.vstack(array)
                    #     d[key]=array[:]
                    # else:
                    #     array = file.root[key_file]
                    #     d[key]=array[:]
                    # # Convert string into utf8
                    # if '|S' in str(d[key].dtype):
                    #     d[key] = str_utf8(d[key])
                    # else:
                    #     if dtype==torch.Tensor:
                    #         d[key] = torch.from_numpy(d[key])
                        
                        
            return d, key0
            
        
        # self=dataaccess
        file = tables.open_file(filepath, mode='r')
        file_keys = [k.name for k in file.root]
        
        # Check if all keys_select exist otherwise return None
        keys_found=True
        for k0 in keys_select:
            key_found=False
            for k1 in file_keys:
                if k0 in k1:
                    key_found=True
            if not key_found:
                print('Key:', k0, 'not found in hdf5 file!')
            keys_found = keys_found and key_found
        if not keys_found:
            file.close()
            return None
            
                    
        
        #print('file_keys123', 'root_ID' in file_keys)
        #print('ID678', type(ID))
        if ID and ('root_ID' in file_keys):
            IDArray = list(file.root['root_ID'][:])
            if ignoreNoID:
                idx = find_idx(ID, IDArray)
            else:
                #idx = [IDArray.index(x) for x in ID]
                idx = find_idx(ID, IDArray)
            idx = list(idx)
            #print('idx123', len(idx))
        
            if not idx:
                return None
                
        for key_file in file_keys:
            d, key0 = read_d(file, key_file, d, key_file, idx)
            #d[key0] = dn
            
        file.close()
        return d['root']
    


    def get_ID_dataset(self, fip_hdf5_list, dataset='train'):
        # Filter ID of samples by processing history
        # self = dataaccess
        IDList=[]
        for fip_hdf5 in fip_hdf5_list[::-1]:
            #print('fip_hdf5123', fip_hdf5)
            if os.path.isfile(fip_hdf5):
                if not IDList:
                    data = self.read_dict(fip_hdf5, keys_select=['ID', 'dataset'])
                    if dataset is not None:
                        dset = data['F']['dataset']
                        ID = data['ID'][dset==dataset]
                    else:
                        ID = data['ID']
                    IDList.append(ID)
                else:
                    data = self.read_dict(fip_hdf5, keys_select=['ID', 'dataset'])
                    if dataset is not None:
                        dset = data['F']['dataset']
                        ID = data['ID'][dset==dataset]
                    else:
                        ID = data['ID']
                    idx = ~np.in1d(ID, IDList[-1])
                    IDList.append(ID[idx])
        IDList = np.hstack(IDList)
        return IDList 
                    
                    
    def read_dict_list(self, filepath, idx=[], ID=[], keys_select=[], dtype=np.ndarray):
        
        # filepath = fip_data
        # self = dataaccess
        # keys_select=['MAINEXIST', 'REGIONEXIST']
        
        def combine_dict(value_base, value):
            if isinstance(value_base, dict):
                for key in value_base: 
                    value_base[key] = combine_dict(value_base[key], value[key])
                return value_base
            else:
                value_base = np.vstack([value_base, value])
                return value_base
        
        data_all = None
        for i,fip in enumerate(filepath):
            data = self.read_dict(fip, ID=ID, keys_select=['ID'], dtype=np.ndarray, ignoreNoID=True)
            # keys = self.read_keys(fip_hdf5_list_read[-1])
            if data is not None:
                if len(data['ID'].shape)==2:
                    IDsel = list(data['ID'][:,0])
                else:
                    IDsel = list(data['ID'])
                if data_all is None:
                    print('fip123', fip)
                    #sys.exit()
                    data_all = self.read_dict(fip, ID=IDsel, keys_select=keys_select, dtype=np.ndarray, ignoreNoID=False)
                    #print('data_all123', data_all.keys())
                    if data_all is not None:
                        # Update ID
                        #ID = [i for i in ID if i not in IDsel]
                        ID = list(np.array(ID)[~np.isin(ID, IDsel)])
                else:
                    data_new = self.read_dict(fip, ID=IDsel, keys_select=keys_select, dtype=np.ndarray, ignoreNoID=False)
                    #print('data_new123', data_new.keys())
                    if data_new is not None:
                        data_all = combine_dict(data_all, data_new)
                        # Update ID
                        #ID = [i for i in ID if i not in IDsel]
                        ID = list(np.array(ID)[~np.isin(ID, IDsel)])
            if not ID:
                return data_all
                #sys.exit()

        return data_all
            
   
    def read_keys(self, filepath):
        file = tables.open_file(filepath, mode='r')
        file_keys = [k.name for k in file.root]
        #keys = [key.split('_')[-1] for key in file_keys]
        keys = [k for k in file_keys]
        file.close()
        return keys
            
    def dataFrameToDict(self, df_features):
        F=dict()
        for col in df_features:
            a = np.array(list(df_features[col]))
            if '<U' in str(a.dtype):
                F[col] = np.array(list(df_features[col]), dtype = '|S100')
            else:
                F[col] = np.array(list(df_features[col]))
        return F
        
                
################################

def test():
    # Save data
    ID = np.array([i+1 for i in range(10)])
    Xtrain = np.ones((10,1,10,1))
    Xvalid = np.ones((10,1,10,1))
    F = dict({'feature0': np.ones((10,1,10,1)), 'feature1': np.ones((10,1,10,1))})
    d = dict({'ID': ID, 'Xtrain': Xtrain, 'Xvalid': Xvalid, 'F': F})
    filepath = '/mnt/SSD2/cloud_data/Projects/CTP/src/tmp/table.hdf5'
    da = DataAccess()
    da.save_dict(filepath, d)
    
    d=da.read_dict(filepath, idx=[3,5])
    
    d=da.read_dict(filepath, idx=[3,5], ID=[6,7])
    
    
    # Load data
    datadict = defaultdict(None, {'Xtrain': [], 'Xvalid': [], 'Xtest': [], 'Ytrain': [], 'Yvalid': [], 'Ytest': [], 'Xlabel': [], 'Ylabel': []})
    da = DataAccess()
    d = da.read(filepath, datadict, NumSamples=[3,10])
    
    
    
    dataaccess = DataAccess()
    fip_hdf5 = ['/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data/SegmentCACS/AL/INIT/INIT_V01/manager/train/data/sl.hdf5',
                '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/data/SegmentCACS/AL/INIT/INIT_V01/manager/valid/data/sl.hdf5']
    

    data = dataaccess.read_dict(fip_hdf5[1], keys_select=['ID'], dtype=np.ndarray)
    
    data = dataaccess.read_dict(fip_hdf5[1], dtype=np.ndarray)
      
    
    ID=[5,6,60,61]
    data = dataaccess.read_dict_list(fip_hdf5, ID=ID, keys_select=['XMask', 'XImage', 'MAINEXIST', 'ID'], dtype=np.ndarray)
    
    
    