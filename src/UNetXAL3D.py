import os, sys
fp_alunet = os.path.dirname(os.path.abspath(os.getcwd()))
#sys.path.insert(0,"/media/foellmer/HHD/code/ALUNET/src/nnunet/nnunetv2")
sys.path.insert(0,os.path.join(fp_alunet, "/media/foellmer/HHD/code/ALUNET/src/nnunet/nnunetv2"))
import shutil
from utils.config import CConfig
import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
from glob import glob
import subprocess
import matplotlib.pyplot as plt
from utils.DataAccess import DataAccess
from strategies.RandomStrategyUNet import RandomStrategy
from strategies.EntropyStrategyUNet import EntropyStrategy
from strategies.MCDStrategyUNet import MCDStrategy
from strategies.FullStrategyUNet import FullStrategy
from strategies.BALDStrategyUNet import BALDStrategy
from strategies.USIMFStrategyUNet import USIMFStrategy
from strategies.USIMCStrategyUNet import USIMCStrategy
from strategies.CORESETStrategyUNet import CORESETStrategy
from strategies.BADGELLStrategyUNet import BADGELLStrategy
from strategies.STBStrategyUNet import STBStrategy
from utils.ct import CTImage, CTRef
from nnunet.nnunetv2.run.run_training import run_training
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from scipy.interpolate import interp1d
import seaborn as sns
import pandas as pd

def create_dataset(opts):
    dataset = opts.CLDataset()
    hdf5filepath = os.path.join(opts.fp_active, 'hdf5_all_'+str(opts.dim)+'D.hdf5')
    dataset.create_dataset_mds(opts, hdf5filepath, NumSamples=None, test_set=True)

def init_dataset(opts):

    # Init dataset
    method = 'INIT'
    NewVersion = True
    man = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = man.createALFolderpath(opts, method=method, NewVersion=NewVersion)
    
    # Init dataset
    man.init_datasets(opts, folderDict, exclude=['labeled', 'unlabeled'])
    man.save(save_dict=dict(), save_class=opts.CLPatch, hdf5=True)
    
    # Init nnUNet datasets
    man.init_dataset_mds(opts, man, folderDict)
    
    # Preprocessing raw data
    cmd = "nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean"
    returned_value = subprocess.call(cmd, shell=True)

    # Train model
    run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                 configuration=opts.configuration, 
                 fold=opts.fold,
                 trainer_class_name=opts.nnUNetTrainer, 
                 plans_identifier=opts.plans_identifier, 
                 pretrained_weights=None,
                 num_gpus=1, 
                 use_compressed_data=False, 
                 export_validation_probabilities=True, 
                 continue_training=False, 
                 only_run_validation=False, 
                 disable_checkpointing=False, 
                 val_with_best=True,
                 device=torch.device('cuda'))
    

    # Copy model back
    dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
    fp_results = os.path.join(opts.fp_nnunet, 'nnUNet_results', dname, opts.nnUNetResults)
    fp_results_new = os.path.join(folderDict['modelpath'], opts.nnUNetResults)
    shutil.copytree(fp_results, fp_results_new, dirs_exist_ok=True)
    # Copy labels into model folder
    fp_nnunet = opts.fp_nnunet
    fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
    fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
    fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
    fp_labelsTr_new = os.path.join(folderDict['modelpath'], 'labelsTr')
    shutil.copytree(fp_labelsTr, fp_labelsTr_new, dirs_exist_ok=True)
    opts_dict['dataset_name_or_id'] = str(100 + opts.dataset_dict[opts.dataset]*10 + list(opts.strategy_dict.keys()).index(opts.strategy))
    return man



def alquery(opts):

    NewVersion = True
    VersionUse = None
    strategy_name = opts.strategy
    man = opts.CLManager(fp_dataset=opts.dataset_data)

    folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True)

    # Load train
    man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=True)
    version = folderDict['version']
    if opts.budget>0:
        NumSamples = opts.budget
    else:
        NumSamples = opts.AL_steps[version]
    man.datasets['action'].data = []

    
    # Preprocessing raw data
    cmd = "nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean"
    returned_value = subprocess.call(cmd, shell=True)
    data_query = man.datasets['query'].data
    

    strategy = opts.strategy_dict[strategy_name]()
    previous=True   
    action_round = strategy.query(opts, folderDict, man, data_query, opts.CLPatch, NumSamples=NumSamples, batchsize=500, pred_class='XMaskPred', previous=previous, save_uc=False)
    
    man.datasets['action_round'].data = action_round
    man.save(include=['action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=True)

    # Manual labeling
    if opts.label_manual:
        fp_manual = opts.fp_manual
        dataset = opts.CLDataset()
        dataset.create_action(opts, man, fp_manual, action_round)

    return man
    
def alupdate_auto(opts):

    NewVersion = False
    VersionUse = None
    strategy_name = opts.strategy
    man = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse)

    # Load train
    man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=True)
    version = folderDict['version']
    man.datasets['action'].data=[]
    
    dataset = opts.CLDataset()
    action_round = man.datasets['action_round'].data
    dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
    fp_nnunet = opts.fp_nnunet
    fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
    fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')
    fp_nnUNet_results = os.path.join(fp_nnunet, 'nnUNet_results')
    fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
    fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')

    plans = load_json(os.path.join(fp_nnUNet_preprocessed, dname, 'nnUNetPlans.json'))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(opts.configuration)
    data_update = man.datasets['action_round'].data
    imagenames = sorted(list(np.unique([s.F['imagename'] for s in man.datasets['action_round'].data])))
    if opts.dim==2:
        dataset.load_dataset_mds3D(man, opts, folderDict, data_update)
    else:
        dataset.load_dataset_mds3D(man, opts, folderDict, data_update)

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
                    #sl = s.F['slice']
                    if 'sliceOrg' in s.F:
                        sl = s.F['sliceOrg']
                    else:
                        sl = s.F['slice']
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
    
    # Preprocessing raw data
    cmd = "nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean"
    returned_value = subprocess.call(cmd, shell=True)
    
    # Copy split file
    fip_split_model = os.path.join(folderDict['modelpath'], 'splits_final.json')
    shutil.copyfile(opts.fip_split, fip_split_model)

    # Update datasets
    man.datasets['action'].data = man.datasets['action'].data + action_round
    man.datasets['train'].data = man.datasets['train'].data + action_round
    man.datasets['query'].delete(action_round)
    man.datasets['action_round'].delete(action_round)
    man.save(include=['action'], save_dict={}, save_class=opts.CLPatch, hdf5=True)
    man.save(include=['train', 'valid', 'query'], save_dict={}, save_class=opts.CLPatch, hdf5=True)
    man.save(include=['train'], save_dict={}, save_class=opts.CLPatch, hdf5=True)
    
    return man


    
def altrain(opts):

    NewVersion = False
    VersionUse = None
    strategy_name = opts.strategy
    man = opts.CLManager(fp_dataset=opts.dataset_data)
    folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse)

    # Load train
    man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=True)

    # Preprocessing raw data
    cmd = "nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean"
    returned_value = subprocess.call(cmd, shell=True)
    
    # Train model
    run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                 configuration=opts.configuration, 
                 fold=opts.fold, 
                 trainer_class_name=opts.nnUNetTrainer, 
                 plans_identifier=opts.plans_identifier, 
                 #pretrained_weights=os.path.join(folderDict['modelpath_prev'], opts.nnUNetResults,'fold_'+str(opts.fold), 'checkpoint_best.pth'),
                 pretrained_weights=os.path.join(folderDict['modelpath_prev'], opts.nnUNetResults,'fold_'+str(opts.fold), 'checkpoint_final.pth'),
                 #pretrained_weights=None,
                 num_gpus=1, 
                 use_compressed_data=False, 
                 export_validation_probabilities=True, 
                 continue_training=False, 
                 only_run_validation=False, 
                 disable_checkpointing=False, 
                 val_with_best=True,
                 device=torch.device('cuda'))

    # Copy model back
    dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
    fp_results = os.path.join(opts.fp_nnunet, 'nnUNet_results', dname, opts.nnUNetResults)
    fp_results_new = os.path.join(folderDict['modelpath'], opts.nnUNetResults)
    shutil.copytree(fp_results, fp_results_new, dirs_exist_ok=True)
    # Copy labels into model folder
    fp_nnunet = opts.fp_nnunet
    fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
    fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
    fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
    fp_labelsTr_new = os.path.join(folderDict['modelpath'], 'labelsTr')
    shutil.copytree(fp_labelsTr, fp_labelsTr_new, dirs_exist_ok=True)
    
    return man

    
def alfull(opts):

    # Query all samples for full training
    man = alquery(opts)
    
    # Update labels
    #man = alupdate_auto(opts) 
    dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
    fp_nnunet = opts.fp_nnunet
    fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
    fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
    fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
    name = 'Task' + str(opts.dataset_dict[opts.dataset]).zfill(2) + '_' + opts.dataset
    fp_labelsTr_org = os.path.join(opts.fp_mds, name, 'labelsTr')
    shutil.rmtree(fp_labelsTr) 
    if os.path.exists(fp_labelsTr):
        shutil.rmtree(fp_labelsTr)
    shutil.copytree(fp_labelsTr_org, fp_labelsTr)
        

    # Preprocessing raw data
    cmd = "nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean"
    returned_value = subprocess.call(cmd, shell=True)
    
    # Train model
    run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                 configuration=opts.configuration, 
                 fold=opts.fold, 
                 trainer_class_name=opts.nnUNetTrainer, 
                 plans_identifier=opts.plans_identifier, 
                 pretrained_weights=None,
                 num_gpus=1, 
                 use_compressed_data=False, 
                 export_validation_probabilities=True, 
                 continue_training=False, 
                 only_run_validation=False, 
                 disable_checkpointing=False, 
                 val_with_best=True,
                 device=torch.device('cuda'))

    # Copy model back
    dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
    fp_results = os.path.join(opts.fp_nnunet, 'nnUNet_results', dname, opts.nnUNetResults)
    fp_results_new = os.path.join(man.folderDict['modelpath'], opts.nnUNetResults)
    shutil.copytree(fp_results, fp_results_new, dirs_exist_ok=True)
    # Copy labels into model folder
    fp_nnunet = opts.fp_nnunet
    fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
    fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
    fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
    fp_labelsTr_new = os.path.join(man.folderDict['modelpath'], 'labelsTr')
    shutil.copytree(fp_labelsTr, fp_labelsTr_new, dirs_exist_ok=True)
    
    return man



if __name__ == '__main__':
    
    fp_alunet = os.path.dirname(os.path.abspath(os.getcwd()))
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name', type=str, default='Hippocampus')
    parser.add_argument('--strategy', help='AL strategy', type=str, default='USIMC')
    parser.add_argument('--emulation', help='Function to execute', type=str, default=False)
    parser.add_argument('--func', help='Function to execute', type=str, default='')
    parser.add_argument('--dim', help='Number of dimensions (2D or 3D model)', type=int, default=2)
    parser.add_argument('--fold', help='Cross validation fold number', type=int, default=0)
    parser.add_argument('--budget', help='Budget', type=int, default=100)
    opts = parser.parse_args()
    opts_dict = vars(opts)

    opts_dict['dataset_dict'] = {'LITS': 0, 'Spleen': 9, 'Hippocampus': 4, 'Liver': 3}
    opts_dict['dataset_mds'] = ['Spleen', 'Hippocampus', 'Liver']
    opts_dict['strategy_dict'] = {
        'INIT': None,                           # 1X0
        'RANDOM': RandomStrategy,               # 1X1
        'ENTROPY': EntropyStrategy,             # 1X2
        'FULL': FullStrategy,                   # 1X3
        'MCD': MCDStrategy,                     # 1X4
        'BALD': BALDStrategy,                   # 1X5
        'USIMF': USIMFStrategy,                 # 1X6
        'CORESET': CORESETStrategy,             # 1X7
        'BADGELL': BADGELLStrategy,             # 1X8
        'STB': STBStrategy,                     # 1X9
        'USIMC': USIMCStrategy}                 # 2X0             

    # Define classes
    from utils.ALUNet import UNetManager, UNetDataset, UNetSample, UNetPatch
    opts_dict['CLDataset'] = UNetDataset
    opts_dict['CLSample'] = UNetSample
    opts_dict['CLPatch'] = UNetPatch
    opts_dict['CLManager'] = UNetManager
    opts_dict['fp_images'] = ''
    opts_dict['fp_references'] = ''  
    opts_dict['fp_references_org'] = '' 
    opts_dict['AL_steps'] = [100] + [100 for i in range(100)]
    opts_dict['epochs_init'] = 500
    opts_dict['epochs_refine'] = 300
    opts_dict['epoch_valid'] = 15
    opts_dict['savePretrained_all'] = True
    strat = list(opts.strategy_dict.keys()).index(opts.strategy)
    if strat>=10:
        strat = 100 + strat-10
    opts_dict['dataset_name_or_id'] = str(100 + opts.dataset_dict[opts.dataset]*10 + strat)
    opts_dict['dataset_name'] = 'ALUNet'
    #opts_dict['nnUNetTrainer'] = 'nnUNetTrainer_3epochs'
    opts_dict['nnUNetTrainer'] = 'nnUNetTrainer_300epochs' 
    if opts.dataset in ['EATAL']:
        opts_dict['label_manual'] = True
    else:
        opts_dict['label_manual'] = False
    opts_dict['configuration'] = '2d'
    opts_dict['nnUNetResults'] = opts_dict['nnUNetTrainer']+'__nnUNetPlans__'+opts_dict['configuration']
    opts_dict['plans_identifier'] = 'nnUNetPlans'
    opts_dict['dataset_data'] = os.path.join(fp_alunet, '/src/modules', opts.dataset, 'data')
    opts_dict['fp_active'] = os.path.join(fp_alunet, 'output', opts.dataset, 'AL')
    opts_dict['fip_split'] = os.path.join(opts_dict['fp_active'], 'INIT', 'INIT_V01', 'model', 'splits_final.json')
    opts_dict['fp_mds'] = os.path.join(fp_alunet,'data','mds')
    opts_dict['fp_nnunet'] = os.path.join(fp_alunet,'data','nnunet')
    os.makedirs(opts_dict['fp_active'], exist_ok=True)
    os.makedirs(opts_dict['fp_mds'], exist_ok=True)

    # Start active learning
    if opts.func=='dataset_patches':      
        create_dataset(opts)
    elif opts.func=='alinit':      
        init_dataset(opts)
    elif opts.func=='alquery':      
        alquery(opts)
    elif opts.func=='alupdate_auto':      
        alupdate_auto(opts) 
    elif opts.func=='altrain':      
        altrain(opts) 
    elif opts.func=='alfull':
        opts_dict['nnUNetTrainer'] = 'nnUNetTrainer' 
        opts_dict['strategy']='FULL'
        opts_dict['dataset_name_or_id'] = str(100 + opts.dataset_dict[opts.dataset]*10 + list(opts.strategy_dict.keys()).index(opts.strategy))
        alfull(opts)
    elif opts.func=='alcont':
        # Create dataset
        fip_dataset = os.path.join(opts_dict['fp_active'], 'hdf5_all_'+str(opts_dict['dim'])+'D.hdf5')
        if not os.path.isfile(fip_dataset):
            create_dataset(opts)
        # Init dataset
        fip_init = os.path.join(opts_dict['fp_active'], 'INIT')
        if not os.path.isdir(fip_init):
            strategyTmp = opts.strategy
            dataset_name_or_idTmp = opts.dataset_name_or_id
            opts.strategy='INIT'
            opts.dataset_name_or_id = '1'+opts.dataset_name_or_id[1]+'0'
            man = init_dataset(opts)
            opts.strategy=strategyTmp
            opts.dataset_name_or_id=dataset_name_or_idTmp
        # Active learning round
        for i in range(10):
            man = alquery(opts)
            man = alupdate_auto(opts) 
            man = altrain(opts)


    
    
