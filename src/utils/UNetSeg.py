# -*- coding: utf-8 -*-
import os, sys
from collections import defaultdict
import shutil
import time
import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
import warnings
from torchmetrics import ConfusionMatrix
from torch import nn, optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from utils.DLBaseModel import DLBaseModel, SaveState
from utils.helper import EarlyStopping
#from config.config import CConfig
import keyboard
#from metrices.metrices_new import accuracy_bin, f1_score, precision_score, recall_score, accuracy_bin_multi, diceCoeff, diceCoeff_multi
#from metrices.metrices_new import precision_score_multi, recall_score_multi, f1_score_multi, specificity, sensitivity, confusion_multi_mat, f1_score_metric
#from modules.ALLiver.ALLiverModel import ALLiverModel, SaveState
from utils.const_dropout import constDropout, BayesianModule
from utils.metrices_new import accuracy, specificity, recall, f1, confusion
from torchcontrib.optim import SWA
import torchcontrib
import copy
from collections import OrderedDict

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )
    
def update_teacher_model(student, teacher, keep_rate=0.996):
    # if comm.get_world_size() > 1:
    #     student_model_dict = {
    #         key[7:]: value for key, value in self.model.state_dict().items()
    #     }
    # else:
    #     student_model_dict = self.model.state_dict()
    
    student_model_dict = student.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    teacher.load_state_dict(new_teacher_dict)
    
class UNetSeg(DLBaseModel):
    
    """
    UNetSeg model
    """

    def __init__(self, settingsfilepath, overwrite=False):
        props = defaultdict(lambda: None,
            NumChannelsIn = 1,
            NumChannelsOut = 2,
            Input_size = (512, 512, 1),
            Output_size = (512, 512, 2),
            device = 'cuda',
            modelname = 'UNetSeg',
            savePretrainedEpochMin=0
        )
        DLBaseModel.__init__(self, settingsfilepath=settingsfilepath, overwrite=overwrite, props=props)
        

       
    def create(self, params):
        """ Create deep learning model 
            
        :param params:  Dictionary of model parameters for model_01
        :type params: dict
        """
            
        props = self.props
        self.params=params

        class Conv_down(nn.Module):
            def __init__(self, in_ch, out_ch, dropout=0.0):
                super(Conv_down, self).__init__()
                self.down = nn.Conv2d(in_ch, out_ch,  kernel_size=4, stride=2, padding=1)
                self.relu1 = nn.LeakyReLU(0.2)
                self.dropout = constDropout(p=dropout, const=params['const_dropout'])
                self.conv = nn.Conv2d(out_ch, out_ch,  kernel_size=3, stride=1, padding=1)
                self.norm = nn.BatchNorm2d(out_ch)
                self.relu2 = nn.LeakyReLU(0.2)
                self.down.weight.data.normal_(0.0, 0.1)
                self.conv.weight.data.normal_(0.0, 0.1)
        
            def forward(self, x):
                x = self.down(x)
                x = self.relu1(x)
                x = self.dropout(x)
                x = self.conv(x)
                x = self.norm(x)
                x = self.relu2(x)
                return x

        class Conv_up(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size_1=3, stride_1=1, padding_1=1, kernel_size_2=3, stride_2=1, padding_2=1, dropout=0.0):
                super(Conv_up, self).__init__()
                self.up = nn.UpsamplingBilinear2d(scale_factor=2)
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size_1, padding=padding_1, stride=stride_1)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size_2, padding=padding_2, stride=stride_2)
                self.relu1 = nn.LeakyReLU(0.2)
                self.relu2 = nn.LeakyReLU(0.2)
                self.dropout = constDropout(p=dropout, const=params['const_dropout'])
                self.norm = nn.BatchNorm2d(out_ch)
                self.conv1.weight.data.normal_(0.0, 0.1)
                self.conv2.weight.data.normal_(0.0, 0.1)
        
            def forward(self, x1, x2):
                x1 = self.up(x1)
                x = torch.cat((x1, x2), dim=1)
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.norm(x)
                x = self.relu2(x)
                return x
            
        class UNet(nn.Module):
            def __init__(self):
                super(UNet, self).__init__()
                
                dropout1 = 0.0

                self.conv00 = nn.Conv2d(1, 8, kernel_size=5, padding=2, stride=1)
                self.relu00 = nn.LeakyReLU(0.2)
                self.conv01 = nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=1)
                self.relu01 = nn.LeakyReLU(0.2)
                
                self.conv_down1 = Conv_down(16, 16, dropout=dropout1)
                self.conv_down2 = Conv_down(16, 32, dropout=dropout1)
                self.conv_down3 = Conv_down(32, 32, dropout=dropout1)
                self.conv_down4 = Conv_down(32, 64, dropout=dropout1)
                self.conv_down5 = Conv_down(64, 64, dropout=dropout1)
                self.conv_down6 = Conv_down(64, 64, dropout=dropout1)
                self.conv_down7 = Conv_down(64, 128, dropout=dropout1)
                self.conv_down8 = Conv_down(128, 128, dropout=dropout1)
                
                self.dropout0 = constDropout(p=0.5, const=params['const_dropout'])
                
                self.conv_up1 = Conv_up(128+128, 128, dropout=dropout1)
                self.conv_up2 = Conv_up(128+64, 64, dropout=dropout1)
                self.conv_up3 = Conv_up(64+64, 64, dropout=dropout1)
                self.conv_up4 = Conv_up(64+64, 64, dropout=dropout1)
                self.conv_up5 = Conv_up(64+32, 32, dropout=dropout1)
                self.conv_up6 = Conv_up(32+32, 32, dropout=dropout1)
                self.conv_up7 = Conv_up(32+16, 16, dropout=dropout1)
                self.conv_up8 = Conv_up(16+16, 16, dropout=dropout1)

                self.conv0_out = nn.Conv2d(16, 8,  kernel_size=3, stride=1, padding=1)
                self.relu0_out = nn.LeakyReLU(0.2)
                self.conv1_out = nn.Conv2d(8, 2,  kernel_size=3, stride=1, padding=1)

                # dropout1 = 0.0

                # self.conv00 = nn.Conv2d(1, 8, kernel_size=5, padding=2, stride=1)
                # self.relu00 = nn.LeakyReLU(0.2)
                # self.conv01 = nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=1)
                # self.relu01 = nn.LeakyReLU(0.2)
                
                # self.conv_down1 = Conv_down(16, 16, dropout=dropout1)
                # self.conv_down2 = Conv_down(16, 32, dropout=dropout1)
                # self.conv_down3 = Conv_down(32, 32, dropout=dropout1)
                # self.conv_down4 = Conv_down(32, 64, dropout=dropout1)
                # self.conv_down5 = Conv_down(64, 64, dropout=dropout1)
                # self.conv_down6 = Conv_down(64, 64, dropout=dropout1)
                # #self.conv_down7 = Conv_down(64, 128, dropout=dropout1)
                # #self.conv_down8 = Conv_down(128, 128, dropout=dropout1)
                
                # self.dropout0 = constDropout(p=0.5, const=params['const_dropout'])
                
                # #self.conv_up1 = Conv_up(128+128, 128, dropout=dropout1)
                # #self.conv_up2 = Conv_up(128+64, 64, dropout=dropout1)
                # self.conv_up3 = Conv_up(64+64, 64, dropout=dropout1)
                # self.conv_up4 = Conv_up(64+64, 64, dropout=dropout1)
                # self.conv_up5 = Conv_up(64+32, 32, dropout=dropout1)
                # self.conv_up6 = Conv_up(32+32, 32, dropout=dropout1)
                # self.conv_up7 = Conv_up(32+16, 16, dropout=dropout1)
                # self.conv_up8 = Conv_up(16+16, 16, dropout=dropout1)

                # self.conv0_out = nn.Conv2d(16, 8,  kernel_size=3, stride=1, padding=1)
                # self.relu0_out = nn.LeakyReLU(0.2)
                # self.conv1_out = nn.Conv2d(8, 2,  kernel_size=3, stride=1, padding=1)
                
            def forward(self, x):
                if params['const_dropout']:
                    k = params['k']
                    return self.forward_const(x, k)
                else:
                    return self.forward_drop(x)
                    

            def forward_const(self, input_B: torch.Tensor, k: int):
    
                BayesianModule.k = k
                mc_input_BK = UNet.mc_tensor(input_B, k)
                mc_output_BK = self.forward_drop(mc_input_BK)
                mc_output_B_K = UNet.unflatten_tensor(mc_output_BK, k)
                return mc_output_B_K
            
            def forward_drop(self, x):
                
                x00 = self.conv00(x)
                x00r = self.relu00(x00)
                x01 = self.conv01(x00r)
                x01r = self.relu01(x01)
                
                x1 = self.conv_down1(x01r)
                x2 = self.conv_down2(x1)
                x3 = self.conv_down3(x2)
                x4 = self.conv_down4(x3)
                x5 = self.conv_down5(x4)
                x6 = self.conv_down6(x5)
                #x7 = self.conv_down7(x6)
                #x8 = self.conv_down8(x7)
                #x8d = self.dropout0(x8)
                
                #x9 = self.conv_up1(x8d, x7)
                #x10 = self.conv_up2(x9, x6)
                #x11 = self.conv_up3(x10, x5)
                
                
                x6d = self.dropout0(x6)
                x11 = self.conv_up3(x6d, x5)
                
                x12 = self.conv_up4(x11, x4)
                x13 = self.conv_up5(x12, x3)
                x14 = self.conv_up6(x13, x2)
                x15 = self.conv_up7(x14, x1)
                x16 = self.conv_up8(x15, x01r)
                
                xw0 = self.conv0_out(x16)
                xw1 = self.relu0_out(xw0)
                xout_out = self.conv1_out(xw1)
                
                if params['embedding']:
                    return xout_out, x8d
                else:
                    return xout_out

            @staticmethod
            def unflatten_tensor(input: torch.Tensor, k: int):
                input = input.view([-1, k] + list(input.shape[1:]))
                return input
        
            @staticmethod
            def flatten_tensor(mc_input: torch.Tensor):
                return mc_input.flatten(0, 1)
        
            @staticmethod
            def mc_tensor(input: torch.tensor, k: int):
                mc_shape = [input.shape[0], k] + list(input.shape[1:])
                return input.unsqueeze(1).expand(mc_shape).flatten(0, 1)
            
            
        unet = UNet()
        unet.train()
        unet.cuda()   
        
        # Create model
        self.model={}
        self.model['unet'] = unet
        self.count_parameters(self.model)
        self.opt_unet = optim.Adam(self.model['unet'].parameters(), lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        #self.opt_unet = optim.Adam(self.model['unet'].parameters(), lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.5)
        
        # base_opt = optim.Adam(self.model['unet'].parameters(), lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        # self.opt_unet = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    
    def shrink_pertub(self, lam=0.3, std=0.000001):
        with torch.no_grad():
            for name, param in self.model['unet'].named_parameters():
                m = torch.zeros(param.shape).to('cuda')
                s = torch.ones(param.shape).to('cuda')*std
                noise = torch.normal(m,s)
                param_shrink = lam*param + noise
                param.copy_(param_shrink.clone())


    def train(self, dataframe, name_training='', saveData=SaveState.CREATE, params=defaultdict(lambda: None), fitFromFolder=False, loadPretrainedWeights=False, NumFilterTrain=None):
        """ Train model unet_lesion
            
        """
        
        # self=net
        
        torch.autograd.set_detect_anomaly(False)
        warnings.filterwarnings('ignore')
        soft = nn.Softmax(dim=1)
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to('cuda')).to('cuda')
        #fc_loss = FocalLoss(weight=torch.tensor([1.0, 5.0]).to('cuda'), gamma=5., reduction='mean').to('cuda')
        
        # Collect losses and accuracies
        losses_train = ['loss_all', 'loss', 'loss_strong', 'loss_reg']
        acc_train = ['acc',
                     'spec',
                     'sen',
                     'confusion',
                     'confusion_sum',
                     'f1']
        
                
        losses_valid = ['loss_all_valid', 'loss_valid', 'loss_strong_valid', 'loss_uc_valid', 'loss_reg_valid',
                        'loss_ent0_valid', 'loss_ent1_valid']
        self.cddLossAcc.init(losses_train, acc_train)
        acc_valid = ['acc_valid',
                     'spec_valid',
                     'sen_valid',
                     'confusion_valid',
                     'confusion_sum_valid',
                     'f1_valid']
        
        self.cddLossAccValid.init(losses_valid, acc_valid)
        
        # Set updateSettings
        self.cddLossAcc.acc['f1'].updateSettings(updateValue=False, updateCounter=False, counter=1)
        self.cddLossAcc.acc['confusion_sum'].updateSettings(updateValue=True, updateCounter=False, counter=1)
        self.cddLossAccValid.acc['f1_valid'].updateSettings(updateValue=False, updateCounter=False, counter=1)
        self.cddLossAccValid.acc['confusion_sum_valid'].updateSettings(updateValue=True, updateCounter=False, counter=1)

        # Create data handler
        #Xlabel = dataframe.Xlabel
        #Ylabel = dataframe.Ylabel

        # Create early stopping
        early_stopping = EarlyStopping(patience=300)

        # Define parameter
        NumSamplesVis = 3
        epoch_load = self.props['epoch_load']
        os.makedirs(self.props['resultsFolder'], exist_ok=True)
        folderpath_results_valid = os.path.join(self.props['resultsFolder'], 'valid')
        folderpath_results_train = os.path.join(self.props['resultsFolder'], 'train')
        
        # Delete old validation results
        if os.path.isdir(folderpath_results_valid):
            shutil.rmtree(folderpath_results_valid)
            time.sleep(1)
            os.makedirs(folderpath_results_valid, exist_ok=True)
        else:
            os.makedirs(folderpath_results_valid, exist_ok=True)
        if os.path.isdir(folderpath_results_train):
            shutil.rmtree(folderpath_results_train)
            time.sleep(1)
            os.makedirs(folderpath_results_train, exist_ok=True)
        else:
            os.makedirs(folderpath_results_train, exist_ok=True)
                                
        # NumSamplesValid = self.props['NumSamplesValidLoad']
        # NumSamplesTrain = self.props['NumSamplesTrainLoad']
        # dataframe.model = self.model
        # dataframe.createIterativeGeneratorValid('generator_valid', params['fip_hdf5_list'], NumSamples=NumSamplesValid)
        # dataframe.createIterativeGeneratorTrain('generator_train', params['fip_hdf5_list'], NumSamples=NumSamplesTrain)

        modelpath = self.props['loadPretrainedFilePath']
        if self.props['loadPretrained']:
            params_train=dict()
            modelpath = os.path.join(self.props['loadPretrainedFolderPath'], self.props['modelname'])
            _, _, _, params_train = self.loadModel(modelpath, optimizer=None, scheduler=None, params_train=params_train, load_model=False)
        else:
            params_train=dict({'log_vars': None})
            
        print('params_train', params_train)

        # Start training
        if self.props['refine']:
            self.epochs_end = min(self.params['epochs'], self.epoch_bias+self.props['epochs_refine'])
        else:
            self.epochs_end = self.params['epochs']
            
        # Load pretrained model
        if self.props['loadPretrained']:
            params_opt = self.model['unet'].parameters()
            
            self.opt_unet = optim.Adam(params_opt, lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.01)
            params_train=dict()
            # if self.props['load_opt_params']:
            #     scheduler = StepLR(self.opt_unet, step_size=self.params['step_size'], gamma=self.params['gamma'])
            #     _, _, scheduler, params_train = self.loadModel(modelpath, optimizer=self.opt_unet, scheduler=scheduler, params_train=params_train)
            # else:
            #     _, _, _, params_train = self.loadModel(modelpath, optimizer=None, scheduler=None, params_train=params_train)
            #     scheduler = None
            _, _, _, params_train = self.loadModel(modelpath, optimizer=self.opt_unet, scheduler=None, params_train=params_train)
            #_, _, _, params_train = self.loadModel(modelpath, optimizer=None, scheduler=None, params_train=params_train)
            scheduler = None
            print('here')
        else:
            params_opt = self.model['unet'].parameters()
            self.opt_unet = optim.Adam(params_opt, lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.01)
            if self.props['load_opt_params']:
                scheduler = StepLR(self.opt_unet, step_size=self.params['step_size'], gamma=self.params['gamma'])
            else:
                scheduler = None

        # !!! TMP
        #base_opt = optim.Adam(self.model['unet'].parameters(), lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        #base_opt = torch.optim.SGD(self.model['unet'].parameters(), lr = self.params['lr'])
        #base_opt = torch.optim.SGD(self.model['unet'].parameters(), lr = self.params['lr'])
        #self.opt_unet = torchcontrib.optim.SWA(self.opt_unet, swa_start=3, swa_freq=5, swa_lr=self.params['lr'])
        #self.opt_unet.defaults = self.opt_unet.optimizer.defaults
    
        # Reset learning rate
        for param_group in self.opt_unet.param_groups:
            param_group['lr']=self.params['lr']
            
        # Init teacher model
        dataframe.teacher = copy.deepcopy(self.model['unet'])
        dataframe.props['SSL'] = self.props['SSL']
        
        NumSamplesValid = self.props['NumSamplesValidLoad']
        NumSamplesTrain = self.props['NumSamplesTrainLoad']
        dataframe.model = self.model
        dataframe.createIterativeGeneratorValid('generator_valid', params['fip_hdf5_list'], NumSamples=NumSamplesValid)
        dataframe.createIterativeGeneratorTrain('generator_train', params['fip_hdf5_list'], NumSamples=NumSamplesTrain)


        self.visualizer.step = self.epoch_bias
        self.model['unet'].train()
        
        if self.props['shrink_pertub']:
            #self.shrink_pertub(lam=0.3, std=0.000001)
            #self.shrink_pertub(lam=0.7, std=0.0)
            self.shrink_pertub(lam=0.7, std=0.0001)
            #self.shrink_pertub(lam=0.0, std=0.0)
        
        print('Press button "ctrl + e" pressed to exit execution.')
        for epoch in range(self.epoch_bias, self.epochs_end):
            print('epoch: ', epoch, ' / ', self.epochs_end)
            # Schedule learning rate of optimizer 
            if self.props['scheduler']: scheduler.step()
            
            # Update iterative dataset every 10 epochs
            if epoch % epoch_load == 0 or epoch==self.epoch_bias:
                print('epoch_load123')
                dataframe.createIterativeGeneratorTrain('generator_train', params['fip_hdf5_list'], NumSamples=NumSamplesTrain)
                self.props['steps_per_epoch'] = int(np.round(dataframe.NumSamplesTrain/dataframe.props['batch_size']))
                self.props['steps_per_vis'] = self.props['steps_per_epoch']
                
            #print('NumSamplesTrain123', dataframe.NumSamplesTrain)
            #sys.exit()
                    
            # Print learning rate
            for param_group in self.opt_unet.param_groups:
                print('lr:', param_group['lr'])

            # Iterate over steps
            print('steps_per_epoch', self.props['steps_per_epoch'])
            pbar = tqdm(total=self.props['steps_per_epoch'])
            pbar.set_description("STEP")
            for step in range(self.props['steps_per_epoch']):
                pbar.update(1)
                
                # Load data from generator
                X, Y = next(dataframe.generator['train'])

                # Convert to torch tensor
                #Ximage = torch.tensor(torch.from_numpy(X[Xlabel['image']]).float(), requires_grad=True).to(self.props['device'])
                #Xmask = torch.FloatTensor(Y[Ylabel['mask']]).to(self.props['device'])
                Ximage = torch.FloatTensor(X['XImage']).to(self.props['device'])
                Xmask = torch.FloatTensor(Y['XMask']).to(self.props['device'])

                # Predict network
                Xout = self.model['unet'](Ximage)      
                #XoutAll = self.model['unet'](Ximage)
                #Xout = XoutAll[0:params['batch_size']]
                #Xoutq = XoutAll[params['batch_size']:]
                
                Xmask_bin_idx = torch.reshape(Xmask.permute(0, 2, 3, 1), (-1,2)).sum(axis=1)>0
                Xout_bin = torch.reshape(Xout.permute(0, 2, 3, 1), (-1,2))[Xmask_bin_idx]
                Xmask_bin = torch.reshape(Xmask.permute(0, 2, 3, 1), (-1,2))[Xmask_bin_idx]
                Xmask_pos = Xmask_bin.max(1)[1]
                #print('Xmask_pos123', Xmask_pos)

                # Backpropagation
                loss = ce_loss(Xout_bin, Xmask_pos)
                loss_all = loss
                #loss = fc_loss(Xout_bin, Xmask_pos)
                
                # # Consistency loss
                # import matplotlib.pyplot as plt
                # K=5
                # alpha_K = 0.5
                # XMaskC = torch.zeros((8,K,2,512,512))
                # for i in range(K):
                #     Xaug, Yaug = dataframe.ttaug(X, Y, X_idx=['XImageQ'], Y_idx=['XMaskQ'])
                #     XimageQ = torch.FloatTensor(Xaug['XImageQ']).to(self.props['device'])
                #     XoutQ = self.model['unet'](XimageQ)
                #     Yaug['XMaskQ'] = XoutQ.detach().cpu().numpy()
                #     Xa, Ya = dataframe.ttaugInv(Xaug, Yaug, X_idx=['XImageQ'], Y_idx=['XMaskQ'])
                #     XMaskQ = torch.FloatTensor(Ya['XMaskQ'])
                #     XMaskC[:,i,:,:,:] = XMaskQ[:,:,:,:]
                # soft2 = nn.Softmax(dim=2)
                # XMaskC = soft2(XMaskC)
                # XMaskCM = XMaskC.mean(axis=1)
                # ent0 = -(XMaskCM * XMaskCM.log()).mean()
                # ent1 = -(XMaskC * XMaskC.log()).mean()
                # loss_c = alpha_K*ent0 - (1-alpha_K)*ent1
                # #sys.exit()af

                # if self.props['grad_reg']==True:
                #     alpha = 0.001
                #     grad_params = torch.autograd.grad(loss, self.model['unet'].parameters(), create_graph=True)
                #     fce=None
                #     for grad in grad_params:
                #         if fce is None:
                #             fce = torch.reshape(grad, (-1,))
                #         else:
                #             fce = torch.cat((fce, torch.reshape(grad, (-1,))))
                #     fce2 = fce*fce
                #     loss_norm = fce2.sum().sqrt()
                #     loss_all = loss + alpha * loss_norm
                # else:
                #     #loss_all = loss
                #     #lambda_c = 0.0
                #     tr=10
                #     if epoch<tr:
                #         lambda_c = np.exp(-5*(1-epoch/tr)**2)
                #     else:
                #         lambda_c=100.0
                #     # !!!
                #     if epoch<50:
                #         lambda_c = 0.0
                #     loss_all = loss + lambda_c*loss_c
                    
                self.opt_unet.zero_grad()
                loss_all.backward()
                self.opt_unet.step()
                
                # for pt, ps in zip(dataframe.teacher.parameters(), self.model['unet'].parameters()):
                #     if pt.requires_grad:
                #         pass
                #print('data12', pt.data)
                
                # Update teacher
                if dataframe.props['SSL']:
                    update_teacher_model(self.model['unet'], dataframe.teacher, keep_rate=0.99)
                # m = 0.0
                # for pt, ps in zip(dataframe.teacher.parameters(), self.model['unet'].parameters()):
                #     if pt.requires_grad:
                #         pt.data = m*pt.data + (1-m)*ps.data
                        #pt.copy_(torch.randn(10, 10))
                
                # with torch.no_grad():
                # for name, param in model.named_parameters():
                #     if 'classifier.weight' in name:
                #         param.copy_(torch.randn(10, 10))

                
                #print('data123', pt.data)
                #print('data1234', ps.data)
                
                # Update loss and accuracy
                C = confusion(Xmask_bin.detach(), soft(Xout_bin).detach(), num_classes=2, binary=True, input_type='one_hot')
                # Xout_pos = soft(Xout_bin).max(1)[1]
                # C00 = ((1-Xout_pos)*(1-Xmask_pos)).sum()
                # C01 = ((Xout_pos)*(1-Xmask_pos)).sum()
                # C10 = ((1-Xout_pos)*(Xmask_pos)).sum()
                # C11 = ((Xout_pos)*(Xmask_pos)).sum()
                # C = [np.array([[C00.cpu().numpy(), C01.cpu().numpy()],[C10.cpu().numpy(), C11.cpu().numpy()]])]
                
                #print('C123', C)
                acc = accuracy(C, num_classes=2, binary_class=-1, mode='binary', class_names=None)
                spec = specificity(C, num_classes=2, binary_class=-1, mode='binary', class_names=None)
                sen = recall(C, num_classes=2, binary_class=-1, mode='binary', class_names=None)           
                C_sum = self.cddLossAcc.get_acc('confusion_sum')
                if type(C_sum)==torch.tensor:
                    f1 = C_sum[1,1]/(C_sum[1,1] + 0.5*(C_sum[1,0]+C_sum[0,1]))
                else:
                    f1 = 0.0

                #lossesd = {'loss_all': loss_all, 'loss': loss, 'loss_reg': loss_c}#
                lossesd = {'loss_all': loss_all, 'loss': loss, 'loss_reg': loss}
                accd = {'acc': acc['1'],
                       'spec': spec['1'],
                       'sen': sen['1'],
                       'confusion': C,
                       'confusion_sum': C,
                       'f1': f1}
                
                self.cddLossAcc.sum_losses(lossesd)
                self.cddLossAcc.sum_acc(accd)

                if (((step+1) % self.props['steps_per_vis'] == 0) and (step>0)) or (epoch==self.epoch_bias and step==0):
                    pbar.close()

                    # Print, Visualize and Reset loss and accuracy
                    self.cddLossAcc.print_losses()
                    self.cddLossAcc.print_acc()
                    self.cddLossAcc.visualize_loss(self.visualizer)
                    self.cddLossAcc.visualize_acc(self.visualizer)
                    self.cddLossAcc.reset_losses()
                    self.cddLossAcc.reset_acc()
                    
                    if (epoch % self.props['epoch_valid'] == 0) and (epoch>0):
                        
                        self.model['unet'].eval()
                        #self.opt_unet.swap_swa_sgd()
                        
                        # Validate dataset
                        #dataframe.createIterativeGeneratorValid('generator_multi_class_2D_multi', params['hdf5filepath'], NumSamples=NumSamplesValid)
                        NumIterValid = int(np.round(dataframe.NumSamplesValid/dataframe.props['batch_size']))
                        pbar = tqdm(total=NumIterValid)
                        pbar.set_description("STEP VALID")
                        for i in range(NumIterValid):
                            #print('Valid i:', i)
                            pbar.update(1)
                            
                            # Load data from generator
                            Xv, Yv = next(dataframe.generator['valid'])
                            #Ximagev = torch.FloatTensor(Xv[Xlabel['image']]).to(self.props['device'])
                            #Xmaskv = torch.FloatTensor(Yv[Ylabel['mask']]).to(self.props['device'])
                            Ximagev = torch.FloatTensor(Xv['XImage']).to(self.props['device'])
                            Xmaskv = torch.FloatTensor(Yv['XMask']).to(self.props['device'])

                            # Predict network
                            Xoutv = self.model['unet'](Ximagev)
                            Xmaskv_bin_idx = torch.reshape(Xmaskv.permute(0, 2, 3, 1), (-1,2)).sum(axis=1)>0
                            Xoutv_bin = torch.reshape(Xoutv.permute(0, 2, 3, 1), (-1,2))[Xmaskv_bin_idx]
                            Xmaskv_bin = torch.reshape(Xmaskv.permute(0, 2, 3, 1), (-1,2))[Xmaskv_bin_idx]
                            Xmask_posv = Xmaskv_bin.max(1)[1]
 
                            # Predict loss
                            lossv = ce_loss(Xoutv_bin, Xmask_posv)
                            #lossv = fc_loss(Xoutv_bin, Xmask_posv)
                            loss_strongv = lossv
                            loss_allv = lossv
                            
                            # Consistency loss valid
                            # import matplotlib.pyplot as plt
                            # K=5
                            # alpha_K = 0.5
                            # XMaskCv = torch.zeros((8,K,2,512,512))
                            # for i in range(K):
                            #     Xaugv, Yaugv = dataframe.ttaug(Xv, Yv, X_idx=['XImage'], Y_idx=['XMask'])
                            #     XimageQ = torch.FloatTensor(Xaugv['XImage']).to(self.props['device'])
                            #     XoutQ = self.model['unet'](XimageQ)
                            #     Yaugv['XMask'] = XoutQ.detach().cpu().numpy()
                            #     Xa, Ya = dataframe.ttaugInv(Xaugv, Yaugv, X_idx=['XImage'], Y_idx=['XMask'])
                            #     XMaskQv = torch.FloatTensor(Ya['XMask'])
                            #     XMaskCv[:,i,:,:,:] = XMaskQv[:,:,:,:]
                            # soft2 = nn.Softmax(dim=2)
                            # XMaskCv = soft2(XMaskCv)
                            # XMaskCMv = XMaskCv.mean(axis=1)
                            # ent0v = -((XMaskCMv * XMaskCMv.log()).mean())
                            # ent1v = -((XMaskCv * XMaskCv.log()).mean())
                            # loss_cv = alpha_K*ent0v - (1-alpha_K)*ent1v
                            
                            # # Update loss and accuracy
                            # accv = accuracy_bin(Xmaskv_bin[:,1], soft(Xoutv_bin)[:,1])
                            # specv = specificity(Xmaskv_bin[:,1], soft(Xoutv_bin)[:,1])
                            # senv = sensitivity(Xmaskv_bin[:,1], soft(Xoutv_bin)[:,1])

                            # Cv = [confusion_multi_mat(Xmaskv_bin, Xoutv_bin, NumClasses=2)]
                            # C_sumv = self.cddLossAccValid.get_acc('confusion_sum_valid')
                            # if type(C_sumv)==list:
                            #     f1v = C_sumv[0][1,1]/(C_sumv[0][1,1] + 0.5*(C_sumv[0][1,0]+C_sumv[0][0,1]))
                            # else:
                            #     f1v = 0.0
                            
                            # Update loss and accuracy
                            #Cv = [confusion(Xmaskv_bin.detach(), soft(Xoutv_bin).detach(), num_classes=2, input_type='one_hot')]
                            Cv = [confusion(Xmaskv_bin.detach(), soft(Xoutv_bin).detach(), num_classes=2, binary=True, input_type='one_hot')]
                            accv = accuracy(Cv[0], num_classes=2, binary_class=-1, mode='binary', class_names=None)
                            #print('accv123', accv)
                            specv = specificity(Cv[0], num_classes=2, binary_class=-1, mode='binary', class_names=None)
                            senv = recall(Cv[0], num_classes=2, binary_class=-1, mode='binary', class_names=None)           
                            C_sumv = self.cddLossAccValid.get_acc('confusion_sum_valid')
                            if type(C_sumv)==list:
                                f1v = C_sumv[0][1,1]/(C_sumv[0][1,1] + 0.5*(C_sumv[0][1,0]+C_sumv[0][0,1]))
                            else:
                                f1v = 0.0
                            
                            ###### Uncertainty loss ######
                            import matplotlib.pyplot as plt
                            self.enable_dropout(drop_rate=0.01) 
                            NumDO=10
                            XoutL=[]
                            for i in range(NumDO):
                                XoutT = self.model['unet'](Ximagev)
                                XoutL.append(XoutT[:,1:,:,:].detach().cpu())
                            XoutL = torch.stack(XoutL, dim=1)
                            var_uc = torch.var(XoutL,dim=1)
                            loss_ucv = var_uc.sum()
                            #print('var_uc123', var_uc.shape)
                            #print('var_uc123', var_uc[0,0,:,:].mean())
                            
                            #plt.imshow(var_uc[0,0,:,:].numpy())
                            #plt.show()
                            #plt.imshow(Ximagev[0,0,:,:].numpy())
                            #plt.show()
                            
                            self.disable_dropout()
                            #sys.exit()
                            ##############################

                                
                            #losses_valid = {'loss_all_valid': loss_allv, 'loss_valid': lossv, 'loss_strong_valid': loss_strongv, 'loss_uc_valid': loss_ucv,
                            #               'loss_reg_valid': loss_cv, 'loss_ent0_valid': ent0v, 'loss_ent1_valid': ent1v}
                            losses_valid = {'loss_all_valid': loss_allv, 'loss_valid': lossv, 'loss_strong_valid': lossv, 'loss_uc_valid': lossv,
                                            'loss_reg_valid': lossv, 'loss_ent0_valid': lossv, 'loss_ent1_valid': lossv}
                            acc_valid = {'acc_valid': accv['1'],
                                       'spec_valid': specv['1'],
                                       'sen_valid': senv['1'],
                                       'confusion_valid': Cv,
                                       'confusion_sum_valid': Cv,
                                       'f1_valid': f1v}
                            
                            #print('acc_valid123', acc_valid)
                            
                            self.cddLossAccValid.sum_losses(losses_valid)
                            self.cddLossAccValid.sum_acc(acc_valid)
                        pbar.close()
                        
                        self.model['unet'].train()
                        
                        print('epoch', epoch)
                        print('epoch_bias', self.epoch_bias)
                        print('savePretrainedEpochMin', self.props['savePretrainedEpochMin'])
                        
                        # Apply early stopping
                        if (epoch-self.epoch_bias)>=self.props['savePretrainedEpochMin']:
                            loss_stop = -self.cddLossAccValid.get_acc('f1_valid')
                            if loss_stop is None:
                                loss_stop=-1000000
                            print('loss_stop', loss_stop)
                            early_stopping(loss_stop, self.model['unet'])
                            if early_stopping.early_stop:
                                print("Early stopping")
                                sys.exit()
                       
                        self.cddLossAccValid.print_losses()
                        self.cddLossAccValid.print_acc()
                        self.cddLossAccValid.visualize_loss(self.visualizer)
                        self.cddLossAccValid.visualize_acc(self.visualizer)
                        
                        # Reset acc and loss
                        self.cddLossAccValid.reset_losses()
                        self.cddLossAccValid.reset_acc()
                        
                        # Show results from validation set
                        image = torch.cat((Ximagev[0:NumSamplesVis,0:1,:,:], Xmaskv[0:NumSamplesVis,0:1,:,:], Xmaskv[0:NumSamplesVis,1:2,:,:]), 0)
                        vutils.save_image(image, folderpath_results_valid + '/mask_' + str(epoch) + '.png', normalize=False, nrow=NumSamplesVis)
                        image = torch.cat((Ximagev[0:NumSamplesVis,0:1,:,:], soft(Xoutv)[0:NumSamplesVis,0:1,:,:], soft(Xoutv)[0:NumSamplesVis,1:2,:,:]), 0)
                        vutils.save_image(image, folderpath_results_valid + '/out_' + str(epoch) + '.png', normalize=False, nrow=NumSamplesVis)

            # Show results from training set
            image = torch.cat((Ximage[0:NumSamplesVis,0:1,:,:], Xmask[0:NumSamplesVis,0:1,:,:], Xmask[0:NumSamplesVis,1:2,:,:]), 0)
            vutils.save_image(image, folderpath_results_train + '/mask_' + str(epoch) + '.png', normalize=False, nrow=NumSamplesVis)
            image = torch.cat((Ximage[0:NumSamplesVis,0:1,:,:], soft(Xout)[0:NumSamplesVis,0:1,:,:], soft(Xout)[0:NumSamplesVis,1:2,:,:]), 0)
            vutils.save_image(image, folderpath_results_train + '/out_' + str(epoch) + '.png', normalize=False, nrow=NumSamplesVis)

            # Save model
            if (epoch-self.epoch_bias)>=self.props['savePretrainedEpochMin']:
                if (self.props['savePretrained'] and early_stopping.save_best) or self.props['savePretrained_all']:
                    #params_train = dict({'log_vars': log_vars.data.tolist()})
                    params_train = dict()
                    modelname = self.props['modelname']
                    modelfolderpath = os.path.join(self.props['savePretrainedFolderPath'], modelname)
                    if os.path.isdir(modelfolderpath):
                        shutil.rmtree(modelfolderpath)
                    os.makedirs(modelfolderpath, exist_ok=True)
                    self.saveModel(modelfolderpath, modelname, epoch, optimizer=self.opt_unet, scheduler=scheduler, params_train=params_train) 
                
            # Update epoch
            self.visualizer.next_epoch()
            self.visualizer.next_step()
            
    def enable_dropout(self, drop_rate=0.5):
        for m in self.model['unet'].modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p=drop_rate
                m.train()
    
    def disable_dropout(self):
        for m in self.model['unet'].modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p=0.0
                m.eval()
            
    # @staticmethod
    # def fim_loss_func(mask, pred, weight, use_mask=False):

    #     batch = mask.shape[0]
    #     loss = torch.zeros((batch,1)).cuda()
    #     loss = torch.zeros((batch,1)).cuda()

    #     for b in range(batch):
    #         mask_bin_idx = torch.reshape(mask[b].permute(1, 2, 0), (-1,2)).sum(axis=1)>0
    #         mask_bin = torch.reshape(mask[b].permute(1, 2, 0), (-1,2))[mask_bin_idx].to('cuda')
    #         pred_bin = torch.reshape(pred[b].permute(1, 2, 0), (-1,2))[mask_bin_idx]
    #         weight_bin = torch.reshape(weight[b].permute(1, 2, 0), (-1,1))[mask_bin_idx].to('cuda')
    #         weight_bin = weight_bin.repeat(1,2)
    #         pred_bin_log = torch.log_softmax(pred_bin, dim=1)
    #         pred_bin_prop = torch.exp(pred_bin_log)
    #         if use_mask: 
    #             loss[b]  = torch.mean(pred_bin_log * mask_bin * weight_bin)
    #         else:
    #             loss[b]  = torch.mean(pred_bin_log * pred_bin_prop * weight_bin)
    #         loss[b] = loss[b]
    #     #print('loss123', loss)
    #     return loss
    
    # @staticmethod
    # def fim_loss_func_all(mask, pred, weight, use_mask=False):

    #     NumClasses = pred.shape[1]
    #     batch = mask.shape[0]
    #     loss = torch.zeros((batch,NumClasses)).cuda()
    #     loss = torch.zeros((batch,NumClasses)).cuda()

    #     for b in range(batch):
    #         mask_bin_idx = torch.reshape(mask[b].permute(1, 2, 0), (-1,2)).sum(axis=1)>0
    #         mask_bin = torch.reshape(mask[b].permute(1, 2, 0), (-1,2))[mask_bin_idx].to('cuda')
    #         pred_bin = torch.reshape(pred[b].permute(1, 2, 0), (-1,2))[mask_bin_idx]
    #         weight_bin = torch.reshape(weight[b].permute(1, 2, 0), (-1,1))[mask_bin_idx].to('cuda')
    #         weight_bin = weight_bin.repeat(1,2)
    #         pred_bin_log = torch.log_softmax(pred_bin, dim=1)
    #         pred_bin_prop = torch.exp(pred_bin_log)
    #         if use_mask: 
    #             for c in range(NumClasses):
    #                 loss[b,c]  = torch.mean(pred_bin_log[:,c] * mask_bin[:,c] * weight_bin[:,c])
    #         else:
    #             for c in range(NumClasses):
    #                 loss[b,c]  = torch.mean(pred_bin_log[:,c] * weight_bin[:,c])
            
    #         loss[b] = loss[b,:]
    #     #print('loss123', loss)
    #     return loss
    
    
    
    