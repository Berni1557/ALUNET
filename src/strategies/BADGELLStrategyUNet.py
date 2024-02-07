
import os, sys
import numpy as np
import torch
from torch import nn
from scipy import stats
from tqdm import tqdm
import random
from sklearn.cluster import KMeans
import pdb
from scipy import stats
import pandas as pd
import math
from nngeometry.layercollection import LayerCollection
from strategies.ALStrategy import ALStrategy
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry')
sys.path.append('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry')

class BADGELLStrategy(ALStrategy):
    def __init__(self, name='BADGELL'):
        self.name = name

    # kmeans ++ initialization
    def init_centers(self, X, K):
        embs = torch.Tensor(X)
        ind = torch.argmax(torch.norm(embs, 2, 1)).item()
        embs = embs.cuda()
        mu = [embs[ind]]
        indsAll = [ind]
        centInds = [0.] * len(embs)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy()
            else:
                newD = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy()
                for i in range(len(embs)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(embs[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

   
    def get_grad_embedding(self, opts, net, folderDict, data_query, loss_func, layers):
        
        #layer_collection = LayerCollection.from_model_BF(net.model['unet'], ignore_unsupported_layers=True,layer_in=layers)
        layer_collection = LayerCollection.from_model_BF(net.model['unet'].network, ignore_unsupported_layers=True,layer_in=layers)
        params = layer_collection.get_parameters_BF(net.model['unet'].network)
        num_params = int(np.sum([x.numel() for x in params]))

        # Order samples by image
        if opts.dim==2:
            imagenames = list(np.unique([s.F['imagename'] for s in data_query]))
            data_load = pd.DataFrame()
            for imn in imagenames:
                data_im = [{'imagename': s.F['imagename'], 'slice': int(s.F['slice']), 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in data_query if s.F['imagename']==imn]
                data_load = pd.concat([data_load, pd.DataFrame(data_im)])
            data_load.reset_index(inplace=True)
        else:
            imagenames = list(np.unique([s.F['imagename'] for s in data_query]))
            data_load = pd.DataFrame()
            for imn in imagenames:
                data_im = [{'imagename': s.F['imagename'], 'ID': int(s.F['ID']), 'IDP': int(s.F['IDP'])} for s in data_query if s.F['imagename']==imn]
                data_load = pd.concat([data_load, pd.DataFrame(data_im)])
            data_load.reset_index(inplace=True)
        
        dataloader_train = net.model['unet'].get_dataloaders_BF(data_load)
        
        #NumParams=1000
        NumBatches = math.ceil(len(data_load)/dataloader_train.data_loader.batch_size)
        #NumFIM = int(min(NumFIM,len(data_load)))
        gradEmb=torch.ones((len(data_load), num_params))
        gradProps=[]
        device='cuda'
        #idxG=None
        K=0
        pbar = tqdm(total=NumBatches)
        pbar.set_description("Compute gradients")
        for b in range(NumBatches):
            pbar.update()
            batch = next(dataloader_train)
            data = batch['data']
            props = batch['properties']
            data = data.to(device, non_blocking=True)
            for i in range(data.shape[0]):
                outputi = net.model['unet'].network(data[i:i+1])
                targeti = [torch.argmax(x, dim=1, keepdim=True) for x in outputi]
                lossi = net.model['unet'].loss(outputi, targeti)
                grad_params = torch.autograd.grad(lossi, params, create_graph=True)
                fce = torch.hstack([torch.reshape(grad, (-1,)) for grad in grad_params])
                gradEmb[K,:] = fce.detach().cpu()
                for k in range(len(outputi)): outputi[k] = outputi[k].detach_().cpu()
                K=K+1
            gradProps = gradProps + props
        pbar.close()

        return gradEmb, gradProps

    def query(self, opts, folderDict, man, data_query, CLSample, NumSamples=10, pred_class='XMaskPred', batchsize=None, save_uc=False, device='cuda', previous=True):
        
        # self=strategy
        
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        loss_func = net.model['unet'].loss

        # Get last layer
        for layer, mod in net.model['unet'].network.named_modules():
            if 'decoder' in layer and 'convs' in layer and '1.conv' in layer:
                lastlayer = layer
        layers=[lastlayer]
                
        gradEmb, gradProps = self.get_grad_embedding(opts, net, folderDict, data_query, loss_func, layers)
        
        # Resort idxs based on selected samples
        idxs=[]
        for prop in gradProps:
            for i,s in enumerate(data_query):
                if s.F['ID']==prop['ID'] and s.F['IDP']==prop['IDP']:
                    idxs.append(i)
                    continue
        
        # Compute centers
        idx = self.init_centers(gradEmb.numpy(), NumSamples)
        
        # Create samples
        samples=[]
        for i in idx:
            s=data_query[idxs[i]]
            p=opts.CLPatch()
            p.ID=s.ID
            p.F=s.F.copy()
            prop=gradProps[i]
            samples.append(p)

        if False:
            #data_A=man.datasets['action_round'].data
            dataset = opts.CLDataset()
            dataset.load_dataset_mds3D(man, opts, folderDict, samples)
            opts.CLSample.predict(samples, opts, net)
            dropout_rate=0.1
            import matplotlib.pyplot as plt
            for s in samples[0:10]:
                s.plotSample(plotlist=['XImage', 'P', 'Y'], color=True)

                
        return samples
