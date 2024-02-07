# -*- coding: utf-8 -*-
import numpy as np
import sklearn
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from utils.helper import compute_one_hot_torch
from utils.helper import compute_one_hot_np
import torchmetrics
import pingouin as pg
import pandas as pd
from torchmetrics import ConfusionMatrix
import time
import statsmodels.api as sm


def confusion(target, pred, num_classes=-1, binary=False, input_type='categoric', device='cuda'):
    """ Compute F1 score
    
    Parameters
    ----------
    target : np.array, torch.tensor [N x C]
        Target class  with N-Number of samples and C classes
    pred : np.array, torch.tensor
        Predicted class with N-Number of samples and C classes
    num_classes : Number of classes, if -1, number of classes is estimate based on target and pred vector
        Define the calculation method for multi class F1 score
    input_type : Type of input can be 'categoric', 'one_hot', 'propability'
        Define the calculation method for multi class F1 score
        
    Returns
    -------
    C : float
        Confusion matrix (rows-true class, columns-predicted class)
    """
    
    # target=torch.ones((3,3))
    # pred=torch.ones((3,3))
    # target[2,2]=4
    # pred[2,2]=4
    # pred[0,2]=4
    # num_classes=3
    #print('pred123', pred.shape)
    
    target = torch.argmax(target, 1)
    pred = torch.argmax(pred, 1)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    C = confmat(pred, target)

    #print('C123', C)
    
    # # Convert torch tensor to numpy array
    # if torch.is_tensor(target):
    #     target = target.cpu().numpy()
    # if torch.is_tensor(pred):
    #     pred = pred.cpu().numpy()
        
    # # Convert input type
    # if input_type=='one_hot':
    #     target = np.argmax(target, 1)
    #     pred = np.argmax(pred, 1)
    # elif input_type=='propability':
    #     target = np.argmax(target, 1)
    #     pred = np.argmax(pred, 1)
    # else:
    #     target = target
    #     pred = pred
    
    # # Define number of classes
    # if num_classes==-1:
    #     num_classes = max(target.max(), pred.max())
    
    # # Compute confusion matrix
    # if binary:
    #     C00 = ((1-pred)*(1-target)).sum()
    #     C01 = ((pred)*(1-target)).sum()
    #     C10 = ((1-pred)*(target)).sum()
    #     C11 = ((pred)*(target)).sum()
    #     C = np.array([[C00, C01],[C10, C11]])
    # else:
    #     labels=np.arange(0,num_classes)
    #     C = confusion_matrix(target, pred, labels=labels)
    return C

# def confusion(target, pred, num_classes=-1, binary=False, input_type='categoric'):
#     """ Compute F1 score
    
#     Parameters
#     ----------
#     target : np.array, torch.tensor [N x C]
#         Target class  with N-Number of samples and C classes
#     pred : np.array, torch.tensor
#         Predicted class with N-Number of samples and C classes
#     num_classes : Number of classes, if -1, number of classes is estimate based on target and pred vector
#         Define the calculation method for multi class F1 score
#     input_type : Type of input can be 'categoric', 'one_hot', 'propability'
#         Define the calculation method for multi class F1 score
        
#     Returns
#     -------
#     C : float
#         Confusion matrix (rows-true class, columns-predicted class)
#     """
    
#     # target=torch.ones((3,3))
#     # pred=torch.ones((3,3))
#     # target[2,2]=4
#     # pred[2,2]=4
#     # pred[0,2]=4
#     # num_classes=3
    
#     # Convert torch tensor to numpy array
#     if torch.is_tensor(target):
#         target = target.cpu().numpy()
#     if torch.is_tensor(pred):
#         pred = pred.cpu().numpy()
        
#     # Convert input type
#     if input_type=='one_hot':
#         target = np.argmax(target, 1)
#         pred = np.argmax(pred, 1)
#     elif input_type=='propability':
#         target = np.argmax(target, 1)
#         pred = np.argmax(pred, 1)
#     else:
#         target = target
#         pred = pred
    
#     # Define number of classes
#     if num_classes==-1:
#         num_classes = max(target.max(), pred.max())
    
#     # Compute confusion matrix
#     if binary:
#         C00 = ((1-pred)*(1-target)).sum()
#         C01 = ((pred)*(1-target)).sum()
#         C10 = ((1-pred)*(target)).sum()
#         C11 = ((pred)*(target)).sum()
#         C = np.array([[C00, C01],[C10, C11]])
#     else:
#         labels=np.arange(0,num_classes)
#         C = confusion_matrix(target, pred, labels=labels)
#     return C

# def confusion_to_target_pred(C):
#     target=[]
#     pred=[]
#     for t in range(C.shape[0]):
#         for p in range(C.shape[1]):
#             v=int(C[t,p])
#             for k in range(v):
#                 target.append(t)
#                 pred.append(p)
#     target = np.array(target)
#     pred = np.array(pred)
#     return target, pred

def confusion_to_target_pred(C):
    target=[]
    pred=[]
    for t in range(C.shape[0]):
        for p in range(C.shape[1]):
            v=int(C[t,p])
            target.append(torch.ones(v)*t)
            pred.append(torch.ones(v)*p)
            # for k in range(v):
            #     target.append(t)
            #     pred.append(p)
    target = torch.hstack(target)
    pred = torch.hstack(pred)
    #target = np.array(target)
    #pred = np.array(pred)
    return target, pred

def confusion_to_target_pred_weight(C, W):
    target=[]
    pred=[]
    weight=[]
    for t in range(C.shape[0]):
        for p in range(C.shape[1]):
            v=int(C[t,p])
            target.append(torch.ones(v)*t)
            pred.append(torch.ones(v)*p)
            weight.append(torch.ones(v)*W[t,p])
            # for k in range(v):
            #     target.append(t)
            #     pred.append(p)
    target = torch.hstack(target)
    pred = torch.hstack(pred)
    weight = torch.hstack(weight)
    #target = np.array(target)
    #pred = np.array(pred)
    return target, pred, weight

def confusion_to_rate(C):
    TP = C.diagonal().sum()
    FP = C.sum()-C.diagonal().sum()
    FN = C.sum()-C.diagonal().sum()
    TN=0
    for i in range(C.shape[0]):
        TN = TN+C.sum()-C[i,:].sum()-C[:,i].sum()+C[i,i]
    return TP, FP, TN, FN
                
def confusion_to_rate_bin(C):
    TP=[]
    FP=[]
    TN=[]
    FN=[]
    for i in range(C.shape[0]):
        TP.append(C[i,i])
        FP.append(C[:,i].sum()-C[i,i])
        FN.append(C[i,:].sum()-C[i,i])
        TN.append(C.sum()-C[i,:].sum()-C[:,i].sum()+C[i,i])
    return TP, FP, TN, FN

def vol(C, num_classes=-1, binary_class=-1, class_names=None):
    """ Compute volumen
    
    Parameters
    ----------
    C : numpy.array, torch.tensor
        Confusion matrix
    binary_class : 1,2,3,.., for each class or -1 to get a list of all accuraciesoptional
        Used for binary F1 score, defines the target class
    mode : Can be 'binary', 'micro' or 'macro',  optional
        Define the calculation method for multi class F1 score
        
    Returns
    -------
    ACC : float, numpy.array
        Computed accuracy or list of accuracies for each class
    """
    
      
    
    # Define number of classes
    if num_classes==-1:
        num_classes = C.shape[0]

    if binary_class==-1:
        VOL=[]
        for c in range(num_classes):
            VOL.append(C[c,:].sum())
        if class_names is not None:
            VOL = {class_names[i]: VOL[i] for i in range(len(class_names))}
        else:
            VOL = {str(i): VOL[i] for i in range(num_classes)}
    elif binary_class is None: 
        VOL = C[1:,:].sum()
    else:
        VOL=[]
        for c in range(num_classes):
            VOL.append(C[c,:].sum())
        VOL[binary_class]
        
    return VOL

def precision(C, num_classes=-1, binary_class=-1, mode='binary', class_names=None):
    """ Compute precision (positive predictive value)
    
    Parameters
    ----------
    C : numpy.array, torch.tensor
        Confusion matrix
    binary_class : None or 1,2,3,.., optional
        Used for binary F1 score, defines the target class
    mode : Can be 'binary', 'micro' or 'macro',  optional
        Define the calculation method for multi class F1 score
        
    Returns
    -------
    PREC : float, numpy.array
        Computed precision or list of precisions for each class
    """
    
    #target, pred = confusion_to_target_pred(C)   
    
    # Define number of classes
    if num_classes==-1:
        num_classes = num_classes = C.shape[0]
    
    if mode=='binary':
        if binary_class==-1:
            if C.sum()>0:
                PREC=[]
                for c in range(num_classes):
                    #TP = C[c,c].sum()
                    #FP = C[:,c].sum() - TP
                    TP = C[c,c]
                    FN = C[c,:].sum() - TP
                    FP = C[:,c].sum() - TP
                    TN = C.sum()-C[c,:].sum()-C[:,c].sum()+TP
    
                    if (TP+FP)>0:
                        PREC.append(TP/(TP+FP))
                    else:
                        PREC.append(None)
            else:
                PREC = [None for i in range(num_classes)]
            if class_names is not None:
                PREC = {class_names[i]: PREC[i] for i in range(len(class_names))}
            else:
                PREC = {str(i): PREC[i] for i in range(num_classes)}
        elif binary_class is None: 
            TP = C[1:,1:].sum()
            FP = C[0,1:].sum()
            if (TP+FP)>0:
                PREC = TP/(TP+FP)
            else:
                PREC = None
        else:
            target, pred = confusion_to_target_pred(C) 
            prec = torchmetrics.Precision(num_classes=num_classes, average='none')
            PREC = float(prec(torch.from_numpy(pred), torch.from_numpy(target))[binary_class])
    if mode=='micro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C) 
            #prec = torchmetrics.Precision(num_classes=num_classes, average='micro', task='multiclass') 
            #PREC = float(prec(pred, target))
            TP, FP, TN, FN = confusion_to_rate(C)
            PREC = TP/(TP+FP)
        else:
            PREC = None
    if mode=='macro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C) 
            #prec = torchmetrics.Precision(num_classes=num_classes, average='macro', task='multiclass') 
            #PREC = float(prec(pred, target))
            TP, FP, TN, FN = confusion_to_rate_bin(C)
            PREC=0
            for i in range(C.shape[0]):
                if TP[i]+FP[i]>0: PREC=PREC+TP[i]/(TP[i]+FP[i])
            PREC=PREC/C.shape[0]
        else:
            PREC = None
    
    return PREC


def recall(C, num_classes=-1, binary_class=-1, mode='binary', class_names=None):
    """ Compute recall (sensitivity, hit rate, true positive rate)
    
    Parameters
    ----------
    C : numpy.array, torch.tensor
        Confusion matrix
    binary_class : None or 1,2,3,.., optional
        Used for binary F1 score, defines the target class
    mode : Can be 'binary', 'micro' or 'macro',  optional
        Define the calculation method for multi class F1 score
        
    Returns
    -------
    REC : float, numpy.array
        Computed precision or list of precisions for each class
    """
    
    #target, pred = confusion_to_target_pred(C)   
    
    # Define number of classes
    if num_classes==-1:
        num_classes = C.shape[0]
    
    if mode=='binary':
        if binary_class==-1:
            if C.sum()>0:
                #rec = torchmetrics.Recall(num_classes=num_classes, average='none')
                #REC = rec(torch.from_numpy(pred), torch.from_numpy(target)).tolist()
                REC=[]
                for c in range(num_classes):
                    #TP = C[c,c].sum()
                    #FN = C[c,:].sum() - TP
                    TP = C[c,c]
                    FN = C[c,:].sum() - TP
                    FP = C[:,c].sum() - TP
                    TN = C.sum()-C[c,:].sum()-C[:,c].sum()+TP
                    if (TP+FN)>0:
                        REC.append(TP/(TP+FN))
                    else:
                        REC.append(None)
            else:
                REC = [None for i in range(num_classes)]
            if class_names is not None:
                REC = {class_names[i]: REC[i] for i in range(len(class_names))}
            else:
                REC = {str(i): REC[i] for i in range(num_classes)}
        elif binary_class is None: 
            TP = C[1:,1:].sum()
            FN = C[1:,0].sum()
            if (TP+FN)>0:
                REC = TP/(TP+FN)
            else:
                REC = None
        else:
            if C.sum()>0:
                target, pred = confusion_to_target_pred(C) 
                rec = torchmetrics.Recall(num_classes=num_classes, average='none', task='binary')
                REC = float(rec(pred, target))

            else:
                REC = None
    if mode=='micro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C) 
            #rec = torchmetrics.Recall(num_classes=num_classes, average='micro', task='multiclass') 
            #REC = float(rec(pred, target))
            TP, FP, TN, FN = confusion_to_rate(C)
            REC = TP/(TP+FP)
        else:
            REC = None
    if mode=='macro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C) 
            #rec = torchmetrics.Recall(num_classes=num_classes, average='macro', task='multiclass') 
            #REC = float(rec(pred, target))
            TP, FP, TN, FN = confusion_to_rate_bin(C)
            REC=0
            for i in range(C.shape[0]):
                if (TP[i]+FN[i])>0:
                    REC=REC + (TP[i]/(TP[i]+FN[i]))
            REC=REC/C.shape[0]
        else:
            REC = None
    return REC

def specificity(C, num_classes=-1, binary_class=-1, mode='binary', class_names=None):
    """ Compute specificity (selectivity, true negative rate) 
    
    Parameters
    ----------
    C : numpy.array, torch.tensor
        Confusion matrix
    binary_class : None or 1,2,3,.., optional
        Used for binary F1 score, defines the target class
    mode : Can be 'binary', 'micro' or 'macro',  optional
        Define the calculation method for multi class F1 score
        
    Returns
    -------
    SPEC : float, numpy.array
        Computed specificity or list of specificities for each class
    """
    
       
    
    # Define number of classes
    if num_classes==-1:
        num_classes = C.shape[0]
    
    if mode=='binary':
        if binary_class==-1:
            if C.sum()>0:
                SPEC=[]
                for c in range(num_classes):
                    TP = C[c,c]
                    FN = C[c,:].sum() - TP
                    FP = C[:,c].sum() - TP
                    TN = C.sum()-C[c,:].sum()-C[:,c].sum()+TP
                    if (TN+FP)>0:
                        SPEC.append(TN/(TN+FP))
                    else:
                        SPEC.append(None)
            else:
                SPEC = [None for i in range(num_classes)]
            if class_names is not None:
                SPEC = {class_names[i]: SPEC[i] for i in range(len(class_names))}
            else:
                SPEC = {str(i): SPEC[i] for i in range(num_classes)}
        elif binary_class is None: 
            TP = C[1:,1:].sum()
            FN = C[1:,0].sum()
            FP = C[0,1:].sum()
            TN = C[0,0].sum()
            if (TN+FP)>0:
                SPEC = TN/(TN+FP)
            else:
                SPEC = None
        else:
            target, pred = confusion_to_target_pred(C)
            spec = torchmetrics.Specificity(num_classes=num_classes, average='none', task='binary')
            SPEC = float(spec(pred, target))
    if mode=='micro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C)
            #spec = torchmetrics.Specificity(num_classes=num_classes, average='micro', task='multiclass') 
            #SPEC = float(spec(pred, target))
            TP, FP, TN, FN = confusion_to_rate(C)
            SPEC = TN/(TN+FP)
        else:
            SPEC = None
    if mode=='macro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C)
            #spec = torchmetrics.Specificity(num_classes=num_classes, average='macro', task='multiclass') 
            #SPEC = float(spec(pred, target))
            TP, FP, TN, FN = confusion_to_rate_bin(C)
            SPEC=0
            for i in range(C.shape[0]):
                if (TN[i]+FP[i])>0:
                    SPEC = SPEC + (TN[i]/(TN[i]+FP[i]))
            SPEC = SPEC/C.shape[0]
        else:
            SPEC = None
    
    return SPEC

def accuracy(C, num_classes=-1, binary_class=-1, mode='binary', class_names=None):
    """ Compute accuracy
    
    Parameters
    ----------
    C : numpy.array, torch.tensor
        Confusion matrix
    binary_class : 1,2,3,.., for each class or -1 to get a list of all accuraciesoptional
        Used for binary F1 score, defines the target class
    mode : Can be 'binary', 'micro' or 'macro',  optional
        Define the calculation method for multi class F1 score
        
    Returns
    -------
    ACC : float, numpy.array
        Computed accuracy or list of accuracies for each class
    """
    
    
    
    # Define number of classes
    if num_classes==-1:
        num_classes = C.shape[0]
    
    if mode=='binary':
        #acc = torchmetrics.Accuracy(num_classes=num_classes, average='none')
        #acc = torchmetrics.Accuracy(num_classes=num_classes, average='none', task='binary')
        if binary_class==-1:
            if C.sum()>0:
                ACC=[]
                for c in range(num_classes):
                    # TP = C[c,c].sum()
                    # TN = C[0,0].sum()
                    # FP = C[:,c].sum() - TP
                    # FN = C[c,:].sum() - TP
                    TP = C[c,c]
                    FN = C[c,:].sum() - TP
                    FP = C[:,c].sum() - TP
                    TN = C.sum()-C[c,:].sum()-C[:,c].sum()+TP
                    if (TP+TN+FP+FN)>0:
                        ACC.append((TP+TN)/(TP+TN+FP+FN))
                    else:
                        ACC.append(None)
            else:
                ACC = [None for i in range(num_classes)]
            if class_names is not None:
                ACC = {class_names[i]: ACC[i] for i in range(len(class_names))}
            else:
                ACC = {str(i): ACC[i] for i in range(num_classes)}
        elif binary_class is None: 
            TP = C[1:,1:].sum()
            TN = C[0,0].sum()
            FP = C[0,1:].sum()
            FN = C[1:,0].sum()
            if (TP+TN+FP+FN)>0:
                ACC = (TP+TN)/(TP+TN+FP+FN)
            else:
                ACC = None
        else:
            acc = torchmetrics.Accuracy(num_classes=num_classes, average='none', task='binary')
            target, pred = confusion_to_target_pred(C)  
            #print('target123', target)
            #print('pred123', pred)
            #print('acc123', acc(torch.from_numpy(pred), torch.from_numpy(target)))
            ACC = float(acc(pred, target))
    if mode=='micro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C)   
            #acc = torchmetrics.Accuracy(num_classes=num_classes, average='micro', task='multiclass') 
            #ACC = float(acc(pred, target))
            TP, FP, TN, FN = confusion_to_rate(C)
            ACC = (TP)/(TP+FP)

        else:
            ACC=None
    if mode=='macro':
        if C.sum()>0:
            target, pred = confusion_to_target_pred(C)   
            acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro', task='multiclass') 
            ACC = float(acc(pred, target))
            # TP, FP, TN, FN = confusion_to_rate_bin(C)
            # ACC=0
            # for i in range(C.shape[0]):
            #     if (TP[i]+TN[i]+FP[i]+FN[i])>0:
            #         ACC=ACC + (TP[i]+TN[i]) / (TP[i]+TN[i]+FP[i]+FN[i])
            # ACC=ACC/C.shape[0]

        else:
            ACC=None
    
    return ACC

def f1(C, num_classes=-1, binary_class=-1, mode='binary', class_names=None):
    """ Compute F1 score
    
    Parameters
    ----------
    C : numpy.array, torch.tensor
        Confusion matrix
    binary_class : None or 1,2,3,.., optional
        Used for binary F1 score, defines the target class
        If binary_class is None, F1 score is estimated by all classes vs. first class (background)
    mode : Can be 'binary', 'micro' or 'macro',  optional
        Define the calculation method for multi class F1 score
        
    Returns
    -------
    F1 : float, numpy.array
        Computed F1 score or list of F1 scores for each class
    """
    
    #target, pred = confusion_to_target_pred(C)   
    
    # Define number of classes
    if num_classes==-1:
        num_classes = C.shape[0]
    
    if mode=='binary':
        if binary_class==-1:
            if C.sum()>0:
                F1=[]
                for c in range(num_classes):
                    # TP = C[c,c].sum()
                    # FP = C[:,c].sum() - TP
                    # FN = C[c,:].sum() - TP
                    TP = C[c,c]
                    FN = C[c,:].sum() - TP
                    FP = C[:,c].sum() - TP
                    TN = C.sum()-C[c,:].sum()-C[:,c].sum()+TP
                    if (TP+FP+FN)>0:
                        F1.append(2*TP/(2*TP+FP+FN))
                    else:
                        F1.append(None)
            else:
                F1 = [None for i in range(num_classes)]
            if class_names is not None:
                F1 = {class_names[i]: F1[i] for i in range(len(class_names))}
            else:
                F1 = {str(i): F1[i] for i in range(num_classes)}
        elif binary_class is None:
            TP = C[1:,1:].sum()
            FP = C[0,1:].sum()
            FN = C[1:,0].sum()
            if (2*TP+FP+FN)>0:
                F1 = float(2*TP/(2*TP+FP+FN))
            else:
                F1 = None
        else:
            if C.sum()>0:
                target, pred = confusion_to_target_pred(C)   
                f1 = torchmetrics.F1(num_classes=num_classes, average='none')
                F1 = float(f1(torch.from_numpy(pred), torch.from_numpy(target))[binary_class])
            else:
                F1=None
    if mode=='micro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C)   
            #f1 = torchmetrics.F1Score(num_classes=num_classes, average='micro', task='multiclass') 
            #F1 = float(f1(pred, target))
            TP, FP, TN, FN = confusion_to_rate(C)
            F1 = (2*TP)/(2*TP+FP+FN)           
        else:
            F1 = None
    if mode=='macro':
        if C.sum()>0:
            #target, pred = confusion_to_target_pred(C)   
            #f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro', task='multiclass') 
            #F1 = float(f1(pred, target))
            TP, FP, TN, FN = confusion_to_rate_bin(C)
            F1=0
            for i in range(C.shape[0]):
                if (2*TP[i]+FP[i]+FN[i])>0:
                    F1 = F1 + (2*TP[i])/(2*TP[i]+FP[i]+FN[i])           
            F1 = F1/C.shape[0]
        else:
            F1 = None
    
    return F1


def kappa(C, num_classes=-1, weights='linear'):
    """ Compute kappa 
    
    Parameters
    ----------
    C : numpy.array, torch.tensor
        Confusion matrix
        
    Returns
    -------
    kappa : float
        Computed kappa
    """
    
    from torchmetrics import CohenKappa
    
    if isinstance(weights, np.ndarray):
        target, pred, wei = confusion_to_target_pred_weight(C, weights)  
        if target.shape[0]==0:
            return None
        
        # Define number of classes
        if num_classes==-1:
            num_classes = C.shape[0]
            
        #cohenkappa = CohenKappa(num_classes=num_classes, weights=weights)
        #cohenkappa = CohenKappa(num_classes=num_classes, sample_weight=wei, task='multiclass')
        #kappa = cohenkappa(pred, target)
        
        
        
        kappa = sm.stats.cohens_kappa(C, weights=weights, return_results=True, wt=None)['kappa']
        #kappa3 = sm.stats.cohens_kappa(C, return_results=True, wt='linear')['kappa']
    else:
        target, pred = confusion_to_target_pred(C)  
        if target.shape[0]==0:
            return None
        
        # Define number of classes
        if num_classes==-1:
            num_classes = C.shape[0]
            
        #cohenkappa = CohenKappa(num_classes=num_classes, weights=weights)
        cohenkappa = CohenKappa(num_classes=num_classes, weights=weights, task='multiclass')
        kappa = cohenkappa(pred, target)

    return kappa

# def kappa(C, num_classes=-1, weights='linear'):
#     """ Compute kappa 
    
#     Parameters
#     ----------
#     C : numpy.array, torch.tensor
#         Confusion matrix
        
#     Returns
#     -------
#     kappa : float
#         Computed kappa
#     """
    
#     from torchmetrics import CohenKappa
    
#     target, pred = confusion_to_target_pred(C)  
#     if target.shape[0]==0:
#         return None
    
#     # Define number of classes
#     if num_classes==-1:
#         num_classes = C.shape[0]
        
#     #cohenkappa = CohenKappa(num_classes=num_classes, weights=weights)
#     cohenkappa = CohenKappa(num_classes=num_classes, weights=weights, task='multiclass')
#     kappa = cohenkappa(pred, target)

#     return kappa

def icc(C_list, icc_type='Single raters absolute', class_names=None, num_classes=-1, binary_class=-1, mode='binary'):
    """ Compute icc 
    
    Parameters
       ----------
    target : numpy.array, torch.tensor
        Target value for  icc
    pred : numpy.array, torch.tensor
        Predicted value for icc
    icc_type : string
        ICC type used for comptation. Check https://pingouin-stats.org/generated/pingouin.intraclass_corr.html for more information.
        
    Returns
    -------
    icc_values : list of float
        Computed icc of type for each class
    """
    
    if num_classes==-1:
        num_classes = C_list[0].shape[0]
    
    if mode=='binary':
        if binary_class==-1:
            icc_values = []
            # Iterate over classes
            for c in range(num_classes):
                target=[]
                pred=[]
                for C in C_list:
                    target.append(C[c,:].sum())
                    pred.append(C[:,c].sum())
                    
                data = pd.DataFrame()
                data['targets'] = [i for i in range(len(target))] + [i for i in range(len(pred))]
                data['raters'] = [0 for i in range(len(target))] + [1 for i in range(len(pred))]
                data['ratings'] = list(target) + list(pred)
                icc = pg.intraclass_corr(data=data, targets='targets', raters='raters',ratings='ratings')
                icc_value = icc.loc[list(icc['Description']).index(icc_type), 'ICC']
                icc_values.append(icc_value)
            
            if class_names is not None:
                icc_values = {class_names[i]: icc_values[i] for i in range(len(class_names))}
            else:
                icc_values = {str(i): icc_values[i] for i in range(num_classes)}
            return icc_values
                
        elif binary_class is None:
            class_names = ['NOCAC', 'CAC']
            icc_values = []
            target=[]
            pred=[]
            for C in C_list:
                target.append(C[0,:].sum())
                pred.append(C[:,0].sum())
            data = pd.DataFrame()
            data['targets'] = [i for i in range(len(target))] + [i for i in range(len(pred))]
            data['raters'] = [0 for i in range(len(target))] + [1 for i in range(len(pred))]
            data['ratings'] = list(target) + list(pred)
            icc = pg.intraclass_corr(data=data, targets='targets', raters='raters',ratings='ratings')
            icc_value = icc.loc[list(icc['Description']).index(icc_type), 'ICC']
            icc_values.append(icc_value)
            
            target=[]
            pred=[]
            for C in C_list:
                target.append(C[1:,:].sum())
                pred.append(C[:,1:].sum())
            data = pd.DataFrame()
            data['targets'] = [i for i in range(len(target))] + [i for i in range(len(pred))]
            data['raters'] = [0 for i in range(len(target))] + [1 for i in range(len(pred))]
            data['ratings'] = list(target) + list(pred)
            icc = pg.intraclass_corr(data=data, targets='targets', raters='raters',ratings='ratings')
            icc_value = icc.loc[list(icc['Description']).index(icc_type), 'ICC']
            icc_values.append(icc_value)
            if class_names is not None:
                icc_values = {class_names[i]: icc_values[i] for i in range(len(class_names))}
            else:
                icc_values = {str(i): icc_values[i] for i in range(num_classes)}
            return icc_values
        
        else:
            icc_values = []
            # Iterate over classes
            for c in range(num_classes):
                target=[]
                pred=[]
                for C in C_list:
                    target.append(C[c,:].sum())
                    pred.append(C[:,c].sum())
                    
                data = pd.DataFrame()
                data['targets'] = [i for i in range(len(target))] + [i for i in range(len(pred))]
                data['raters'] = [0 for i in range(len(target))] + [1 for i in range(len(pred))]
                data['ratings'] = list(target) + list(pred)
                icc = pg.intraclass_corr(data=data, targets='targets', raters='raters',ratings='ratings')
                icc_value = icc.loc[list(icc['Description']).index(icc_type), 'ICC']
                icc_values.append(icc_value)
            
            if class_names is not None:
                icc_values = {class_names[i]: icc_values[i] for i in range(len(class_names))}
                icc_value = icc_values[list(icc_values.keys())[binary_class]]
            else:
                icc_values = {str(i): icc_values[i] for i in range(num_classes)}
                icc_value = icc_values[list(icc_values.keys())[binary_class]]
    
            return icc_value
    elif mode=='micro':
        # Iterate over classes
        target=[]
        pred=[]
        for c in range(num_classes):
            for C in C_list:
                target.append(C[c,:].sum())
                pred.append(C[:,c].sum())
                
        data = pd.DataFrame()
        data['targets'] = [i for i in range(len(target))] + [i for i in range(len(pred))]
        data['raters'] = [0 for i in range(len(target))] + [1 for i in range(len(pred))]
        data['ratings'] = list(target) + list(pred)
        icc = pg.intraclass_corr(data=data, targets='targets', raters='raters',ratings='ratings')
        icc_value = icc.loc[list(icc['Description']).index(icc_type), 'ICC']

        return icc_value
        