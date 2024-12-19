import numpy as np
import pandas as pd
import warnings

from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, auc

import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine

import random
random.seed(42)

pyro.set_rng_seed(42)

# given a set of predictions of class Predict(), computes fit statistics:

def convert_onehot_K(intvec, K):
    N = len(intvec)
    return np.array([np.array([1 if i==intvec[j] else 0 for i in range(K)]) for j in range(N)])
    
def mad(data, pred):
    return np.mean(np.abs(data - pred))
    
class Metrics():
    def __init__(self, data, pred, K_vec):
        
        self.data = data
        self.pred = pred
        
        self.bin_mse = MSE(data.bin.detach().cpu().numpy(), pred.bin)
        self.cat1_mse = MSE(convert_onehot_K(data.cat1.cpu().int().flatten().numpy(), K_vec[0]), pred.cat1)
        self.cat2_mse = MSE(convert_onehot_K(data.cat2.cpu().int().flatten().numpy(), K_vec[1]), pred.cat2)
        self.cat3_mse = MSE(convert_onehot_K(data.cat3.cpu().int().flatten().numpy(), K_vec[2]), pred.cat3)
        self.cat4_mse = MSE(convert_onehot_K(data.cat4.cpu().int().flatten().numpy(), K_vec[3]), pred.cat4)
        self.cat5_mse = MSE(convert_onehot_K(data.cat5.cpu().int().flatten().numpy(), K_vec[4]), pred.cat5)
        

        self.bin_mad = mad(data.bin.detach().cpu().numpy(), pred.bin)
        self.cat1_mad = mad(convert_onehot_K(data.cat1.cpu().int().flatten().numpy(), K_vec[0]), pred.cat1)
        self.cat2_mad = mad(convert_onehot_K(data.cat2.cpu().int().flatten().numpy(), K_vec[1]), pred.cat2)
        self.cat3_mad = mad(convert_onehot_K(data.cat3.cpu().int().flatten().numpy(), K_vec[2]), pred.cat3)
        self.cat4_mad = mad(convert_onehot_K(data.cat4.cpu().int().flatten().numpy(), K_vec[3]), pred.cat4)
        self.cat5_mad = mad(convert_onehot_K(data.cat5.cpu().int().flatten().numpy(), K_vec[4]), pred.cat5)

        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.bin_report = classification_report(data.bin.detach().cpu().numpy(), pred.bin > 0.5, output_dict = True)
            self.cat1_report = classification_report(data.cat1.cpu().int().flatten().numpy(), np.argmax(pred.cat1, 1), output_dict = True)
            self.cat2_report = classification_report(data.cat2.cpu().int().flatten().numpy(), np.argmax(pred.cat2, 1), output_dict = True)
            self.cat3_report = classification_report(data.cat3.cpu().int().flatten().numpy(), np.argmax(pred.cat3, 1), output_dict = True)
            self.cat4_report = classification_report(data.cat4.cpu().int().flatten().numpy(), np.argmax(pred.cat4, 1), output_dict = True)
            self.cat5_report = classification_report(data.cat5.cpu().int().flatten().numpy(), np.argmax(pred.cat5, 1), output_dict = True)


            
            fpr, tpr, thresholds = roc_curve(data.bin.cpu().numpy().flatten(), pred.bin.flatten())
            self.bin_auc = auc(fpr, tpr)
            

            
    def summarize(self, path, index=0, givens=pd.DataFrame([])):
        if givens.empty:
            metrics_table = pd.DataFrame(index = [index])
        else:
            metrics_table = givens
            metrics_table.set_index(np.array([index]), inplace=True)

        for attr, value in self.__dict__.items():

            if len(attr.split("_")) > 1:

                if attr.split("_")[1] == "mse":
                    cols = pd.DataFrame([value.overall], columns = [attr], index = [index])
                    metrics_table = pd.concat([metrics_table, cols], axis=1)

                if attr.split("_")[1] == "report":
                    dom = attr.split("_")[0]
            
                    if 'micro avg' in value.keys():
                        cols = pd.DataFrame([value['micro avg']], index = [index]).iloc[:,:3]
                        cols.columns = dom + '_micro_' + cols.columns
                        metrics_table = pd.concat([metrics_table, cols], axis=1)
            
                    if 'accuracy' in value.keys():
                        cols = pd.DataFrame({dom + '_accuracy': value['accuracy']}, index = [index]).iloc[:,:3]
                        metrics_table = pd.concat([metrics_table, cols], axis=1)
            
                    if 'macro avg' in value.keys():
                        cols = pd.DataFrame([value['macro avg']], index = [index]).iloc[:,:3]
                        cols.columns = dom + '_macro_' + cols.columns
                        metrics_table = pd.concat([metrics_table, cols], axis=1)

                if attr.split("_")[1] == "auc":
                    cols = pd.DataFrame([value], columns = [attr], index = [index])
                    metrics_table = pd.concat([metrics_table, cols], axis=1)  

                if attr.split("_")[1] == "mad":
                    cols = pd.DataFrame([value], columns = [attr], index = [index])
                    metrics_table = pd.concat([metrics_table, cols], axis=1)

        if path == None:
            return metrics_table

        else: 
            with open(path, 'a') as f:
                metrics_table.to_csv(f, header=f.tell()==0)
                
                
    def save_features_table(self, path, names, givens=pd.DataFrame([]), index=0):
        cr_dict = classification_report(self.data.bin.detach().cpu().numpy(), self.pred.bin > 0.5, target_names = names, output_dict = True)
        bin_feature_metrics = pd.DataFrame(cr_dict).T
        bin_feat_auc = []
        for j in range(self.pred.bin.shape[1]):
            fpr, tpr, thresholds = roc_curve(self.data.bin.cpu().numpy()[:,j], self.pred.bin[:,j])
            bin_feat_auc.append(auc(fpr, tpr))
            
        bin_feat_auc_avg = np.mean(bin_feat_auc)
        for j in range(4):
            bin_feat_auc.append(bin_feat_auc_avg)
        
        bin_feature_metrics["auc"] = bin_feat_auc
        bin_feature_metrics["index"] = index
        
        if not givens.empty:
            for col in givens.columns:
                bin_feature_metrics[col] = givens[col].values[0]
        
        if path == None:
            return bin_feature_metrics
        else:
            with open(path, 'a') as f:
                bin_feature_metrics.to_csv(f, header=f.tell()==0)
        
        
        
            

class MSE():
    def __init__(self, true, pred):
        self.features = np.mean((pred - true)**2, 0)
        self.overall = np.mean(self.features)


