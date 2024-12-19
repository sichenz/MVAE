import os

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.set_rng_seed(42)

import random
random.seed(42)

import pandas as pd

from skimage import io
from sklearn import metrics
from matplotlib import pyplot as plt

from tqdm import tqdm

from sklearn.metrics import classification_report, roc_curve, auc

import warnings

class CompanyData():
    pass

class Data():
    def __init__(self, x_text, x_bin, x_cat1, x_cat2, x_cat3, x_cat4, x_cat5, x_bp, x_indus, names, noptions):
        self.x_text = x_text
        self.x_bin = x_bin
        self.x_cat1 = x_cat1
        self.x_cat2 = x_cat2
        self.x_cat3 = x_cat3
        self.x_cat4 = x_cat4
        self.x_cat5 = x_cat5
        self.x_bp = x_bp
        self.x_indus = x_indus
        
        self.x_names = names
        self.names = names
        self.N = x_text.shape[0]      
        self.noptions = noptions  
        
    def get_company(self, index=None, name=None):
        if (index == None) and (name == None):
            raise Exception("Need either an index or a name")
        
        if (index != None) and (name != None):
            raise Exception("Can't have both an index and a name")
        
        company = CompanyData()
        if (index != None):
            company.text = self.x_text[index]
            company.bin = self.x_bin[index]
            company.cat1 = self.x_cat1[index]
            company.cat2 = self.x_cat2[index]
            company.cat3 = self.x_cat3[index]
            company.cat4 = self.x_cat4[index]
            company.cat5 = self.x_cat5[index]
            company.bp = self.x_bp[index]
            company.indus = self.x_indus[index]
        
        if (name != None):
            company.text = self.x_text[self.x_names == name]
            company.bin = self.x_bin[self.x_names == name]
            company.cat1 = self.x_cat1[self.x_names == name]
            company.cat2 = self.x_cat2[self.x_names == name]
            company.cat3 = self.x_cat3[self.x_names == name]
            company.cat4 = self.x_cat4[self.x_names == name]
            company.cat5 = self.x_cat5[self.x_names == name]
            company.bp = self.x_bp[self.x_names == name]
            company.indus = self.x_indus[self.x_names == name]
            
        return company

    def make_torch(self):
        self.text = torch.tensor(self.x_text, dtype = torch.float).cuda()
        self.bin = torch.tensor(self.x_bin, dtype = torch.float).cuda()
        self.cat1 = torch.tensor(self.x_cat1, dtype = torch.float).cuda()
        self.cat2 = torch.tensor(self.x_cat2, dtype = torch.float).cuda()
        self.cat3 = torch.tensor(self.x_cat3, dtype = torch.float).cuda()
        self.cat4 = torch.tensor(self.x_cat4, dtype = torch.float).cuda()
        self.cat5 = torch.tensor(self.x_cat5, dtype = torch.float).cuda()
        self.bp = torch.tensor(self.x_bp, dtype = torch.float).cuda()
        self.indus = torch.tensor(self.x_indus, dtype = torch.float).cuda()
        
        self.cat1_hot = torch.nn.functional.one_hot(torch.tensor(self.x_cat1, dtype = torch.int).long(), self.noptions[0]).float().squeeze().cuda()
        self.cat2_hot = torch.nn.functional.one_hot(torch.tensor(self.x_cat2, dtype = torch.int).long(), self.noptions[1]).float().squeeze().cuda()
        self.cat3_hot = torch.nn.functional.one_hot(torch.tensor(self.x_cat3, dtype = torch.int).long(), self.noptions[2]).float().squeeze().cuda()
        self.cat4_hot = torch.nn.functional.one_hot(torch.tensor(self.x_cat4, dtype = torch.int).long(), self.noptions[3]).float().squeeze().cuda()
        self.cat5_hot = torch.nn.functional.one_hot(torch.tensor(self.x_cat5, dtype = torch.int).long(), self.noptions[4]).float().squeeze().cuda()
    
    def shuffle(self):
        self.make_torch()
        self.order = np.random.choice(np.arange(self.N), replace=False, size=self.N)
        
        self.text = self.text[self.order]
        self.bin = self.bin[self.order]
        self.cat1 = self.cat1[self.order]
        self.cat2 = self.cat2[self.order]
        self.cat3 = self.cat3[self.order]
        self.cat4 = self.cat4[self.order]
        self.cat5 = self.cat5[self.order]
        self.bp = self.bp[self.order]
        self.indus = self.indus[self.order]
        
        self.cat1_hot = self.cat1_hot[self.order]
        self.cat2_hot = self.cat2_hot[self.order]
        self.cat3_hot = self.cat3_hot[self.order]
        self.cat4_hot = self.cat4_hot[self.order]
        self.cat5_hot = self.cat5_hot[self.order]
        
        self.names = self.x_names[self.order]
        
class SplitData():
    def __init__(self, x_text, x_bin, x_cat1, x_cat2, x_cat3, x_cat4, x_cat5, x_bp, x_indus, names, noptions, test_indices=np.array([])):
        if (test_indices.shape[0] == 0):
            self.training = Data(x_text, x_bin, x_cat1, x_cat2, x_cat3, 
                                 x_cat4, x_cat5, x_bp, x_indus, names, noptions)
        else:
            self.test = Data(x_text[test_indices], x_bin[test_indices], x_cat1[test_indices], 
                             x_cat2[test_indices], x_cat3[test_indices], x_cat4[test_indices], 
                             x_cat5[test_indices], x_bp[test_indices], x_indus[test_indices], 
                             names[test_indices], noptions)
            
            training_indices = np.setdiff1d(np.arange(x_text.shape[0]), test_indices)
            self.training = Data(x_text[training_indices], x_bin[training_indices], 
                                 x_cat1[training_indices], x_cat2[training_indices], 
                                 x_cat3[training_indices], x_cat4[training_indices], 
                                 x_cat5[training_indices], x_bp[training_indices], 
                                 x_indus[training_indices], names[training_indices], noptions)