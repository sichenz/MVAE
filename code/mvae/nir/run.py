#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser

parser = ArgumentParser()


# inference parameters
parser.add_argument("-f","--folds", dest="FOLDS", type=int, default=4)

# data / misc
parser.add_argument("--no_center_bp", dest="CENTER_BP", action='store_false')


args = parser.parse_args()

FOLDS = args.FOLDS

CENTER_BP = args.CENTER_BP



from datetime import datetime
save_index = str('x'+datetime.now().strftime("%m%d%y-%H%M%S"))

import os
os.mkdir(save_index)

import sys
sys.path.insert(1, '../')


import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine

pyro.set_rng_seed(42)

import random
random.seed(42)

import pandas as pd

from skimage import io
from sklearn import metrics
from matplotlib import pyplot as plt

from tqdm.auto import tqdm, trange

from sklearn.metrics import classification_report

from data import SplitData
from metrics import Metrics


assert pyro.__version__.startswith('1.3.0')



## Helper functions:

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def compute_distance(z):
    b = z.reshape(z.shape[0], 1, z.shape[1])
    return np.sqrt(np.einsum('ijk, ijk->ij', z-b, z-b))


## Data Loading

# First, load text data, and apply word filter. Note on notation: `tx` stands for "true x," because the model variables are also called x.

textdf = pd.read_csv("../../../data/web_dtfm20_binary.csv", index_col=0)
tx_text = textdf.values
seltext = tx_text.sum(0) > 0.05
tx_text = textdf.values[:,seltext]

gt20words = tx_text.sum(1) > 20
tx_text = tx_text[gt20words,:]

words = textdf.columns[seltext]
N, V = tx_text.shape

binfeats = pd.read_csv("../../../data/y_bin_all_py2.csv", index_col=0)
tx_b = binfeats.values
tx_b = tx_b[gt20words,:]
M_b = tx_b.shape[1]

catfeats = pd.read_csv("../../../data/y_mult_ncolors_py2.csv", index_col=0)

tx_c1 = catfeats.values[:,0][gt20words]
M_c1 = len(np.unique(tx_c1))
tx_c1 = np.expand_dims(tx_c1, 1)

tx_c2 = catfeats.values[:,1][gt20words]
M_c2 = len(np.unique(tx_c2))
tx_c2 = np.expand_dims(tx_c2, 1)

tx_c3 = catfeats.values[:,2][gt20words]
M_c3 = len(np.unique(tx_c3))
tx_c3 = np.expand_dims(tx_c3, 1)

tx_c4 = catfeats.values[:,3][gt20words]
M_c4 = len(np.unique(tx_c4))
tx_c4 = np.expand_dims(tx_c4, 1)

tx_c5 = catfeats.values[:,4][gt20words]
M_c5 = len(np.unique(tx_c5))
tx_c5 = np.expand_dims(tx_c5, 1)

c1_labels = np.array(["black","blue_dark","blue_light","blue_medium","brown","green_dark",
                      "green_light","grey_dark","grey_light","orange","red","red_dark",
                      "yellow"])

c2_labels = np.array(["circle","rect-oval_medium","rect-oval_large","rect-oval_thin",
                      "square","triangle"])

c3_labels = np.array(["bad_letters","bulky_hollow_geometric","circular","dense_simple_geom",
                      "detailed_circle","hollow_circle","detailed_hor","long_hor","no_mark",
                      "simple","square","thin_vert_rect","vert_narrow","detailed","thin",
                      "hor_wispy"])

c4_labels = np.array(["nochars","sans","serif"])

c5_labels = np.array(["one_color","two_colors","three_colors","many_colors"])

bp = pd.read_csv("../../../data/bp_avg_all_traits.csv", index_col=0)

bp_labels = bp.columns

tx_bp = bp.values
tx_bp = tx_bp[gt20words]
if CENTER_BP:
    tx_bp = (tx_bp - tx_bp.mean(0)) / tx_bp.std(0)
M_bp = tx_bp.shape[1]

indus = pd.read_csv("../../../data/industry_codes_b2bc.csv", index_col=0)
indus = indus.iloc[np.in1d(indus.index, bp.index),:]
indus = indus.sort_index()

tx_indus = indus.values.astype('int')
tx_indus = tx_indus[:, tx_indus.sum(0) > 9]
tx_indus = tx_indus[gt20words,:]
M_indus = tx_indus.shape[1]

indus_labels = indus.columns[indus.values.sum(0) > 9]

allnames = binfeats.index.values[gt20words]

x_sizes = {"text": V, 
           "bin": M_b, 
           "cat1": M_c1, 
           "cat2": M_c2, 
           "cat3": M_c3, 
           "cat4": M_c4, 
           "cat5": M_c5, 
           "bp": M_bp, 
           "indus": M_indus, 
           "logo": M_b + M_c1 + M_c2 + M_c3 + M_c4 + M_c5, 
           "all": V + M_b + M_c1 + M_c2 + M_c3 + M_c4 + M_c5 + M_bp + M_indus}

noptions = np.array([M_c1, M_c2, M_c3, M_c4, M_c5])


## Training: Instantiate Model and Run

givens = pd.DataFrame(np.concatenate(([[FOLDS]]))).T
givens.columns = ["folds"]


# Create holdout and cross-validation subsets (just the indices):

if FOLDS > 1:
    holdout_indices = list(split(np.arange(N), FOLDS))
    holdout_indices.append(np.array([]))
    fold_indices = [np.setdiff1d(np.arange(N), holdout_indices[i]) for i in range(FOLDS)]
    fold_indices.append(np.arange(N))
else:
    holdout_indices = [np.array([])]

def convert_onehot_K(intvec, K):
    N = len(intvec)
    return np.array([np.array([1 if i==intvec[j] else 0 for i in range(K)]) for j in range(N)])

class NoInfoPredict():
    def __init__(self, data, N, K_vec):
        self.text = np.tile(data.x_text.mean(0), (N,1))
        self.bin = np.tile(data.x_bin.mean(0), (N,1))
        self.cat1 = np.tile(convert_onehot_K(data.x_cat1.flatten(), K_vec[0]).mean(0), (N,1))
        self.cat2 = np.tile(convert_onehot_K(data.x_cat2.flatten(), K_vec[1]).mean(0), (N,1))
        self.cat3 = np.tile(convert_onehot_K(data.x_cat3.flatten(), K_vec[2]).mean(0), (N,1))
        self.cat4 = np.tile(convert_onehot_K(data.x_cat4.flatten(), K_vec[3]).mean(0), (N,1))
        self.cat5 = np.tile(convert_onehot_K(data.x_cat5.flatten(), K_vec[4]).mean(0), (N,1))
        self.bp = np.tile(data.x_bp.mean(0), (N,1))
        self.indus = np.tile(data.x_indus.mean(0), (N,1))


# Run the model across all folds (sequentially):
for fold in tqdm(range(FOLDS+1), desc="Folds"):

    data = SplitData(tx_text, tx_b, tx_c1, tx_c2, tx_c3, tx_c4, tx_c5, tx_bp, tx_indus, 
                     allnames, test_indices = holdout_indices[fold])     

    data.training.make_torch()

    nipred = NoInfoPredict(data.training, data.training.N, noptions)
    nir_training = Metrics(data.training, nipred, noptions)
    nir_training.summarize(path = str(save_index) + "/" + str(save_index) + "_training_metrics.csv", index = fold, givens = givens)
    nir_training.save_features_table(path = str(save_index) + "/" + str(save_index) + "_training_bin_features.csv", names = binfeats.columns, index = fold, givens = givens)

    if hasattr(data, 'test'):
        data.test.make_torch()

        nipred = NoInfoPredict(data.training, data.test.N, noptions)
        nir_test = Metrics(data.test, nipred, noptions)
        nir_test.summarize(path = str(save_index) + "/" + str(save_index) + "_test_metrics.csv", index = fold, givens = givens)
        nir_test.summarize(path = str(save_index) + "/" + str(save_index) + "_res_metrics.csv", index = fold, givens = givens)
        nir_test.summarize(path = str(save_index) + "/" + str(save_index) + "_des_metrics.csv", index = fold, givens = givens)
        nir_test.summarize(path = str(save_index) + "/" + str(save_index) + "_mgr_metrics.csv", index = fold, givens = givens)
        nir_test.save_features_table(path = str(save_index) + "/" + str(save_index) + "_test_bin_features.csv", names = binfeats.columns, index = fold, givens = givens)
        nir_test.save_features_table(path = str(save_index) + "/" + str(save_index) + "_des_bin_features.csv", names = binfeats.columns, index = fold, givens = givens)
        
