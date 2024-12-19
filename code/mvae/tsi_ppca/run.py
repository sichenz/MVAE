#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser

parser = ArgumentParser()

# dimensionality: latent space
parser.add_argument("--k", dest="K", type=int, default=20)

# dimensionality: decoders
parser.add_argument("--text_dec", dest="TEXT_DEC_DIMS", type=int, default=400)
parser.add_argument("--logo_dec", dest="LOGO_DEC_DIMS", type=int, default=400)
parser.add_argument("--bp_dec", dest="BP_DEC_DIMS", type=int, default=400)
parser.add_argument("--ind_dec", dest="INDUS_DEC_DIMS", type=int, default=400)

# dimensionality: encoders (defaults = sum of relevant decoders)
parser.add_argument("--full_enc", dest="FULL_ENC_DIMS", type=int, default=400)
parser.add_argument("--des_enc", dest="DES_ENC_DIMS", type=int, default=200)
parser.add_argument("--mgr_enc", dest="MGR_ENC_DIMS", type=int, default=200)
parser.add_argument("--res_enc", dest="RES_ENC_DIMS", type=int, default=50)

# inference parameters
parser.add_argument("-f","--folds", dest="FOLDS", type=int, default=4)
parser.add_argument("-b", "--batches", dest="BATCHES", type=int, default=5000)
parser.add_argument("-i", "--iters", dest="ITERS", type=int, default=10)
parser.add_argument("-lr", "--adam_lr", dest="ADAM_LR", type=float, default=1e-5)
parser.add_argument("-p", "--num_particles", dest="NUM_PARTICLES", type=int, default=1)
parser.add_argument("-wd", "--weight_decay", dest="WEIGHT_DECAY", type=float, default=0.)

# data / misc
parser.add_argument("--no_center_bp", dest="CENTER_BP", action='store_false')
parser.add_argument("--enable_tqdm", dest="DISABLE_TQDM", action='store_false')
parser.add_argument("--first_fold", dest="FIRST_FOLD", action='store_true')
parser.add_argument("--track", dest="TRACK", action='store_true')

args = parser.parse_args()

DECODER_DIMS = {"text": args.TEXT_DEC_DIMS, "logo": args.LOGO_DEC_DIMS, "bp": args.BP_DEC_DIMS, "indus": args.INDUS_DEC_DIMS}
ENCODER_DIMS = {"full": args.FULL_ENC_DIMS, "res": args.RES_ENC_DIMS, "mgr": args.MGR_ENC_DIMS, "design": args.DES_ENC_DIMS}
K = args.K

FOLDS = args.FOLDS
BATCHES = args.BATCHES
ITERS = args.ITERS

ADAM_LR = args.ADAM_LR
MIN_AF = 1e-6
ANNEALING_BATCHES = max(args.BATCHES - 1000, 0)
NUM_PARTICLES = args.NUM_PARTICLES

CENTER_BP = args.CENTER_BP

WEIGHT_DECAY = args.WEIGHT_DECAY

DISABLE_TQDM = args.DISABLE_TQDM

FIRST_FOLD = args.FIRST_FOLD
TRACK = args.TRACK



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
from model import LogoMVAE

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

task_sizes = {"full": x_sizes["all"], 
              "res": x_sizes["logo"] + x_sizes["indus"], 
              "design": x_sizes["text"] + x_sizes["bp"] + x_sizes["indus"], 
              "mgr": x_sizes["all"] - x_sizes["bp"]}

noptions = np.array([M_c1, M_c2, M_c3, M_c4, M_c5])


## Training: Instantiate Model and Run

givens = pd.DataFrame(np.concatenate(([[K], list(DECODER_DIMS.values()), list(ENCODER_DIMS.values()), [BATCHES], [ITERS], [ADAM_LR], [ANNEALING_BATCHES], [NUM_PARTICLES], [CENTER_BP], [WEIGHT_DECAY], [FOLDS]]))).T
givens.columns = ["K", "text_dec", "logo_dec", "bp_dec", "indus_dec", "full_enc", "logo_enc", "mgr_enc", "des_enc", "batches", "iters", "adam_lr", "annealing_batches", "num_particles", "center_bp", "weight_decay", "folds"]


# Create holdout and cross-validation subsets (just the indices):

if FOLDS > 1:
    holdout_indices = list(split(np.arange(N), FOLDS))
    holdout_indices.append(np.array([]))
    fold_indices = [np.setdiff1d(np.arange(N), holdout_indices[i]) for i in range(FOLDS)]
    fold_indices.append(np.arange(N))
else:
    holdout_indices = [np.array([])]


# Set the KL annealing schedule (same across each fold):

schedule = np.linspace(MIN_AF, 1., ANNEALING_BATCHES)
# schedule = np.concatenate([np.linspace(MIN_AF, 1., round(ANNEALING_BATCHES/4.)) for _ in range(4)])

# Containers for tracking (if used):
track_training = []
track_test = []
track_mgr_bp = []
track_des_bin = []
track_res_bp = []
track_des_cat1 = []
track_full_bp = []
track_full_bin = []

# Run the model across all folds (sequentially):
if FIRST_FOLD:
    folds_to_run = 1
else:
    folds_to_run = FOLDS + 1
    
for fold in tqdm(range(folds_to_run), desc="Folds", disable=DISABLE_TQDM):
    
    pyro.clear_param_store()

    data = SplitData(tx_text, tx_b, tx_c1, tx_c2, tx_c3, tx_c4, tx_c5, tx_bp, tx_indus, 
                     allnames, noptions, test_indices = holdout_indices[fold])   
    
    has_test = hasattr(data, 'test')
    if has_test:
        data.test.make_torch()
        
    lmvae = LogoMVAE(K, ENCODER_DIMS, DECODER_DIMS, x_sizes, task_sizes, use_cuda=True)
    optimizer = Adam({"lr": ADAM_LR}) #, "weight_decay": 0.4})
    svi = SVI(lmvae.model, lmvae.guide, optimizer, loss=Trace_ELBO(num_particles = NUM_PARTICLES))

    for i in tqdm(range(BATCHES), desc="Batches", leave=False, disable=DISABLE_TQDM):

        if i < ANNEALING_BATCHES:
            annealing_factor = schedule[i]
        else:
            annealing_factor = 1.

        data.training.shuffle()

        for j in tqdm(range(ITERS), desc="Iters", leave=False, disable=True):
            svi.step(data.training, annealing_factor)
            
        if (i % 200 == 0) or (i == BATCHES-1):
            if has_test: 
                print('fold ' + str(fold) + ', batch ' + str(i))
                print(svi.evaluate_loss(data.test, annealing_factor))
                
        if TRACK:
            if (i % 50 == 0) or (i == BATCHES-1):
                track_training.append(svi.evaluate_loss(data.training, annealing_factor))
                if has_test: 
                    track_test.append(svi.evaluate_loss(data.test, annealing_factor))
                    
                    lmvae.eval();
                    
                    # Predictions for full task:
                    lmvae.predict(data.test, network = "full")
                    track_full_bin.append(lmvae.pred.metrics.bin_report['macro avg']['f1-score'])
                    track_full_bp.append(lmvae.pred.metrics.bp_mse.features.mean())

                    # Predictions for res task:
                    lmvae.predict(data.test, network = "res")
                    track_res_bp.append(lmvae.pred.metrics.bp_mse.features.mean())
                
                    # Predictions for des task:        
                    lmvae.predict(data.test, network = "des")
                    track_des_bin.append(lmvae.pred.metrics.bin_report['macro avg']['f1-score'])
                    track_des_cat1.append(lmvae.pred.metrics.cat1_report['macro avg']['f1-score'])
                
                    # Predictions for mgr task:
                    lmvae.predict(data.test, network = "mgr")
                    track_mgr_bp.append(lmvae.pred.metrics.bp_mse.features.mean())
                
                    lmvae.train();
    
    # Final save of stats
    lmvae.eval()
    
    lmvae.predict(data.training)
    lmvae.pred.metrics.summarize(path = save_index + "/" + save_index + "_training_metrics.csv", index = fold, givens = givens)
    lmvae.pred.metrics.save_features_table(path = save_index + "/" + save_index + "_training_bin_features.csv", names = binfeats.columns, index = fold, givens = givens)
        
    if hasattr(data, 'test'):
        data.test.make_torch()
        lmvae.predict(data.test)
        lmvae.pred.metrics.summarize(path = save_index + "/" + save_index + "_test_metrics.csv", index = fold, givens = givens)
        lmvae.pred.metrics.save_features_table(path = save_index + "/" + save_index + "_test_bin_features.csv", names = binfeats.columns, index = fold, givens = givens)
        lmvae.pred.ll.summarize(path = save_index + "/" + save_index + "_test_ll.csv", index = fold, givens = givens)
        
        lmvae.predict(data.test, network = "res")
        lmvae.pred.metrics.summarize(path = save_index + "/" + save_index + "_res_metrics.csv", index = fold, givens = givens)
        lmvae.pred.ll.summarize(path = save_index + "/" + save_index + "_res_ll.csv", index = fold, givens = givens)
                
        lmvae.predict(data.test, network = "des")
        lmvae.pred.metrics.summarize(path = save_index + "/" + save_index + "_des_metrics.csv", index = fold, givens = givens)
        lmvae.pred.metrics.save_features_table(path = save_index + "/" + save_index + "_des_bin_features.csv", names = binfeats.columns, index = fold, givens = givens)
        lmvae.pred.ll.summarize(path = save_index + "/" + save_index + "_des_ll.csv", index = fold, givens = givens)
                
        lmvae.predict(data.test, network = "mgr")
        lmvae.pred.metrics.summarize(path = save_index + "/" + save_index + "_mgr_metrics.csv", index = fold, givens = givens)
        lmvae.pred.ll.summarize(path = save_index + "/" + save_index + "_mgr_ll.csv", index = fold, givens = givens)
        

## Save image
pyro.get_param_store().save(save_index + "/" + save_index + ".pt")


# Construct and save z-space neighbors table

if not TRACK:

    lmvae.predict(data.training)

    z = lmvae.pred.z.z_loc.cpu().numpy()
    end_names = data.training.names

    dist_z = compute_distance(z)

    test_firms = ['itw','harman-intl','lilly','goldman-sachs','21st-century-fox','facebook','gucci','old-navy','3m','actavis','mcdonalds', 'kfc']
    test_neighbors = [end_names[dist_z[np.where(end_names == test_firms[i])[0][0],:].argsort()][1:5] for i in range(len(test_firms))]
    test_dist = [np.sort(dist_z[np.where(end_names == test_firms[i])[0][0],:].round(2))[1:5] for i in range(len(test_firms))]
    formatted_neighbors = [", ".join(test_neighbors[i].tolist()) for i in range(len(test_neighbors))]

    neighbors_df = pd.DataFrame(test_neighbors)
    neighbors_df.index = test_firms
    neighbors_df.columns = np.arange(1,5)

    neighbors_df.to_csv(save_index + "/" + save_index + "_neighbors_table.csv")


# Save tracking plots (if used):
if TRACK:
    plt.figure()
    plt.plot(track_training)
    plt.savefig(save_index + "/" +"track_training.png")
    
    plt.figure()
    plt.plot(track_test)
    plt.savefig(save_index + "/" +"track_test.png")
    
    plt.figure()
    plt.plot(track_mgr_bp)
    plt.savefig(save_index + "/" +"track_mgr_bp.png")
    
    plt.figure()
    plt.plot(track_des_bin)
    plt.savefig(save_index + "/" +"track_des_bin.png")    
    
    plt.figure()
    plt.plot(track_res_bp)
    plt.savefig(save_index + "/" +"track_res_bp.png")    
    
    plt.figure()
    plt.plot(track_des_cat1)
    plt.savefig(save_index + "/" +"track_des_cat1.png")  
    
    plt.figure()  
    plt.plot(track_full_bp)
    plt.savefig(save_index + "/" +"track_full_bp.png")   
    
    plt.figure() 
    plt.plot(track_full_bin)
    plt.savefig(save_index + "/" +"track_full_bin.png")
