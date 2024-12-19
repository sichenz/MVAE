import sys

import numpy as np
import pandas as pd
import warnings

from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, auc

import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine


import random
random.seed(42)

pyro.set_rng_seed(42)

from metrics_just_des import *
    
    
# main model:
class LogoMVAE(nn.Module):
    
    def __init__(self, z_dim, encoder_dims, decoder_dims, x_sizes, task_sizes, use_cuda=True):
        super().__init__()
        # create the encoder and decoder networks        
        self.encoder = TaskEncoder(z_dim, encoder_dims["design"], task_sizes)
        self.logo_decoder = LogoDecoder(z_dim, decoder_dims["logo"], x_sizes)

        self.z_dim = z_dim
        self.x_sizes = x_sizes
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
                
        self.K_vec = np.array([x_sizes['cat1'], x_sizes['cat2'], x_sizes['cat3'], x_sizes['cat4'], x_sizes['cat5']])


    def model(self, data, annealing_factor=1.):
        
        # register all decoders with Pyro
        pyro.module("logo_decoder", self.logo_decoder)
        
        
        # observation-specific priors/likelihoods:
        with pyro.plate("data", data.text.shape[0], dim=-1):

            z_loc = torch.zeros(torch.Size((data.text.shape[0], self.z_dim)), device='cuda')
            z_scale = torch.ones(torch.Size((data.text.shape[0], self.z_dim)), device='cuda')
            with poutine.scale(None, annealing_factor):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            # define domain-specific likelihoods:
            mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = self.logo_decoder.forward(z)
            pyro.sample("bin_obs", dist.Bernoulli(mu_bin).to_event(1), obs=data.bin)
            pyro.sample("cat1_obs", dist.Categorical(mu_cat1), obs=torch.flatten(data.cat1))
            pyro.sample("cat2_obs", dist.Categorical(mu_cat2), obs=torch.flatten(data.cat2))
            pyro.sample("cat3_obs", dist.Categorical(mu_cat3), obs=torch.flatten(data.cat3))
            pyro.sample("cat4_obs", dist.Categorical(mu_cat4), obs=torch.flatten(data.cat4))
            pyro.sample("cat5_obs", dist.Categorical(mu_cat5), obs=torch.flatten(data.cat5))


    def guide(self, data, annealing_factor=1.):
        
        # register the encoder with pyro:
        pyro.module("encoder", self.encoder)        

        N = data.text.shape[0]

        x_design = torch.cat((data.text, data.bp, data.indus), 1)
        
        task_xs = x_design

        with pyro.plate("data", N, dim=-1):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(task_xs)
            # sample the latent code z
            with poutine.scale(None, annealing_factor):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            
    def est_z(self, test_data, network="des"):
        self.z = NewZ(self, test_data, network)
        
    def pred_x(self, test_data, network="des"):
        self.x_pred = Predict(self, NewZ(self, test_data, network))
    
    def predict(self, test_data, network="des"):
        self.pred = LMVAE_Test(self, test_data, self.K_vec, network)

        
class TaskEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, task_sizes):
        super().__init__()
        self.task_design = Encoder(z_dim, hidden_dim, task_sizes["design"])
        
    def forward(self, task_xs):
        z_loc, z_scale = self.task_design(task_xs)        
        return z_loc, z_scale
	
	
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_size):
        super().__init__()
        # set up the three linear transformations used
        self.fc1 = nn.Linear(x_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # set up the non-linearities
        self.activation = nn.ReLU()
        self.softplus = nn.Softplus()
        # set up regularizers:
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        # compute the hidden units
        hidden = self.dropout(self.activation(self.fc1(x)))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = self.softplus(self.fc22(hidden))
        return z_loc, z_scale
	
	
class LogoDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_size):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2_bin = nn.Linear(hidden_dim + z_dim, x_size["bin"])
        self.fc2_cat1 = nn.Linear(hidden_dim + z_dim, x_size["cat1"])
        self.fc2_cat2 = nn.Linear(hidden_dim + z_dim, x_size["cat2"])
        self.fc2_cat3 = nn.Linear(hidden_dim + z_dim, x_size["cat3"])
        self.fc2_cat4 = nn.Linear(hidden_dim + z_dim, x_size["cat4"])
        self.fc2_cat5 = nn.Linear(hidden_dim + z_dim, x_size["cat5"])
        
        self.dropout = nn.Dropout()
        
    def forward(self, z):
        hidden = self.dropout(F.relu(self.fc1(z)))
        
        mu_bin = torch.sigmoid(self.fc2_bin(torch.cat((hidden, z), 1)))
        mu_cat1 = F.softmax(self.fc2_cat1(torch.cat((hidden, z), 1)), dim=1)
        mu_cat2 = F.softmax(self.fc2_cat2(torch.cat((hidden, z), 1)), dim=1)
        mu_cat3 = F.softmax(self.fc2_cat3(torch.cat((hidden, z), 1)), dim=1)
        mu_cat4 = F.softmax(self.fc2_cat4(torch.cat((hidden, z), 1)), dim=1)
        mu_cat5 = F.softmax(self.fc2_cat5(torch.cat((hidden, z), 1)), dim=1)
        
        return mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5
	
	
# estimates the z vectors for new data:
class NewZ():
    def __init__(self, lmvae, data, network="full"):
        N = data.text.shape[0]
        with torch.no_grad():
            x_design = torch.cat((data.text, data.bp, data.indus), 1)
            z_loc, z_scale = lmvae.encoder.task_design.forward(x_design)
                
            self.z_loc = z_loc
            self.z_scale = z_scale
	    
	    
# given z, predicts the average feature values (or feature probabilities), and saves them:
class Predict():
    def __init__(self, lmvae, z, samples = 100):
        with torch.no_grad():    
            z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
            mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = lmvae.logo_decoder.forward(z_draw)
            self.bin = mu_bin.detach().cpu().numpy()/samples
            self.cat1 = mu_cat1.detach().cpu().numpy()/samples
            self.cat2 = mu_cat2.detach().cpu().numpy()/samples
            self.cat3 = mu_cat3.detach().cpu().numpy()/samples
            self.cat4 = mu_cat4.detach().cpu().numpy()/samples
            self.cat5 = mu_cat5.detach().cpu().numpy()/samples

            for _ in range(samples-1):
                z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
                mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = lmvae.logo_decoder.forward(z_draw)
                self.bin += mu_bin.detach().cpu().numpy()/samples
                self.cat1 += mu_cat1.detach().cpu().numpy()/samples
                self.cat2 += mu_cat2.detach().cpu().numpy()/samples
                self.cat3 += mu_cat3.detach().cpu().numpy()/samples
                self.cat4 += mu_cat4.detach().cpu().numpy()/samples
                self.cat5 += mu_cat5.detach().cpu().numpy()/samples




class Likelihood():
    def __init__(self, lmvae, z, data, samples = 100):
        with torch.no_grad():    
            z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
            mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = lmvae.logo_decoder.forward(z_draw)
            self.bin = dist.Bernoulli(mu_bin).log_prob(data.bin).sum().cpu().numpy()/samples
            self.cat1 = dist.Categorical(mu_cat1).log_prob(data.cat1.squeeze()).sum().cpu().numpy()/samples
            self.cat2 = dist.Categorical(mu_cat2).log_prob(data.cat2.squeeze()).sum().cpu().numpy()/samples
            self.cat3 = dist.Categorical(mu_cat3).log_prob(data.cat3.squeeze()).sum().cpu().numpy()/samples
            self.cat4 = dist.Categorical(mu_cat4).log_prob(data.cat4.squeeze()).sum().cpu().numpy()/samples
            self.cat5 = dist.Categorical(mu_cat5).log_prob(data.cat5.squeeze()).sum().cpu().numpy()/samples

            
            for _ in range(samples-1):
                z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
                mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = lmvae.logo_decoder.forward(z_draw)
                self.bin += dist.Bernoulli(mu_bin).log_prob(data.bin).sum().cpu().numpy()/samples
                self.cat1 += dist.Categorical(mu_cat1).log_prob(data.cat1.squeeze()).sum().cpu().numpy()/samples
                self.cat2 += dist.Categorical(mu_cat2).log_prob(data.cat2.squeeze()).sum().cpu().numpy()/samples
                self.cat3 += dist.Categorical(mu_cat3).log_prob(data.cat3.squeeze()).sum().cpu().numpy()/samples
                self.cat4 += dist.Categorical(mu_cat4).log_prob(data.cat4.squeeze()).sum().cpu().numpy()/samples
                self.cat5 += dist.Categorical(mu_cat5).log_prob(data.cat5.squeeze()).sum().cpu().numpy()/samples
    
    def summarize(self, path=None, index=0, givens=pd.DataFrame([])):
        
        out_dict = {
                    "bin": self.bin, 
                    "cat1": self.cat1, 
                    "cat2": self.cat2, 
                    "cat3": self.cat3, 
                    "cat4": self.cat4, 
                    "cat5": self.cat5, 
                    }
        
        summary = pd.DataFrame(out_dict, [index])
        
        if givens.empty:
            metrics_table = pd.DataFrame(index = [index])
        else:
            metrics_table = givens
            metrics_table.set_index(np.array([index]), inplace=True)
            
        metrics_table = pd.concat([metrics_table, summary], axis=1)
        
        if path == None:
            return metrics_table
    
        else: 
            with open(path, 'a') as f:
                metrics_table.to_csv(f, header=f.tell()==0)


# for new data, compute the z vectors, then use that to make predictions and compute metrics
class LMVAE_Test():
    def __init__(self, lmvae, data, K_vec, network="full"):
        self.z = NewZ(lmvae, data, network)      
        self.values = Predict(lmvae, self.z)
        self.metrics = Metrics(data, self.values, K_vec)
        self.ll = Likelihood(lmvae, self.z, data)