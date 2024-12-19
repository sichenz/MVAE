import sys
sys.path.insert(1, '../')

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

from metrics import *
    
    
# main model:
class LogoMVAE(nn.Module):
    
    def __init__(self, z_dim, encoder_dims, decoder_dims, x_sizes, task_sizes, use_cuda=True):
        super().__init__()
        # create the encoder and decoder networks        
        self.encoder = TaskEncoder(z_dim, encoder_dims, task_sizes)
        self.text_decoder = BernoulliDecoder(z_dim, x_sizes["text"])
        self.bin_decoder = BernoulliDecoder(z_dim, x_sizes["bin"])
        self.cat1_decoder = CategoricalDecoder(z_dim, x_sizes["cat1"])
        self.cat2_decoder = CategoricalDecoder(z_dim, x_sizes["cat2"])
        self.cat3_decoder = CategoricalDecoder(z_dim, x_sizes["cat3"])
        self.cat4_decoder = CategoricalDecoder(z_dim, x_sizes["cat4"])
        self.cat5_decoder = CategoricalDecoder(z_dim, x_sizes["cat5"])
        self.bp_decoder = GaussianDecoder(z_dim, x_sizes["bp"])
        self.indus_decoder = BernoulliDecoder(z_dim, x_sizes["indus"])

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
        pyro.module("text_decoder", self.text_decoder)
        pyro.module("bin_decoder", self.bin_decoder)
        pyro.module("cat1_decoder", self.cat1_decoder)
        pyro.module("cat2_decoder", self.cat2_decoder)
        pyro.module("cat3_decoder", self.cat3_decoder)
        pyro.module("cat4_decoder", self.cat4_decoder)
        pyro.module("cat5_decoder", self.cat5_decoder)
        pyro.module("bp_decoder", self.bp_decoder)
        pyro.module("indus_decoder", self.indus_decoder)
        
        
        # observation-specific priors/likelihoods:
        with pyro.plate("data", data.text.shape[0], dim=-1):

            z_loc = torch.zeros(torch.Size((data.text.shape[0], self.z_dim)), device='cuda')
            z_scale = torch.ones(torch.Size((data.text.shape[0], self.z_dim)), device='cuda')
            with poutine.scale(None, annealing_factor):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            # define domain-specific likelihoods:
            mu_text = self.text_decoder.forward(z)
            pyro.sample("text_obs", dist.Bernoulli(mu_text).to_event(1), obs=data.text)
            
            mu_bin = self.bin_decoder.forward(z)
            pyro.sample("bin_obs", dist.Bernoulli(mu_bin).to_event(1), obs=data.bin)
            
            mu_cat1 = self.cat1_decoder.forward(z)
            pyro.sample("cat1_obs", dist.Categorical(mu_cat1), obs=torch.flatten(data.cat1))
            
            mu_cat2 = self.cat2_decoder.forward(z)
            pyro.sample("cat2_obs", dist.Categorical(mu_cat2), obs=torch.flatten(data.cat2))
            
            mu_cat3 = self.cat3_decoder.forward(z)
            pyro.sample("cat3_obs", dist.Categorical(mu_cat3), obs=torch.flatten(data.cat3))
            
            mu_cat4 = self.cat4_decoder.forward(z)
            pyro.sample("cat4_obs", dist.Categorical(mu_cat4), obs=torch.flatten(data.cat4))
            
            mu_cat5 = self.cat5_decoder.forward(z)
            pyro.sample("cat5_obs", dist.Categorical(mu_cat5), obs=torch.flatten(data.cat5))
            
            mu_bp, sigma_bp = self.bp_decoder.forward(z)
            pyro.sample("bp_obs", dist.Normal(mu_bp, sigma_bp).to_event(1), obs=data.bp)
            
            mu_indus = self.indus_decoder.forward(z)
            pyro.sample("indus_obs", dist.Bernoulli(mu_indus).to_event(1), obs=data.indus)
            

    def guide(self, data, annealing_factor=1.):
        
        # register the encoder with pyro:
        pyro.module("encoder", self.encoder)        

        N = data.text.shape[0]
            
        task_indices = torch.split(torch.tensor(np.arange(N)), int(np.ceil(N / 4.)), dim=0)
        
        x_full = torch.cat((data.text, data.bin, data.cat1_hot, 
                            data.cat2_hot, data.cat3_hot, data.cat4_hot, 
                            data.cat5_hot, data.bp, data.indus), 1)[task_indices[0]]
        
        x_res = torch.cat((data.bin, data.cat1_hot, 
                            data.cat2_hot, data.cat3_hot, data.cat4_hot, 
                            data.cat5_hot, data.indus), 1)[task_indices[1]]
        
        x_design = torch.cat((data.text, data.bp, data.indus), 1)[task_indices[2]]
        
        x_mgr = torch.cat((data.text, data.bin, data.cat1_hot, 
                           data.cat2_hot, data.cat3_hot, data.cat4_hot, 
                           data.cat5_hot, data.indus), 1)[task_indices[3]]
        
        task_xs = [x_full, x_res, x_design, x_mgr]

        with pyro.plate("data", N, dim=-1):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(task_xs)
            # sample the latent code z
            with poutine.scale(None, annealing_factor):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            
    def est_z(self, test_data, network="full"):
        self.z = NewZ(self, test_data, network)
        
    def pred_x(self, test_data, network="full"):
        self.x_pred = Predict(self, NewZ(self, test_data, network))
    
    def predict(self, test_data, network="full"):
        self.pred = LMVAE_Test(self, test_data, self.K_vec, network)

        
class TaskEncoder(nn.Module):
    def __init__(self, z_dim, encoder_dims, task_sizes):
        super().__init__()
        # set up the three linear transformations used
        self.task_full = Encoder(z_dim, encoder_dims["full"], task_sizes["full"])
        self.task_res = Encoder(z_dim, encoder_dims["res"], task_sizes["res"])
        self.task_design = Encoder(z_dim, encoder_dims["design"], task_sizes["design"])
        self.task_mgr = Encoder(z_dim, encoder_dims["mgr"], task_sizes["mgr"])
        
    def forward(self, task_xs):
        z_loc_full, z_scale_full = self.task_full(task_xs[0])
        z_loc_res, z_scale_res = self.task_res(task_xs[1])
        z_loc_design, z_scale_design = self.task_design(task_xs[2])
        z_loc_mgr, z_scale_mgr = self.task_mgr(task_xs[3])
        
        z_loc = torch.cat((z_loc_full, z_loc_res, z_loc_design, z_loc_mgr), 0)
        z_scale = torch.cat((z_scale_full, z_scale_res, z_scale_design, z_scale_mgr), 0)
        
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
	
	
class BernoulliDecoder(nn.Module):
    def __init__(self, z_dim, x_size):
        super().__init__()
        self.fc = nn.Linear(z_dim, x_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x_probs = self.sigmoid(self.fc(z))
        return x_probs
	
	
class CategoricalDecoder(nn.Module):
    def __init__(self, z_dim, x_size):
        super().__init__()
        # set up the two linear transformations used
        self.fc = nn.Linear(z_dim, x_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        x_probs = self.softmax(self.fc(z))
        return x_probs
	
	
class GaussianDecoder(nn.Module):
    def __init__(self, z_dim, x_size):
        super().__init__()
        # set up the two linear transformations used
        self.fc1 = nn.Linear(z_dim, x_size)
        self.fc2 = nn.Linear(z_dim, x_size)
        # set up the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        x_loc = self.fc1(z)
        x_scale = self.softplus(self.fc2(z))
        return x_loc, x_scale
	
	
# estimates the z vectors for new data:
class NewZ():
    def __init__(self, lmvae, data, network="full"):
        N = data.text.shape[0]
        with torch.no_grad():
            if network == "full":
                x_full = torch.cat((data.text, data.bin, data.cat1_hot, 
                                    data.cat2_hot, data.cat3_hot, data.cat4_hot, 
                                    data.cat5_hot, data.bp, data.indus), 1)
                z_loc, z_scale = lmvae.encoder.task_full.forward(x_full)
            
            elif network == "res":
                x_res = torch.cat((data.bin, data.cat1_hot, 
                                   data.cat2_hot, data.cat3_hot, data.cat4_hot, 
                                   data.cat5_hot, data.indus), 1)
                z_loc, z_scale = lmvae.encoder.task_res.forward(x_res)
                
            elif network == "mgr":
                x_mgr = torch.cat((data.text, data.bin, data.cat1_hot, 
                                   data.cat2_hot, data.cat3_hot, data.cat4_hot, 
                                   data.cat5_hot, data.indus), 1)
                z_loc, z_scale = lmvae.encoder.task_mgr.forward(x_mgr)
                
            elif network == "des":
                x_design = torch.cat((data.text, data.bp, data.indus), 1)
                z_loc, z_scale = lmvae.encoder.task_design.forward(x_design)
                
            else:
                raise ValueError("network must = full, res, mgr, or des")
                
            self.z_loc = z_loc
            self.z_scale = z_scale
	    
	    
# given z, predicts the average feature values (or feature probabilities), and saves them:
class Predict():
    def __init__(self, lmvae, z, samples = 100):
        with torch.no_grad():    
            z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
            self.text = lmvae.text_decoder(z_draw).detach().cpu().numpy()/samples
            self.bin = lmvae.bin_decoder(z_draw).detach().cpu().numpy()/samples
            self.cat1 = lmvae.cat1_decoder(z_draw).detach().cpu().numpy()/samples
            self.cat2 = lmvae.cat2_decoder(z_draw).detach().cpu().numpy()/samples
            self.cat3 = lmvae.cat3_decoder(z_draw).detach().cpu().numpy()/samples
            self.cat4 = lmvae.cat4_decoder(z_draw).detach().cpu().numpy()/samples
            self.cat5 = lmvae.cat5_decoder(z_draw).detach().cpu().numpy()/samples
            self.bp = lmvae.bp_decoder(z_draw)[0].detach().cpu().numpy()/samples
            self.indus = lmvae.indus_decoder(z_draw).detach().cpu().numpy()/samples

            for _ in range(samples-1):
                z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
                self.text += lmvae.text_decoder(z_draw).detach().cpu().numpy()/samples
                self.bin += lmvae.bin_decoder(z_draw).detach().cpu().numpy()/samples
                self.cat1 += lmvae.cat1_decoder(z_draw).detach().cpu().numpy()/samples
                self.cat2 += lmvae.cat2_decoder(z_draw).detach().cpu().numpy()/samples
                self.cat3 += lmvae.cat3_decoder(z_draw).detach().cpu().numpy()/samples
                self.cat4 += lmvae.cat4_decoder(z_draw).detach().cpu().numpy()/samples
                self.cat5 += lmvae.cat5_decoder(z_draw).detach().cpu().numpy()/samples
                self.bp += lmvae.bp_decoder(z_draw)[0].detach().cpu().numpy()/samples
                self.indus += lmvae.indus_decoder(z_draw).detach().cpu().numpy()/samples



class Likelihood():
    def __init__(self, lmvae, z, data, samples = 100):
        with torch.no_grad():    
            z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
            self.text = dist.Bernoulli(lmvae.text_decoder(z_draw)).log_prob(data.text).sum().cpu().numpy()/samples
            self.bin = dist.Bernoulli(lmvae.bin_decoder(z_draw)).log_prob(data.bin).sum().cpu().numpy()/samples
            self.indus = dist.Bernoulli(lmvae.indus_decoder(z_draw)).log_prob(data.indus).sum().cpu().numpy()/samples
            
            self.cat1 = dist.Categorical(lmvae.cat1_decoder(z_draw)).log_prob(data.cat1.squeeze()).sum().cpu().numpy()/samples
            self.cat2 = dist.Categorical(lmvae.cat2_decoder(z_draw)).log_prob(data.cat2.squeeze()).sum().cpu().numpy()/samples
            self.cat3 = dist.Categorical(lmvae.cat3_decoder(z_draw)).log_prob(data.cat3.squeeze()).sum().cpu().numpy()/samples
            self.cat4 = dist.Categorical(lmvae.cat4_decoder(z_draw)).log_prob(data.cat4.squeeze()).sum().cpu().numpy()/samples
            self.cat5 = dist.Categorical(lmvae.cat5_decoder(z_draw)).log_prob(data.cat5.squeeze()).sum().cpu().numpy()/samples

            bp_loc = lmvae.bp_decoder(z_draw)[0]
            bp_scale = lmvae.bp_decoder(z_draw)[1]

            self.bp = dist.Normal(bp_loc, bp_scale).log_prob(data.bp).sum().cpu().numpy()/samples
            
            for _ in range(samples-1):
                z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
                self.text += dist.Bernoulli(lmvae.text_decoder(z_draw)).log_prob(data.text).sum().cpu().numpy()/samples
                self.bin += dist.Bernoulli(lmvae.bin_decoder(z_draw)).log_prob(data.bin).sum().cpu().numpy()/samples
                self.indus += dist.Bernoulli(lmvae.indus_decoder(z_draw)).log_prob(data.indus).sum().cpu().numpy()/samples
    
                self.cat1 += dist.Categorical(lmvae.cat1_decoder(z_draw)).log_prob(data.cat1.squeeze()).sum().cpu().numpy()/samples
                self.cat2 += dist.Categorical(lmvae.cat2_decoder(z_draw)).log_prob(data.cat2.squeeze()).sum().cpu().numpy()/samples
                self.cat3 += dist.Categorical(lmvae.cat3_decoder(z_draw)).log_prob(data.cat3.squeeze()).sum().cpu().numpy()/samples
                self.cat4 += dist.Categorical(lmvae.cat4_decoder(z_draw)).log_prob(data.cat4.squeeze()).sum().cpu().numpy()/samples
                self.cat5 += dist.Categorical(lmvae.cat5_decoder(z_draw)).log_prob(data.cat5.squeeze()).sum().cpu().numpy()/samples

                bp_loc = lmvae.bp_decoder(z_draw)[0]
                bp_scale = lmvae.bp_decoder(z_draw)[1]

                self.bp += dist.Normal(bp_loc, bp_scale).log_prob(data.bp).sum().cpu().numpy()/samples
    
    def summarize(self, path=None, index=0, givens=pd.DataFrame([])):
        
        out_dict = {"text": self.text,
                    "bin": self.bin, 
                    "cat1": self.cat1, 
                    "cat2": self.cat2, 
                    "cat3": self.cat3, 
                    "cat4": self.cat4, 
                    "cat5": self.cat5, 
                    "bp": self.bp, 
                    "indus": self.indus}
        
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