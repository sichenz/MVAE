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

from metrics import *
    
    
# main model:
class LogoMVAE(nn.Module):
    
    def __init__(self, z_dim, encoder_dims, decoder_dims, x_sizes, task_sizes, use_cuda=True, domain_scaling = {"logo": 1., "text": 1., "bp": 1., "indus": 1.}):
        super().__init__()
        # create the encoder and decoder networks        
        self.encoder = TaskEncoder(z_dim, encoder_dims, task_sizes)
        self.text_decoder = BernoulliDecoder(z_dim, decoder_dims["text"], x_sizes["text"])
        self.logo_decoder = LogoDecoder(z_dim, decoder_dims["logo"], x_sizes)
        self.bp_decoder = GaussianDecoder(z_dim, decoder_dims["bp"], x_sizes["bp"])
        self.indus_decoder = BernoulliDecoder(z_dim, decoder_dims["indus"], x_sizes["indus"])

        self.z_dim = z_dim
        self.x_sizes = x_sizes
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
                
        self.K_vec = np.array([x_sizes['cat1'], x_sizes['cat2'], x_sizes['cat3'], x_sizes['cat4'], x_sizes['cat5']])
        
        self.domain_scaling = domain_scaling


    def model(self, data, annealing_factor=1.):
        
        # register all decoders with Pyro
        pyro.module("text_decoder", self.text_decoder)
        pyro.module("logo_decoder", self.logo_decoder)
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
            with poutine.scale(None, self.domain_scaling["text"]):
                pyro.sample("text_obs", dist.Bernoulli(mu_text).to_event(1), obs=data.text)
            
            mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = self.logo_decoder.forward(z)
            with poutine.scale(None, self.domain_scaling["logo"]):
                pyro.sample("bin_obs", dist.Bernoulli(mu_bin).to_event(1), obs=data.bin)
                pyro.sample("cat1_obs", dist.Categorical(mu_cat1), obs=torch.flatten(data.cat1))
                pyro.sample("cat2_obs", dist.Categorical(mu_cat2), obs=torch.flatten(data.cat2))
                pyro.sample("cat3_obs", dist.Categorical(mu_cat3), obs=torch.flatten(data.cat3))
                pyro.sample("cat4_obs", dist.Categorical(mu_cat4), obs=torch.flatten(data.cat4))
                pyro.sample("cat5_obs", dist.Categorical(mu_cat5), obs=torch.flatten(data.cat5))
            
            mu_bp, sigma_bp = self.bp_decoder.forward(z)
            with poutine.scale(None, self.domain_scaling["bp"]):
                pyro.sample("bp_obs", dist.Normal(mu_bp, sigma_bp).to_event(1), obs=data.bp)
            
            mu_indus = self.indus_decoder.forward(z)
            with poutine.scale(None, self.domain_scaling["indus"]):
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
        
	
	
class BernoulliDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_size):
        super().__init__()
        # set up the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim + z_dim, x_size)
        # set up the non-linearities
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # set up dropout
        self.dropout = nn.Dropout()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.dropout(self.activation(self.fc1(z)))
        # return the parameters for the output Bernoulli
        x_probs = self.sigmoid(self.fc21(torch.cat((hidden, z), 1)))
        return x_probs
	
	
class CategoricalDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_size):
        super().__init__()
        # set up the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim + z_dim, x_size)
        # set up the non-linearities
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        # set up dropout
        self.dropout = nn.Dropout()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.dropout(self.activation(self.fc1(z)))
        # return the parameter for the output categorical (i.e., softmax probs)
        x_probs = self.softmax(self.fc21(torch.cat((hidden, z), 1)))
        return x_probs
	
	
class GaussianDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_size):
        super().__init__()
        # set up the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim + z_dim, x_size)
        self.fc22 = nn.Linear(hidden_dim + z_dim, x_size)
        # set up the non-linearities
        self.activation = nn.ReLU()
        self.softplus = nn.Softplus()
        # set up dropout
        self.dropout = nn.Dropout()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.dropout(self.activation(self.fc1(z)))
        # return the parameter for the output reals
        x_loc = self.fc21(torch.cat((hidden, z), 1))
        x_scale = self.softplus(self.fc22(torch.cat((hidden, z), 1)))
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
            
            mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = lmvae.logo_decoder.forward(z_draw)
            self.bin = mu_bin.detach().cpu().numpy()/samples
            self.cat1 = mu_cat1.detach().cpu().numpy()/samples
            self.cat2 = mu_cat2.detach().cpu().numpy()/samples
            self.cat3 = mu_cat3.detach().cpu().numpy()/samples
            self.cat4 = mu_cat4.detach().cpu().numpy()/samples
            self.cat5 = mu_cat5.detach().cpu().numpy()/samples
            
            self.bp = lmvae.bp_decoder(z_draw)[0].detach().cpu().numpy()/samples
            self.indus = lmvae.indus_decoder(z_draw).detach().cpu().numpy()/samples

            for _ in range(samples-1):
                z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
                self.text += lmvae.text_decoder(z_draw).detach().cpu().numpy()/samples
                
                mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = lmvae.logo_decoder.forward(z_draw)
                self.bin += mu_bin.detach().cpu().numpy()/samples
                self.cat1 += mu_cat1.detach().cpu().numpy()/samples
                self.cat2 += mu_cat2.detach().cpu().numpy()/samples
                self.cat3 += mu_cat3.detach().cpu().numpy()/samples
                self.cat4 += mu_cat4.detach().cpu().numpy()/samples
                self.cat5 += mu_cat5.detach().cpu().numpy()/samples
                
                self.bp += lmvae.bp_decoder(z_draw)[0].detach().cpu().numpy()/samples
                self.indus += lmvae.indus_decoder(z_draw).detach().cpu().numpy()/samples



class Likelihood():
    def __init__(self, lmvae, z, data, samples = 100):
        with torch.no_grad():    
            z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
            mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = lmvae.logo_decoder.forward(z_draw)
            
            self.text = dist.Bernoulli(lmvae.text_decoder(z_draw)).log_prob(data.text).sum().cpu().numpy()/samples
            self.bin = dist.Bernoulli(mu_bin).log_prob(data.bin).sum().cpu().numpy()/samples
            self.indus = dist.Bernoulli(lmvae.indus_decoder(z_draw)).log_prob(data.indus).sum().cpu().numpy()/samples
            
            self.cat1 = dist.Categorical(mu_cat1).log_prob(data.cat1.squeeze()).sum().cpu().numpy()/samples
            self.cat2 = dist.Categorical(mu_cat2).log_prob(data.cat2.squeeze()).sum().cpu().numpy()/samples
            self.cat3 = dist.Categorical(mu_cat3).log_prob(data.cat3.squeeze()).sum().cpu().numpy()/samples
            self.cat4 = dist.Categorical(mu_cat4).log_prob(data.cat4.squeeze()).sum().cpu().numpy()/samples
            self.cat5 = dist.Categorical(mu_cat5).log_prob(data.cat5.squeeze()).sum().cpu().numpy()/samples

            bp_loc = lmvae.bp_decoder(z_draw)[0]
            bp_scale = lmvae.bp_decoder(z_draw)[1]

            self.bp = dist.Normal(bp_loc, bp_scale).log_prob(data.bp).sum().cpu().numpy()/samples
            
            for _ in range(samples-1):
                z_draw = dist.Normal(z.z_loc, z.z_scale).sample()
                mu_bin, mu_cat1, mu_cat2, mu_cat3, mu_cat4, mu_cat5 = lmvae.logo_decoder.forward(z_draw)
                
                self.text += dist.Bernoulli(lmvae.text_decoder(z_draw)).log_prob(data.text).sum().cpu().numpy()/samples
                self.bin += dist.Bernoulli(mu_bin).log_prob(data.bin).sum().cpu().numpy()/samples
                self.indus += dist.Bernoulli(lmvae.indus_decoder(z_draw)).log_prob(data.indus).sum().cpu().numpy()/samples
    
                self.cat1 += dist.Categorical(mu_cat1).log_prob(data.cat1.squeeze()).sum().cpu().numpy()/samples
                self.cat2 += dist.Categorical(mu_cat2).log_prob(data.cat2.squeeze()).sum().cpu().numpy()/samples
                self.cat3 += dist.Categorical(mu_cat3).log_prob(data.cat3.squeeze()).sum().cpu().numpy()/samples
                self.cat4 += dist.Categorical(mu_cat4).log_prob(data.cat4.squeeze()).sum().cpu().numpy()/samples
                self.cat5 += dist.Categorical(mu_cat5).log_prob(data.cat5.squeeze()).sum().cpu().numpy()/samples

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