


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
from pyro import poutine

pyro.set_rng_seed(42)

import random
random.seed(42)

import pandas as pd

from skimage import io
from sklearn import metrics

from model import Predict, NewZ


# We have to redo some stuff to get things like "words":

textdf = pd.read_csv("../../../data/web_dtfm20_binary.csv", index_col=0)
tx_text = textdf.values
seltext = tx_text.sum(0) > 0.05
tx_text = textdf.values[:,seltext]

gt20words = tx_text.sum(1) > 20
tx_text = tx_text[gt20words,:]

words = textdf.columns[seltext]

stem2tok = pd.read_csv("/home/ryandew/Dropbox/1_proj/logos/data/stem2tok.csv", index_col = 0)
new_words = []
for stem in words:
    if any(stem2tok.index == stem):
        new_words.append(stem2tok.loc[stem][0])
    else:
        new_words.append(stem)
        
words = new_words

# Copied clean labels:

c1_labels_clean = np.array(['Black','Dark Blue','Light Blue','Blue','Brown','Dark Green','Light Green','Dark Grey', 
                            'Light Grey', 'Orange', 'Red', 'Dark Red', 'Yellow'])

c2_labels_clean = np.array(['Circle', 'Rectangle/Oval', 'Wide Rectangle/Oval', 'Narrow Rectangle/Oval', 
                            'Square', 'Triangle'])

c3_labels_clean = np.array(['Unique-font letter(s)', 'Large, hollow, geometric shape', 'Dense and circular', 
                            'Dense, simple, geometric shape', 'Detailed and circular', 'Hollow and circular',
                            'Detailed with a horizontal orientation', 'Long and horizontal', 'No mark', 'Simple design', 
                            'Square shape', 'Thin vertical rectangles', 'Vertical and narrow design', 'Very detailed',
                            'Thin design', 'Horizontal and wispy'])

c4_labels_clean = np.array(['No characters in the logo', 'Sans-serif font (e.g., Arial, Helvetica)', 'Serif font (e.g., Times New Roman)'])

c5_labels_clean = np.array(["One color", "Two colors", "Three colors", "More than three colors"])

bin_labels_clean = np.array(["Mark position: underneath the brand name", "Mark position: to the left of the brand name", 
                             "Mark position: to the right of the brand name", "Mark position: above the brand name",
                             "Mark position: toward the bottom of the logo", "Mark position: toward the left of the logo",
                             "Mark position: toward the right of the logo", "Mark position: toward the top of the logo",
                             "Font width: Condensed", "Font width: Wide", "Font width: Ordinary",
                             "Font weight: Bold", "Font weight: Light", "Font weight: Ordinary (not bold, not light)", 
                             "Font slant: Italics", "Font slant: None (non-italics)", 
                             "Has the color: Dark Blue", "Has the color: Light Grey", "Has the color: Red",
                             "Has the color: Light Blue", "Has the color: Black", "Has the color: Blue", 
                             "Has the color: Orange", "Has the color: Dark Red", "Has the color: Light Green",
                             "Has the color: Dark Grey", "Has the color: Dark Green",
                             "Has the color: Brown", "Has the color: Yellow", "Mark: Has a mark", 
                             "Font class: Grotesque",
                             "Font class: Geometric", "Font class: Square Geometric", "Font class: Humanist",
                             "Font class: Transitional", "Font class: Old Style", "Font class: Clarendon",
                             "Font class: Slab", "Font class: Didone", 
                             "Accent color: Black", "Accent color: Dark Blue", 
                             "Accent color: Light Blue", "Accent color: Blue", "Accent color: Brown", 
                             "Accent color: Dark Green", "Accent color: Light Green", "Accent color: Dark Grey",
                             "Accent color: Light Grey", "Accent color: Orange", "Accent color: Red", 
                             "Accent color: Dark Red", "Accent color: Yellow",
                             "Color saturation: Grayscale", "Color saturation: High", 
                             "Color saturation: High variation", "Color saturation: Low", 
                             "Color saturation: Low variation",
                             "Edges: Few downward diagonal edges", 
                             "Design: Low entropy (i.e., not very detailed)",
                             "Design: Low GPC (i.e., not very complex)", "Symmetry: low horizontal symmetry",
                             "Edges: Few horizontal edges", "Color lightness: Low (i.e., dark colors)", 
                             "Design: Few corners", "Design: Few distinct regions",
                             "Design: Low percentage of whitespace", "Color lightness: Low variation",
                             "Edges: Few upward diagonal edges", "Symmetry: Low vertical symmetry",
                             "Edges: Many downward diagonal edges",
                             "Design: High entropy (i.e., very detailed)",
                             "Design: High GPC (i.e., very complex)",
                             "Symmetry: High horizontal symmetry",
                             "Edges: Many horizontal edges",
                             "Color lightness: High (i.e., bright colors)",
                             "Design: Many corners", "Design: Many distinct regions",
                             "Design: High percentage of whitespace", "Color lightness: High variation", 
                             "Edges: Many upward diagonal edges", "Symmetry: High vertical symmetry",
                             "Design: No characters (just a mark)", "Design: Many characters", "Design: Many marks"
                            ])

indus_labels_clean = np.array(['B2C', 'B2B', 'Administrative Services', 'Biotechnology',
       'Clothing and Apparel', 'Commerce and Shopping',
       'Community and Lifestyle', 'Consumer Electronics',
       'Consumer Goods', 'Data and Analytics', 'Education', 'Energy',
       'Financial Services', 'Food and Beverage',
       'Government and Military', 'Hardware', 'Health Care',
       'Information Technology', 'Internet Services',
       'Lending and Investments', 'Manufacturing',
       'Media and Entertainment', 'Mobile', 'Natural Resources',
       'Payments', 'Platforms', 'Privacy and Security',
       'Professional Services', 'Real Estate', 'Sales and Marketing',
       'Software', 'Sports', 'Sustainability', 'Telecommunications',
       'Transportation', 'Travel and Tourism'])

bp_labels_clean = np.array(['charming', 'cheerful', 'confident', 'contemporary', 'cool',
       'corporate', 'daring', 'down-to-earth', 'exciting',
       'family-oriented', 'feminine', 'friendly', 'glamorous',
       'good-looking', 'hard-working', 'honest', 'imaginative',
       'independent', 'intelligent', 'leader', 'masculine', 'original',
       'outdoorsy', 'real', 'reliable', 'rugged', 'secure', 'sentimental',
       'sincere', 'small-town', 'smooth', 'spirited', 'successful',
       'technical', 'tough', 'trendy', 'unique', 'up-to-date',
       'upper-class', 'western', 'wholesome', 'young'])



class CompanyData():
    pass

def get_company(data, index=None, name=None, cuda=False):
    if (index == None) and (name == None):
        raise Exception("Need either an index or a name")

    if (index != None) and (name != None):
        raise Exception("Can't have both an index and a name")

    company = CompanyData()
    if (index != None):
        company.text = torch.tensor(data.x_text[index], dtype = torch.float)
        company.bin = torch.tensor(data.x_bin[index], dtype = torch.float)
        company.cat1 = torch.tensor(data.x_cat1[index], dtype = torch.float)
        company.cat2 = torch.tensor(data.x_cat2[index], dtype = torch.float)
        company.cat3 = torch.tensor(data.x_cat3[index], dtype = torch.float)
        company.cat4 = torch.tensor(data.x_cat4[index], dtype = torch.float)
        company.cat5 = torch.tensor(data.x_cat5[index], dtype = torch.float)
        company.bp = torch.tensor(data.x_bp[index], dtype = torch.float)
        company.indus = torch.tensor(data.x_indus[index], dtype = torch.float)

    if (name != None):
        company.text = torch.tensor(data.x_text[data.x_names == name], dtype = torch.float)
        company.bin = torch.tensor(data.x_bin[data.x_names == name], dtype = torch.float)
        company.cat1 = torch.tensor(data.x_cat1[data.x_names == name], dtype = torch.float)
        company.cat2 = torch.tensor(data.x_cat2[data.x_names == name], dtype = torch.float)
        company.cat3 = torch.tensor(data.x_cat3[data.x_names == name], dtype = torch.float)
        company.cat4 = torch.tensor(data.x_cat4[data.x_names == name], dtype = torch.float)
        company.cat5 = torch.tensor(data.x_cat5[data.x_names == name], dtype = torch.float)
        company.bp = torch.tensor(data.x_bp[data.x_names == name], dtype = torch.float)
        company.indus = torch.tensor(data.x_indus[data.x_names == name], dtype = torch.float)
        
    if cuda:
        company.text = company.text.cuda()
        company.bin = company.bin.cuda()
        company.cat1 = company.cat1.cuda()
        company.cat2 = company.cat2.cuda()
        company.cat3 = company.cat3.cuda()
        company.cat4 = company.cat4.cuda()
        company.cat5 = company.cat5.cuda()
        company.indus = company.indus.cuda()
        company.bp = company.bp.cuda()

    return company
    

class PredictNoVar():
    def __init__(self, lmvae, Z):
        bp = lmvae.bp_decoder(Z.cuda())
        self.bp = bp[0].cpu().detach()
        self.indus = lmvae.indus_decoder(Z.cuda()).cpu().detach()
        self.text = lmvae.text_decoder(Z.cuda()).cpu().detach()
        
        bin_gpu, cat1_gpu, cat2_gpu, cat3_gpu, cat4_gpu, cat5_gpu = lmvae.logo_decoder(Z.cuda())
        self.bin = bin_gpu.cpu().detach()
        self.cat1 = cat1_gpu.cpu().detach()
        self.cat2 = cat2_gpu.cpu().detach()
        self.cat3 = cat3_gpu.cpu().detach()
        self.cat4 = cat4_gpu.cpu().detach()
        self.cat5 = cat5_gpu.cpu().detach()


class RandomBrand():
    def __init__(self, lmvae, K, N = 100):
        self.K = K
        self.N = N
        self.Z = dist.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.)).sample([N,self.K])
        self.pred = PredictNoVar(lmvae, self.Z)


def raw_profile(pred, i = 0):

    return {'bp': pred.bp[i],
            'bin': pred.bin[i],
            'indus': pred.indus[i],
            'text': pred.text[i],
            'cat1': pred.cat1[i],
            'cat2': pred.cat2[i],
            'cat3': pred.cat3[i],
            'cat4': pred.cat4[i],
            'cat5': pred.cat5[i]}


def relevance(brand_prob, overall_prob, weight = 0.6):
    return weight * np.log(brand_prob) + (1 - weight) * np.log(brand_prob / overall_prob)

def profile(pred, data, i = 0):
    raw = raw_profile(pred, i)

    # Binary logo feats:
    prob = pd.DataFrame(raw["bin"])
    prob.index = bin_labels_clean
    prob.columns = ["Prob"]

    rel = pd.DataFrame(relevance(raw["bin"], data.bin.mean(0).cpu().numpy()))
    rel.index = bin_labels_clean
    rel.columns = ["Rel"]

    bin_profile = pd.concat([rel, prob], axis=1)
    bin_profile = bin_profile.sort_values(by="Rel", ascending=False)

    # Cat 1:
    prob = pd.DataFrame(raw["cat1"])
    prob.index = c1_labels_clean
    prob.columns = ["Prob"]

    c1_probs = pd.Series(data.cat1.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat1.shape[0]

    rel = pd.DataFrame(relevance(raw["cat1"],c1_probs))
    rel.index = c1_labels_clean
    rel.columns = ["Rel"]

    cat1_profile = pd.concat([rel, prob], axis=1)
    cat1_profile = cat1_profile.sort_values(by="Rel", ascending=False)

    # Cat 2:
    prob = pd.DataFrame(raw["cat2"])
    prob.index = c2_labels_clean
    prob.columns = ["Prob"]

    c2_probs = pd.Series(data.cat2.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat2.shape[0]

    rel = pd.DataFrame(relevance(raw["cat2"], c2_probs))
    rel.index = c2_labels_clean
    rel.columns = ["Rel"]

    cat2_profile = pd.concat([rel, prob], axis=1)
    cat2_profile = cat2_profile.sort_values(by="Prob", ascending=False)

    # Cat 3:
    prob = pd.DataFrame(raw["cat3"])
    prob.index = c3_labels_clean
    prob.columns = ["Prob"]

    c3_probs = pd.Series(data.cat3.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat3.shape[0]

    rel = pd.DataFrame(relevance(raw["cat3"], c3_probs))
    rel.index = c3_labels_clean
    rel.columns = ["Rel"]

    cat3_profile = pd.concat([rel, prob], axis=1)
    cat3_profile = cat3_profile.sort_values(by="Prob", ascending=False)

    # Cat 4:
    prob = pd.DataFrame(raw["cat4"])
    prob.index = c4_labels_clean
    prob.columns = ["Prob"]

    c4_probs = pd.Series(data.cat4.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat4.shape[0]

    rel = pd.DataFrame(relevance(raw["cat4"], c4_probs))
    rel.index = c4_labels_clean
    rel.columns = ["Rel"]

    cat4_profile = pd.concat([rel, prob], axis=1)
    cat4_profile = cat4_profile.sort_values(by="Prob", ascending=False)

    # Cat 5:
    prob = pd.DataFrame(raw["cat5"])
    prob.index = c5_labels_clean
    prob.columns = ["Prob"]

    c5_probs = pd.Series(data.cat5.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat5.shape[0]

    rel = pd.DataFrame(relevance(raw["cat5"], c5_probs))
    rel.index = c5_labels_clean
    rel.columns = ["Rel"]

    cat5_profile = pd.concat([rel, prob], axis=1)
    cat5_profile = cat5_profile.sort_values(by="Prob", ascending=False)

    # Indus tags:
    prob = pd.DataFrame(raw["indus"])
    prob.index = indus_labels_clean
    prob.columns = ["Prob"]

    rel = pd.DataFrame(raw["indus"] - data.indus.mean(0).cpu().numpy())
    rel.index = indus_labels_clean
    rel.columns = ["Rel"]

    indus_profile = pd.concat([rel, prob], axis=1)
    indus_profile = indus_profile.sort_values(by="Prob", ascending=False)

    # BP:
    bp_profile = pd.DataFrame(raw["bp"])
    bp_profile.index = bp_labels_clean
    bp_profile.columns = ["Rel Values"]
    bp_profile = bp_profile.sort_values(by="Rel Values", ascending=False)

    # Text:
    prob = pd.DataFrame(raw["text"])
    prob.index = words
    prob.columns = ["Prob"]

    rel = pd.DataFrame(relevance(raw["text"], data.text.mean(0).cpu().numpy()))
    rel.index = words
    rel.columns = ["Rel"]

    text_profile = pd.concat([rel, prob], axis=1)
    text_profile = text_profile.sort_values(by="Rel", ascending=False)


    return {"bp": bp_profile, "text": text_profile, "indus": indus_profile,
            "bin": bin_profile, "cat1": cat1_profile, "cat2": cat2_profile,
            "cat3": cat3_profile, "cat4": cat4_profile, "cat5": cat5_profile}
            

def profile_OLD(pred, data, i = 0):
    raw = raw_profile(pred, i)

    # Binary logo feats:
    act_probs = pd.DataFrame(raw["bin"])
    act_probs.index = bin_labels_clean
    act_probs.columns = ["Prob"]

    rel_probs = pd.DataFrame(raw["bin"] - data.bin.mean(0).cpu().numpy())
    rel_probs.index = bin_labels_clean
    rel_probs.columns = ["Rel Prob"]

    bin_profile = pd.concat([rel_probs, act_probs], axis=1)
    bin_profile = bin_profile.sort_values(by="Prob", ascending=False)

    # Cat 1:
    act_probs = pd.DataFrame(raw["cat1"])
    act_probs.index = c1_labels_clean
    act_probs.columns = ["Prob"]

    c1_probs = pd.Series(data.cat1.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat1.shape[0]

    rel_probs = pd.DataFrame(raw["cat1"] - c1_probs)
    rel_probs.index = c1_labels_clean
    rel_probs.columns = ["Rel Prob"]

    cat1_profile = pd.concat([rel_probs, act_probs], axis=1)
    cat1_profile = cat1_profile.sort_values(by="Prob", ascending=False)

    # Cat 2:
    act_probs = pd.DataFrame(raw["cat2"])
    act_probs.index = c2_labels_clean
    act_probs.columns = ["Prob"]

    c2_probs = pd.Series(data.cat2.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat2.shape[0]

    rel_probs = pd.DataFrame(raw["cat2"] - c2_probs)
    rel_probs.index = c2_labels_clean
    rel_probs.columns = ["Rel Prob"]

    cat2_profile = pd.concat([rel_probs, act_probs], axis=1)
    cat2_profile = cat2_profile.sort_values(by="Prob", ascending=False)

    # Cat 3:
    act_probs = pd.DataFrame(raw["cat3"])
    act_probs.index = c3_labels_clean
    act_probs.columns = ["Prob"]

    c3_probs = pd.Series(data.cat3.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat3.shape[0]

    rel_probs = pd.DataFrame(raw["cat3"] - c3_probs)
    rel_probs.index = c3_labels_clean
    rel_probs.columns = ["Rel Prob"]

    cat3_profile = pd.concat([rel_probs, act_probs], axis=1)
    cat3_profile = cat3_profile.sort_values(by="Prob", ascending=False)

    # Cat 4:
    act_probs = pd.DataFrame(raw["cat4"])
    act_probs.index = c4_labels_clean
    act_probs.columns = ["Prob"]

    c4_probs = pd.Series(data.cat4.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat4.shape[0]

    rel_probs = pd.DataFrame(raw["cat4"] - c4_probs)
    rel_probs.index = c4_labels_clean
    rel_probs.columns = ["Rel Prob"]

    cat4_profile = pd.concat([rel_probs, act_probs], axis=1)
    cat4_profile = cat4_profile.sort_values(by="Prob", ascending=False)

    # Cat 5:
    act_probs = pd.DataFrame(raw["cat5"])
    act_probs.index = c5_labels_clean
    act_probs.columns = ["Prob"]

    c5_probs = pd.Series(data.cat5.cpu().numpy().flatten()).value_counts().sort_index().to_numpy() / data.cat5.shape[0]

    rel_probs = pd.DataFrame(raw["cat5"] - c5_probs)
    rel_probs.index = c5_labels_clean
    rel_probs.columns = ["Rel Prob"]

    cat5_profile = pd.concat([rel_probs, act_probs], axis=1)
    cat5_profile = cat5_profile.sort_values(by="Prob", ascending=False)

    # Indus tags:
    act_probs = pd.DataFrame(raw["indus"])
    act_probs.index = indus_labels_clean
    act_probs.columns = ["Prob"]

    rel_probs = pd.DataFrame(raw["indus"] - data.indus.mean(0).cpu().numpy())
    rel_probs.index = indus_labels_clean
    rel_probs.columns = ["Rel Prob"]

    indus_profile = pd.concat([rel_probs, act_probs], axis=1)
    indus_profile = indus_profile.sort_values(by="Prob", ascending=False)

    # BP:
    bp_profile = pd.DataFrame(raw["bp"])
    bp_profile.index = bp_labels_clean
    bp_profile.columns = ["Rel Values"]
    bp_profile = bp_profile.sort_values(by="Rel Values", ascending=False)

    # Text:
    act_probs = pd.DataFrame(raw["text"])
    act_probs.index = words
    act_probs.columns = ["Prob"]

    rel_probs = pd.DataFrame(raw["text"] - data.text.mean(0).cpu().numpy())
    rel_probs.index = words
    rel_probs.columns = ["Rel Prob"]

    text_profile = pd.concat([rel_probs, act_probs], axis=1)
    text_profile = text_profile.sort_values(by="Rel Prob", ascending=False)


    return {"bp": bp_profile, "text": text_profile, "indus": indus_profile,
            "bin": bin_profile, "cat1": cat1_profile, "cat2": cat2_profile,
            "cat3": cat3_profile, "cat4": cat4_profile, "cat5": cat5_profile}
            
class NewCompany(CompanyData):
    def __init__(self, name, noptions, read_dir = "../../code/extract_features/new_logo_outputs/"):
    
        self.bp = pd.read_csv(read_dir + name + "_rel_bp.csv", header=None, index_col=0).values.T

        indus_df = pd.read_csv(read_dir + name + "_indus.csv", header=None, index_col=0)
        self.indus = indus_df.values.T

        new_bin = pd.read_csv(read_dir + name + "_y_bin.csv", index_col=0)
        self.bin = new_bin.values

        new_mult = pd.read_csv(read_dir + name + "_y_mult.csv", index_col=0)

        self.cat1 = np.expand_dims(new_mult.values[:,0], 1)
        self.cat2 = np.expand_dims(new_mult.values[:,1], 1)
        self.cat3 = np.expand_dims(new_mult.values[:,2], 1)
        self.cat4 = np.expand_dims(new_mult.values[:,3], 1)
        self.cat5 = np.expand_dims(new_mult.values[:,4], 1)

        new_text_df = pd.read_csv(read_dir + name + "_newrow_binary.csv", index_col=0)
        self.text = new_text_df.values
        
        self.noptions = noptions
        
    def make_torch(self, cuda = False):
        if cuda:
            
            self.cat1_hot = torch.nn.functional.one_hot(torch.tensor(self.cat1, dtype = torch.int).long(), self.noptions[0]).float().squeeze(0).cuda()
            self.cat2_hot = torch.nn.functional.one_hot(torch.tensor(self.cat2, dtype = torch.int).long(), self.noptions[1]).float().squeeze(0).cuda()
            self.cat3_hot = torch.nn.functional.one_hot(torch.tensor(self.cat3, dtype = torch.int).long(), self.noptions[2]).float().squeeze(0).cuda()
            self.cat4_hot = torch.nn.functional.one_hot(torch.tensor(self.cat4, dtype = torch.int).long(), self.noptions[3]).float().squeeze(0).cuda()
            self.cat5_hot = torch.nn.functional.one_hot(torch.tensor(self.cat5, dtype = torch.int).long(), self.noptions[4]).float().squeeze(0).cuda()
            
            self.text = torch.tensor(self.text, dtype = torch.float).cuda()
            self.bin = torch.tensor(self.bin, dtype = torch.float).cuda()
            self.cat1 = torch.tensor(self.cat1, dtype = torch.float).cuda()
            self.cat2 = torch.tensor(self.cat2, dtype = torch.float).cuda()
            self.cat3 = torch.tensor(self.cat3, dtype = torch.float).cuda()
            self.cat4 = torch.tensor(self.cat4, dtype = torch.float).cuda()
            self.cat5 = torch.tensor(self.cat5, dtype = torch.float).cuda()
            self.bp = torch.tensor(self.bp, dtype = torch.float).cuda()
            self.indus = torch.tensor(self.indus, dtype = torch.float).cuda()
            
        else:
            
            self.cat1_hot = torch.nn.functional.one_hot(torch.tensor(self.cat1.squeeze(0), dtype = torch.int).long(), self.noptions[0]).float().squeeze(0)
            self.cat2_hot = torch.nn.functional.one_hot(torch.tensor(self.cat2.squeeze(0), dtype = torch.int).long(), self.noptions[1]).float().squeeze(0)
            self.cat3_hot = torch.nn.functional.one_hot(torch.tensor(self.cat3.squeeze(0), dtype = torch.int).long(), self.noptions[2]).float().squeeze(0)
            self.cat4_hot = torch.nn.functional.one_hot(torch.tensor(self.cat4.squeeze(0), dtype = torch.int).long(), self.noptions[3]).float().squeeze(0)
            self.cat5_hot = torch.nn.functional.one_hot(torch.tensor(self.cat5.squeeze(0), dtype = torch.int).long(), self.noptions[4]).float().squeeze(0)
       
    
            self.text = torch.tensor(self.text, dtype = torch.float)
            self.bin = torch.tensor(self.bin, dtype = torch.float)
            self.cat1 = torch.tensor(self.cat1, dtype = torch.float)
            self.cat2 = torch.tensor(self.cat2, dtype = torch.float)
            self.cat3 = torch.tensor(self.cat3, dtype = torch.float)
            self.cat4 = torch.tensor(self.cat4, dtype = torch.float)
            self.cat5 = torch.tensor(self.cat5, dtype = torch.float)
            self.bp = torch.tensor(self.bp, dtype = torch.float)
            self.indus = torch.tensor(self.indus, dtype = torch.float)
            
            
class MultiviewZ():
    def __init__(self, lmvae, data):
        self.full = NewZ(lmvae, data, network = "full")
        self.mgr = NewZ(lmvae, data, network = "mgr")
        self.des = NewZ(lmvae, data, network = "des")
        self.logo = NewZ(lmvae, data, network = "res")