import time
import os
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from PIL import Image
import sklearn.cluster as cluster
from skimage import io

try:
    os.chdir("/media/ryan/hdd/Dropbox/1_proj/logos/")
except:
    os.chdir("/Users/ryan/Dropbox/1_proj/logos/")

features = pd.read_csv("data/marks_erosion-char_higher_fixed_labeled.csv")
nclust = np.max(features['cluster'])

for j in range(nclust+1):
    features_mat = features.loc[features['cluster']==j,'0':'624'].as_matrix()
    fracs = features.loc[features['cluster']==j,'frac'].as_matrix()
    sample_examples = np.random.choice(np.arange(len(fracs)), size=15, replace=False)
    joined_examples = np.hstack([np.hstack([features_mat[i,:].reshape(25,25), 1.0*np.zeros(shape=(25,10))]) for i in sample_examples])
    
    plt.figure(figsize=(15,2))
    plt.imshow(joined_examples, cmap="gray")
    plt.show()
    
    scipy.misc.imsave('data/clustering/marks-cluster/'+str(j)+'.jpg', joined_examples)

    plt.figure()
    sns.distplot(fracs)
    plt.show()
    
    short_desc = input('short cluster description:  ')
    long_desc = input('long cluster description:  ')
    
    if j == 0:
        cluster_labels = pd.DataFrame({'short_desc':short_desc,'long_desc':long_desc}, index=[j])
    else:
        cluster_labels = pd.concat([cluster_labels, pd.DataFrame({'short_desc':short_desc,'long_desc':long_desc}, index=[j])])


cluster_labels.to_csv("data/clustering/marks-cluster_descs.csv")




