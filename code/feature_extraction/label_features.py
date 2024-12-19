import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from PIL import Image
import sklearn.cluster as cluster
from skimage import io

try:
    os.chdir("/media/ryan/hdd/Dropbox/1_proj/logos/")
except:
    os.chdir("/Users/ryan/Dropbox/1_proj/logos/")

features = pd.read_csv("data/mark_features_erosion-char_higher_fixed_labeled.csv")
nclust = np.max(features['cluster'])

for j in range(nclust):
    explore_features = features.loc[scaled_fit==j,:]
    features_mat = explore_features.loc[:,'0':'624'].as_matrix()
    
    for i in range(min(explore_features.shape[0],25)):
        plt.figure(figsize=(2,2))
        plt.imshow(features_mat[i,:].reshape(25,25), cmap="gray")
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.15)

    time.sleep(0.3)

    plt.figure()
    sns.distplot(explore_features.frac)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    time.sleep(0.1)

    short_desc = input('short cluster description')
    long_desc = input('long cluster description')
    labfontdf.loc[j,'letter'] = letter
    print(j)
    if j % 100 == 0:
        labfontdf.to_csv("data/labeled_font_add1.csv", index=False)


lower_labels = np.char.lower(labfontdf['letter'].as_matrix().astype('str'))
labfontdf.loc[:,'letter'] = lower_labels
labfontdf.to_csv("data/labeled_font_add1.csv", index=False)


