
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

fontdf = pd.read_csv("data/new_sources/font_features.csv")
labfontdf = fontdf.copy(deep=True)
labfontdf.insert(labfontdf.shape[1], 'letter', 'NA')

for j in range(5): #range(fontmat.shape[0]):
    plt.close("all")
    cur_letter = 'NA'
    plt.figure()
    plt.imshow(fontdf.iloc[j,0:625].values.astype('float').reshape(25,25), cmap="gray")
    plt.show()
    plt.pause(0.001)
    letter = input('what letter is this?')
    labfontdf.loc[j,'letter'] = letter

labfontdf.to_csv("data/labeled_fonts.csv", index=False)


