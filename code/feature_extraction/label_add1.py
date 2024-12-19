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

fontdf = pd.read_csv("data/font_add1.csv")
# labfontdf = fontdf.copy(deep=True)
labfontdf = pd.read_csv("data/labeled_font_add1.csv")

## LEFT OFF AT 2000
for j in range(2001,labfontdf.shape[0]):
    plt.figure()
    plt.imshow(fontdf.iloc[j,0:625].values.astype('float').reshape(25,25), cmap="gray")
    plt.show()
    letter = input('what letter is this?')
    labfontdf.loc[j,'letter'] = letter
    print(j)
    if j % 100 == 0:
        labfontdf.to_csv("data/labeled_font_add1.csv", index=False)


lower_labels = np.char.lower(labfontdf['letter'].as_matrix().astype('str'))
labfontdf.loc[:,'letter'] = lower_labels
labfontdf.to_csv("data/labeled_font_add1.csv", index=False)


