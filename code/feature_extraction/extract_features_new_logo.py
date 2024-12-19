import os
import sys

import warnings
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

from PIL import Image

from skimage import measure, io, color, feature, morphology
from skimage.transform import resize, rescale
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import convex_hull_image, disk
from skimage.filters.rank import entropy

from scipy.stats import mode
import scipy.misc

from sklearn.cluster import DBSCAN, KMeans

from collections import OrderedDict

try:
    os.chdir("/media/ryan/hdd/Dropbox/1_proj/logos/code/extract_features/")
except:
    os.chdir("/Users/ryan/Dropbox/1_proj/logos/code/extract_features/")

from segmentation_functions import *
from logo_features_wrapper_erosion import *

try:
    os.chdir("/media/ryan/hdd/Dropbox/1_proj/logos/")
except:
    os.chdir("/Users/ryan/Dropbox/1_proj/logos/")


fontdf = pd.read_csv("data/labeled_fonts_amp_wide.csv")
fontmat = fontdf.iloc[:,0:625].values.astype('float')

name = sys.argv[1]
filepath = sys.argv[2]

img = io.imread(filepath)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    
    (big_marks, allchars, color_df, mark_features, marks, mark_hulls, brand_font_df, brand_summary_df, 
    brand_hull_df) = logo_features(img, name, fontdf, fontmat,
                                    name_low_pc_score=0.7,
                                    name_high_pc_score=0.65,
                                    low_pc_score=0.85,
                                    high_pc_score=0.8)

color_matrix = color_df.values[:,3:]

lab_matrix = np.apply_along_axis(lambda x: color.rgb2lab([[[x[0],x[1],x[2]]]]).flatten(), 1, color_matrix)

color_df["lab-l"] = lab_matrix[:,0]
color_df["lab-a"] = lab_matrix[:,1]
color_df["lab-b"] = lab_matrix[:,2]

color_df.to_csv("code/extract_features/new_logo_outputs/" + name + "_colors.csv", index = False)
mark_features.to_csv("code/extract_features/new_logo_outputs/" + name + "_mark_features.csv", index = False)
marks.to_csv("code/extract_features/new_logo_outputs/" + name + "_marks.csv", index = False)
mark_hulls.to_csv("code/extract_features/new_logo_outputs/" + name + "_mark_hulls.csv", index = False)
brand_font_df.to_csv("code/extract_features/new_logo_outputs/" + name + "_fonts.csv", index = False)
brand_summary_df.to_csv("code/extract_features/new_logo_outputs/" + name + "_summary_feats.csv", index = False)
brand_hull_df.to_csv("code/extract_features/new_logo_outputs/" + name + "_hull.csv", index = False)








