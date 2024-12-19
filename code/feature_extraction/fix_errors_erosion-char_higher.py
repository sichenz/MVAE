
## -------------------------------------- ##
## EXTRACT ALL FEATURES AND FORM DATASETS ##
## -------------------------------------- ##


# Preliminary: load required functions

import os
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


# Load data about the firms/logos:

logodata = pd.read_csv("data/new_sources/final_cleaned.csv")
logonames = logodata.name


# Load the font dictionary:

fontdf = pd.read_csv("data/labeled_fonts_amp_wide.csv")
fontmat = fontdf.iloc[:,0:625].values.astype('float')





##############################
#          MAIN LOOP         #
##############################

# Load past run if applicable:

errors = pd.read_csv("data/errors_erosion-char_higher.csv", header=None)
combined_colors = pd.read_csv("data/colors_erosion-char_higher.csv")
combined_mark_features = pd.read_csv("data/mark_features_erosion-char_higher.csv")
combined_marks = pd.read_csv("data/marks_erosion-char_higher.csv")
combined_mark_hulls = pd.read_csv("data/mark_hulls_erosion-char_higher.csv")
combined_fonts = pd.read_csv("data/fonts_erosion-char_higher.csv")
combined_summaries = pd.read_csv("data/summaries_erosion-char_higher.csv")
combined_hulls = pd.read_csv("data/hulls_erosion-char_higher.csv")

#error_names = errors.as_matrix().flatten().astype('str') 

#error_names = np.array(list(set(logodata['name'])-set(combined_hulls['id']))).astype('str')

error_names = errors.as_matrix().astype('str')
errors = pd.Series()

for l, name in enumerate(error_names):

    print(str(l) + ". " + name)
    
    try:
        img = io.imread("data/new_sources/logos/" + name + ".png")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            
            (big_marks, allchars, color_df, mark_features, marks, mark_hulls, brand_font_df, brand_summary_df, 
            brand_hull_df) = logo_features(img, name, fontdf, fontmat,
                                            name_low_pc_score=0.7,
                                            name_high_pc_score=0.65,
                                            low_pc_score=0.85,
                                            high_pc_score=0.8)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            scipy.misc.imsave('data/test_extraction/erosion-char_higher/'+name+'.jpg', 1-np.hstack([big_marks, allchars]))
        
        combined_colors = pd.concat([combined_colors, color_df])
        combined_mark_features = pd.concat([combined_mark_features, mark_features])
        combined_marks = pd.concat([combined_marks, marks])
        combined_mark_hulls = pd.concat([combined_mark_hulls, mark_hulls])
        combined_fonts = pd.concat([combined_fonts, brand_font_df])
        combined_summaries = pd.concat([combined_summaries, brand_summary_df])
        combined_hulls = pd.concat([combined_hulls, brand_hull_df])
            
    except:
        errors = errors.append(pd.Series(name))
        continue
    

# # Save everything:
        
errors.to_csv("data/errors_erosion-char_higher_fixed.csv", index = False)
combined_colors.to_csv("data/colors_erosion-char_higher_fixed.csv", index = False)
combined_mark_features.to_csv("data/mark_features_erosion-char_higher_fixed.csv", index = False)
combined_marks.to_csv("data/marks_erosion-char_higher_fixed.csv", index = False)
combined_mark_hulls.to_csv("data/mark_hulls_erosion-char_higher_fixed.csv", index = False)
combined_fonts.to_csv("data/fonts_erosion-char_higher_fixed.csv", index = False)
combined_summaries.to_csv("data/summaries_erosion-char_higher_fixed.csv", index = False)
combined_hulls.to_csv("data/hulls_erosion-char_higher_fixed.csv", index = False)








