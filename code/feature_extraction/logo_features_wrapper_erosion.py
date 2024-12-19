# DEFINING ONE SUPER EXTRACTION FUNCTION WITH THE TUNING PARAMETERS EXPLICITLY DEFINED

import os
import warnings
import numpy as np
import pandas as pd
import math

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



def logo_features(img, name, fontdf, fontmat,
bgval=255,
subsample=10000,
erosion_radius=3,
pc_cutoff=2.5, # 3 before seemed okay
name_low_pc_score=0.65,
name_high_pc_score=0.55, # prev val 0.6 was pretty good
low_pc_score=0.85,
high_pc_score=0.8,
i_score=0.99,
o_score=0.85,
amp_score=0.6,
top_tolerance=0.05, 
region_cutoff_perc=0.001,
mark_eps_frac=0.0001, # mark_eps = 15
mark_frac_cutoff=0.005, # this was 0.01
min_dist_corners=5,
entropy_disk_size=20,
grey_thresh=0.9,
qmr_cutoff=0.9,
pc_diff_limit=2.0):


    ### Pre-process the logo:  ----------------------------------------------------------------------

    img = standardize(img)
    img = remove_white_bg(img)
    img = add_padding(img, bgval=bgval)



    ### Segment the image and isolate characters  ----------------------------------------------------------------------

    # Segment the image by color regions:

    quantized, centers, color_df = hsv_segmentation(img, name=name, subsample=subsample)
    small_feats, feat_cols, areas, fracs, regionalized = extract_features(quantized, centers, region_cutoff_perc=region_cutoff_perc)


    # Match features to fonts:

    is_char, char_match, remaining_letters = detect_chars(small_feats, fontdf, name=name, 
    pc_cutoff=pc_cutoff, name_low_pc_score=name_low_pc_score, name_high_pc_score=name_high_pc_score, 
    low_pc_score=low_pc_score, high_pc_score=high_pc_score, i_score=i_score, o_score=o_score, 
    amp_score=amp_score, top_tolerance=top_tolerance, pc_diff_limit=pc_diff_limit)
    brand_font_df = fontdf.iloc[char_match[char_match < 9999.].astype('int'),625:]
    brand_font_df.insert(0, 'name', name)
    brand_font_df.insert(1, 'match', char_match[char_match < 9999.].astype('int'))


    # Isolate the part of the image that corresponds to the characters:

    allchars = np.isin(regionalized-1, np.where(is_char))


    # Isolate the remainder, which should be the marks:

    marks = np.copy(regionalized)
    if np.sum(is_char) > 0:
        for j in np.nditer(np.where(is_char)[0]):
            ch = convex_hull_image(((regionalized-1)==j).astype('float'))
            border_regions, region_counts = np.unique(regionalized[np.bitwise_xor(morphology.dilation(ch, selem=disk(3)), ch)], return_counts=True)
            fill = border_regions[np.argmax(region_counts)]
            marks[ch] = fill

    # Apply erosion to try to separate stuck-together characters:
    
    if np.sum(marks) > 0:
        eroded_labels = np.zeros_like(marks)

        cur_max_lab = 0
        for mr in np.unique(marks[marks>0]):
            mr_eroded_labeled = label(morphology.erosion((marks==mr).astype('bool'), selem=morphology.disk(erosion_radius)))
            mr_eroded_labeled[mr_eroded_labeled > 0] = cur_max_lab + mr_eroded_labeled[mr_eroded_labeled > 0]
            cur_max_lab = np.max(mr_eroded_labeled)
            eroded_labels += mr_eroded_labeled
        
        if np.sum(eroded_labels) > 0:    
            eroded_segmentation = np.zeros((25,25,np.max(eroded_labels)))
            for lab in range(1,np.max(eroded_labels)+1):
                eroded_segmentation[:,:,lab-1] = make_small_feature(remove_segmentation_bg(
                    morphology.dilation(eroded_labels==lab, selem=morphology.disk(erosion_radius))))
                
            eroded_chars, eroded_match, remaining_letters = detect_chars(eroded_segmentation, fontdf=fontdf, name=name, remaining_letters=remaining_letters,
                pc_cutoff=pc_cutoff, name_low_pc_score=name_low_pc_score, name_high_pc_score=name_high_pc_score, 
                low_pc_score=low_pc_score, high_pc_score=high_pc_score, i_score=i_score, o_score=o_score, 
                amp_score=amp_score, top_tolerance=top_tolerance, pc_diff_limit=pc_diff_limit)
        
            font_add_df = fontdf.iloc[eroded_match[eroded_match < 9999.].astype('int'),625:]
            font_add_df.insert(0, 'name', name)
            font_add_df.insert(1, 'match', eroded_match[eroded_match < 9999.].astype('int'))
        
            brand_font_df = pd.concat([brand_font_df, font_add_df])
        
            for lab in (np.where(eroded_chars)[0]):
                ch = convex_hull_image(morphology.dilation(eroded_labels==(lab+1), selem=morphology.disk(erosion_radius+2)))
                border_regions, region_counts = np.unique(marks[np.bitwise_xor(morphology.dilation(ch, selem=disk(3)), ch)], return_counts=True)
                fill = border_regions[np.argmax(region_counts)]
                marks[ch] = fill
                allchars[morphology.dilation(eroded_labels==(lab+1), selem=morphology.disk(erosion_radius))] = 1
    

    # Compute the location of the characters (this will be used later on):

    h,w = regionalized.shape


    h_charfirstp = 0
    h_charlastp = w
    v_charfirstp = 0
    v_charlastp = h

    for i in range(w):
        if np.sum(allchars[:,i])>0:
            h_charfirstp = i
            break

    for i in reversed(range(w)):
        if np.sum(allchars[:,i])>0:
            h_charlastp = i
            break

    for j in range(h):
        if np.sum(allchars[j,:])>0:
            v_charfirstp = j
            break

    for j in reversed(range(h)):
        if np.sum(allchars[j,:])>0:
            v_charlastp = j
            break
        
    if np.sum(is_char) > 0:
        h_charpos = (h_charfirstp+h_charlastp)/2./w
        v_charpos = (v_charfirstp+v_charlastp)/2./h
    else: 
        h_charpos = 0.5
        v_charpos = 0.5



    ### Features of the marks ----------------------------------------------------------------------
    
    ## BEGIN MARK LOOP ##  ----------------------------------------------------------------------

    marks_df = pd.DataFrame()
    mark_features = pd.DataFrame()
    mark_hulls = pd.DataFrame()
    
    big_marks = np.zeros_like(marks)
    if np.sum(marks) > 0:
        
        # Re-zero out background regions:
        qbg = detect_quantized_bg(quantized)
        quantized = add_quantized_padding(quantized, bgval=qbg, pad_size=20)
        
        for mr in np.unique(marks[marks>0]):
            qmr = quantized[marks==mr]
            if np.sum(qmr==qbg)/qmr.size > qmr_cutoff:
                marks[marks==mr] = 0
    
    if np.sum(marks) > 0:
        mark_segments = np.copy(marks)    
        
        # Cluster the marks based on position, and eliminate very small outliers:
        mark_pix = np.transpose(np.vstack(np.where(marks > 0)))
        mark_eps = np.max((quantized.size*mark_eps_frac, 15))
        
        mark_clusters = DBSCAN(algorithm='ball_tree', eps=mark_eps, min_samples=1) 
        mark_clusters.fit(mark_pix)
        mark_segments[mark_segments > 0] = mark_clusters.labels_+1
        
        h,w = img.shape[0:2]
        img_size = h*w
        
        mark_fracs = np.array([np.sum(mark_segments == s) / img_size for s in range(np.max(mark_segments)+1)])
        
        index = 0
        for s in range(np.max(mark_segments)+1):
            if mark_fracs[s] > mark_frac_cutoff: 
                big_marks[mark_segments==s] = index
                index += 1
                
        for m in (np.arange(np.max(big_marks))+1):
        
            # Isolate the mark in the original image:
            origmark = np.copy(add_padding(img, bgval=255, pad_size=20))
            origmark[big_marks != m] = 255
        
            # Grayscale representation:
            graymark = color.rgb2gray(origmark)

            # Binarized version of the mark:
            binmark = (big_marks==m)
        
            # Save the mark:
            std_mark = make_small_feature(remove_segmentation_bg(binmark), end_size = 25)
            mark_df = pd.DataFrame(columns=np.arange(25*25).astype('str'), index=[str(m)])
            mark_df.iloc[0,:] = np.reshape(std_mark, (1,(25*25)))
            mark_df.insert(0, 'name', name)
            mark_df.insert(1, 'index', m)
            marks_df = pd.concat([marks_df, mark_df])
        
            # Which of the original regions correspond to this mark?
            whichregions = np.unique(regionalized[binmark])
        
            # Bumber of colors in the mark:
            ncolors = np.unique(feat_cols[whichregions-1], axis=0).shape[0]
        
            # Number of subregions in the mark?
            nregions = whichregions.size
        
            # Mark's fraction of the total image:
            frac = np.sum(big_marks==m)/img_size
        
            # Perimetric complexity of the mark:
            pcm = perimetric_complexity(remove_segmentation_bg(binmark).astype("float"))
        
            # Horizontal, vertical symmetry
            h_sym, v_sym = compute_symmetry(binmark)
        
            # Repetition metrics:
            if nregions > 1:
                rep_frac = np.std([np.sum(regionalized==j)/np.sum(binmark) for j in whichregions])
                rep_pcm = np.std([perimetric_complexity(add_quantized_padding(
                    remove_segmentation_bg(regionalized==j),pad_size=5).astype("float")) for j in whichregions])
            else:
                rep_frac = 0.
                rep_pcm = 0.
            
            # Position of the mark:
            h,w = regionalized.shape
        
            ## horizontal position:
            for j in range(w):
                if np.sum(binmark[:,j])>0:
                    h_firstp = j
                    break

            for j in reversed(range(w)):
                if np.sum(binmark[:,j])>0:
                    h_lastp = j
                    break
                
            h_pos = (h_firstp+h_lastp)/2./w

            ## vertical position:
            for j in range(h):
                if np.sum(binmark[j,:])>0:
                    v_firstp = j
                    break
                
            for j in reversed(range(h)):
                if np.sum(binmark[j,:])>0:
                    v_lastp = j
                    break

            v_pos = (v_firstp+v_lastp)/2./h
        
            # Relative position:
            avg_left = False
            avg_right = False
            avg_top = False
            avg_bot = False

            abs_left = False
            abs_right = False
            abs_top = False
            abs_bot = False

            if (h_charpos - h_pos) > 0.2:
                avg_left = True
            elif (h_pos - h_charpos) > 0.2:
                avg_right = True

            if (v_charpos - v_pos) > 0.2:
                avg_top = True
            elif (v_pos - v_charpos) > 0.2:
                avg_bot = True

            if h_lastp < h_charfirstp:
                abs_left = True
            elif h_firstp > h_charlastp:
                abs_right = True

            if v_lastp < v_charfirstp:
                abs_top = True
            elif v_firstp > v_charlastp:
                abs_bot = True
        
            # Convex hull of the mark:
            ch = convex_hull_image(binmark)
        
            # Save the convex hull:
            ch_df = pd.DataFrame(columns=np.arange(25*25).astype('str'), index=[str(m)])
            ch_df.iloc[0,:] = np.reshape(make_small_feature(remove_segmentation_bg(ch), end_size = 25), (1,25*25))
            ch_df.insert(0, 'name', name)
            ch_df.insert(1, 'index', m)
            mark_hulls = pd.concat([mark_hulls, ch_df])
        
            # Density of the mark within its hull:
            ch_density = np.sum(binmark[ch])/np.sum(ch)
        
            # Entropy of the mark within its hull:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ch_entropy = np.mean(entropy(graymark, disk(5))[ch])

            # Aspect ratio of the mark:
            ar = np.abs(h_firstp-h_lastp)/np.abs(v_firstp-v_lastp)
        
            # Number of corners in the mark:
            ncorners = feature.corner_peaks(feature.corner_harris(binmark)).shape[0]
        
            # Gradient information to capture orientation:
            grad_mean, grad_bins = gradient_info(binmark)
            (down_diag, vert, up_diag, hor) = grad_bins
        
            features_df = pd.DataFrame({'name':name,
                                        'index':m,
                                        'ncolors':ncolors,
                                        'nregions':nregions,
                                        'frac':frac,
                                        'pcm':pcm,
                                        'h_sym':h_sym,
                                        'v_sym':v_sym,
                                        'rep_frac':rep_frac,
                                        'rep_pcm':rep_pcm,
                                        'h_pos':h_pos,
                                        'v_pos':v_pos,
                                        'avg_left':avg_left,
                                        'avg_right':avg_right,
                                        'avg_top':avg_top,
                                        'avg_bot':avg_bot,
                                        'abs_left':abs_left,
                                        'abs_right':abs_right,
                                        'abs_top':abs_top,
                                        'abs_bot':abs_bot,
                                        'ch_density':ch_density,
                                        'ch_entropy':ch_entropy,
                                        'ar':ar,
                                        'ncorners':ncorners,
                                        'down_diag':down_diag,
                                        'vert':vert,
                                        'up_diag':up_diag,
                                        'hor':hor}, 
                                      index=[m], 
                                      columns=['name','index','ncolors','nregions',
                                               'frac','pcm','h_sym','v_sym','rep_frac', 
                                               'rep_pcm','h_pos','v_pos',
                                               'avg_left','avg_right','avg_top','avg_bot',
                                               'abs_left','abs_right','abs_top','abs_bot',
                                               'ch_density','ch_entropy','ar','ncorners',
                                               'down_diag','vert','up_diag','hor'])
            mark_features = pd.concat([mark_features,features_df])

    ## END MARK LOOP ##  ----------------------------------------------------------------------


    ### Global Features  ----------------------------------------------------------------------

    # To begin, we will look at summaries of the metrics computed above, to capture complexity, like how many marks there are, how many letters, how many regions, and how many distinct colors:

    ncolors = color_df.shape[0]

    nmarks = marks_df.shape[0]

    nregions = np.max(regionalized)


    # Compute greyscale summaries:
    grey_img = color.rgb2grey(img)


    # Average local greyscale entropy:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grey_entropy = np.mean(entropy(remove_white_bg(grey_img), disk(entropy_disk_size)))


    # Perimetric complexity:
    binary_img = (grey_img < grey_thresh).astype("float")
    gpc = perimetric_complexity(binary_img)


    # Total number of corners in the img:
    ncorners = feature.corner_peaks(feature.corner_harris(binary_img), min_distance=min_dist_corners).shape[0]


    # HSV summaries:
    hsv_img = color.rgb2hsv(remove_white_bg(img))
    hsv_means = np.mean(hsv_img, axis=(0,1))
    mean_sat = hsv_means[1]
    mean_light = hsv_means[2]

    hsv_sds = np.std(hsv_img, axis=(0,1))
    sd_sat = hsv_sds[1]
    sd_light = hsv_sds[2]


    # Another metric of lightness (and perhaps of complexity) is the percentage of whitespace:

    perc_white = 1.0 - np.mean(remove_segmentation_bg(binary_img))


    # Horizontal, vertical symmetry
    h_sym, v_sym = compute_symmetry(binary_img)


    # The aspect ratio of the logo as a whole:

    h,w = remove_segmentation_bg(regionalized).shape
    ar = w/h


    # Gradient information to capture orientation:
    grad_mean, grad_bins = gradient_info(binary_img)
    (down_diag, vert, up_diag, hor) = grad_bins


    # Another measure of orientation, looking at where the main (alt. max) mark (shortened as `mmark`) falls relative to the text:
    (mmark_avgleft, mmark_avgright, mmark_avgtop, mmark_avgbot, mmark_absleft, mmark_absright, mmark_abstop, 
         mmark_absbot) = np.zeros(8).astype('bool')
    if mark_features.shape[0] > 0:
        (mmark_avgleft, mmark_avgright, mmark_avgtop, mmark_avgbot, mmark_absleft, mmark_absright, mmark_abstop, 
         mmark_absbot) = mark_features.iloc[mark_features['frac'].idxmax()-1,:].loc['avg_left':'abs_bot']



    ## Save summary feats:
    brand_summary_df = pd.DataFrame({'name': name, 
                                     'h': h, 
                                     'w': w,
                                     'ncolors': ncolors,
                                     'nmarks': nmarks,
                                     'nregions': nregions,
                                     'entropy': grey_entropy,
                                     'gpc': gpc,
                                     'ncorners': ncorners,
                                     'mean_sat': mean_sat,
                                     'mean_light': mean_light,
                                     'sd_sat': sd_sat,
                                     'sd_light': sd_light,
                                     'perc_white': perc_white,
                                     'h_sym': h_sym,
                                     'v_sym': v_sym,
                                     'ar': ar,
                                     'down_diag': down_diag,
                                     'vert': vert,
                                     'up_diag': up_diag,
                                     'hor': hor,                  
                                     'mmark_avgleft': mmark_avgleft, 
                                     'mmark_avgright': mmark_avgright, 
                                     'mmark_avgtop': mmark_avgtop, 
                                     'mmark_avgbot': mmark_avgbot, 
                                     'mmark_absleft': mmark_absleft, 
                                     'mmark_absright': mmark_absright, 
                                     'mmark_abstop': mmark_abstop, 
                                     'mmark_absbot': mmark_absbot,                        
                                     'nchars': brand_font_df.shape[0]}, index=[i])
                                     
                                     

    # Finally, we can still compute the hull and all of that:

    hull = make_small_feature(convex_hull_image(binary_img))
    brand_hull_df = pd.DataFrame(columns=np.arange(25*25).astype('str'), index=[str(i)])
    brand_hull_df.iloc[0,:] = np.transpose(np.reshape(hull, 25*25))
    brand_hull_df['id'] = name

    return big_marks, allchars, color_df, mark_features, marks_df, mark_hulls, brand_font_df, brand_summary_df, brand_hull_df

