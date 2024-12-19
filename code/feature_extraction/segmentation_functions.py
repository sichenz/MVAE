import os
import warnings
import numpy as np
import pandas as pd
import math
from PIL import Image
from skimage import measure, io, color
from skimage.transform import resize, rescale
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import convex_hull_image
from scipy.stats import mode
from sklearn.cluster import DBSCAN, KMeans
from skimage import feature

def alpha_to_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

## Standardizes the color formatting:
def standardize(image):
    if len(image.shape)==2:
        image = color.gray2rgb(image)

    if len(image.shape)>3:
        image = image[:,:,:,0]

    if image.shape[2]==4:
        image = Image.fromarray(image)
        image = alpha_to_color(image)
        image = np.array(image)
        
    return image


## Removes the extra whitespace along the borders of the image:
def remove_white_bg(img):
    
    img_gray = color.rgb2gray(img)
    
    h, w = tuple(img_gray.shape)

    keep_row = np.ones(h, dtype='Bool')
    for row in range(h):
        keep_row[row] = (np.sum(np.sum(img_gray[row,:]<0.98,axis=0))>0)
        if keep_row[row]:
            break

    for row in reversed(range(h)):
        keep_row[row] = (np.sum(np.sum(img_gray[row,:]<0.98,axis=0))>0)
        if keep_row[row]:
            break

    keep_col = np.ones(w, dtype='Bool')
    for col in range(w):
        keep_col[col] = (np.sum(np.sum(img_gray[:,col]<0.98,axis=0))>0)
        if keep_col[col]:
            break
            
    for col in reversed(range(w)):
        keep_col[col] = (np.sum(np.sum(img_gray[:,col]<0.98,axis=0))>0)
        if keep_col[col]:
            break
                    

    img_nobg = img[keep_row,:]
    img_nobg = img_nobg[:,keep_col]
    
    return img_nobg


## Removes the extra background around a feature:
## (Why is this different than above???)
def remove_segmentation_bg(img):
      
    h, w = tuple(img.shape)

    keep_row = np.zeros(h, dtype='Bool')
    for row in range(h):
        keep_row[row] = (np.sum(img[row,:])>0)

    keep_col = np.ones(w, dtype='Bool')
    for col in range(w):
        keep_col[col] = (np.sum(img[:,col])>0)

    img_nobg = img[keep_row,:]
    img_nobg = img_nobg[:,keep_col]
    
    return img_nobg


## Adds some padding (extra white space) around an RGB feature:
## (Kind of the opposite of the above)
def add_padding(img, bgval, pad_size=20):
    side_pad = np.full((img.shape[0], pad_size, 3), bgval, dtype='uint8')
    tb_pad = np.full((pad_size, img.shape[1] + 2*pad_size, 3), bgval, dtype='uint8')
    padded_img = np.vstack((tb_pad, np.hstack((side_pad, img, side_pad)), tb_pad))
    return padded_img


## Same thing as above, but assuming a quantized image (integer pixels):
def add_quantized_padding(img, bgval=0, pad_size=20):
    side_pad = np.full((img.shape[0], pad_size), bgval, dtype='uint8')
    tb_pad = np.full((pad_size, img.shape[1] + 2*pad_size), bgval, dtype='uint8')
    padded_img = np.vstack((tb_pad, np.hstack((side_pad, img, side_pad)), tb_pad))
    return padded_img


## Detect what the background color is:
def detect_quantized_bg(img):
    from scipy.stats import mode
    corners = np.vstack((np.vstack(img[:10,:10]),np.vstack(img[:10,-10:]),np.vstack(img[-10:,:10]),np.vstack(img[-10:,-10:])))
    rowsums = np.sum(corners, axis=1)
    return mode(corners.flatten())[0][0]


## Rescale an image to have a given number of pixels (approximately):
def rescale_img(img, tot_pix=10000):
    h, w = tuple(img.shape[0:2])
    scale_factor = (h*w) / tot_pix
    return resize(img, output_shape=(np.ceil(h / np.sqrt(scale_factor)).astype('uint8'), np.ceil(w / np.sqrt(scale_factor)).astype('uint8')), mode='constant')


## Rescale a feature to a standard small size:
def make_small_feature(feature, end_size = 25, max_size = 100.):

    h, w = tuple(feature.shape[0:2])

    if w==h:
        resized = resize(feature, output_shape=(max_size, max_size), mode='constant')

    if w>h: 
        scale_factor = w/max_size
        resized = resize(feature, output_shape=(np.ceil(h/scale_factor), max_size), mode='constant')

        padding_amount = max_size - np.ceil(h/scale_factor)
        top_pad = np.floor(padding_amount/2)
        bottom_pad = np.ceil(padding_amount/2)

        resized = np.concatenate([np.zeros((int(top_pad),int(max_size))), 
                                    resized, 
                                    np.zeros((int(bottom_pad),int(max_size)))], axis=0)

    if h>w: 
        scale_factor = h/max_size
        resized = resize(feature, output_shape=(max_size, np.ceil(w/scale_factor)), mode='constant')

        padding_amount = max_size - np.ceil(w/scale_factor)
        left_pad = np.floor(padding_amount/2)
        right_pad = np.ceil(padding_amount/2)

        resized = np.concatenate([np.zeros((int(max_size),int(left_pad))), 
                                    resized, 
                                    np.zeros((int(max_size),int(right_pad)))], axis=1)

    resized = resize(resized, output_shape=(end_size, end_size), mode='constant')
    return resized
    
def perimetric_complexity(charimg, sigma=1):
    padded = add_quantized_padding(charimg, pad_size=1)
    area = np.sum(padded)
    perim = np.sum(feature.canny(padded, sigma=sigma))
    return perim**2/(4.0*math.pi*area)
    
    
def hsv_segmentation(image, name="NA", subsample=10000):

    # WARNING: THIS SHOULD ONLY BE USED WITH SCIKIT-LEARN 0.15 OR 0.19; 
    # OTHER VERSIONS HAVE LESS EFFICIENT IMPLEMENTATIONS OF DBSCAN THAT 
    # CAN CRASH YOUR COMPUTER DUE TO MEMORY CONSUMPTION!!

    from sklearn.cluster import DBSCAN, KMeans

    unsupervised = False
    ncolors = 0

    h, w, c = tuple(image.shape)
    rgbstacked = np.reshape(image, (h*w, c))

    hsvimage = color.rgb2hsv(image)
    hsvstacked = np.reshape(hsvimage, (h*w, c))
    
    labimage = color.rgb2lab(image)
    labstacked = np.reshape(labimage, (h*w, c))
    
    ## Look for non-greyscale pixels:
    goodpix = hsvstacked[np.bitwise_and(hsvstacked[:,1]>0.1, hsvstacked[:,2]>0.1),:]
    ngood = goodpix.shape[0]

    ## If there are non-grey pixels, figure out how many colors there are using DBSCAN:
    if ngood > 25:
        bw = False
        
        ## HSV is the cylindrical coordinate transformation of RGB, and Hue is defined as the 
        ## position around the circle of the coordinate system; therefore, to use DBSCAN clustering,
        ## we should look at the circular coordinate system 
        ## (This avoids unwanted things, like the color red existing at both h=0 and h=1)
        
        polar_hues = np.transpose(np.vstack((np.cos(2*math.pi*goodpix[:,0]),np.sin(2*math.pi*goodpix[:,0]))))
        
        if ngood > subsample:
            pix_sample = np.random.choice(polar_hues.shape[0], subsample, replace=False)
        else:
            pix_sample = np.arange(polar_hues.shape[0])
            
        round_cluster = DBSCAN(algorithm='ball_tree', eps=0.05, min_samples=25)
        round_cluster.fit(polar_hues[pix_sample,:])

        ncolors = np.sum(np.unique(round_cluster.labels_) != -1)

        ## Even if we find there is only one color, there could still be two colors that are
        ## very close together. In this step, we check if there is a large standard deviation 
        ## along any of the polar dimensions. If there is, let's assume there are actually two 
        ## colors. Then we will use a totally unsupervised kmeans clustering with two colors 
        ## plus whatever black/white centers we add below. 
        
        if ncolors==1:
            if np.any(np.std(polar_hues, axis=0) > 0.1):
                unsupervised = True
                
        if unsupervised == False:
            rgbgood = rgbstacked[np.bitwise_and(hsvstacked[:,1]>0.1, hsvstacked[:,2]>0.1),:]
            centers = np.vstack([np.median(rgbgood[pix_sample][round_cluster.labels_ == c,:], axis=0) for c in range(ncolors)])
            
    else:
        ncolors = 0
        bw = True


    # In this section, we try to figure out how many shades of gray there are. Usually, it's just white/black,
    # but we can use properties of the color distribution to guess whether there might also be a grey hidden there.
    # The problem is that grey is also the transition color between black/white, so we may find residual
    # traces of grey even if there is no actual grey.

    if unsupervised == False:
        bg = np.array([255.,255.,255.])
        n_nonbg = np.sum(np.any(rgbstacked != bg, axis=1))

        badpix = hsvstacked[np.bitwise_or(hsvstacked[:,1]<0.1, hsvstacked[:,2]<0.1),:]
        val = badpix[:,2]
        n_nonbg = np.sum(np.any(rgbstacked < 245, axis=1))
    
        if bw:
            centers = np.array([[255,255,255],
                                [0,0,0]])
        else:
            centers = np.vstack((centers, [255, 255, 255]))
            if np.sum(val < 0.2)/n_nonbg > 0.05:
                centers = np.vstack((centers, [0, 0, 0]))

        # If there are mid-range values, there is probably a grey in there; add a center:
        if np.sum(np.bitwise_and(val < 0.8, val > 0.2))/n_nonbg > 0.05:
            centers = np.vstack((centers, [125, 125, 125]))

    # Suppress warnings because kmeans doesn't like that we are using our own initial values:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if unsupervised:
            km = KMeans(n_clusters=4)
            quantized = km.fit_predict(rgbstacked)
            centers = km.cluster_centers_

            coldist = np.bincount(quantized)/len(quantized)

            while np.any(coldist < 0.01):
                centers = centers[coldist > 0.01,:]
                km = KMeans(n_clusters=centers.shape[0], init=centers)
                quantized = km.fit_predict(rgbstacked)
                centers = km.cluster_centers_

                coldist = np.bincount(quantized)/len(quantized)            

        else:
            km = KMeans(n_clusters=centers.shape[0], init=centers)
            quantized = km.fit_predict(rgbstacked)
            centers = km.cluster_centers_

            coldist = np.bincount(quantized)/len(quantized)

            while np.any(coldist < 0.01):
                centers = centers[coldist > 0.01,:]
                km = KMeans(n_clusters=centers.shape[0], init=centers)
                quantized = km.fit_predict(rgbstacked)
                centers = km.cluster_centers_

                coldist = np.bincount(quantized)/len(quantized)

    df = pd.DataFrame({"name": name,
                       "r": km.cluster_centers_[:,0],
                       "g": km.cluster_centers_[:,1],
                       "b": km.cluster_centers_[:,2],
                       "area": np.bincount(quantized),
                       "frac": np.bincount(quantized)/len(quantized)})

    return (np.reshape(quantized, (h,w)), km.cluster_centers_, df[["name","area","frac","r","g","b"]])
    

def extract_features(quantized, centers, end_size=25, region_cutoff_perc=0.005):
    
    h, w = quantized.shape
    
    color_segments = np.unique(quantized)

    if np.any(np.sum(centers, axis=1)>(235.0*3)):
        qbg = np.argmax(np.sum(centers, axis=1))
    else:
        qbg = np.max(quantized)+1

    big_region = 1
    areas = []
    fracs = []
        
    quantized = add_quantized_padding(quantized, bgval=qbg)
    regionalized = np.zeros_like(quantized)
    
    for r in range(len(color_segments)):        

        sub_labels = label((quantized == color_segments[r]))

        index = 1
    
        for region in regionprops(sub_labels):
        
            new_labels = np.zeros(sub_labels.shape)
            
            if (color_segments[r] == qbg) & (np.sum((sub_labels == index)[0:5,0:5]) > 5):
                index += 1
                continue
        
            if region.area > max(region_cutoff_perc*h*w,20):
            
                if (big_region == 1):
                    segmentation = np.expand_dims((sub_labels == index), 2)
                    segment_colors = centers[r,:]
                    
                else:
                    segmentation = np.append(segmentation, np.expand_dims((sub_labels == index), 2), axis=2)
                    segment_colors = np.vstack((segment_colors, centers[r,:]))
                
                areas = np.append(areas, region.area)
                fracs = np.append(fracs, region.area/(h*w))
                
                regionalized[sub_labels == index] = big_region
                
                big_region += 1
                
            index += 1

    small_features = np.zeros((end_size, end_size, segmentation.shape[2]))
    for r in range(segmentation.shape[2]):
        small_features[:,:,r] = make_small_feature(remove_segmentation_bg(segmentation[:,:,r]), end_size = end_size)

    return small_features, segment_colors, areas, fracs, regionalized
    
    
def detect_chars(small_feats, fontdf, name='NA', remaining_letters=None, pc_cutoff=2.0, name_low_pc_score=0.75, name_high_pc_score=0.65, 
low_pc_score=0.85, high_pc_score=0.75, i_score=0.99, o_score=0.9, amp_score=0.65, top_tolerance=0.025, pc_diff_limit=2.0):
    
    #--------------------------------------------------------------------------------------------------------
    # If name is not 'NA', then we will do template matching using the name, allowing for
    # lower thresholds for letters that are in the name (up to the number of letters actually 
    # contained in the name), and raising the threshold for letters not in the name. This also
    # raises the threshold for the letters o, i, l, and 1, as these are often confused. This
    # can be effectively disabled by setting simple_char_score=low_pc_score (or high_pc_score).
    #
    # NOTE: Using this function with name != 'NA' requires that the fontdf have a column called 
    # 'letter' containing the actual letter assignment of each reference character.
    #--------------------------------------------------------------------------------------------------------
    
    fontmat = fontdf.iloc[:,0:625].values.astype('float')
    nfeats = small_feats.shape[2]
    
    tm = np.array([[np.corrcoef(small_feats[:,:,j].flatten(),
                                fontmat[i,:].reshape(25,25).flatten())[0,1]
                    for i in range(fontmat.shape[0])]
                   for j in range(nfeats)])
        
    scores = np.amax(tm, axis=1)
    assignments = np.argmax(tm, axis=1)
    pc_actual = np.array([perimetric_complexity(small_feats[:,:,s]) for s in range(nfeats)])
    assigned_pc = [perimetric_complexity(fontmat[assignments[s],:].reshape((25,25))) 
                   for s in range(small_feats.shape[2])]
    pc_diff = np.abs(pc_actual - assigned_pc)
        
    if name == 'NA':
        is_char = np.zeros(nfeats).astype("bool")
        for s in range(nfeats):
            if pc_actual[s] < pc_cutoff:
                if (scores[s] > low_pc_score) and (pc_diff[s] < pc_diff_limit):
                    is_char[s] = True
            else:
                if scores[s] > high_pc_score:
                    is_char[s] = True
    
    else:
        if remaining_letters is None:
            name_letters = list(name)
        else:
            name_letters = remaining_letters

        sorted_score_indices = np.argsort(-scores)
        
        # In this section, we assign the letter, based on whether or not we find that character in the name:
        # if there are many letters within top_tolerance of the max score, then we look through all of those letters.
        # If we find only one of those letters in the name of the firm, we assume that that letter is the correct 
        # one. Otherwise, we take the max.
        
        max_letter = fontdf['letter'].as_matrix()[assignments]
        top_indices = [np.where(tm[s,:] > np.max(tm[s,:])-top_tolerance)[0] for s in range(nfeats)]
        top_scores = [tm[s,:][top_indices[s]] for s in range(nfeats)]
        top_letters = [fontdf['letter'].as_matrix()[top_indices[s]][np.argsort(-top_scores[s])] for s in range(nfeats)]
        sorted_top_indices = [top_indices[s][np.argsort(-top_scores[s])] for s in range(nfeats)]
        in_name = np.array([np.any(np.isin(top_letters[s], name_letters)) for s in range(nfeats)])
        
        is_char = np.zeros(nfeats).astype("bool")
        for s in sorted_score_indices:
            if in_name[s]==True:
                if np.any(np.isin(top_letters[s], name_letters)):
                    if pc_actual[s] < pc_cutoff:
                        if (scores[s] > name_low_pc_score) and (pc_diff[s] < pc_diff_limit):
                            is_char[s] = True
                            assigned_letter = np.unique(top_letters[s][np.isin(top_letters[s], name_letters)][0])
                            assignments[s] = sorted_top_indices[s][np.isin(top_letters[s], name_letters)][0]
                            name_letters.remove(assigned_letter) 
                    else:
                        if scores[s] > name_high_pc_score:
                            is_char[s] = True
                            assigned_letter = np.unique(top_letters[s][np.isin(top_letters[s], name_letters)][0])
                            assignments[s] = sorted_top_indices[s][np.isin(top_letters[s], name_letters)][0]
                            name_letters.remove(assigned_letter)
                               
                elif np.any(np.isin(top_letters[s], ['i'])):
                    if (scores[s] > i_score) and (pc_diff[s] < pc_diff_limit):
                        is_char[s] = True
                
                elif np.any(np.isin(top_letters[s], ['o'])):
                    if (scores[s] > o_score) and (pc_diff[s] < pc_diff_limit):
                        is_char[s] = True
                        
                else:
                    if pc_actual[s] < pc_cutoff:
                        if (scores[s] > low_pc_score) and (pc_diff[s] < pc_diff_limit):
                            is_char[s] = True
                    else:
                        if scores[s] > high_pc_score:
                            is_char[s] = True
                            
            elif np.any(np.isin(top_letters[s], ['&'])):
                if (scores[s] > amp_score):
                    is_char[s] = True
                                                                       
            elif np.any(np.isin(top_letters[s], ['i'])) :
                if (scores[s] > i_score) and (pc_diff[s] < pc_diff_limit):
                    is_char[s] = True
            
            elif np.any(np.isin(top_letters[s], ['o'])) :
                if (scores[s] > o_score) and (pc_diff[s] < pc_diff_limit):
                    is_char[s] = True
                    
            else:
                if pc_actual[s] < pc_cutoff:
                    if (scores[s] > low_pc_score) and (pc_diff[s] < pc_diff_limit):
                        is_char[s] = True
                else:
                    if scores[s] > high_pc_score:
                        is_char[s] = True
                        
        remaining_letters = name_letters

    char_match = np.ones(scores.size)*9999
    char_match[is_char] = assignments[is_char]

    return is_char, char_match, remaining_letters


def gradient_info(binmark):
    
    gy, gx = [np.ascontiguousarray(g, dtype=np.double) for g in np.gradient(binmark.astype("float"))]
    
    has_grad = np.bitwise_or(gx!=0.0,gy!=0.0)
    
    grad = (np.arctan2(gy[has_grad], gx[has_grad]) * 180 / np.pi)

    # Convert everything to the upper half-circle:
    grad[grad < 0] += 180.

    # Convert everything in quadrant 2 to quadrant 4:
    grad[grad > 90] += -180

    grad_bins = (np.histogram(grad, bins = [-90., -15., 15., 89., 90.])/np.sum(has_grad))[0]
    grad_mean = np.mean(grad)
    
    return grad_mean, grad_bins
    
def compute_symmetry(binmark):
    nobg_binmark = remove_segmentation_bg(binmark)

    mh, mw = nobg_binmark.shape

    if mw/2 == mw//2:
        lhalf = nobg_binmark[:,:(mw//2)]
        rhalf = nobg_binmark[:,(mw//2):]
    else:
        lhalf = nobg_binmark[:,:(mw//2+1)]
        rhalf = nobg_binmark[:,(mw//2):]
        
    h_sym = np.corrcoef(lhalf.ravel(),np.flip(rhalf,1).ravel())[0,1]
    
    if mh/2 == mh//2:
        top = nobg_binmark[:(mh//2),:]
        bot = nobg_binmark[(mh//2):,:]
    else:
        top = nobg_binmark[:(mh//2+1),:]
        bot = nobg_binmark[(mh//2):,:]

    v_sym = np.corrcoef(top.ravel(),np.flip(bot,0).ravel())[0,1]

    return (h_sym, v_sym)
