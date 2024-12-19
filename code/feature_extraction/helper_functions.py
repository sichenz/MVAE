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
    return resize(img, (np.ceil(h / np.sqrt(scale_factor)).astype('uint8'), np.ceil(w / np.sqrt(scale_factor)).astype('uint8')))


## Rescale a feature to a standard small size:
def make_small_feature(feature, end_size = 25, max_size = 100.):

    h, w = tuple(feature.shape[0:2])

    if w==h:
        resized = resize(feature, (max_size, max_size))

    if w>h: 
        scale_factor = w/max_size
        resized = resize(feature, (np.ceil(h/scale_factor), max_size)) 

        padding_amount = max_size - np.ceil(h/scale_factor)
        top_pad = np.floor(padding_amount/2)
        bottom_pad = np.ceil(padding_amount/2)

        resized = np.concatenate([np.zeros((int(top_pad),int(max_size))), 
                                    resized, 
                                    np.zeros((int(bottom_pad),int(max_size)))], axis=0)

    if h>w: 
        scale_factor = h/max_size
        resized = resize(feature, (max_size, np.ceil(w/scale_factor))) 

        padding_amount = max_size - np.ceil(w/scale_factor)
        left_pad = np.floor(padding_amount/2)
        right_pad = np.ceil(padding_amount/2)

        resized = np.concatenate([np.zeros((int(max_size),int(left_pad))), 
                                    resized, 
                                    np.zeros((int(max_size),int(right_pad)))], axis=1)

    resized = resize(resized, (end_size, end_size))
    return resized
    
def perimetric_complexity(charimg, sigma=1):
    padded = add_quantized_padding(charimg, pad_size=1)
    area = np.sum(padded)
    perim = np.sum(feature.canny(padded, sigma=sigma))
    return perim**2/(4.0*math.pi*area)
