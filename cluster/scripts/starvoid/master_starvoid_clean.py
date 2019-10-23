from csbdeep.models import Config, CARE
import numpy as np
from csbdeep.utils.n2v_utils import manipulate_val_data
from skimage.segmentation import find_boundaries
import json
from os.path import join, dirname
import pickle
from sklearn.feature_extraction import image
from Segmentation import Segmentation
from Denoising import Denoising

with open('experiment.json', 'r') as f:
    exp_params = json.load(f)
    print("Here!")
    
def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean / std
    
def create_patches(images, masks, size):
    patchesimages = image.extract_patches_2d(images, (size, size), 10, 0)  
    patchesmasks = image.extract_patches_2d(masks, (size, size), 10, 0)
    
    return patchesimages, patchesmasks
    
if(exp_params['scheme'] == 'denoising'):
    denoising = Denoising()
    denoising.compute()

if(exp_params['scheme'] == 'segmentation'):
    segmentation = Segmentation()
    segmentation.compute()
