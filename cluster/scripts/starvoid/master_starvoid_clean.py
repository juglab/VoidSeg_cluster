from csbdeep.models import Config, CARE
import numpy as np
from csbdeep.utils.n2v_utils import manipulate_val_data
from skimage.segmentation import find_boundaries
import json
from os.path import join
import pickle
from trainn2v import TrainN2V
from trainseg import TrainSeg
from predictn2v import PredictN2V
from predictseg import PredictSeg
from sklearn.feature_extraction import image
from Baseline import Baseline
from Sequential import Sequential
from Joint import Joint

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
    
if(exp_params['scheme'] == 'sequential'): 
    sequential = Sequential()
    sequential.compute()   
    
if(exp_params['scheme'] == 'baseline'): 
    baseline = Baseline()
    baseline.compute()
