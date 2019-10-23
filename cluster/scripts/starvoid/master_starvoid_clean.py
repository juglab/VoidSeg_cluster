from csbdeep.models import Config, CARE
import numpy as np
from csbdeep.utils.n2v_utils import manipulate_val_data
from skimage.segmentation import find_boundaries
import json
from os.path import join, dirname
import pickle
from sklearn.feature_extraction import image
from Baseline import Baseline
from Sequential import Sequential

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
    print(exp_params)
    sequential = Sequential()
    sequential.compute()
    ### The denoising part is finished here, now do the segmentation by creating a baseline object. But first change train_data path and test_data path along with Scheme in config
    exp_params['scheme'] = 'baseline'
    exp_params['train_path'] = dirname(exp_params['train_path'])+'/N2V_TrainVal_'+exp_params['exp_name']+'.npz'
    exp_params['test_path'] = dirname(exp_params['test_path']) + '/N2V_Test_' + exp_params['exp_name'] + '.npz'

    print("Changing exp conf scheme to baseline and the training and test data paths")
    print(exp_params, flush = True)

    with open('experiment.json','w') as file:
        json.dump(exp_params, file)
    baseline = Baseline()
    baseline.compute()

    
if(exp_params['scheme'] == 'baseline'): 
    baseline = Baseline()
    baseline.compute()
