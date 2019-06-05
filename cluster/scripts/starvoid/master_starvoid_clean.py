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

with open('experiment.json', 'r') as f:
    exp_params = json.load(f)
    
    
def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean / std


if(exp_params['scheme'] == 'finetune_denoised'):
    
    # Config from json file
    with open(exp_params['model_name']+str('_denoise_model')+ '/config.json', 'r') as f:
        denoise_conf = json.load(f)
    with open(exp_params['model_name']+str('_n2v_init_model')+ '/config.json', 'r') as f:
        n2v_init_conf = json.load(f)
    with open(exp_params['model_name']+str('_seg_model')+ '/config.json', 'r') as f:
        seg_conf = json.load(f)
        
    denoise_train_files = np.load(exp_params['train_path'])
    denoise_X_train = denoise_train_files['X_train']
    denoise_Y_train = denoise_train_files['Y_train']
    denoise_X_val = denoise_train_files['X_val'].astype(np.float32)
    denoise_Y_val = denoise_train_files['Y_val'].astype(np.float32)
    denoise_test_files = np.load(exp_params['test_path'])
    denoise_X_test = denoise_test_files['X_test']
    denoise_Y_test = denoise_test_files['Y_test']
    X_train_N2V = np.concatenate(denoise_X_train, denoise_X_test)
    Y_train_N2V = np.concatenate(denoise_Y_train, denoise_Y_test)
    X_val_N2V = denoise_X_val
    Y_val_N2V = denoise_Y_val 
    mean, std = np.mean(X_train_N2V), np.std(X_train_N2V)
    denoise_obj = TrainN2V(denoise_conf, exp_params) 
    model = denoise_obj.prepare_data_and_denoise(X_train_N2V, Y_train_N2V, X_val_N2V, Y_val_N2V, '_denoise_model')
    
    denoise_pred_train = PredictN2V(exp_params, model) #Load above model
    X_train_d, X_val_d, X_test_d = denoise_pred_train.predict(denoise_X_train, denoise_X_val, denoiseX_test, '_denoise_model', mean, std)
    
    n2v_init_obj = TrainN2V(n2v_init_conf, exp_params) 
    n2v_init_X_train = X_train_d
    n2v_init_Y_train = denoise_Y_train
    n2v_init_X_val = X_val_d.astype(np.float32)
    n2v_init_Y_val = denoise_Y_val
    X_train_N2V = np.concatenate(n2v_init_X_train, X_test_d)
    Y_train_N2V = np.concatenate(n2v_init_Y_train, denoise_Y_test)
    mean, std = np.mean(X_train_N2V), np.std(X_train_N2V)
    model = n2v_init_obj.prepare_data_and_denoise(n2v_init_X_train, n2v_init_Y_train, n2v_init_X_val, n2v_init_Y_val, '_n2v_init_model')
     
    
    seg_obj = TrainSeg(seg_conf, exp_params, load_weights = True)
    seg_X_train = n2v_init_X_train 
    seg_Y_train = n2v_init_Y_train
    seg_X_val = n2v_init_X_val
    seg_Y_val = n2v_init_Y_val
    seg_obj.prepare_data_and_segment(seg_X_train, seg_Y_train, seg_X_val, seg_Y_val)
    
    import compute_seg_threshold
    
     ###TODO predict 
    X_test = X_test_d
    seg_pred = PredictSeg(exp_params)
    seg_pred.predict(X_test, seg_X_train, '_seg_model')
    


if(exp_params['scheme'] == 'finetune'):
    with open(exp_params['model_name']+str('_n2v_init_model')+ '/config.json', 'r') as f:
        n2v_init_conf = json.load(f)
    with open(exp_params['model_name']+str('_seg_model')+ '/config.json', 'r') as f:
        seg_conf = json.load(f)
        
    
    n2v_init_train_files = np.load(exp_params['train_path'])
    n2v_init_X_train = n2v_init_train_files ['X_train']
    n2v_init_Y_train = n2v_init_train_files ['Y_train']
    n2v_init_X_val = n2v_init_train_files ['X_val'].astype(np.float32)
    n2v_init_Y_val = n2v_init_train_files ['Y_val'].astype(np.float32)
    n2v_init_test_files = np.load(exp_params['test_path'])
    n2v_init_X_test = n2v_init_test_files['X_test']
    n2v_init_Y_test = n2v_init_test_files['Y_test']
    X_train_N2V = np.concatenate(n2v_init_X_train, n2v_init_X_test)
    Y_train_N2V = np.concatenate(n2v_init_Y_train, n2v_init_Y_test)
    X_val_N2V = n2v_init_X_val
    Y_val_N2V = n2v_init_Y_val 
    mean, std = np.mean(X_train_N2V), np.std(X_train_N2V)
    n2v_init_obj = TrainN2V(n2v_init_conf, exp_params)
    model = n2v_init_obj.prepare_data_and_denoise(n2v_init_X_train, n2v_init_Y_train, n2v_init_X_val, n2v_init_Y_val, '_n2v_init_model')
    
    seg_obj = TrainSeg(seg_conf, exp_params, load_weights = True)
    seg_train_files = np.load(exp_params['train_path']) 
    seg_X_train = seg_train_files['X_train']
    seg_Y_train = seg_train_files['Y_train']
    seg_X_val = seg_train_files['X_val'].astype(np.float32)
    seg_Y_val = seg_train_files['Y_val'].astype(np.float32)
    seg_obj.prepare_data_and_segment(seg_X_train, seg_Y_train, seg_X_val, seg_Y_val)
    
    import compute_seg_threshold
    
    files = np.load(exp_params["test_path"]) 
    X_test = files['X_test']
    seg_pred = PredictSeg(exp_params)
    seg_pred.predict(X_test, seg_X_train, '_seg_model')
     
    

if(exp_params['scheme'] == 'sequential'):
    with open(exp_params['model_name']+str('_denoise_model')+ '/config.json', 'r') as f:
        denoise_conf = json.load(f)
    with open(exp_params['model_name']+str('_seg_model')+ '/config.json', 'r') as f:
        seg_conf = json.load(f)
    
    denoise_train_files = np.load(exp_params['train_path'])
    denoise_X_train = denoise_train_files['X_train']
    denoise_Y_train = denoise_train_files['Y_train']
    denoise_X_val = denoise_train_files['X_val'].astype(np.float32)
    denoise_Y_val = denoise_train_files['Y_val'].astype(np.float32)
    denoise_test_files = np.load(exp_params['test_path'])
    denoise_X_test = denoise_test_files['X_test']
    denoise_Y_test = denoise_test_files['Y_test']
    X_train_N2V = np.concatenate(denoise_X_train, denoise_X_test)
    Y_train_N2V = np.concatenate(denoise_Y_train, denoise_Y_test)
    X_val_N2V = denoise_X_val
    Y_val_N2V = denoise_Y_val 
    mean, std = np.mean(X_train_N2V), np.std(X_train_N2V)
    denoise_obj = TrainN2V(denoise_conf, exp_params) 
    model = denoise_obj.prepare_data_and_denoise(X_train_N2V, Y_train_N2V, X_val_N2V, Y_val_N2V, '_denoise_model')
    
    denoise_X_train = normalize(denoise_X_train, mean, std) 
    denoise_X_val = normalize(denoise_X_val, mean, std) 
    denoise_X_test = normalize(denoise_X_test, mean, std) 
    
    denoise_pred_train = PredictN2V(exp_params, model) #Load above model
    X_train_d, X_val_d, X_test_d = denoise_pred_train.predict(denoise_X_train, denoise_X_val, denoiseX_test, '_denoise_model', mean, std)
    
    seg_obj = TrainSeg(seg_conf, exp_params, load_weights = False)
    seg_X_train =  X_train_d
    seg_Y_train =  denoise_Y_train
    seg_X_val = X_val_d.astype(np.float32)
    seg_Y_val = denoise_Y_val
    seg_obj.prepare_data_and_segment(seg_X_train, seg_Y_train, seg_X_val, seg_Y_val)
    
    import compute_seg_threshold
    
    files = np.load(exp_params["test_path"]) 
    X_test = files['X_test']
    seg_pred = PredictSeg(exp_params, model2)
    seg_pred.predict(X_test, seg_X_train, '_seg_model')
    
    
    
if(exp_params['scheme'] == 'baseline'): 
    # Config from json file
    
    with open(exp_params['model_name']+str('_seg_model')+ '/config.json', 'r') as f:
        seg_conf = json.load(f)
        
    seg_obj = TrainSeg(seg_conf, exp_params, load_weights = False)
    seg_train_files = np.load(exp_params['train_path']) 
    seg_X_train = seg_train_files['X_train']
    seg_Y_train = seg_train_files['Y_train']
    seg_X_val = seg_train_files['X_val'].astype(np.float32)
    seg_Y_val = seg_train_files['Y_val'].astype(np.float32)
    seg_obj.prepare_data_and_segment(seg_X_train, seg_Y_train, seg_X_val, seg_Y_val)
    
    import compute_seg_threshold
    
    files = np.load(exp_params["test_path"])
    X_test = files['X_test']
    seg_pred = PredictSeg(exp_params)
    seg_pred.predict(X_test, seg_X_train, '_seg_model')
