from csbdeep.models import Config, CARE
import numpy as np
from csbdeep.utils.n2v_utils import manipulate_val_data
from skimage.segmentation import find_boundaries
import json
from os.path import join
import pickle


def add_boundary_label(lbl, dtype=np.uint16):
    """ lbl is an integer label image (not binarized) """
    b = find_boundaries(lbl,mode='outer')
    res = (lbl>0).astype(dtype)
    res[b] = 2
    return res


def onehot_encoding(lbl, n_classes=3, dtype=np.uint32):
    """ n_classes will be determined by max lbl value if its value is None """
    from keras.utils import to_categorical
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot


def cutHalf(data, repeat):
    data_=data
    print(data_.shape)
    for i in range(repeat):
        newsize=int(data_.shape[1]/2)
        a=data_[:,:newsize,:]
        b=data_[:,newsize:,:]
  #      print(a.shape,b.shape)
        data_=np.concatenate((a,b), axis=0)

        newsize=int(data_.shape[2]/2)
        a=data_[:,:,:newsize]
        b=data_[:,:,newsize:]
        data_=np.concatenate((a,b), axis=0)
        print(data_.shape)
    return data_


def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std


def denormalize(x, mean, std):
    return x*std + mean


def convert_to_oneHot(data):
    data_oneHot = np.zeros((*data.shape, 3), dtype=np.float32)
    for i in range(data.shape[0]):
        data_oneHot[i] = onehot_encoding(add_boundary_label(data[i].astype(np.int32)))
    return data_oneHot


with open('experiment.json', 'r') as f:
    exp_params = json.load(f)


#Config from json file
with open(exp_params['model_name']+'/config.json', 'r') as f:
    conf = json.load(f)


use_denoising = conf['use_denoising']

# Read training images and GT
train_files = np.load(exp_params['train_path'])
X_train = train_files['X_train']
Y_train = train_files['Y_train']
X_val = cutHalf(train_files['X_val'][:640, :640], 2).astype(np.float32)
Y_val = cutHalf(train_files['Y_val'][:640, :640], 2).astype(np.float32)
mean, std = np.mean(X_train), np.std(X_train)
X_train = normalize(X_train, mean, std)
X_val = normalize(X_val, mean, std)

# Select fraction

train_frac = int(np.round((exp_params['train_frac']/100)*X_train.shape[0]))
X_train[train_frac:]