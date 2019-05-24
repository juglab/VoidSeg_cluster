from csbdeep.models import Config, CARE
import numpy as np
from csbdeep.utils.n2v_utils import manipulate_val_data
from skimage.segmentation import find_boundaries
import json
from os.path import join
import pickle


def add_boundary_label(lbl, dtype=np.uint16):
    """ lbl is an integer label image (not binarized) """
    b = find_boundaries(lbl, mode='outer')
    res = (lbl > 0).astype(dtype)
    res[b] = 2
    return res


def onehot_encoding(lbl, n_classes=3, dtype=np.uint32):
    """ n_classes will be determined by max lbl value if its value is None """
    from keras.utils import to_categorical
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot


def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean / std


def denormalize(x, mean, std):
    return x * std + mean


def convert_to_oneHot(data):
    data_oneHot = np.zeros((*data.shape, 3), dtype=np.float32)
    for i in range(data.shape[0]):
        data_oneHot[i] = onehot_encoding(add_boundary_label(data[i].astype(np.int32)))
    return data_oneHot


with open('experiment.json', 'r') as f:
    exp_params = json.load(f)

# Config from json file
with open(exp_params['model_name'] + '/config.json', 'r') as f:
    conf = json.load(f)

use_denoising = conf['use_denoising']


num_pix = 1
pixelsInPatch = conf['n2v_patch_shape'][0] * conf['n2v_patch_shape'][1]
    
    
################This is where the fine tuning stuff begins ################################################################

train_files = np.load(exp_params['train_path'])
X_train = train_files['X_train']
Y_train = train_files['Y_train']
if 'CTC' in exp_params['exp_name']:
    X_val = cutHalf(train_files['X_val'][:640, :640], 2).astype(np.float32)
    Y_val = cutHalf(train_files['Y_val'][:640, :640], 2).astype(np.float32)
else:
    X_val = train_files['X_val'].astype(np.float32)
    Y_val = train_files['Y_val'].astype(np.float32)

mean, std = np.mean(X_train), np.std(X_train)
X_train = normalize(X_train, mean, std)
X_val = normalize(X_val, mean, std)

Y_train_oneHot = convert_to_oneHot(Y_train)
Y_val_oneHot = convert_to_oneHot(Y_val)

Y_train = np.concatenate(
    (X_train[..., np.newaxis], np.zeros(X_train.shape, dtype=np.float32)[..., np.newaxis], Y_train_oneHot), axis=3)
    
# Select fraction
print('X_train.shape:', X_train.shape[0])
train_frac = int(np.round((exp_params['train_frac'] / 100) * X_train.shape[0]))

if 'is_seeding' in exp_params.keys():
    if exp_params['is_seeding']:
        print('seeding training data')
        np.random.seed(exp_params['random_seed'])
        seed_ind = np.random.permutation(X_train.shape[0])
        X_train = X_train[seed_ind]
        Y_train = Y_train[seed_ind]
        
if use_denoising:
    Y_train[train_frac:, ..., 1:] *= 0
else:
    X_train = X_train[:train_frac]
    Y_train = Y_train[:train_frac]

print('X_train.shape:', X_train.shape[0])

X_validation = X_val[..., np.newaxis]
Y_validation = Y_val[..., np.newaxis]
Y_validation = np.concatenate((X_validation.copy(), np.ones(X_validation.shape, dtype=np.float32)), axis=3)
Y_validation = np.concatenate((Y_validation, Y_val_oneHot), axis=3)

# Augment validation
if 'augment' in exp_params.keys():
    if exp_params['augment']:
        print('augmenting training data')
        X_ = X_train.copy()
        X_train_aug = np.concatenate((X_train, np.rot90(X_, 2, (1, 2))))
        X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1), np.flip(X_train_aug, axis=2)))
        Y_ = Y_train.copy()
        Y_train_aug = np.concatenate((Y_train, np.rot90(Y_, 2, (1, 2))))
        Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1), np.flip(Y_train_aug, axis=2)))
        print('Training data size after augmentation', X_train.shape)
        print('Training data size after augmentation', Y_train.shape)

# Augment validation
if 'augment' in exp_params.keys():
    if exp_params['augment']:
        print('augment validation data')
        X_ = X_validation.copy()
        X_validation_aug = np.concatenate((X_validation, np.rot90(X_, 2, (1, 2))))
        X_validation_aug = np.concatenate(
            (X_validation_aug, np.flip(X_validation_aug, axis=1), np.flip(X_validation_aug, axis=2)))
        Y_ = Y_validation.copy()
        Y_validation_aug = np.concatenate((Y_validation, np.rot90(Y_, 2, (1, 2))))
        Y_validation_aug = np.concatenate(
            (Y_validation_aug, np.flip(Y_validation_aug, axis=1), np.flip(Y_validation_aug, axis=2)))

model = CARE(None, name= 'n2vpart_finetune', basedir= exp_params['base_dir'])
model.load_weights('/lustre/projects/juglab/StarVoid/n2vpart_VoidSeg/weights_best.h5') ## Set path to n2v pretrained model

#####Scheme 1 (Retraining all weights with n2v initialization)
model_1 = CARE(None, name= 'segpart_finetune_scheme1', basedir= exp_params['base_dir'])

i = 0;
for i in range(len(model_1.keras_model.layers[:-2])):
    model_1.keras_model.layers[i].set_weights(model.keras_model.layers[i].get_weights())
    i = i+1
hist1 = model_1.train(X_train_aug[..., np.newaxis],Y_train_aug,validation_data=(X_validation_aug,Y_validation_aug))

#####Scheme 2 (Freezing only downsampling weights)

model_2 = CARE(None, name= 'segpart_finetune_scheme2', basedir= exp_params['base_dir'])
j = 0;
for j in range(len(model_2.keras_model.layers[:-2])):
    model_2.keras_model.layers[j].set_weights(model.keras_model.layers[j].get_weights())
    j = j+1

for layer in model_2.keras_model.layers[:15]:
    layer.trainable = False

for layer in model_2.keras_model.layers:
    print(layer, layer.trainable)
hist2 = model_2.train(X_train_aug[..., np.newaxis],Y_train_aug,validation_data=(X_validation_aug,Y_validation_aug))   

#####Scheme 3 (Freezing all weights except last layer)

model_3 = CARE(None, name= 'segpart_finetune_scheme3', basedir= exp_params['base_dir'])
k = 0;
for j in range(len(model_3.keras_model.layers[:-2])):
    model_3.keras_model.layers[k].set_weights(model.keras_model.layers[k].get_weights())
    k = k+1

for layer in model_3.keras_model.layers[:-2]:
    layer.trainable = False

for layer in model_3.keras_model.layers:
    print(layer, layer.trainable)
hist3 = model_3.train(X_train_aug[..., np.newaxis],Y_train_aug,validation_data=(X_validation_aug,Y_validation_aug))  
