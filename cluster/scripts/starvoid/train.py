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


def cutHalf(data, repeat):
    data_ = data
    print(data_.shape)
    for i in range(repeat):
        newsize = int(data_.shape[1] / 2)
        a = data_[:, :newsize, :]
        b = data_[:, newsize:, :]
        #      print(a.shape,b.shape)
        data_ = np.concatenate((a, b), axis=0)

        newsize = int(data_.shape[2] / 2)
        a = data_[:, :, :newsize]
        b = data_[:, :, newsize:]
        data_ = np.concatenate((a, b), axis=0)
        print(data_.shape)
    return data_


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
is_seeding = conf['is_seeding']

# Read training images and GT
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

# convert to oneHot
Y_train_oneHot = convert_to_oneHot(Y_train)
Y_val_oneHot = convert_to_oneHot(Y_val)

Y_train = np.concatenate(
    (X_train[..., np.newaxis], np.zeros(X_train.shape, dtype=np.float32)[..., np.newaxis], Y_train_oneHot), axis=3)

# Select fraction
train_frac = int(np.round((exp_params['train_frac'] / 100) * X_train.shape[0]))

if is_seeding:
    print('seeding training data')
    np.random.seed(exp_params['seed'])
    rng = np.random.RandomState(exp_params['seed'])
    print(rng)
    seed_ind = rng.permutation(X_train.shape[0])
    print(seed_ind)
    X_train, Y_train = [X_train[i] for i in seed_ind], [Y_train[i] for i in seed_ind]


# if 'is_seeding' in exp_params.keys():
#     print('Here!')
#     if exp_params['is_seeding']:
#         print('seeding training data')
#         np.random.seed(exp_params['seed'])
#         rng = np.random.RandomState(exp_params['seed'])
#         print(rng)
#         seed_ind = rng.permutation(X_train.shape[0])
#         print(seed_ind)
#         X_train, Y_train = [X_train[i] for i in seed_ind], [Y_train[i] for i in seed_ind]
        
if use_denoising:
    Y_train[train_frac:, ..., 1:] *= 0
else:
    X_train = X_train[:train_frac]
    Y_train = Y_train[:train_frac]


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

# prepare validation data
X_validation = X_val[..., np.newaxis]
Y_validation = Y_val[..., np.newaxis]

num_pix = conf['n2v_num_pix']
pixelsInPatch = conf['n2v_patch_shape'][0] * conf['n2v_patch_shape'][1]

# 1. Option
Y_validation = np.concatenate((X_validation.copy(), np.zeros(X_validation.shape, dtype=np.float32)), axis=3)
manipulate_val_data(X_validation, Y_validation, num_pix=int(num_pix * X_validation.shape[1] * X_validation.shape[2] / float(pixelsInPatch)),
                    shape=(X_validation.shape[1], X_validation.shape[2]))

Y_validation = np.concatenate((Y_validation, Y_val_oneHot), axis=3)

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

model = CARE(None, name= exp_params['model_name'], basedir= exp_params['base_dir'])
print(conf)

hist = model.train(X_train[..., np.newaxis],Y_train,validation_data=(X_validation,Y_validation))

with open(join(exp_params['base_dir'], exp_params['model_name'], 'history_' + exp_params['model_name'] + '.dat'),
          'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
