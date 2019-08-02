from csbdeep.models import Config, CARE
import numpy as np
from csbdeep.utils.n2v_utils import manipulate_val_data
from skimage.segmentation import find_boundaries
import json
import os
from os.path import join
import pickle
from keras.layers import Input, Conv2D, Conv3D, Activation, Lambda
from keras.models import Model
from keras.layers.merge import Add, Concatenate
from csbdeep.internals.blocks import unet_block
from csbdeep.utils import _raise, backend_channels_last


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

# Read training images and GT
train_files = np.load(exp_params['train_path'])
X_train = train_files['X_train']
Y_train = train_files['Y_train']

X_val = train_files['X_val'].astype(np.float32)
Y_val = train_files['Y_val'].astype(np.float32)

mean, std = np.mean(X_train), np.std(X_train)
X_train = normalize(X_train, mean, std)
X_val = normalize(X_val, mean, std)

if 'is_seeding' in exp_params.keys():
    if exp_params['is_seeding']:
        print('seeding training data')
        np.random.seed(exp_params['random_seed'])
        seed_ind = np.random.permutation(X_train.shape[0])
        X_train = X_train[seed_ind]
        Y_train = Y_train[seed_ind]


if 'augment' in exp_params.keys():
    if exp_params['augment']:
        print('augmenting training data')
        X_ = X_train.copy()
        X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (1, 2))))
        X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
        X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))
        X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))

        Y_ = Y_train.copy()
        Y_train_aug = np.concatenate((Y_train, np.rot90(Y_, 1, (1, 2))))
        Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 2, (1, 2))))
        Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 3, (1, 2))))
        Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1)))
        print('Training data size after augmentation', X_train_aug.shape)
        print('Training data size after augmentation', Y_train_aug.shape)

        X_ = X_val.copy()
        X_val_aug = np.concatenate((X_val, np.rot90(X_, 1, (1, 2))))
        X_val_aug = np.concatenate((X_val_aug, np.rot90(X_, 2, (1, 2))))
        X_val_aug = np.concatenate((X_val_aug, np.rot90(X_, 3, (1, 2))))
        X_val_aug = np.concatenate((X_val_aug, np.flip(X_val_aug, axis=1)))

        Y_ = Y_val.copy()
        Y_val_aug = np.concatenate((Y_val, np.rot90(Y_, 1, (1, 2))))
        Y_val_aug = np.concatenate((Y_val_aug, np.rot90(Y_, 2, (1, 2))))
        Y_val_aug = np.concatenate((Y_val_aug, np.rot90(Y_, 3, (1, 2))))
        Y_val_aug = np.concatenate((Y_val_aug, np.flip(Y_val_aug, axis=1)))

# convert to oneHot
Y_train_oneHot = convert_to_oneHot(Y_train_aug)
Y_val_oneHot = convert_to_oneHot(Y_val_aug)

Y_train_aug = np.concatenate(
    (X_train_aug[..., np.newaxis], np.zeros(X_train_aug.shape, dtype=np.float32)[..., np.newaxis], Y_train_oneHot), axis=3)

# Select fraction
print('X_train.shape:', X_train.shape[0])
train_frac = int(np.round((exp_params['train_frac'] / 100) * X_train.shape[0]))
if use_denoising:
    Y_train_aug[train_frac:, ..., 1:] *= 0
else:
    X_train_aug = X_train_aug[:train_frac]
    Y_train_aug = Y_train_aug[:train_frac]

print('X_train.shape:', X_train.shape[0])


# prepare validation data
X_validation = X_val_aug[..., np.newaxis]
Y_validation = Y_val_aug[..., np.newaxis]

num_pix = conf['n2v_num_pix']
pixelsInPatch = conf['n2v_patch_shape'][0] * conf['n2v_patch_shape'][1]

Y_validation = np.concatenate((X_validation.copy(), np.ones(X_validation.shape, dtype=np.float32)), axis=3)
# manipulate_val_data(X_validation, Y_validation, num_pix=int(num_pix * X_validation.shape[1] * X_validation.shape[2] / float(pixelsInPatch)), #Not manipulating val_data as it is closer to testing
#                     shape=(X_validation.shape[1], X_validation.shape[2]))

Y_validation = np.concatenate((Y_validation, Y_val_oneHot), axis=3)

class CARE_SEQ(CARE):
    def _build(self):
        def seq_unet(input_shape, last_activation, n_depth=2, n_filter_base=32, kernel_size=(3,3),
             n_conv_per_depth=2, activation="relu", batch_norm=True, pool_size=(2,2),
             n_channel_out=4, eps_scale=1e-3):
            if last_activation is None:
                raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

            all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

            channel_axis = -1 if backend_channels_last() else 1

            n_dim = len(kernel_size)
            conv = Conv2D if n_dim==2 else Conv3D

            input = Input(input_shape, name = "input")
            unet = unet_block(2, 32, (3,3),
                              activation="relu", dropout=False, batch_norm=True,
                              n_conv_per_depth=2, pool=(2,2), prefix='n2v')(input)

            # final_n2v = conv(1, (1,)*n_dim, activation='linear')(unet)

            unet_seg = unet_block(2, 32, (3,3),
                              activation="relu", dropout=False, batch_norm=True,
                              n_conv_per_depth=2, pool=(2,2), prefix='seg')(unet)

            final_seg = conv(4, (1,)*n_dim, activation='linear')(unet_seg) #Changed output dimensions

            # final_n2v = Activation(activation=last_activation)(final_n2v)
            final = Activation(activation=last_activation)(final_seg)

            # final = Concatenate(axis=channel_axis)([final_n2v, final_seg])
            return Model(inputs=input, outputs=final)
        return seq_unet((None, None, 1), 'linear')


model = CARE_SEQ(None, name= exp_params['model_name'], basedir= exp_params['base_dir'])
print(conf)
model.keras_model.summary()

if(exp_params['scheme'] == "finetune_both"):
    os.makedirs((exp_params['base_dir']+'/temp_model'),mode=0o775)
    with open(join(exp_params['base_dir'], 'temp_model', 'config.json'),'w') as file:
        json.dump(conf, file)
    model2 = CARE(None, name='temp_model', basedir=exp_params['base_dir'])
    model2.load_weights('/lustre/projects/juglab/StarVoid/outdata/finN2V_dsb_n20_run8sequential/train_100.0/finN2V_model_denoise/weights_best.h5') #Chnage path depending on run number and seed
    for la in range(len(model2.keras_model.layers[:37])):
        model.keras_model.layers[la].set_weights(model2.keras_model.layers[la].get_weights())
    for sla in range(2,len(model2.keras_model.layers)):
        model.keras_model.layers[sla+36].set_weights(model2.keras_model.layers[sla].get_weights())

hist = model.train(X_train_aug[..., np.newaxis],Y_train_aug,validation_data=(X_validation,Y_validation))

with open(join(exp_params['base_dir'], exp_params['model_name'], 'history_' + exp_params['model_name'] + '.dat'),
          'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
