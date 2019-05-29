from csbdeep.models import Config, CARE
import numpy as np

from scipy import ndimage

from os.path import join
from skimage import io

import pickle

import json
from keras.layers import Input, Conv2D, Conv3D, Activation, Lambda
from keras.models import Model
from keras.layers.merge import Add, Concatenate
from csbdeep.internals.blocks import unet_block
from csbdeep.utils import _raise, backend_channels_last


def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std

def denormalize(x, mean, std):
    return x*std + mean

with open('experiment.json', 'r') as f:
    exp_params = json.load(f)


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

            final_n2v = conv(1, (1,)*n_dim, activation='linear')(unet)

            unet_seg = unet_block(2, 32, (3,3),
                              activation="relu", dropout=False, batch_norm=True,
                              n_conv_per_depth=2, pool=(2,2), prefix='seg')(unet)

            final_seg = conv(3, (1,)*n_dim, activation='linear')(unet_seg)

            final_n2v = Activation(activation=last_activation)(final_n2v)
            final_seg = Activation(activation=last_activation)(final_seg)

            final = Concatenate(axis=channel_axis)([final_n2v, final_seg])
            return Model(inputs=input, outputs=final)
        return seq_unet((None, None, 1), 'linear')


files = np.load(exp_params["test_path"])
X_test = files['X_test']


train_files = np.load(exp_params["train_path"])
X_trn = train_files['X_train']

mean, std = np.mean(X_trn), np.std(X_trn)
X = normalize(X_test, mean, std)

model = CARE_SEQ(None, name= exp_params['model_name'], basedir= exp_params['base_dir'])

with open('best_score.dat', 'rb') as best_score_file:
    ts = pickle.load(best_score_file)[0]

print('Use threshold =', ts)

for i in range(X.shape[0]):
    prediction = model.predict(X[i], axes='YX',normalizer=None )
    denoised = prediction[...,0]
    prediction_exp = np.exp(prediction[...,1:])
    prediction_seg = prediction_exp/np.sum(prediction_exp, axis = 2)[...,np.newaxis]
    predicton_denoise = denormalize(denoised, mean, std)
    prediction_bg = prediction_seg[...,0]
    prediction_fg = prediction_seg[...,1]
    prediction_b = prediction_seg[...,2]
    pred_thresholded = prediction_fg>ts
    labels, nb = ndimage.label(pred_thresholded)
#    predictions.append(pred)
    io.imsave(join(exp_params['base_dir'], 'mask'+str(i).zfill(3)+'.tif'), labels.astype(np.int16))
    io.imsave(join(exp_params['base_dir'], 'foreground'+str(i).zfill(3)+'.tif'), prediction_fg)
    io.imsave(join(exp_params['base_dir'], 'background'+str(i).zfill(3)+'.tif'), prediction_bg)
    io.imsave(join(exp_params['base_dir'], 'border'+str(i).zfill(3)+'.tif'), prediction_b)
