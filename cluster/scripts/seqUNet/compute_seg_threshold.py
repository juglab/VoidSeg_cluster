
from os.path import join
import pickle

from csbdeep.models import Config, CARE
import numpy as np
import json

from scipy import ndimage

from numba import jit
from keras.layers import Input, Conv2D, Conv3D, Activation, Lambda
from keras.models import Model
from keras.layers.merge import Add, Concatenate
from csbdeep.internals.blocks import unet_block
from csbdeep.utils import _raise, backend_channels_last

@jit
def pixel_sharing_bipartite(lab1, lab2):
    assert lab1.shape == lab2.shape
    psg = np.zeros((lab1.max()+1, lab2.max()+1), dtype=np.int)
    for i in range(lab1.size):
        psg[lab1.flat[i], lab2.flat[i]] += 1
    return psg

def intersection_over_union(psg):
    rsum = np.sum(psg, 0, keepdims=True)
    csum = np.sum(psg, 1, keepdims=True)
    return psg / (rsum + csum - psg)

def matching_overlap(psg, fractions=(0.5,0.5)):
    """
    create a matching given pixel_sharing_bipartite of two label images based on mutually overlapping regions of sufficient size.
    NOTE: a true matching is only gauranteed for fractions > 0.5. Otherwise some cells might have deg=2 or more.
    NOTE: doesnt break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
    """
    afrac, bfrac = fractions
    tmp = np.sum(psg, axis=1, keepdims=True)
    m0 = np.where(tmp==0,0,psg / tmp)
    tmp = np.sum(psg, axis=0, keepdims=True)
    m1 = np.where(tmp==0,0,psg / tmp)
    m0 = m0 > afrac
    m1 = m1 > bfrac
    matching = m0 * m1
    matching = matching.astype('bool')
    return matching

def seg(lab_gt, lab, partial_dataset=False):
    """
    calculate seg from pixel_sharing_bipartite
    seg is the average conditional-iou across ground truth cells
    conditional-iou gives zero if not in matching
    ----
    calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
    for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
    IoU as low as 1/2 that don't match, and thus have CIoU = 0.
    """
    psg = pixel_sharing_bipartite(lab_gt, lab)
    iou = intersection_over_union(psg)
    matching = matching_overlap(psg, fractions=(0.5, 0))
    matching[0,:] = False
    matching[:,0] = False
    n_gt = len(set(np.unique(lab_gt)) - {0})
    n_matched = iou[matching].sum()
    if partial_dataset:
        return n_matched , n_gt
    else:
        return n_matched / n_gt


def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std

def denormalize(x, mean, std):
    return x*std + mean


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


with open('experiment.json', 'r') as f:
    exp_params = json.load(f)

train_files = np.load(join('..', '..', '..', *exp_params['train_path'].split('/')[4:]))
X_train = train_files['X_train']
mean, std = np.mean(X_train), np.std(X_train)
X_val = train_files['X_val']
Y_val = train_files['Y_val']
X_val = normalize(X_val, mean, std)
model = CARE_SEQ(None, name=exp_params['model_name'], basedir='')

print('Compute best threshold:')
seg_scores = []
for ts in np.linspace(0, 1, 21):
    seg_score = 0
    for idx in range(X_val.shape[0]):
        img, gt = X_val[idx], Y_val[idx]
        prediction = model.predict(img, axes='YX', normalizer=None)
        prediction_exp = np.exp(prediction[..., 1:])
        prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
        prediction_fg = prediction_seg[..., 1]
        pred_thresholded = prediction_fg > ts
        labels, _ = ndimage.label(pred_thresholded)
        tmp_score = seg(gt, labels)
        if not np.isnan(tmp_score):
            seg_score += tmp_score

    seg_score /= float(X_val.shape[0])
    seg_scores.append((ts, seg_score))
    print('Seg-Score for threshold =', ts, 'is', seg_score)

with open('seg_scores.dat', 'wb') as file_segScores:
    pickle.dump(seg_scores, file_segScores)

best_score = sorted(seg_scores, key=lambda tup: tup[1])[-1]

print('Best Seg-Score is', best_score[1], 'achieved with threshold =', best_score[0])
with open('best_score.dat', 'wb') as file_bestScore:
    pickle.dump(best_score, file_bestScore)
