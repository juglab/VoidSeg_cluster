
from os.path import isfile, join, basename
from glob import glob
import pickle

from csbdeep.models import Config, CARE
import numpy as np
import json

from scipy import ndimage

from numba import jit


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-exp')

args = parser.parse_args()

exp_path = args.exp

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


subdirs = {int(basename(x).split('_')[1]) : x for x in glob(join(exp_path, '*'))}
subdirs_keys = [x for x in subdirs.keys()]
subdirs_keys.sort()

for sk in subdirs_keys:
    with open(join(subdirs[sk], 'experiment.json'), 'r') as f:
        exp_params = json.load(f)
        train_files = np.load(exp_params['train_path'])
        X_train = train_files['X_train']
        mean, std = np.mean(X_train), np.std(X_train)
        X_val = train_files['X_val']
        Y_val = train_files['Y_val']
        X_val = normalize(X_val, mean, std)

        model = CARE(None, name=exp_params['model_name'], basedir=exp_params['base_dir'])

        print('Compute best threshold for:')
        print(subdirs[sk])
        seg_scores = []
        for ts in np.linspace(0, 1, 21):
            seg_score = 0
            for img, gt in zip(X_val, Y_val):
                prediction = model.predict(img, axes='YX', normalizer=None)
                prediction_exp = np.exp(prediction[..., 1:])
                prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
                prediction_fg = prediction_seg[..., 1]
                pred_thresholded = prediction_fg > ts
                labels, _ = ndimage.label(pred_thresholded)
                seg_score += seg(gt, labels)

            seg_score /= float(X_val.shape[0])
            seg_scores.append((ts, seg_score))
            print('Seg-Score for threshold =', ts, 'is', seg_score)

        with open(join(exp_params['base_dir'], 'seg_scores.dat'), 'wb') as file_segScores:
            pickle.dump(seg_scores, file_segScores)

        best_score = sorted(seg_scores, key=lambda tup: tup[1])[0]
        print('Best Seg-Score is', best_score[1], 'achieved with threshold =', best_score[0])
        with open(join(exp_params['base_dir'], 'best_score.dat'), 'wb') as file_bestScore:
            pickle.dump(best_score, file_bestScore)
