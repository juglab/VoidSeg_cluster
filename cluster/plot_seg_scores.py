from tifffile import imread
from os.path import join, basename
from glob import glob
import numpy as np
import pickle
from scipy import ndimage

from numba import jit

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gt')
parser.add_argument('-out')
parser.add_argument('exp', nargs=argparse.REMAINDER)

args = parser.parse_args()

gt_path = args.gt
out_path = args.out
exp_paths = args.exp

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

gt_files = glob(join(gt_path, 'man_seg*.tif'))
gt_images = {int(basename(x)[8:-4]) : imread(x) for x in gt_files}
gt_images_keys = [x for x in gt_images.keys()]
gt_images_keys.sort()

exp_list = []
for exp in exp_paths:
    subdirs = {int(basename(x).split('_')[1]) : x for x in glob(join(exp, '*'))}
    subdirs_keys = [x for x in subdirs.keys()]
    subdirs_keys.sort()
    list_SEGs = []
    for k in subdirs_keys:
        seg_score = 0
        result_files = glob(join(subdirs[k], 'foreground*.tif'))
        result_images = {int(basename(x)[10:-4]) : imread(x) for x in result_files}
        result_image_keys = [x for x in result_images.keys()]
        result_image_keys.sort()

        with open(join(subdirs[k], 'best_score.dat'), 'rb') as best_score_file:
            ts = pickle.load(best_score_file)[0]

        print('Use threshold =', ts)
        for gt_img_key, result_img_key in zip(gt_images_keys, result_image_keys):
            prediction_fg = result_images[result_img_key]
            pred_thresholded = prediction_fg > ts
            labels, _ = ndimage.label(pred_thresholded)
            score = seg(gt_images[gt_img_key], labels)
            seg_score = seg_score + score
        list_SEGs.append(seg_score/float(len(gt_files)))
    exp_list.append(list_SEGs)

X_train_frac = subdirs_keys
plt.figure(figsize=(10,5))
for exp, path in zip(exp_list, exp_paths):
    plt.plot(X_train_frac, exp, label=basename(path))

plt.ylim(0.55, 0.8)
plt.xlabel('Training Data Fraction')
plt.ylabel('SEG Score')
plt.legend(loc='lower right')
plt.savefig(out_path, pad_inches=0.0, bbox_inches='tight')
