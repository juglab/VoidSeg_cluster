
from os.path import join
import pickle

from stardist import Config, StarDist
from csbdeep.utils import Path, normalize
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
import numpy as np
import json
import sys

from scipy import ndimage

from numba import jit

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


with open('experiment.json', 'r') as f:
    exp_params = json.load(f)
print("Opened json!")

train_files = np.load(join('..', '..', '..', *exp_params['train_path'].split('/')[4:]))
# X_train = train_files['X_train']
# mean, std = np.mean(X_train), np.std(X_train)
X_val = train_files['X_val']
Y_val = train_files['Y_val']
# X_val = normalize(X_val, mean, std)
model = StarDist(None, name=exp_params['model_name'], basedir='')

print('Compute best threshold:')

seg_scores = []
ts_range = np.array([0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1 ])
for ts in ts_range:
# for ts in np.linspace(0, 1, 21):
    print(ts)
    sys.stdout.flush()
    seg_score = 0
    for idx in range(X_val.shape[0]):
        print(idx)
        sys.stdout.flush()
        img, gt = X_val[idx], Y_val[idx]
        img = normalize(img, 1, 99.8)
        probability, distance = model.predict(img)
        coordinates = dist_to_coord(distance)
        point = non_maximum_suppression(coordinates, probability, prob_thresh=ts)
        label = polygons_to_label(coordinates, probability, point)
        tmp_score = seg(gt, label)
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
