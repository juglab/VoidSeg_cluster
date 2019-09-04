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

def matching_iou(psg, fraction=0.5):
  iou = intersection_over_union(psg)
  matching = iou > 0.5
  matching[:,0] = False
  matching[0,:] = False
  return matching

def precision(lab_gt, lab, iou=0.5, partial_dataset=False):
  """
  precision = TP / (TP + FP + FN) i.e. "intersection over union" for a graph matching
  """
  psg = pixel_sharing_bipartite(lab_gt, lab)
  matching = matching_iou(psg, fraction=iou)
  assert matching.sum(0).max() < 2
  assert matching.sum(1).max() < 2
  n_gt  = len(set(np.unique(lab_gt)) - {0})
  n_hyp = len(set(np.unique(lab)) - {0})
  n_matched = matching.sum()
  if partial_dataset:
    return n_matched , (n_gt + n_hyp - n_matched)
  else:
    return n_matched / (n_gt + n_hyp - n_matched)


with open('experiment.json', 'r') as f:
    exp_params = json.load(f)
print("Opened json!")

train_files = np.load(exp_params['train_path'])
# X_train = train_files['X_train']
# mean, std = np.mean(X_train), np.std(X_train)
X_val = train_files['X_val']
Y_val = train_files['Y_val']
# X_val = normalize(X_val, mean, std)
model = StarDist(None, name=exp_params['model_name'], basedir='')

print('Compute best threshold:')

precision_scores = []
# ts_range = np.array([0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1 ])
# for ts in ts_range:
for ts in np.linspace(0.1, 1, 19):
    print(ts)
    sys.stdout.flush()
    precision_score = 0
    for idx in range(X_val.shape[0]):
        print(idx)
        sys.stdout.flush()
        img, gt = X_val[idx], Y_val[idx]
        img = normalize(img, 1, 99.8)
        probability, distance = model.predict(img)
        coordinates = dist_to_coord(distance)
        point = non_maximum_suppression(coordinates, probability, prob_thresh=ts)
        label = polygons_to_label(coordinates, probability, point)
        tmp_score = precision(gt, label)
        if not np.isnan(tmp_score):
            precision_score += tmp_score

    precision_score /= float(X_val.shape[0])
    precision_scores.append((ts, precision_score))
    print('Precision-Score for threshold =', ts, 'is', precision_score)

with open('precision_scores.dat', 'wb') as file_precisionScores:
    pickle.dump(precision_scores, file_precisionScores)

best_score = sorted(precision_scores, key=lambda tup: tup[1])[-1]

print('Best Precision-Score is', best_score[1], 'achieved with threshold =', best_score[0])
with open('best_score.dat', 'wb') as file_bestScore:
    pickle.dump(best_score, file_bestScore)
