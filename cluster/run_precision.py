from os.path import join
import pickle

# from csbdeep.models import Config, CARE
import numpy as np
import json

from scipy import ndimage

from numba import jit

from os.path import isdir, exists, join, basename
import os
from shutil import copy as cp
from shutil import move as mv
import glob
import sys
from PyInquirer import prompt, Validator, ValidationError
import argparse as ap


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

def normalize(img, mean, std):
  zero_mean = img - mean
  return zero_mean / std

def denormalize(x, mean, std):
  return x * std + mean


def predict_seg(model, test, mean_std):
    mean, std = mean_std[0], mean_std[1]
    X_test = test['X_test']
    X = normalize(X_test, mean, std)
    with open('best_precision_score.dat', 'rb') as best_score_file:
        ts = pickle.load(best_score_file)[0]
    print('Use threshold =', ts)
    for i in range(X.shape[0]):
        prediction = model.predict(X[i], axes='YX', normalizer=None)
        prediction_exp = np.exp(prediction[..., 1:])
        prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
        prediction_bg = prediction_seg[..., 0]
        prediction_fg = prediction_seg[..., 1]
        prediction_b = prediction_seg[..., 2]
        pred_thresholded = prediction_fg > ts
        labels, nb = ndimage.label(pred_thresholded)
        #    predictions.append(pred)
        io.imsave(join(exp_conf['base_dir'], 'mask' + str(i).zfill(3) + '.tif'), labels.astype(np.int16))
        io.imsave(join(exp_conf['base_dir'], 'foreground' + str(i).zfill(3) + '.tif'), prediction_fg)
        io.imsave(join(exp_conf['base_dir'], 'background' + str(i).zfill(3) + '.tif'), prediction_bg)
        io.imsave(join(exp_conf['base_dir'], 'border' + str(i).zfill(3) + '.tif'), prediction_b)


import sys
print(sys.argv)

for i in range(1, len(sys.argv)):
    directory = "/lustre/projects/juglab/StarVoid/outdata/" + sys.argv[i] +'/'
    print(os.walk(directory))

# with open('experiment.json', 'r') as f:
#   exp_params = json.load(f)
#
# train_files = np.load(join('..', '..', '..', *exp_params['train_path'].split('/')[4:]))
# X_train = train_files['X_train']
# seg_test_d = np.load(exp_conf['test_path'])
# mean, std = np.mean(X_train), np.std(X_train)
# X_val = train_files['X_val']
# Y_val = train_files['Y_val']
# X_val = normalize(X_val, mean, std)
# model = CARE(None, name=exp_params['model_name'] + str('_seg'), basedir='')
#
# print('Compute best threshold:')
# precision_scores = []
# for ts in np.linspace(0.1, 1, 19):
#   precision_score = 0
#   for idx in range(X_val.shape[0]):
#       img, gt = X_val[idx], Y_val[idx]
#       prediction = model.predict(img, axes='YX', normalizer=None)
#       prediction_exp = np.exp(prediction[..., 1:])
#       prediction_precision = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
#       prediction_fg = prediction_precision[..., 1]
#       pred_thresholded = prediction_fg > ts
#       labels, _ = ndimage.label(pred_thresholded)
#       tmp_score = precision(gt, labels)
#       if not np.isnan(tmp_score):
#           precision_score += tmp_score
#
#   precision_score /= float(X_val.shape[0])
#   precision_scores.append((ts, precision_score))
#   print('Precision-Score for threshold =', ts, 'is', precision_score)
#
# with open('precision_scores.dat', 'wb') as file_precisionScores:
#   pickle.dump(precision_scores, file_precisionScores)
#
# best_score = sorted(precision_scores, key=lambda tup: tup[1])[-1]
#
# print('Best Precision-Score is', best_score[1], 'achieved with threshold =', best_score[0])
# with open('best_precision_score.dat', 'wb') as file_bestScore:
#   pickle.dump(best_score, file_bestScore)
#
# predict_seg(model, seg_test_d, (mean, std))
#
