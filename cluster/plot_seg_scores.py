from tifffile import imread
from os.path import join, basename
from glob import glob

import pickle
from scipy import ndimage

from scripts.utils.utils import seg



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



gt_files = sorted(glob(join(gt_path, 'man_seg*.tif')))
gt_images=[]
for x in gt_files:
    gt_images.append(imread(x))

exp_list = []
for exp in exp_paths:
    subdirs = {(float(basename(x).split('_')[1])) : x for x in glob(join(exp, '*'))}
    subdirs_keys = [x for x in subdirs.keys()]
    subdirs_keys.sort()
    list_SEGs = []
    for k in subdirs_keys:

        seg_score = 0
        result_files = sorted(glob(join(subdirs[k], 'foreground*.tif')))
        result_images=[]
        for j in result_files:
            result_images.append(imread(j))

        with open(join(subdirs[k], 'best_score.dat'), 'rb') as best_score_file:
            ts = pickle.load(best_score_file)[0]

        print('Use threshold =', ts)
        
        for index in range(len(gt_images)):
            gt_img=gt_images[index]
            prediction_fg=result_images[index]
            # print(gt_files[index],result_files[index])
            pred_thresholded = prediction_fg > ts
            labels, _ = ndimage.label(pred_thresholded)
            score = seg(gt_img, labels)
            seg_score = seg_score + score
        list_SEGs.append(seg_score/float(len(gt_files)))
    exp_list.append(list_SEGs)

X_train_frac = subdirs_keys
plt.figure(figsize=(10,5))
for exp, path in zip(exp_list, exp_paths):
    print(exp)
    plt.plot(X_train_frac, exp, label=basename(path))

plt.ylim(0.45, 0.77)
plt.xlabel('Training Data Fraction')
plt.ylabel('SEG Score')
plt.legend(loc='lower right')
plt.savefig(out_path, pad_inches=0.0, bbox_inches='tight')
