from tifffile import imread
from os.path import join, basename
from glob import glob
import numpy as np

from scripts.utils.utils import seg

import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-config')

args = parser.parse_args()

with open(args.config, 'r') as f:
    conf = json.load(f)

gt_files = glob(join(conf['gt'], 'man_seg*.tif'))
gt_images = {int(basename(x)[8:-4]) : imread(x) for x in gt_files}
gt_image_keys = [x for x in gt_images.keys()]
gt_image_keys.sort()

exp_names = conf['exp_names']

evaluation_scores = {}
for exp in exp_names:
    run_seg_scores = {}
    for run in conf[exp]:
        print('Compute on', run)
        fractions = {float(basename(x).split('_')[1]) : x for x in glob(join(run, '*'))}
        fraction_keys = [x for x in fractions.keys()]
        fraction_keys.sort()

        for frac in fraction_keys:
            print('-', frac)
            seg_score = 0
            result_files = glob(join(fractions[frac], 'mask*.tif'))
            result_images = {int(basename(x)[4:-4]) : imread(x) for x in result_files}
            result_image_keys = [x for x in result_images.keys()]
            result_image_keys.sort()

            for gt_k, result_k in zip(gt_image_keys, result_image_keys):
                seg_score += seg(gt_images[gt_k], result_images[result_k])

            seg_score /= float(len(result_image_keys))
            if frac in run_seg_scores.keys():
                run_seg_scores[frac].append(seg_score)
            else:
                run_seg_scores[frac] = [seg_score]

    mean_run = {}
    std_run = {}
    low_perc = {}
    up_perc = {}
    min = {}
    max = {}
    n_run = {}
    for r in run_seg_scores.keys():
        ar = np.array(run_seg_scores[r])
        n_run[r] = ar.shape[0]
        mean_run[r] = np.mean(ar)
        std_run[r] = np.std(ar)
        low_perc[r] = np.percentile(ar, 25)
        up_perc[r] = np.percentile(ar, 75)
        min[r] = np.min(ar)
        max[r] = np.max(ar)

    keys = [x for x in run_seg_scores.keys()]
    keys.sort()
    stats = np.zeros((len(keys), 9))
    print('Stats for', exp)
    for i in range(len(keys)):
        key = keys[i]
        print(key, mean_run[key], std_run[key], low_perc[key], up_perc[key], min[key], max[key])
        stats[i,0] = key
        stats[i,1] = n_run[key]
        stats[i,2] = mean_run[key]
        stats[i,3] = std_run[key]/np.sqrt(n_run[key])
        stats[i,4] = std_run[key]
        stats[i,5] = low_perc[key]
        stats[i,6] = up_perc[key]
        stats[i,7] = min[key]
        stats[i,8] = max[key]

    np.save(conf['output'][:-4]+'_'+exp+'.npy', stats)
    evaluation_scores[exp] = stats

plt.figure(figsize=(10,5))
for exp in exp_names:
    stats = evaluation_scores[exp]
    plt.errorbar(stats[:,0], stats[:,2], yerr=stats[:,3], label=exp)

plt.ylim(0.5, 0.8)
plt.xlabel('Training Data Fractions')
plt.ylabel('SEG Score')
plt.legend(loc='lower right')
plt.savefig(conf['output'], pad_inches=0.0, bbox_inches='tight')