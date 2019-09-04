from os.path import join
import pickle

from csbdeep.models import Config, CARE
from csbdeep.utils import Path, normalize
from stardist import Config, StarDist
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
import numpy as np
import json

from scipy import ndimage

from os.path import isdir, exists, join, basename
from skimage import io
import os
from shutil import copy as cp
from shutil import move as mv
import glob
import sys
from PyInquirer import prompt, Validator, ValidationError
import argparse as ap


import sys
print(sys.argv)

for i in range(1, len(sys.argv)):
    directory = "/lustre/projects/juglab/StarVoid/outdata/" + sys.argv[i] +'/'
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]

    for j in subfolders:
        print("Working on subfolder"+j)
        with open(j+'/'+'best_score.dat', 'rb') as best_score_file:
            ts = pickle.load(best_score_file)[0]
        print('Use threshold =', ts)
        seg_directory = 'seg_masks'
        os.makedirs(j+'/'+seg_directory)
        img = []
        prob = []
        dist = []
        coord = []
        points = []
        labels = []

        with open(j+'/'+'experiment.json', 'r') as f:
            exp_params = json.load(f)

        files = np.load(exp_params["test_path"])
        X_test = files['X_test']
        model_no_sc = StarDist(None, name=exp_params['model_name'], basedir=exp_params['base_dir'])
        model_no_sc.load_weights(j+'/'+exp_params["model_name"]+'_seg/weights_best.h5')

        for k in range(X_test.shape[0]):
            image = normalize(X_test[k], 1, 99.8)
            probability, distance = model_no_sc.predict(image)
            coordinates = dist_to_coord(distance)
            point = non_maximum_suppression(coordinates, probability, prob_thresh=ts)
            label = polygons_to_label(coordinates, probability, point)

            img.append(image)
            prob.append(probability)
            dist.append(distance)
            coord.append(coordinates)
            points.append(point)
            labels.append(label)

        for l in range(0, len(img)):
            io.imsave(j+'/'+seg_directory+'mask'+ str(l).zfill(3) + '.tif', labels[l].astype(np.int16))
