from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from skimage.segmentation import find_boundaries

from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap, draw_polygons, sample_points
from stardist import Config, StarDist

import json
from os.path import join
from skimage import io

with open('experiment.json', 'r') as f:
    exp_params = json.load(f)


files = np.load(exp_params["test_path"])
X_test = files['X_test']
model_no_sc = StarDist(None, name = exp_params['model_name'], basedir = exp_params['base_dir'])


img = []
prob = []
dist = []
coord = []
points = []
labels = []

for i in range(X_test.shape[0]):

    image = normalize(X_test[i],1,99.8)
    probability, distance = model_no_sc.predict(image)
    coordinates = dist_to_coord(distance)
    point = non_maximum_suppression(coordinates,probability,prob_thresh=0.4)
    label = polygons_to_label(coordinates,probability,point)

    img.append(image)
    prob.append(probability)
    dist.append(distance)
    coord.append(coordinates)
    points.append(point)
    labels.append(label)

for i in range(0,len(img)):
    io.imsave(join(exp_params['base_dir'], str(i).zfill(3)+'.tif'), labels[i])
