from __future__ import (print_function, division)

import os
import json
import math
import itertools
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from PIL import Image, ImageDraw

###########################################################################
## local imports

from noaa.utils import (lookup_whale_images, load_images,
                   plot_images, plot_whales_by_id)
from config import BASE_DIR

###########################################################################
## slic segmentation


def interior_direction(l1, l2, p1, p2, flip=True):
    if flip:
        return ((l2[0] - l1[0])*(p1 - l1[1]) - (l2[1] - l1[1])*(p2 - l1[0])) > 0
    else:
        return ((l2[0] - l1[0])*(p1 - l1[1]) - (l2[1] - l1[1])*(p2 - l1[0])) <= 0


def convex_hull_mask_slow(data, val):
    from scipy.spatial import ConvexHull

    segm = np.argwhere(data == val)
    hull = ConvexHull(segm)
    coor = np.argwhere(data >= 0)
    n = len(hull.vertices)
    rtn = np.ones(data.shape) == 1

    for i, ver in enumerate(hull.vertices):
        j = (i + 1) % n
        k = (i + 2) % n
        p1 = coor[i,:]
        p2 = coor[j,:]
        p3 = coor[j,:]
        direction = interior_direction(p1, p2, p3[0], p3[1])
        f = lambda x, y: interior_direction(p1, p2, x, y, direction)
        rtn = rtn & np.fromfunction(f, rtn.shape)

    return rtn


def convex_hull_mask(data):
    from scipy.spatial import ConvexHull

    segm = np.argwhere(data)
    hull = ConvexHull(segm)
    img = Image.new('L', data.shape, 0)
    verts = [(segm[v,0], segm[v,1]) for v in hull.vertices]
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask


def compute_shape_regularity(data):
    vals = np.unique(data)
    d = dict()
    for val in vals:
        segm = data == val
        hull = convex_hull_mask(segm)
        d[val] = hull.sum()

    return d


def numpy_isin(array, keeps):
    rtn = np.zeros(array.shape)
    for k in keeps:
        rtn[array==k] = k

    return rtn


def detect_regularity_outliers(d, retain=3, cut=None):
    m = np.mean(d.values())
    s = np.std(d.values())
    if cut is not None:
        return [v for k,v in d.iteritems() if (v-m) > cut * s]
    for i in range(1000):
        cnt = len([v for k,v in d.iteritems() if (v-m) > i * 0.1 * s])
        if cnt < retain:
            i -= 1
            break

    return {k:v for k,v in d.iteritems() if (v-m) > i * 0.1 * s}


def slic_segmentation(retain=3, n_segments=20, compactness=20, sigma=0):
    csv_path=os.path.join(BASE_DIR, 'data/train.csv')
    img_path=os.path.join(BASE_DIR, 'data/imgs/')

    images = os.listdir(img_path)
    shuffle(images)
    for im in images:
        print('loading image ', im)
        im = os.path.join(img_path, im)
        image = load_images([im])[0]
        print('computing segments')
        segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
        print('computing shape regularity')
        segments_regularity_map = compute_shape_regularity(segments)
        segments_to_keep = detect_regularity_outliers(segments_regularity_map, retain=2)
        retained_segments = numpy_isin(segments, segments_to_keep)
        print('plotting')
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        ax[0].imshow(image)
        ax[1].imshow(retained_segments)
        plt.show()
        plt.clf()
