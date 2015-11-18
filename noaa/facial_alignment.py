from __future__ import (print_function, division)

import os
import json
import math
import itertools
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.filters import threshold_yen
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_opening


###########################################################################
## local imports

from noaa.utils import (lookup_whale_images, load_images,
                   plot_images, plot_whales_by_id)
from config import BASE_DIR

###########################################################################
## transforms

def mask_polygon(verts, shape):
    img = Image.new('L', shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T


def convex_hull_mask(data, mask=True):
    from scipy.spatial import ConvexHull
    segm = np.argwhere(data)
    hull = ConvexHull(segm)
    verts = [(segm[v,0], segm[v,1]) for v in hull.vertices]
    if mask:
        return mask_polygon(verts, data.shape)

    return verts


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


def image_generator():
    csv_path=os.path.join(BASE_DIR, 'data/train.csv')
    img_path=os.path.join(BASE_DIR, 'data/imgs/')

    images = os.listdir(img_path)
    shuffle(images)
    for idx, im in enumerate(images):
        im = os.path.join(img_path, im)
        image = load_images([im])[0].astype('int32')
        yield im, image


def smallest_partition(data, channel):
    orig = data[:,:,channel].sum()
    invs = np.abs(255-data[:,:,channel]).sum()
    rtn = np.copy(data)
    if invs < orig:
        rtn[:,:,channel] = np.abs(rtn[:,:,channel] - 255)
        return rtn

    return rtn


from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
def denoise_image(data, type=None):
    if type == 'tv':
        return denoise_tv_chambolle(data, weight=0.2, multichannel=True)

    return denoise_bilateral(data, sigma_range=0.1, sigma_spatial=15)


from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
def detect_blobs(data_gray):
    # takes grayscale
    blobs = blob_doh(data_gray, min_sigma=10, num_sigma=3, max_sigma=50, threshold=.1)
    mask = np.zeros(data_gray)
    n, m = mask.shape
    for blob in blobs:
        y,x = np.ogrid[-a:n-a, -b:m-b]
        mask = mask * (x*x + y*y <= r*r)

    return mask


from skimage.segmentation import slic
def segment_image_slic(image, retain=None, n_segments=20, compactness=20, sigma=5):
        segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
        if retain is not None:
            segments_regularity_map = compute_shape_regularity(segments)
            segments_to_keep = detect_regularity_outliers(segments_regularity_map, retain=retain)
            retained_segments = numpy_isin(segments, segments_to_keep)
            return retained_segments

        return segments


###########################################################################
## feature extractions

def segmentation(retain=3, n_segments=20, compactness=20, sigma=0):
    images = image_generator()

    for im in images:
        retained_segments = segment_image_slic(image, retain=retain, n_segments=n_segments, compactness=compactness, sigma=sigma)
        plot_images([image, retained_segments], ['Original', 'Segmented'])


def thresholding(votes_min=3):
    from skimage.filters import threshold_otsu
    from skimage.filters import threshold_li
    from skimage.filters import threshold_yen
    from skimage.filters import threshold_adaptive
    from scipy.ndimage import median_filter

    images = image_generator()

    for fn, im in images:
        print('inspecting image: ', fn)
        print('computing otsu threshold')
        otsu = threshold_otsu(im)
        otsu_ch1 = np.zeros(im.shape)
        otsu_ch2 = np.zeros(im.shape)
        otsu_ch3 = np.zeros(im.shape)
        otsu_ch1[im[:,:,0] > otsu, 0] = 255
        otsu_ch2[im[:,:,1] > otsu, 1] = 255
        otsu_ch3[im[:,:,2] > otsu, 2] = 255
        otsu_ch1 = smallest_partition(otsu_ch1, 0)
        otsu_ch2 = smallest_partition(otsu_ch2, 1)
        otsu_ch3 = smallest_partition(otsu_ch3, 2)

        print('computing yen threshold')
        yen = threshold_yen(im)
        yen_ch1 = np.zeros(im.shape)
        yen_ch2 = np.zeros(im.shape)
        yen_ch3 = np.zeros(im.shape)
        yen_ch1[im[:,:,0] > yen, 0] = 255
        yen_ch2[im[:,:,1] > yen, 1] = 255
        yen_ch3[im[:,:,2] > yen, 2] = 255
        yen_ch1 = smallest_partition(yen_ch1, 0)
        yen_ch2 = smallest_partition(yen_ch2, 1)
        yen_ch3 = smallest_partition(yen_ch3, 2)

        print('computing li threshold')
        li = threshold_li(im)
        li_ch1 = np.zeros(im.shape)
        li_ch2 = np.zeros(im.shape)
        li_ch3 = np.zeros(im.shape)
        li_ch1[im[:,:,0] > li, 0] = 255
        li_ch2[im[:,:,1] > li, 1] = 255
        li_ch3[im[:,:,2] > li, 2] = 255
        li_ch1 = smallest_partition(li_ch1, 0)
        li_ch2 = smallest_partition(li_ch2, 1)
        li_ch3 = smallest_partition(li_ch3, 2)

        print('computing average threshold')
        av_ch1 = np.zeros(im.shape)
        av_ch2 = np.zeros(im.shape)
        av_ch3 = np.zeros(im.shape)
        votes1 = (otsu_ch1 + yen_ch1 + li_ch1)
        votes1 = (otsu_ch1 + yen_ch1 + li_ch1)
        votes2 = (otsu_ch2 + yen_ch2 + li_ch2)
        votes3 = (otsu_ch3 + yen_ch3 + li_ch3)
        av_ch1[votes1[:,:,0] >= (255 * votes_min), 0] = 255
        av_ch2[votes2[:,:,1] >= (255 * votes_min), 1] = 255
        av_ch3[votes3[:,:,2] >= (255 * votes_min), 2] = 255

        thresholded_images = [
            otsu_ch1, otsu_ch2, otsu_ch3,
            yen_ch1, yen_ch2, yen_ch3,
            li_ch1, li_ch2, li_ch3,
            av_ch1, av_ch2, av_ch3
        ]

        print('filtering out specks')
        for idx, im in enumerate(thresholded_images):
            thresholded_images[idx] = median_filter(im, size=3)

        titles = [
            'Channel 1 Otsu', 'Channel 2 Otsu', 'Channel 3 Otsu',
            'Channel 1 Yen', 'Channel 2 Yen', 'Channel 3 Yen',
            'Channel 1 Li', 'Channel 2 Li', 'Channel 3 Li',
            'Channel 1 Avg', 'Channel 2 Avg', 'Channel 3 Avg'
        ]

        print('plotting')
        plot_images(
            thresholded_images,
            titles,
            suptitle=fn.split('/')[-1]
        )


def threshold_by_channel(image, threshold, n_channels=3):
    rtn = []
    for i in xrange(n_channels):
        ch = np.zeros(image.shape)
        ch[image[:,:,i] > threshold, :] = 255
        ch = smallest_partition(ch, i)
        rtn.append(ch)

    return rtn


def extract_largest_region(mask):
    rtn = np.copy(mask)
    regions, n_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, regions, range(n_labels+1))
    max_size = sorted(sizes)[-2]
    mask_on_size = sizes < max_size

    remove = np.ones(regions.shape)
    for i in range(1, n_labels+1):
        if sizes[i] >= max_size:
            remove[regions==i] = 0

    rtn[remove==1] = 0
    return rtn


def minimum_bounding_rectangle(verts):
    pi2 = np.pi/2.

    # compute rotation angles
    verts = np.array(verts)
    angles = np.empty(verts.shape[0])
    angles = np.arctan2(verts[:, 1], verts[:, 0])

    # compute rotations
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)
    ]).T
    rotations = rotations.reshape((-1, 2, 2))

    #apply rotations to the verts
    rot_points = np.dot(rotations, verts.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the smallest area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # extract the best points
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rtn = np.zeros((4,2))
    rtn[0,:] = np.dot([x1, y2], r)
    rtn[1,:] = np.dot([x2, y2], r)
    rtn[2,:] = np.dot([x2, y1], r)
    rtn[3,:] = np.dot([x1, y1], r)

    return rtn

def extract_variants(binary_image, rgb_image):

    bin = binary_image > 0
    s = 1 + 10000 * (bin.sum()/bin.size) ** 1.4
    s = max(5, 3 * np.log(s))
    structure = np.ones((s, s))
    bin = binary_opening(bin, structure=structure)
    rem = np.zeros(binary_image.shape + (3,))
    rem[bin > 0, :] = 255

    print('dilation')
    structure = np.ones((5, 5))
    dilated = np.zeros(binary_image.shape + (3,))
    dilated[binary_dilation(rem[:,:,0], iterations=40) > 0, : ] = 255

    print('dilation mask')
    mask1 = np.zeros(rgb_image.shape)
    mask1[dilated > 0] = rgb_image[dilated > 0]

    print('largest regions')
    largest = extract_largest_region(dilated[:,:,0])
    mask2 = np.zeros(rgb_image.shape)
    for i in xrange(3):
        mask2[largest > 0,i] = rgb_image[largest > 0,i]

    print('convex hull')
    hull =  convex_hull_mask(largest>0)
    mask3 = np.zeros(rgb_image.shape)
    for i in xrange(3):
        mask3[hull > 0,i] = rgb_image[hull > 0,i]

    print('minimal bounding rectangle')
    verts =  convex_hull_mask(largest>0, mask=False)
    verts = minimum_bounding_rectangle(verts)
    verts = [(verts[i,0], verts[i,1]) for i in range(4)]
    hull = mask_polygon(verts, binary_image.shape)
    mask4 = np.zeros(rgb_image.shape)
    for i in xrange(3):
        mask4[hull > 0,i] = rgb_image[hull > 0,i]

    return [rem, dilated, mask1, mask2, mask3, mask4]


from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

def threshold_and_filter(size=15):
    images = image_generator()

    for fn, im in images:
        print('inspecting image: ', fn)

        print('computing yen threshold')
        yen = threshold_yen(im)
        channels = threshold_by_channel(im ,yen)
        imgs = []
        chan = channels[0]
        imgs.append(chan)
        imgs.extend(extract_variants(chan[:,:,0], im))
        print('plotting')
        plot_images(
            [imgs[0],imgs[-1]],
            suptitle=fn.split('/')[-1]
        )
