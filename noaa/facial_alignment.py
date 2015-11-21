from __future__ import (print_function, division)

import os
import json
import math
import itertools
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

from scipy import ndimage
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_opening
from scipy.ndimage import binary_closing
from scipy.spatial import ConvexHull

from skimage.transform import warp
from skimage.transform import SimilarityTransform
from skimage import io, color
from skimage.segmentation import slic
from skimage.filters import threshold_yen
import skimage.transform
import skimage.feature
from skimage import exposure

import sklearn.utils
from sklearn.cluster import KMeans

###########################################################################
## local imports

from noaa.utils import (lookup_whale_images, load_images,
                        plot_images, subplot_images,
                        plot_images_and_histograms,
                        plot_whales_by_id)
from config import BASE_DIR, SEED

###########################################################################
## i/o

def add_image_to_image_generator(images, file=None):
    if file is not None:
        path = os.path.join(BASE_DIR, 'data/imgs/')
        path = os.path.join(path, file)
        image = load_images(path).astype('int32')
        images = itertools.chain([(path, image)], images)

    return images


def binary2gray(binary_image):
    gray_image = np.zeros(binary_image.shape + (3,))
    for i in range(3):
        gray_image[binary_image > 0, i] = binary_image[binary_image > 0]
    return gray_image


def image_generator():
    csv_path=os.path.join(BASE_DIR, 'data/train.csv')
    img_path=os.path.join(BASE_DIR, 'data/imgs/')

    images = os.listdir(img_path)
    shuffle(images)
    for idx, im in enumerate(images):
        im = os.path.join(img_path, im)
        image = load_images([im])[0].astype('int32')
        yield im, image


###########################################################################
## numpy utils

def numpy_isin(array, keeps):
    rtn = np.zeros(array.shape)
    for k in keeps:
        rtn[array==k] = k

    return rtn


def extract_largest_regions(mask, num_regions=2):
    rtn = np.copy(mask)
    regions, n_labels = ndimage.label(mask)
    label_list = range(1, n_labels+1)
    sizes = []
    for label in label_list:
        size = (regions==label).sum()
        sizes.append((size, label))

    sizes = sorted(sizes, reverse=True)
    num_regions = min(num_regions, n_labels-1)
    min_size = sizes[num_regions][0]

    labels = []
    for size, label in sizes:
        if size < min_size:
            regions[regions==label] = 0
        else:
            labels.append(label)

    return regions, labels


def smallest_partition(data, channel):
    orig = data[:,:,channel].sum()
    invs = np.abs(255-data[:,:,channel]).sum()
    rtn = np.copy(data)
    if invs < orig:
        rtn[:,:,channel] = np.abs(rtn[:,:,channel] - 255)
        return rtn

    return rtn


def is_mad_outlier(points, thresh=3.5):
    # source: http://stackoverflow.com/a/22357811/759442
    if len(points.shape) == 1:
        points = points[:,None]

    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

###########################################################################
## masks

def mask_polygon(verts, shape):
    img = Image.new('L', shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T


def convex_hull_mask(data, mask=True):
    segm = np.argwhere(data)
    hull = ConvexHull(segm)
    verts = [(segm[v,0], segm[v,1]) for v in hull.vertices]
    if mask:
        return mask_polygon(verts, data.shape)

    return verts


def mask_largest_regions(mask, num_regions=2):
    rtn = np.copy(mask)
    regions, n_labels = ndimage.label(mask)
    label_list = range(1, n_labels+1)
    sizes = []
    for label in label_list:
        size = (regions==label).sum()
        sizes.append((size, label))

    sizes = sorted(sizes, reverse=True)
    num_regions = min(num_regions, n_labels-1)
    min_size = sizes[num_regions][0]

    remove = np.ones(regions.shape)
    for size, label in sizes:
        if size < min_size:
            remove[regions==label] = 0

    rtn[remove==1] = 0
    return rtn


###########################################################################
## transforms

def calculate_binary_opening_structure(binary_image, weight=1, hollow=False):
    s = 1 + 10000 * (binary_image.sum()/binary_image.size) ** 1.4
    s = int(max(5, 3 * np.log(s) * weight))

    if hollow:
        structure = np.ones((s, s))
        structure[s/4:3*s/4,s/4:3*s/4] = 0
        return structure

    return np.ones((s, s))




def compute_shape_regularity(data):
    vals = np.unique(data)
    d = dict()
    for val in vals:
        segm = data == val
        hull = convex_hull_mask(segm)
        d[val] = hull.sum()

    return d


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


def denoise_image(data, type=None):
    from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
    if type == 'tv':
        return denoise_tv_chambolle(data, weight=0.2, multichannel=True)

    return denoise_bilateral(data, sigma_range=0.1, sigma_spatial=15)



def detect_blobs(data_gray):
    from skimage.feature import blob_dog, blob_log, blob_doh
    from skimage.color import rgb2gray
    # takes grayscale
    blobs = blob_doh(data_gray, min_sigma=10, num_sigma=3, max_sigma=50, threshold=.1)
    mask = np.zeros(data_gray)
    n, m = mask.shape
    for blob in blobs:
        y,x = np.ogrid[-a:n-a, -b:m-b]
        mask = mask * (x*x + y*y <= r*r)

    return mask


def segment_image_slic(image, retain=None, n_segments=20, compactness=20, sigma=5):
        segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
        if retain is not None:
            segments_regularity_map = compute_shape_regularity(segments)
            segments_to_keep = detect_regularity_outliers(segments_regularity_map, retain=retain)
            retained_segments = numpy_isin(segments, segments_to_keep)
            return retained_segments

        return segments


def segmentation(retain=3, n_segments=20, compactness=20, sigma=0):
    images = image_generator()

    for im in images:
        retained_segments = segment_image_slic(image, retain=retain, n_segments=n_segments, compactness=compactness, sigma=sigma)
        plot_images([image, retained_segments], ['Original', 'Segmented'])


def threshold_by_channel(image, threshold, n_channels=3):
    rtn = []
    for i in xrange(n_channels):
        ch = np.zeros(image.shape)
        ch[image[:,:,i] > threshold, :] = 255
        ch = smallest_partition(ch, i)
        rtn.append(ch)

    return rtn


###########################################################################
## cropping

def minimum_bounding_rectangle(verts, return_rotation_matrix=False, return_rotation_angle=False):
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
    angle = angles[best_idx]

    rtn = np.zeros((4,2))
    rtn[0,:] = np.dot([x1, y2], r)
    rtn[1,:] = np.dot([x2, y2], r)
    rtn[2,:] = np.dot([x2, y1], r)
    rtn[3,:] = np.dot([x1, y1], r)
    rtn = tuple(tuple(pt) for pt in rtn.tolist())

    r = np.linalg.inv(r)
    angle = math.atan2(r[1,0], r[0,0])
    r = tuple(tuple(el) for el in r.tolist())
    return rtn, r, angle


def similarity_transform_from_mbr(verts, rot, angle, shape):
    verts = np.array(verts)
    rot = np.array(rot)
    verts = np.dot(verts, rot).astype(int)
    inv_rot = np.linalg.inv(rot)

    # translation
    xtrans = np.abs(min(0,np.min(verts[:,0])))
    xtrans = xtrans if xtrans > 0 else shape[0] - max(np.max(verts[:,0]), shape[0])
    ytrans = np.abs(min(0,np.min(verts[:,1])))
    ytrans = ytrans if ytrans > 0 else shape[1] - max(np.max(verts[:,1]), shape[1])
    translation = -1*np.dot(np.array([xtrans, ytrans]), rot.T).astype(int)

    # crop window
    xmax = np.max(verts[:,0]) + xtrans
    xmin = np.min(verts[:,0]) + xtrans
    ymax = np.max(verts[:,1]) + ytrans
    ymin = np.min(verts[:,1]) + ytrans

    # transform matrix
    transform = np.eye(3)
    transform[0:2,0:2] = rot.T
    transform[0:2, 2] = [translation[1], translation[0]]

    # return transform and window
    transform = SimilarityTransform(matrix=transform)
    corners = [xmin, xmax, ymin, ymax]
    return transform, corners


def rotate_crop_gray_image_from_mbr(image, verts, rot, angle):
    transform, corners = similarity_transform_from_mbr(verts, rot, angle, image.shape)
    xmin, xmax, ymin, ymax = corners

    crop = warp(image, transform)
    crop = crop[xmin:xmax, ymin:ymax]

    return crop.astype('uint8')


def rotate_crop_rgb_image_from_mbr(image, verts, rot, angle):
    rgb_crop = None
    shape = image.shape[:-1]
    for i in range(3):
        chan = np.zeros(shape)
        chan[:,:] = image[:,:,i]
        chan_crop = rotate_crop_gray_image_from_mbr(chan, verts, rot, angle)
        if rgb_crop is None:
            rgb_crop = np.zeros(chan_crop.shape + (3,), dtype='uint8')

        rgb_crop[:,:,i] = chan_crop

    return rgb_crop


###########################################################################
## color reduction


def segment_image(image, n_segments=400, compactness=30, sigma=5, verbose=True):
    if verbose:
        print('segmenting image')
        print('n_segments=%d, compactness=%d, sigma=%d' % (n_segments, compactness, sigma))

    image = image.astype('float64')
    labels = slic(image, compactness=compactness, n_segments=n_segments, sigma=sigma)
    segmented = color.label2rgb(labels, image, kind='avg')
    return  segmented.astype('uint8')


def extract_colors_grays(image, n_grays=4, verbose=True):
    if verbose:
        print('extracting grays')
        print('n_grays=%d' % (n_grays, ))

    colors = pd.DataFrame(image.reshape(-1,3)).drop_duplicates().values
    inv_grayness = colors.std(axis=1)
    brightness = colors.mean(axis=1)

    grayest_colors = colors[inv_grayness.argsort()[:n_grays]]

    darkest_gray = grayest_colors[grayest_colors.mean(axis=1).argsort()[0]]
    mid_gray = grayest_colors[grayest_colors.mean(axis=1).argsort()[int(round(n_grays/2))]]
    brightest_gray = grayest_colors[grayest_colors.mean(axis=1).argsort()[1]]


    return darkest_gray, mid_gray, brightest_gray


def extract_colors_brightness(image, thresh=3.5, verbose=True):
    if verbose:
        print('extracting intensities')
        print('outlier threshold=%f' % (thresh, ))

    colors = pd.DataFrame(image.reshape(-1,3)).drop_duplicates().values
    inv_grayness = colors.std(axis=1)
    brightness = colors.mean(axis=1)

    is_outlier = is_mad_outlier(brightness, thresh=thresh)
    brightest_colors = colors[is_outlier]

    least_gray = brightest_colors[np.newaxis, brightest_colors.std(axis=1).argsort()[-1]]

    return brightest_colors


def extract_brightest_colors(image, thresh=3.5, verbose=True):
    if verbose:
        print('extracting intensities')
        print('outlier threshold=%f' % (thresh, ))

    colors = pd.DataFrame(image.reshape(-1,3)).drop_duplicates().values
    brightness = colors.mean(axis=1)
    is_outlier = is_mad_outlier(brightness, thresh=thresh)
    brightest_colors = colors[is_outlier]

    return brightest_colors


def extract_brightest_reds(image, thresh=100, verbose=True):
    if verbose:
        print('extracting reds')
        print('threshold=%f' % (thresh, ))

    image = image.astype('uint8')
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    lower_red = np.array([0,thresh,0])
    upper_red = np.array([255,255,255])
    mask = cv2.inRange(ycrcb, lower_red, upper_red)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res.astype('uint8')


def list_colors_exclude_black(image):
    colors = pd.DataFrame(image.reshape(-1,3)).drop_duplicates().values
    colors = colors[np.logical_and.reduce(colors[:,:] > 10, axis=1),:]

    return colors


def select_colors(image, colors):
    rtn = np.zeros_like(image)
    for color in list(colors):
        rtn[np.logical_and.reduce(image[:,:] == color, axis=2),:] = color

    return rtn


def quantize_colors(image, n_colors=3, n_samples=1000,
                   max_iter=300, n_init=10, n_jobs=1,
                   random_state=SEED, verbose=False, split=True):
    if verbose:
        print('color quantizing image')
        print('n_colors=%d, n_samples=%d, n_init=%d' % (n_colors, n_samples, n_init))

    w, h, d = image.shape; assert(d == 3)
    pixels = image.reshape(w*h, d).astype('float64')
    if n_samples is not None:
        pixel_sample = sklearn.utils.shuffle(pixels, random_state=SEED)[:n_samples]
    else:
        pixel_sample = pixels

    kmeans = KMeans(n_clusters=n_colors, max_iter=max_iter, n_init=n_init, n_jobs=n_jobs, random_state=random_state).fit(pixel_sample)
    labels = kmeans.predict(pixels)
    if not split:
        quantized = np.zeros(image.shape, dtype='float64')
        colors = kmeans.cluster_centers_.tolist()
        label_idx = 0
        for i in range(w):
            for j in range(h):
                quantized[i][j] = colors[labels[label_idx]]
                label_idx += 1

        quantized = quantized.astype('uint8')
    else:
        # TODO: debug this
        quantized = [np.zeros(image.shape[:-1], dtype='float64')] * n_colors
        colors = kmeans.cluster_centers_.tolist()
        label_idx = 0
        for i in range(w):
            for j in range(h):
                c = colors.index(colors[labels[label_idx]])
                quantized[c][i][j] = 255
                label_idx += 1

        quantized = [binary2gray(im.astype('uint8')) for im in quantized]

    return quantized


###########################################################################
## gabor filters

def build_gabor_filters():
    filters = []
    ksize = 31
    for theta in (np.pi * np.array(range(3)) / 3).tolist():
        for lambd in [0.1, 0.2, 0.4]:
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)

    return filters


def apply_gabor_filters(image, filters):
    results = []
    for kernel in filters:
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        results.append(filtered)

    return results


###########################################################################
## examine feature

def inspect_gabor_patterns(file=None, dilation_iterations=40, num_regions=4):
    images = image_generator()
    if file is not None:
        images = add_image_to_image_generator(images, file)

    gabor_filters = build_gabor_filters()
    for fn, im in images:
        image_arrays = []
        titles = []

        # plot original image
        titles.append('Original Image')
        image_arrays.append(im)

        # segmented image
        segmented = segment_image(im, n_segments=20, compactness=20, sigma=2)
        titles.append('Segmented Image')
        image_arrays.append(segmented)

        # re-segment image
        brightest_colors = extract_brightest_colors(segmented, thresh=3.5)
        bright_regions = select_colors(segmented, brightest_colors)
        resegmented = segmented.copy()
        resegmented[bright_regions <= 0] = 0
        resegmented = segment_image(resegmented, n_segments=5, compactness=30, sigma=2)
        titles.append('Re-Segmented Image')
        image_arrays.append(resegmented)

        colors = list_colors_exclude_black(segmented)[[0, 4, 8, 12, 16]]
        rgb_crops = []
        rgb_labels = []
        crop_titles = []
        for color in list(colors):
            mask = select_colors(segmented, [color]) > 0
            x, y = ndimage.measurements.center_of_mass((mask[:,:,0] > 0).astype(int))
            xmin = int(max(x-50,0))
            xmax = int(min(x+50, im.shape[0]-1))
            ymin = int(max(y-50,0))
            ymax = int(min(y+50, im.shape[1]-1))
            rgb_crops.append(im[xmin:xmax,ymin:ymax, :].copy())
            rgb_labels.append(str(color))
            crop_titles.append('Crop for Color ' + str(color))

        subplot_images(
            rgb_crops,
            titles=crop_titles,
            suptitle='Crops'
        )

        for crop, color in zip(rgb_crops, rgb_labels):
            rgb_gabor = apply_gabor_filters(crop, gabor_filters)
            subplot_images(rgb_gabor, suptitle='RGB Gabor Filters for Color ' + color)

        subplot_images(
            image_arrays,
            titles=titles,
            suptitle=fn.split('/')[-1],
            show_plot=True
        )


def yen_mask_and_normalize(file=None):
    images = image_generator()
    if file is not None:
        images = add_image_to_image_generator(images, file)

    for fn, im in images:
        image_arrays = []
        titles = []

        # plot original image
        titles.append('Original')
        image_arrays.append(im)

        # thresholded
        yen = threshold_yen(im)
        thresholded = threshold_by_channel(im, yen)[0]
        titles.append('Yen Thresholded')
        image_arrays.append(thresholded)

        # masked and equalized
        masked = im.copy()
        masked[thresholded<=0] = 0

        titles.append('Masked')
        image_arrays.append(masked)

        # equalize
        from skimage.util import img_as_ubyte
        equalized = masked.copy()
        for chan in xrange(3):
            img = masked[:,:,chan].astype('uint8')
            range = (200, 255) if chan == 0 else (100, 155) if chan == 1 else (0, 55)
            rescaled = exposure.rescale_intensity(img, in_range=range)
            equalized[:,:,chan] = rescaled
            break

        titles.append('Equalied')
        image_arrays.append(equalized)

        # plot
        subplot_images(
            image_arrays,
            titles=titles,
            show_plot=True,
            suptitle=fn.split('/')[-1]
        )
        continue


def inspect_color_quantization(file=None):
    images = image_generator()
    if file is not None:
        images = add_image_to_image_generator(images, file)

    for fn, im in images:
        image_arrays = []
        titles = []

        # plot original image
        titles.append('Original')
        image_arrays.append(im)

        yen = threshold_yen(im)
        thresholded = threshold_by_channel(im, yen)[0]
        titles.append('Yen Thresholded')
        image_arrays.append(thresholded)

        # equalize
        equalized = np.zeros_like(im)
        for chan in xrange(3):
            equalized[:,:,chan] = (exposure.equalize_hist(im[:,:,chan]) * 255).astype('uint8')

        titles.append('Equalized')
        image_arrays.append(equalized)

        # experiments point to a reasonable threshold of 10 or 20
        thresh = 160
        reds_ycbcr = extract_brightest_reds(equalized, thresh=thresh, verbose=False)
        titles.append('Reds Thresholded YCBCR %d' % thresh)
        image_arrays.append(reds_ycbcr)

        smoothed_ycbcr = reds_ycbcr.copy()
        smoothed_ycbcr[:,:,0] = binary_closing(smoothed_ycbcr[:,:,0]>0, iterations=3) * 255
        mask = (np.amax(smoothed_ycbcr, axis=2) > 0)
        structure = calculate_binary_opening_structure(mask, weight=1, hollow=False)
        mask = binary_opening(mask, structure=structure, iterations=1)
        smoothed_ycbcr[mask <= 0,:] = 0

        titles.append('Smoothed YCBCR Reds')
        image_arrays.append(smoothed_ycbcr)

        # plot
        subplot_images(
            image_arrays,
            titles=titles,
            show_plot=True,
            suptitle=fn.split('/')[-1]
        )
        continue

        # segmented image
        segmented = segment_image(smoothed_ycbcr, n_segments=3, compactness=100, sigma=2)
        titles.append('Segmented')
        image_arrays.append(segmented)

        titles.append('Least Gray Bright Region')
        image_arrays.append(bright)

        d, m, b = extract_colors_grays(segmented, n_grays=6)
        grays = select_colors(segmented, [d, m, b])

        titles.append('Gray Regions of Varied Intensity')
        image_arrays.append(grays)

        quantized = quantize_colors(segmented, n_colors=3, n_samples=1000,
                                    max_iter=300, n_init=10, n_jobs=1,
                                    random_state=SEED, verbose=True, split=False)

        titles.append('Quantized Image %d Colors' % 3)
        image_arrays.append(quantized)

        quantized = quantize_colors(segmented, n_colors=4, n_samples=1000,
                                    max_iter=300, n_init=10, n_jobs=1,
                                    random_state=SEED, verbose=True, split=False)

        titles.append('Quantized Image %d Colors' % 4)
        image_arrays.append(quantized)

        quantized = quantize_colors(segmented, n_colors=5, n_samples=1000,
                                    max_iter=300, n_init=10, n_jobs=1,
                                    random_state=SEED, verbose=True, split=False)

        titles.append('Quantized Image %d Colors' % 5)
        image_arrays.append(quantized)

        subplot_images(
            image_arrays,
            titles=titles,
            show_plot=True,
            suptitle=fn.split('/')[-1]
        )


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
            titles=titles,
            suptitle=fn.split('/')[-1]
        )


def inspect_glare_patterns(image=None):
    images = image_generator()
    if image is not None:
        img_path=os.path.join(BASE_DIR, 'data/imgs/')
        fn = os.path.join(img_path, image)
        im = load_images([fn])[0].astype('int32')
        images = itertools.chain([(fn, im)], images)

    for fn, im in images:
        imgs = [im]
        titles = ['Original']
        print('computing yen threshold')
        yen = threshold_yen(im)
        channels = threshold_by_channel(im ,yen)
        titles.append('Yen Thresholded')
        imgs.append(channels[0])

        for cp in [175, 200, 225, 250]:
            titles.append('Cutpoint %d' % cp)
            print('cutting')
            rtn = np.zeros(im.shape)
            rtn[np.average(im, axis=2, weights=[.7, .1, .2]) > cp,0] = 255
            for i in xrange(1,3):
                rtn[:, :, i] = rtn[:, :, 0]

            imgs.append(rtn)

        print('plotting')
        plot_images(
            imgs,
            titles=titles,
            suptitle=fn.split('/')[-1]
        )


def generate_masks(binary_image, rgb_image):

    bin = binary_image > 0
    structure = calculate_binary_opening_structure(bin)
    bin = binary_opening(bin, structure=structure)
    rem = np.zeros(binary_image.shape + (3,))
    rem[bin > 0, :] = 255

    print('dilation')
    strutcure = np.ones((5, 5))
    dilated = np.zeros(binary_image.shape + (3,))
    dilated[binary_dilation(rem[:,:,0], iterations=40) > 0, : ] = 255

    print('dilation mask')
    mask1 = np.zeros(rgb_image.shape)
    mask1[dilated > 0] = rgb_image[dilated > 0]

    print('largest regions')
    largest = mask_largest_regions(dilated[:,:,0])
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
    verts, _, _ = minimum_bounding_rectangle(verts)
    verts = [(verts[i,0], verts[i,1]) for i in range(4)]
    hull = mask_polygon(verts, binary_image.shape)
    mask4 = np.zeros(rgb_image.shape)
    for i in xrange(3):
        mask4[hull > 0,i] = rgb_image[hull > 0,i]

    return [rem, dilated, mask1, mask2, mask3, mask4]


def threshold_and_generate_objects(size=15):
    images = image_generator()

    for fn, im in images:
        print('inspecting image: ', fn)

        print('computing yen threshold')
        yen = threshold_yen(im)
        channels = threshold_by_channel(im ,yen)
        imgs = []
        chan = channels[0]
        imgs.append(chan)
        imgs.extend(generate_masks(chan[:,:,0], im))
        print('plotting')
        plot_images(
            [imgs[0],imgs[-1]],
            suptitle=fn.split('/')[-1]
        )



def inspect_local_binary_patterns(file=None, dilation_iterations=40, num_regions=5, radius=2):
    from skimage.transform import warp
    from skimage.transform import SimilarityTransform
    images = image_generator()
    images = add_image_to_image_generator(images, file)
    n_points = 8*radius

    for fn, im in images:
        image_arrays = []
        histograms = []
        image_titles = []
        hist_titles = []

        image_arrays.append(im)
        image_titles.append('original image')

        yen = threshold_yen(im)
        yen_channels = threshold_by_channel(im, yen)
        image_titles.append('Yen Thresholded Channel 1')
        image_arrays.append(yen_channels[0])

        binary_image = yen_channels[0][:,:,0] > 0
        structure = calculate_binary_opening_structure(binary_image)
        binary_image = binary_opening(binary_image, structure=structure)
        binary_image = binary_dilation(binary_image, iterations=dilation_iterations)
        regions, labels = extract_largest_regions(binary_image, num_regions=num_regions)

        for label in labels:
            print('Cropping region %d' % label)
            region = np.zeros(yen_channels[0].shape[:-1])
            region[regions == label] = yen_channels[0][regions==label,0]

            # convex hull
            verts = convex_hull_mask(region>0, mask=False)
            corners, rot, angle  = minimum_bounding_rectangle(verts)

            yen_crop = rotate_crop_gray_image_from_mbr(region, corners, rot, angle)
            image_titles.append('Yen Region %d Crop' % label)
            image_arrays.append(yen_crop)

            rgb_crop = rotate_crop_rgb_image_from_mbr(im, corners, rot, angle)
            image_titles.append('RGB Crop %d' % label)
            image_arrays.append(rgb_crop)


        plot_images_and_histograms(
            image_arrays=image_arrays,
            histograms=histograms,
            image_titles=image_titles,
            hist_titles=hist_titles,
            image_suptitle=fn.split('/')[-1],
            hist_suptitle=fn.split('/')[-1]
        )
