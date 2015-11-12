from __future__ import (print_function, division)

import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
import skimage.util
from numpy.random import RandomState
import math
import scipy

###########################################################################
## local imports and config

from apply_parallel import apply_parallel
rng = RandomState()

###########################################################################
## i/o

def lookup_whale_images(frame, whale_id='whale_78785', source_path='imgs/', append_path=True):
    image_list = list(frame[frame['whaleID']==whale_id]['Image'].values)
    return [os.path.join(source_path, im) if append_path else im for im in image_list]


def load_images(image_list, source_path='imgs/', use_path=False):
    images = []
    for image in image_list:
        file = os.path.join(source_path, image) if use_path else image
        with open(file, 'rb') as f:
            images.append(np.asarray(Image.open(f)))

    return images


###########################################################################
## plot

def display_images(image_arrays, image_names):
    if not isinstance(image_arrays, list):
        image_arrays=[image_arrays]
        image_names=[image_names]

    n = len(image_arrays)
    height = int(np.sqrt(n))
    width = int(np.ceil(n/height))
    fig, axes = plt.subplots(
        nrows=height,
        ncols=width,
        figsize=(8, 8),
        sharex=True,
        sharey=True,
        subplot_kw={'adjustable':'box-forced'}
    )
    if n > 1:
        axes = axes.ravel()
    else:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(image_arrays[i].astype('uint8'))
        ax.set_title(image_names[i])
        ax.axis('off')

    #fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()
    plt.clf()


def display_whales_by_id(csv_path='train.csv', img_path='imgs/', while_id='whale_78785'):
    whale_index = pd.read_csv(csv_path)
    image_file_names = lookup_whale_images(whale_index, whale_id=whale_id)
    image_arrays = load_images(image_file_names)
    display_images(image_arrays, image_file_names)


###########################################################################
## transform

def resize_images(image_arrays, size=300):
    n = len(image_arrays)
    rtn_array = np.zeros((n, 3 * size * size))
    for i, image in enumerate(image_arrays):
        crop_idx = np.argmax(image.shape)
        scale_idx  = 0 if crop_idx else 1

        # scale
        scale = [0, 0]
        scale[scale_idx] = size
        scale[crop_idx] = int((image.shape[crop_idx] / image.shape[scale_idx]) * size)
        transformed = scipy.misc.imresize(image, scale)

        # crop
        crop_width = (scale[crop_idx] - size) / 2.0
        crop_width = tuple([
            (int(np.floor(crop_width)), int(np.ceil(crop_width)))
            if j == crop_idx else
            (0, 0)
            for j in range(3)
        ])
        transformed = skimage.util.crop(transformed, crop_width=crop_width)
        # display_images([transformed],['transformed image'])

        # stash
        rtn_array[i, :] = transformed.transpose(2, 0, 1).flatten()

    return rtn_array


def transform_images(image_array, shape=(3, 300, 300)):

    def f(image):
        rotation_angle = rng.uniform(low=-180, high=180)
        x_shift = rng.uniform(low=-5, high=5)
        y_shift = rng.uniform(low=-5, high=5)
        scale = 1/1.3
        tr_scale_trans = tf.SimilarityTransform(
            scale=scale,
            translation=(x_shift, y_shift)
        )
        image = image[0].reshape(shape).transpose(1, 2, 0)
        transformed = tf.rotate(image, angle=rotation_angle)
        transformed = tf.warp(transformed, tr_scale_trans)
        transformed = transformed.transpose(2, 0, 1).flatten()
        return np.expand_dims(transformed, axis=0)

    transformed_array = apply_parallel(f, image_array, chunks=(1, ) + image_array.shape[1:])
    return transformed_array

###########################################################################
## main

def main(csv_path='train.csv', img_path='imgs/'):
    #    csv_path='train.csv'; img_path='imgs/'
    whale_index = pd.read_csv(csv_path)
    image_file_names = lookup_whale_images(whale_index, whale_id='whale_78785')
    image_arrays = load_images(image_file_names)

    image_file = image_file_names[0]
    image = image_arrays[0]

    resized_array = resize_images(image_arrays)
    transformed_array = transform_images(resized_array)

    t_im_1 = transformed_array[0].reshape((3, 300, 300)).transpose(1, 2, 0)
    t_im_2 = transformed_array[1].reshape((3, 300, 300)).transpose(1, 2, 0)
    display_images([t_im_1, t_im_2],['transformed image 1','transformed image 2'])


if __name__ == '__main__':
    main()
