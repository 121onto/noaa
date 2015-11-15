from __future__ import (print_function, division)

import os
import operator
import pandas as pd
import numpy as np
from skimage import transform as tf
import skimage.util
from numpy.random import RandomState
import scipy

###########################################################################
## local imports

from apply_parallel import apply_parallel
from preproc import load_pca
from config import BASE_DIR, SEED

###########################################################################
## config

rng = RandomState(SEED)
pca_val, pca_vec = load_pca()

###########################################################################
## geometric transforms

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


        # stash
        rtn_array[i, :] = transformed.transpose(2, 0, 1).flatten()

    return rtn_array


def transform_images(image_array, shape=(3, 300, 300)):

    def f(image):
        # initialization
        rotation_angle = rng.uniform(low=-180, high=180)
        x_shift = rng.uniform(low=-5, high=5)
        y_shift = rng.uniform(low=-5, high=5)
        scale = 1/1.3
        tr_scale_trans = tf.SimilarityTransform(
            scale=scale,
            translation=(x_shift, y_shift)
        )
        image = image[0].reshape(shape).transpose(1, 2, 0)

        # color transformation
        trans = np.multiply(
            pca_val,
            np.array([
                rng.normal(loc=0.0, scale=0.02),
                rng.normal(loc=0.0, scale=0.02),
                rng.normal(loc=0.0, scale=0.02)
            ])
        )
        trans = np.dot(pca_vec, trans)
        image = image + trans

        # geometric transformation
        transformed = tf.rotate(image, angle=rotation_angle)
        transformed = tf.warp(transformed, tr_scale_trans)
        transformed = transformed.transpose(2, 0, 1).flatten()
        return np.expand_dims(transformed, axis=0)

    transformed_array = apply_parallel(f, image_array, chunks=(1, ) + image_array.shape[1:])
    return transformed_array.astype('float32')


def equalize_image(image_array):
    from PIL import Image
    im = Image.fromarray(np.uint8(image_array))
    h = im.convert("L").histogram()
    lut = []
    for b in range(0, len(h), 256):
        # step size
        step = reduce(operator.add, h[b:b+256]) / 255
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]

    # map image through lookup table
    return np.asarray(im.point(lut*3))

###########################################################################
## main

from utils import (lookup_whale_images, load_images, plot_images, plot_whales_by_id)

csv_path=os.path.join(BASE_DIR, 'data/train.csv')
img_path=os.path.join(BASE_DIR, 'data/imgs/')

def main(csv_path=os.path.join(BASE_DIR, 'data/train.csv'),
         img_path=os.path.join(BASE_DIR, 'data/imgs/')):

    whale_index = pd.read_csv(csv_path)
    image_file_names = lookup_whale_images(whale_index, whale_id='whale_78785')
    image_arrays = load_images(image_file_names)

    image_file = image_file_names[0]
    image = image_arrays[0]

    resized_array = resize_images(image_arrays)
    transformed_array = transform_images(resized_array)

    t_im_1 = transformed_array[0].reshape((3, 300, 300)).transpose(1, 2, 0)
    t_im_2 = transformed_array[1].reshape((3, 300, 300)).transpose(1, 2, 0)
    plot_images([t_im_1, t_im_2],['transformed image 1','transformed image 2'])


if __name__ == '__main__':
    main()
