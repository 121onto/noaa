from __future__ import (print_function, division)

import os
import pandas as pd
import numpy as np
from skimage import transform as tf
import skimage.util
from numpy.random import RandomState
import scipy

###########################################################################
## local imports

from apply_parallel import apply_parallel
from config import BASE_DIR, SEED

###########################################################################
## local imports

rng = RandomState(SEED)

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

from utils import (lookup_whale_images, load_images, plot_images)

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
