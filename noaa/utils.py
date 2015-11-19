from __future__ import (print_function, division)

import os
import sys
import select
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###########################################################################
## local imports

from config import BASE_DIR

###########################################################################
## general

def prompt_for_quit_or_timeout(msg='Stop execution?', timeout=2):
    """Adapted from: http://stackoverflow.com/a/2904057/759442
    """
    print("%s (y/N) " % msg)
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        if sys.stdin.readline().strip() == 'y':
            return True

    return False

###########################################################################
## i/o

def lookup_whale_images(frame, whale_id='whale_78785',
                        source_path=os.path.join(BASE_DIR, 'data/imgs/'),
                        append_path=True):

    image_list = list(frame[frame['whaleID']==whale_id]['Image'].values)
    return [os.path.join(source_path, im) if append_path else im for im in image_list]


def load_images(image_list,
                source_path=os.path.join(BASE_DIR, 'data/imgs/'),
                use_path=False):

    is_list = type(image_list) is list
    image_list = image_list if is_list else [image_list]

    images = []
    for image in image_list:
        file = os.path.join(source_path, image) if use_path else image
        with open(file, 'rb') as f:
            images.append(np.asarray(Image.open(f)))

    return images if is_list else images[0]


###########################################################################
## plot

def plot_images(image_arrays, dims=None, titles=None, suptitle=None):
    if not isinstance(image_arrays, list):
        image_arrays=[image_arrays]
        if titles is not None:
            titles=[titles]

    n = len(image_arrays)
    if dims is None:
        width = int(np.sqrt(n))
        height = int(np.ceil(n/width))
    else:
        height, width = dims

    fig, axes = plt.subplots(
        nrows=height,
        ncols=width,
        figsize=(16 * height / (width + height), 16 * width / (width + height)),
        sharex=True,
        sharey=True,
        subplot_kw={'adjustable':'box-forced'}
    )
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    if width > 1 or height > 1:
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        print(image_arrays[i].shape)
        if i >= n:
            ax.imshow(np.zeros(image_arrays[0].shape).astype('uint8'))
        else:
            ax.imshow(image_arrays[i].astype('uint8'))
            if titles is not None: ax.set_title(titles[i])

        ax.axis('off')

    plt.show()
    plt.clf()


def plot_images_and_histograms(image_arrays, histograms,
                               image_titles=None, hist_titles=None,
                               image_suptitle=None, hist_suptitle=None):

    # plot images
    if not isinstance(image_arrays, list):
        image_arrays=[image_arrays]
        if image_titles is not None:
            image_titles=[image_titles]

    n_images = len(image_arrays)
    width = int(round(np.sqrt(n_images), 0))
    height = int(np.ceil(n_images/width))

    fig, axes = plt.subplots(
        nrows=height,
        ncols=width,
        figsize=(16 * height / (width + height), 16 * width / (width + height)),
        sharex=True,
        sharey=True,
        subplot_kw={'adjustable':'box-forced'}
    )
    if image_suptitle is not None:
        fig.suptitle(image_suptitle, fontsize=14)

    if width > 1 or height > 1:
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= n_images:
            ax.imshow(np.zeros(image_arrays[0].shape).astype('uint8'))
        else:
            ax.imshow(image_arrays[i].astype('uint8'))
            if image_titles is not None: ax.set_title(image_titles[i])

        ax.axis('off')

    if histograms is None:
        plt.show()

    # plot histograms
    plt.figure()
    if not isinstance(histograms, list):
        histograms=[histograms]
        if hist_titles is not None:
            hist_titles=[hist_titles]

    n_hists = len(histograms)
    width = int(round(np.sqrt(n_hists), 0))
    height = int(np.ceil(n_hists/width))

    fig, axes = plt.subplots(
        nrows=height,
        ncols=width,
        figsize=(16 * height / (width + height), 16 * width / (width + height)),
        sharex=True,
        sharey=True,
        subplot_kw={'adjustable':'box-forced'}
    )
    if image_suptitle is not None:
        fig.suptitle(image_suptitle, fontsize=14)

    if width > 1 or height > 1:
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= n_hists:
            break
        else:
            hist, bins = histograms[i]
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            ax.bar(center, hist, align='center', width=width)
            if hist_titles is not None: ax.set_title(hist_titles[i])



    plt.show()
    plt.clf()


def plot_whales_by_id(
        csv_path=os.path.join(BASE_DIR, 'data/train.csv'),
        img_path=os.path.join(BASE_DIR, 'data/imgs/'),
        whale_id='whale_78785'):

    whale_index = pd.read_csv(csv_path)
    image_file_names = lookup_whale_images(whale_index, whale_id=whale_id)
    image_arrays = load_images(image_file_names)
    plot_images(image_arrays, image_file_names)
