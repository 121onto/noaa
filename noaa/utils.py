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

    images = []
    for image in image_list:
        file = os.path.join(source_path, image) if use_path else image
        with open(file, 'rb') as f:
            images.append(np.asarray(Image.open(f)))

    return images


###########################################################################
## plot

def plot_images(image_arrays, dims=None, image_names=None, suptitle=None):
    if not isinstance(image_arrays, list):
        image_arrays=[image_arrays]
        if image_names is not None:
            image_names=[image_names]

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
    if width > 1 or height > 1:
        axes = axes.ravel()
    else:
        axes = [axes]
    for i, ax in enumerate(axes):
        if i >= n:
            break
        ax.imshow(image_arrays[i].astype('uint8'))
        if image_names is not None:
            ax.set_title(image_names[i])
        ax.axis('off')

    #fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

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
