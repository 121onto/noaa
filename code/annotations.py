from __future__ import (print_function, division)

import os
import json
import math
import itertools
import numpy as np
from random import shuffle
from PIL import Image
import cv2

from config import BASE_DIR

###########################################################################
## annotations markup

def modify_sloth_head_annotations():
    # original file from Kaggle user Vinh Nguyen
    a_path = os.path.join(BASE_DIR, 'data/head_annotations.json')
    img_path = os.path.join(BASE_DIR, 'data/imgs')

    annotations = []
    with open(a_path, 'r') as i:
        annotations = json.load(i)

    for i, a in enumerate(annotations):
        f = a['filename']
        f = f.split('/')[-1]
        annotations[i]['filename'] = os.path.join(img_path, f)

    with open(a_path, 'w') as o:
        json.dump(annotations, o, indent=4, sort_keys=True)


def write_opencv_info_files():
    heads_p_path = os.path.join(BASE_DIR, 'data/heads_p')
    heads_n_path = os.path.join(BASE_DIR, 'data/heads_n')
    ex_path = os.path.join(BASE_DIR, 'data/head_examples.info')
    bg_path = os.path.join(BASE_DIR, 'data/head_backgrounds.info')

    with open(ex_path, 'w') as file:
        images = os.listdir(heads_p_path)
        shuffle(images)
        for idx, im in enumerate(images):
            if idx >= 2000:
                break

            if im.endswith('.jpg'):
                print(os.path.join(heads_p_path, im) + '\t1\t0 0 256 256', file=file)


    with open(bg_path, 'w') as file:
        images = os.listdir(heads_n_path)
        shuffle(images)
        for idx, im in enumerate(images):
            if idx >= 8000:
                break

            if im.endswith('.jpg'):
                print(os.path.join(heads_n_path, im), file=file)


###########################################################################
## geometry

def smallest_enclosing_square(pp):
    pw, ph = pp['width'], pp['height']
    x, y = pp['x'], pp['y']
    jumpw, jumph = 0, 0

    if pw > ph:
        jumph = int(math.ceil((pw - ph) / 2))
    else:
        jumpw = int(math.ceil((ph - pw) / 2))

    x, y = x - jumpw, y - jumph
    pw, ph = pw + jumpw, ph + jumph
    return dict(x=x, y=y, width=pw, height=ph)


def get_adjacent_tiles(pp, width, height):
    pw, ph = pp['width'], pp['height']
    x, y = pp['x'], pp['y']

    xlb = max(x - pw, 0)
    xub = min(x + 2 * pw, width)
    ylb = max(y - ph, 0)
    yub = min(y + 2 * ph, height)

    xvals = [xlb, x, x + pw, xub]
    yvals = [ylb, y, y + ph, yub]

    grid = list(itertools.product(xvals, yvals))
    grid = np.array(grid).reshape(4,4,2)
    pp_neg = [
        dict(
            x=grid[i,j,0],
            y=grid[i,j,1],
            width=grid[i+1,j,0] - grid[i,j,0],
            height=grid[i,j+1,1] - grid[i,j,1]
        )
        for i,j in itertools.product(xrange(3), xrange(3))
        if not i == j == 1
    ]
    return pp_neg


def bounding_box_from_points(pp):
    left, upper = pp['x'], pp['y']
    right, lower = pp['x'] + pp['width'], pp['y'] + pp['height']
    return left, upper, right, lower


###########################################################################
## images

def crop_resize_image(file, box, mw=256, mh=256):
    # crop image (may not be a square)
    im = Image.open(file)
    thumb = im.crop(box)
    thumb.thumbnail((mw, mh), Image.ANTIALIAS)

    # create a new image of size mw, mh with a blank background
    bg = (255, 255, 255)
    cropped = Image.new(thumb.mode, (mw, mh), bg)

    # paste the thumbnail centered against the background
    thumb_width, thumb_height = thumb.size
    x1 = int(math.ceil(math.ceil((mw - thumb_width) / 2)))
    y1 = int(math.ceil(math.ceil((mh - thumb_height) / 2)))
    cropped.paste(thumb, (x1, y1, x1 + thumb_width, y1 + thumb_height))
    return cropped


def generate_examples_from_head_annotations():
    # TODO: breaks with IndexError: list index out of range after cropping 2928 heads
    # Adapted from: https://github.com/Smerity/right_whale_hunt
    a_path = os.path.join(BASE_DIR, 'data/head_annotations.json')
    heads_p_path = os.path.join(BASE_DIR, 'data/heads_p')
    heads_n_path = os.path.join(BASE_DIR, 'data/heads_n')

    annotations = []
    with open(a_path, 'r') as i:
        annotations = json.load(i)

    for a in annotations:
        f = a['filename']
        pp = [x for x in a['annotations']][0]

        # crop and save positive examples
        pp = smallest_enclosing_square(pp)
        box = map(int, bounding_box_from_points(pp))
        cropped = crop_resize_image(f, box)
        path = os.path.join(heads_p_path, f.split('/')[-1])
        cropped.save(path)

        # generate negative examples
        with Image.open(f) as im:
            width, height = im.size

        pp_negs = get_adjacent_tiles(pp, width, height)

        # crop and save negative examples
        for j, pp in enumerate(pp_negs):
            pp = smallest_enclosing_square(pp)
            box = map(int, bounding_box_from_points(pp))

            cropped = crop_resize_image(f, box)
            path = os.path.join(heads_n_path, f.split('/')[-1].split('.')[0] + ('_%d.jpg' % j))
            cropped.save(path)


def predict_crop_heads_from_cascade():
    cascade_path = os.path.join(BASE_DIR, 'data/heads/cascade.xml')
    heads_path = os.path.join(BASE_DIR, 'data/heads')
    img_path = os.path.join(BASE_DIR, 'data/imgs')

    cascade = cv2.CascadeClassifier(cascade_path)
    images = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg')]
    for in_file in images:
        img = cv2.imread(in_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop = cascade.detectMultiScale(gray)[0]
        head = img[crop[1]:crop[3],crop[0]:crop[2]]
        out_file = os.path.join(heads_path, file)
        cv2.imwrite(out_file, head)


if __name__ == '__main__':
    modify_head_annotations()
    generate_examples_from_head_annotations()
    write_opencv_info_files()
