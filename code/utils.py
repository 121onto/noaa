from __future__ import (print_function, division)

import os
import sys
import gzip
import select
import cPickle
import random
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from PIL import Image
from sklearn.utils import gen_batches

###########################################################################
## config and local imports

from preproc import IncrementalZCA
SEED = 1234

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
## preproc

def build_submission_stub(csv_path='../data/train.csv', img_path='../data/imgs-proc/',
                    out_path='../output/submission_stub.csv'):

    train = pd.read_csv(csv_path)
    train_files = pd.unique(train['Image'].values)
    train['whaleID'] = train['whaleID'].astype('category')
    train['whaleCode'] = train['whaleID'].cat.codes
    labels_dict = dict(zip(train.whaleCode, train.whaleID))

    submission_stub = []
    for idx, file in enumerate(sorted(os.listdir(img_path))):
        if not file.startswith('w_'):
            continue
        if file.endswith('.jpg'):
            if file not in train_files:
                submission_stub.append(file)

    frame = pd.DataFrame(submission_stub, columns=['Image'])
    return labels_dict, frame


def build_memmap_arrays(csv_path='../data/train.csv', img_path='../data/imgs-proc/',
                    out_path='../data/memmap/', image_size=3*300*300):

    train = pd.read_csv(csv_path)
    train['whaleID'] = train['whaleID'].astype('category')
    train['whaleID'] = train['whaleID'].cat.codes
    labels_dict = dict(zip(train.Image, train.whaleID))

    tn_x_path = os.path.join(out_path, 'tn_x.dat')
    tn_y_path = os.path.join(out_path, 'tn_y.dat')
    v_x_path = os.path.join(out_path, 'v_x.dat')
    v_y_path = os.path.join(out_path, 'v_y.dat')
    tt_x_path = os.path.join(out_path, 'tt_x.dat')
    tt_y_path = os.path.join(out_path, 'tt_y.dat')

    tn_x = np.memmap(tn_x_path, dtype=theano.config.floatX, mode='w+', shape=(4044,image_size))
    tn_y = np.memmap(tn_y_path, dtype=theano.config.floatX, mode='w+', shape=(4044,))
    v_x = np.memmap(v_x_path, dtype=theano.config.floatX, mode='w+', shape=(500,image_size))
    v_y = np.memmap(v_y_path, dtype=theano.config.floatX, mode='w+', shape=(500,))
    tt_x = np.memmap(tt_x_path, dtype=theano.config.floatX, mode='w+', shape=(6925,image_size))
    tt_y = np.memmap(tt_y_path, dtype=theano.config.floatX, mode='w+', shape=(6925,))

    # randomly allocate 500 samples to the validation dataset
    v_batch = np.random.choice(range(4544), size=500, replace=False)
    a_idx = 0
    tn_idx = 0
    v_idx = 0
    tt_idx = 0

    terminate = False
    for idx, file in enumerate(sorted(os.listdir(img_path))):
        if not file.startswith('w_'):
            continue
        if idx % 1000 == 0:
            print(file)
            np.memmap.flush(tn_x)
            np.memmap.flush(v_x)
            np.memmap.flush(tt_x)
            terminate = prompt_for_quit_or_timeout()
            if terminate:
                print('Exiting gracefully...')
                break
            else:
                print('Program will continue...')
        if file.endswith('.jpg'):
            with open(os.path.join(img_path,file), 'rb') as f:
                im = Image.open(f)
                im = np.asarray(im).T.flatten()

                if file in labels_dict:
                    if a_idx in v_batch:
                        v_x[v_idx,:] = im[:]
                        v_y[v_idx] = labels_dict[file]
                        v_idx += 1
                        a_idx += 1
                    else:
                        tn_x[tn_idx,:] = im[:]
                        tn_y[tn_idx] = labels_dict[file]
                        tn_idx += 1
                        a_idx += 1
                else:
                    tt_x[tt_idx,:] = im[:]
                    tt_idx += 1

    if terminate:
        sys.exit('')


def whiten_images(path='../data/memmap/', batch_size=500, n_components=500, image_size=3*300*300):
    tn_x_path = os.path.join(path, 'tn_x.dat')
    v_x_path = os.path.join(path, 'v_x.dat')
    tt_x_path = os.path.join(path, 'tt_x.dat')

    tn_x = np.memmap(tn_x_path, dtype=theano.config.floatX, mode='r+', shape=(4044,image_size))
    v_x = np.memmap(v_x_path, dtype=theano.config.floatX, mode='r+', shape=(500,image_size))
    tt_x = np.memmap(tt_x_path, dtype=theano.config.floatX, mode='r+', shape=(6925,image_size))

    # fit
    print('Fitting on training data')
    izca = IncrementalZCA(n_components=n_components, batch_size=batch_size)
    izca.fit(tn_x)
    print('Fitting on validation data')
    n_samples, n_features = v_x.shape
    for idx, batch in enumerate(gen_batches(n_samples, batch_size)):
        izca.partial_fit(v_x[batch])

    # transform
    n_samples, n_features = tn_x.shape
    print('Transforming training data')
    for batch in gen_batches(n_samples, batch_size):
        tn_x[batch][:] = izca.transform(tn_x[batch])
        np.memmap.flush(tn_x)

    print('Transforming validation data')
    n_samples, n_features = v_x.shape
    for batch in gen_batches(n_samples, batch_size):
        v_x[batch][:] = izca.transform(v_x[batch])
        np.memmap.flush(v_x)

    print('Transforming test data')
    n_samples, n_features = tt_x.shape
    for batch in gen_batches(n_samples, batch_size):
        tt_x[batch][:] = izca.transform(tt_x[batch])
        np.memmap.flush(tt_x)


###########################################################################
## i/o

def load_data(path='../data/memmap/', image_size=3*300*300):
    tn_x_path = os.path.join(path, 'tn_x.dat')
    tn_y_path = os.path.join(path, 'tn_y.dat')
    v_x_path = os.path.join(path, 'v_x.dat')
    v_y_path = os.path.join(path, 'v_y.dat')
    tt_x_path = os.path.join(path, 'tt_x.dat')
    tt_y_path = os.path.join(path, 'tt_y.dat')

    tn_x = np.memmap(tn_x_path, dtype=theano.config.floatX, mode='r', shape=(4044,image_size))
    tn_y = np.memmap(tn_y_path, dtype=theano.config.floatX, mode='r', shape=(4044,))
    v_x = np.memmap(v_x_path, dtype=theano.config.floatX, mode='r', shape=(500,image_size))
    v_y = np.memmap(v_y_path, dtype=theano.config.floatX, mode='r', shape=(500,))
    tt_x = np.memmap(tt_x_path, dtype=theano.config.floatX, mode='r', shape=(6925,image_size))
    tt_y = np.memmap(tt_y_path, dtype=theano.config.floatX, mode='r', shape=(6925,))

    tn_x, tn_y = make_shared((tn_x, tn_y))
    v_x , v_y  = make_shared((v_x, v_y))
    tt_x, tt_y = make_shared((tt_x, tt_y))

    return [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)]


def make_shared(data, borrow=True):
    """
    Function that loads the dataset into shared variables.

    Parameters
    ----------
    data: tuple of numpy arrays
      data = (x, y) where x is an np.array of predictors and y is an np array
      of outcome variables

    """
    x, y = data
    sx = theano.shared(
        np.asarray(x, dtype=theano.config.floatX),
        borrow=borrow
    )
    sy = theano.shared(
        np.asarray(y, dtype=theano.config.floatX),
        borrow=borrow
    )
    return sx, T.cast(sy, 'int32')


def display_results(best_validation_loss, elapsed_time, epoch):
    print(
        'Optimization complete with best validation score of %f'
        % (best_validation_loss)
    )
    print(
        'The code run for %d epochs, with %f epochs/sec'
        % (epoch, 1. * epoch / (elapsed_time))
    )
