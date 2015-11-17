# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import os
import numpy as np
import numpy.random as rng
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

###########################################################################
## local imports

from noaa.nnet.preproc import load_data, build_submission_stub
from noaa.nnet.layers import LeNet
from noaa.nnet.solvers import SupervisedRandomMSGD, display_results

###########################################################################
## local imports

from config import BASE_DIR, SEED

###########################################################################
## main

def fit_lenet(image_shape=(300, 300), n_image_channels=3, randomize=None,
              datasets=os.path.join(BASE_DIR,'data/memmap/'),
              outpath=os.path.join(BASE_DIR,'output/noaa_lenet.params'),
              filter_shapes=[(5, 5), (3,3), (3,3)], nkerns=(32, 64, 128),
              pool_sizes=[(2, 2), (2, 2), (2, 2)],
              n_hidden=1000, n_out=447,
              learning_rate=0.01, L1_reg=0.00, L2_reg=0.00,
              n_epochs=1000, batch_size=128, patience=10000,
              patience_increase=2, improvement_threshold=0.995):

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LeNet(
        rng=rng.RandomState(SEED),
        input=x,
        batch_size=batch_size,
        image_shape=image_shape,
        n_image_channels=n_image_channels,
        nkerns=nkerns,
        filter_shapes=filter_shapes,
        pool_sizes=pool_sizes,
        n_hidden=n_hidden,
        n_out=n_out
    )
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2
    )
    learner = SupervisedRandomMSGD(
        index,
        x,
        y,
        batch_size,
        load_data(datasets),
        outpath,
        classifier,
        cost,
        learning_rate=learning_rate
    )

    best_validation_loss, best_iter, epoch, elapsed_time = learner.fit(
        n_epochs=n_epochs,
        patience=patience,
        patience_increase=patience_increase,
        improvement_threshold=improvement_threshold
    )
    display_results(best_validation_loss, elapsed_time, epoch)

    return learner

###########################################################################
## main

def main():
    ln = fit_lenet()

    # TODO: build output suitable for kaggle entry

if __name__ == '__main__':
    main()
