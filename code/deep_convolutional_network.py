# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import numpy as np
import numpy.random as rng
import theano.tensor as T

###########################################################################
## local imports

from utils import load_data, display_results, build_submission_stub
from layers import LeNet
from solvers import SupervisedMSGD, SupervisedRandomMSGD

###########################################################################
## config

SEED = 1234

###########################################################################
## main

def fit_lenet(image_shape=(300, 300), n_image_channels=3, randomize=False,
              datasets='../data/memmap/', outpath='../output/noaa_lenet.params',
              filter_shapes=[(5, 5),(5,5),(3,3)], nkerns=(6, 6, 10),
              pool_sizes=[(2, 2), (2, 2), (2, 2)],
              n_hidden=1000, n_out=447,
              learning_rate=0.01, L1_reg=0.00, L2_reg=0.001,
              n_epochs=1000, batch_size=200, patience=10000,
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
    if randomize:
        learner = SupervisedRandomMSGD(
            index,
            x,
            y,
            batch_size,
            learning_rate,
            load_data(datasets),
            outpath,
            classifier,
            cost,
            rng=rng.RandomState(SEED)
        )
    else:
        learner = SupervisedMSGD(
            index,
            x,
            y,
            batch_size,
            learning_rate,
            load_data(datasets),
            outpath,
            classifier,
            cost
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
    predicted_values = ln.predict()
    if True:
        return ln, predicted_values

    # something goes here: ??? stub['whaleCode'] = 1
    labels_dict, stub = build_submission_stub()
    stub['whileID'] = stub['whaleCode'].apply(lambda x: labels_dict[x])
    stub = stub[['Image', 'whileID']]
    stub.to_csv('../output/init_preds.csv', index=False)


if __name__ == '__main__':
    main()
