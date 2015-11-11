# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import numpy as np
import numpy.random as rng
import theano.tensor as T

###########################################################################
## local imports

from utils import load_data, display_results, build_submission_stub
from layers import LogisticRegression
from solvers import SupervisedMSGD

###########################################################################
## config

SEED = 1234

###########################################################################
## fit

def fit_logistic_regression(image_shape=(300, 300), n_image_channels=3, n_out=447,
              datasets='../data/memmap/', outpath='../output/noaa_lenet.params',
              learning_rate=0.01, L1_reg=0.00, L2_reg=0.001,
              n_epochs=1000, batch_size=200, patience=10000,
              patience_increase=2, improvement_threshold=0.995):

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(
        input=x,
        n_in=n_image_channels * reduce(np.multiply, image_shape),
        n_out=n_out
    )
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2
    )
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
    lr = fit_logistic_regression()
    labels_dict, stub = build_submission_stub()
    predicted_values = lr.predict()
    if True:
        return lr, predicted_values

    # something goes here: ??? stub['whaleCode'] = 1
    stub['whileID'] = stub['whaleCode'].apply(lambda x: labels_dict[x])
    stub = stub[['Image', 'whileID']]
    stub.to_csv('../output/init_preds.csv', index=False)


if __name__ == '__main__':
    main()
