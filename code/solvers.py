# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import math
import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from skimage import transform as tf

###########################################################################
## local imports

from config import SEED
from data_augmentation import transform_images
from utils import prompt_for_quit_or_timeout

###########################################################################
## solvers

def fit_msgd_early_stopping(outpath, n_batches,
                            models, classifier, n_epochs=1000,
                            patience=5000, patience_increase=2,
                            improvement_threshold=0.995):

    # unpack parameters
    n_tn_batches, n_v_batches, n_tt_batches = n_batches
    tn_model, v_model = models

    validation_frequency = min(n_tn_batches, patience//20)

    # initialize some variables
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    # main loop
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_tn_batches):
            finished = prompt_for_quit_or_timeout(msg='Stop learning?', timeout=5)
            if finished:
                end_time = timeit.default_timer()
                return best_validation_loss, best_iter, epoch, (end_time - start_time)

            minibatch_avg_cost = tn_model(minibatch_index)
            iter = (epoch - 1) * n_tn_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [v_model(i) for i in xrange(n_v_batches)]
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f, minibatch average cost %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_tn_batches,
                        this_validation_loss,
                        minibatch_avg_cost
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    #best_params = copy.deepcopy(params)
                    best_validation_loss = this_validation_loss
                    best_iter = iter

            else:
                print(
                    'epoch %i, minibatch %i/%i, minibatch average cost %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_tn_batches,
                        minibatch_avg_cost
                    )
                )


            if patience <= iter:
                done_looping = True
                break

        if outpath is not None:
            classifier.save_params(path=outpath)

    end_time = timeit.default_timer()
    return best_validation_loss, best_iter, epoch, (end_time - start_time)


def fit_random_msgd_early_stopping(x, datasets, outpath, n_batches,
                                   models, classifier, n_epochs=1000,
                                   patience=5000, patience_increase=2,
                                   improvement_threshold=0.995):

    # unpack parameters
    [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)] = datasets
    n_tn_batches, n_v_batches, n_tt_batches = n_batches
    tn_model, v_model = models

    validation_frequency = min(n_tn_batches, patience//20)

    # initialize some variables
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    # main loop
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_tn_batches):
            finished = prompt_for_quit_or_timeout(msg='Stop learning?', timeout=5)
            if finished:
                end_time = timeit.default_timer()
                return best_validation_loss, best_iter, epoch, (end_time - start_time)

            minibatch = tn_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            minibatch = transform_images(minibatch)
            minibatch_avg_cost = tn_model(minibatch, minibatch_index)

            iter = (epoch - 1) * n_tn_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [v_model(i) for i in xrange(n_v_batches)]
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f, minibatch average cost %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_tn_batches,
                        this_validation_loss,
                        minibatch_avg_cost
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    #best_params = copy.deepcopy(params)
                    best_validation_loss = this_validation_loss
                    best_iter = iter

            else:
                print(
                    'epoch %i, minibatch %i/%i, minibatch average cost %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_tn_batches,
                        minibatch_avg_cost
                    )
                )


            if patience <= iter:
                done_looping = True
                break

        if outpath is not None:
            classifier.save_params(path=outpath)

    end_time = timeit.default_timer()
    return best_validation_loss, best_iter, epoch, (end_time - start_time)


def display_results(best_validation_loss, elapsed_time, epoch):
    print(
        'Optimization complete with best validation score of %f'
        % (best_validation_loss)
    )
    print(
        'The code run for %d epochs, with %f epochs/sec'
        % (epoch, 1. * epoch / (elapsed_time))
    )

###########################################################################
## sdg via mini batches

class MiniBatchSGD(object):
    def __init__(self, x, y, batch_size,
                 datasets, outpath, learner, cost, updates=None,
                 learning_rate=0.01, momentum=0.9, weight_decay=0.0005):


        self.x = x
        self.y = y

        self.datasets = datasets
        self.outpath = outpath
        self.batch_size = batch_size

        self.learner = learner
        self.cost = cost

        if updates is None:
            dparams = [T.grad(cost, param) for param in learner.params]
            momentum_updates = [
                (v, momentum * v - weight_decay * learning_rate * v - learning_rate * dp)
                for p, v, dp in zip(learner.params, learner.momentum, dparams)
            ]
            updates = [
                (p, p + momentum * v - weight_decay * learning_rate * v - learning_rate * dp)
                for p, v, dp in zip(learner.params, learner.momentum, dparams)
            ] + momentum_updates

        self.updates = updates
        self.n_batches = self._compute_n_batches()
        self.models = self._compile_models()

    def _compute_n_batches(self):
        tn, _ = self.datasets[0]
        v, _ = self.datasets[1]
        tt, _ = self.datasets[2]

        n_tn_batches = int(tn.get_value(borrow=True).shape[0] / self.batch_size)
        n_v_batches = int(v.get_value(borrow=True).shape[0] / self.batch_size)
        n_tt_batches = int(tt.get_value(borrow=True).shape[0] / self.batch_size)
        return [n_tn_batches, n_v_batches, n_tt_batches]

    def _compile_models():
        pass

    def fit(self, patience=5000, n_epochs=1000,
            patience_increase=2, improvement_threshold=0.995):

        return fit_msgd_early_stopping(
            self.datasets,
            self.outpath,
            self.n_batches,
            self.models,
            self.learner,
            n_epochs=n_epochs,
            patience=patience,
            patience_increase=patience_increase,
            improvement_threshold=improvement_threshold
        )


    def predict(self, params=None):

        if params is not None:
            self.learner.load_params(path=params)

        tt_x, _ = self.datasets[2]
        prediction_model = theano.function(
            inputs=[],
            outputs=self.learner.y_pred,
            givens={self.x: tt_x}
        )

        predicted_values = prediction_model()
        return predicted_values


class SupervisedMSGD(MiniBatchSGD):
    def __init__(self, index, x, y, batch_size,
                 datasets, outpath, learner, cost, updates=None,
                 learning_rate=0.01, momentum=0.9, weight_decay=0.0005):

        self.index = index

        super(SupervisedMSGD, self).__init__(
            x, y, batch_size,
            datasets, outpath, learner, cost, updates,
            learning_rate, momentum, weight_decay)

    def _compile_models(self):
        tn_x, tn_y = self.datasets[0]
        v_x, v_y = self.datasets[1]

        tn_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: tn_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: tn_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        v_model = theano.function(
            inputs=[self.index],
            outputs=self.learner.errors(self.y),
            givens={
                self.x: v_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: v_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        return [tn_model, v_model]


class SupervisedRandomMSGD(MiniBatchSGD):
    def __init__(self, index, x, y, batch_size,
                 datasets, outpath, learner, cost, updates=None,
                 learning_rate=0.01, momentum=0.9, weight_decay=0.0005):

        self.index = index

        super(SupervisedRandonMSGD, self).__init__(
            x, y, batch_size,
            datasets, outpath, learner, cost, updates,
            learning_rate, momentum, weight_decay)

    def _compile_models(self):
        tn_x, tn_y = self.datasets[0]
        v_x, v_y = self.datasets[1]

        tn_model = theano.function(
            inputs=[theano.In(self.x, borrow=True), self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.y: tn_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        v_model = theano.function(
            inputs=[self.index],
            outputs=self.learner.errors(self.y),
            givens={
                self.x: v_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: v_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        return [tn_model, v_model]


    def fit(self, patience=5000, n_epochs=1000,
            patience_increase=2, improvement_threshold=0.995):

        return fit_random_msgd_early_stopping(
            self.x,
            self.datasets,
            self.outpath,
            self.n_batches,
            self.models,
            self.learner,
            n_epochs=n_epochs,
            patience=patience,
            patience_increase=patience_increase,
            improvement_threshold=improvement_threshold
        )
