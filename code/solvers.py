# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from skimage import transform as tf

###########################################################################
## config

SEED = 1234

###########################################################################
## solvers

def fit_msgd_early_stopping(datasets, outpath, n_batches,
                            models, classifier, n_epochs=1000,
                            patience=5000, patience_increase=2,
                            improvement_threshold=0.995):

    # unpack parameters
    [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)] = datasets
    n_tn_batches, n_v_batches, n_tt_batches = n_batches
    tn_model, v_model = models

    validation_frequency = min(n_tn_batches, patience/20)

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


            if patience <= iter:
                done_looping = True
                break

        if outpath is not None:
            classifier.save_params(path=outpath)

    end_time = timeit.default_timer()
    return best_validation_loss, best_iter, epoch, (end_time - start_time)


def fit_random_msgd_early_stopping(datasets, outpath, models, classifier,
                                   n_v_batches, n_epochs=1000,
                                   patience=5000, patience_increase=2,
                                   improvement_threshold=0.995):

    # unpack parameters
    [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)] = datasets
    tn_model, v_model = models

    validation_frequency = patience/200

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
        minibatch_avg_cost = tn_model()

        if (epoch + 1) % validation_frequency == 0:
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
                    patience = max(patience, int(epoch * patience_increase))

                best_validation_loss = this_validation_loss
                best_epoch = epoch


        if patience <= epoch:
            done_looping = True
            break

        if outpath is not None:
            classifier.save_params(path=outpath)

    end_time = timeit.default_timer()
    return best_validation_loss, best_epoch, epoch, (end_time - start_time)


###########################################################################
## sdg via mini batches

class MiniBatchSGD(object):
    def __init__(self, index, x, y, batch_size, learning_rate,
                 datasets, outpath, learner, cost, updates=None):

        self.x = x
        self.y = y
        self.index = index

        self.datasets = datasets
        self.outpath = outpath
        self.batch_size = batch_size

        self.learner = learner
        self.cost = cost

        if updates is None:
            dparams = [T.grad(cost, param) for param in learner.params]
            updates = [
                (p, p - learning_rate * dp)
                for p, dp in zip(learner.params, dparams)
            ]
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
    def __init__(self, index, x, y, batch_size, learning_rate,
                 datasets, outpath, learner, cost):

        super(SupervisedMSGD, self).__init__(
            index, x, y, batch_size, learning_rate,
            datasets, outpath, learner, cost)


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
    def __init__(self, index, x, y, batch_size, learning_rate,
                 datasets, outpath, learner, cost, rng):

        self.rng = rng

        super(SupervisedMSGD, self).__init__(
            index, x, y, batch_size, learning_rate,
            datasets, outpath, learner, cost)


    def _compile_models(self):
        tn_x, tn_y = self.datasets[0]
        v_x, v_y = self.datasets[1]

        tn_range = tn_x.get_value(borrow=True).shape[0]
        v_range = v_x.get_value(borrow=True).shape[0]

        # select a random batch of images
        tn_random_idx = self.rng.choice(size=self.batch_size, a=tn_range, replace=True)

        # apply a random transformation
        trans_x =  self.rng.random_integers(size=1, low=-4, high=4)
        trans_y =  self.rng.random_integers(size=1, low=-4, high=4)
        scale =  self.rng.uniform(size=1, low=1/1.3, high=1.3)
        rotation = self.rng.uniform(size=1, low=0.0, high=2*math.pi)
        tform = tf.SimilarityTransform(scale=scale, rotation=rotation,
                                       translation=(trans_x, trans_y))

        # TODO: testing this... if it fails, try playing around with theano.clone
        tn_model = theano.function(
            inputs=[],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: tf.warp(tn_x[tn_random_idx], tform),
                self.y: tn_y[tn_idx]
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
