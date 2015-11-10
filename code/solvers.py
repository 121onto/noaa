# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import timeit
import numpy as np
import theano
import theano.tensor as T

###########################################################################
## solvers

def fit_msgd_early_stopping(datasets, outpath, n_batches,
                            models, classifier, n_epochs=1000,
                            patience=5000, patience_increase=2, improvement_threshold=0.995):

    # unpack parameters
    [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)] = datasets
    n_tn_batches, n_v_batches, n_tt_batches = n_batches
    tn_model, v_model, tt_model = models

    validation_frequency = min(n_tn_batches, patience/2)

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
                    'epoch %i, minibatch %i/%i, validation error %f ' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_tn_batches,
                        this_validation_loss
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

    def predict(self, dataset='test', datasets=None, params=None):
        if datasets is not None:
            datasets = datasets
        else:
            datasets = self.datasets

        if params is not None:
            self.learner.load_params(path=params)

        prediction_model = theano.function(
            inputs=[self.learner.input],
            outputs=self.learner.y_pred
        )

        index = 2 if dataset=='test' else 1 if dataset=='valid' else 0
        x, y = datasets[index]
        x = x.get_value()

        predicted_values = prediction_model(x)
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
        tt_x, tt_y = self.datasets[2]

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
        tt_model = theano.function(
            inputs=[self.index],
            outputs=self.learner.errors(self.y),
            givens={
                self.x: tt_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: tt_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        return [tn_model, v_model, tt_model]


class UnsupervisedMSGD(MiniBatchSGD):
    def __init__(self, index, x, batch_size, learning_rate,
                 datasets, outpath, learner, cost):

        super(UnsupervisedMSGD, self).__init__(
            index, x, None, batch_size, learning_rate,
            datasets, outpath, learner, cost)

    def _compile_models(self):
        tn_x, _ = self.datasets[0]
        v_x, _ = self.datasets[1]
        tt_x, _ = self.datasets[2]

        tn_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: tn_x[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        v_model = theano.function(
            inputs=[self.index],
            outputs=self.learner.error,
            givens={
                self.x: v_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
            }
        )
        tt_model = theano.function(
            inputs=[self.index],
            outputs=self.learner.error,
            givens={
                self.x: tt_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
            }
        )
        return [tn_model, v_model, tt_model]
