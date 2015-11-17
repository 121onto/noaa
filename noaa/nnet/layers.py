# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import cPickle
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams

###########################################################################
## activation functions

def relu(input):
    return T.switch(input<0, 0, input)

def maxout(input, size=1):
    rtn = None
    for i in xrange(size):
        tmp = input[:,i::size]
        if rtn is None:
            rtn = tmp
        else:
            rtn = T.maximum(rtn, tmp)

    return rtn

###########################################################################
## helper functions

def dropout(rng, layer, p=0.5):
    """source https://github.com/mdenil/dropout/blob/master/mlp.py
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


def initialize_tensor(shape, scale=None, dtype=theano.config.floatX, rng=None, dist='uniform'):
    if dist=='uniform':
        rtn = np.asarray(
            rng.uniform(
                low=-1.0 * scale,
                high=1.0 * scale,
                size=shape
            ),
        dtype=dtype
        )
    elif dist=='normal':
        rtn = np.asarray(
            rng.normal(
                loc=0.0,
                scale=scale,
                size=shape
            ),
        dtype=dtype
        )
    elif dist=='zeros':
        rtn = np.zeros(
            shape,
            dtype=dtype
        )
    elif dist=='ones':
        rtn = np.ones(
            shape,
            dtype=dtype
        )
        rtn = rtn if scale is None else rtn * scale
    else:
        raise NotImplementedError()
    return rtn

###########################################################################
## base layer

class BaseLayer(object):
    def __init__(self):
        pass

    def save_params(self, path):
        with open(path, 'wb') as file:
            for v in self.params:
                cPickle.dump(v.get_value(borrow=True), file, -1)

    def load_params(self, path, scale=0.5):
        with open(path) as file:
            for v in self.params:
                v.set_value(cPickle.load(file) * scale, borrow=True)

###########################################################################
## hidden

class HiddenLayer(BaseLayer):

    def __init__(self, rng, input, n_in, n_out,
                 W=None, b=None, activation=maxout,
                 with_dropout=True, dropout_rate=0.5):

        self.input = input
        W_shape = (n_in, n_out)
        b_shape = (n_out,)

        if W is None:
            # suggested initial weights larger for sigmoid activiation
            W = theano.shared(
                value=initialize_tensor(W_shape, scale=0.01, dtype=theano.config.floatX, rng=rng, dist='normal'),
                name='W',
                borrow=True
            )

        if b is None:
            b = theano.shared(
                value=initialize_tensor(b_shape, scale=0.01, dtype=theano.config.floatX, dist='ones'),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.weights = [self.W]

        v_W = theano.shared(
            value=initialize_tensor(W_shape, dtype=theano.config.floatX, dist='zeros'),
            name='v_W',
            borrow=True
        )
        v_b = theano.shared(
            value=initialize_tensor(b_shape, dtype=theano.config.floatX, dist='zeros'),
            name='v_W',
            borrow=True
        )
        self.v_W = v_W
        self.v_b = v_b
        self.momentum = [self.v_W, self.v_b]

        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            #else 1.7159 * activation( (2./3) * lin_output) if activation == T.tanh # transformation
            else activation(lin_output)
        )
        if with_dropout:
            self.dropout_rate = dropout_rate
            self.output = dropout(rng, self.output, p=dropout_rate)

        self.L1 = abs(self.W).sum()
        self.L2 = abs(self.W ** 2).sum()

###########################################################################
## Logistic regression

class LogisticRegression(BaseLayer):
    def __init__(self, rng, input, n_in, n_out):
        """
        Parameters
        ----------
        n_in: tuple
          the dimention of the input
        n_out: integer
          the number of classes
        """

        self.input = input
        W_shape = (n_in, n_out)
        b_shape = (n_out,)

        self.W = theano.shared(
            value=initialize_tensor(W_shape, scale=0.01, dtype=theano.config.floatX, rng=rng, dist='normal'),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=initialize_tensor(b_shape, scale=0.01, dtype=theano.config.floatX, dist='zeros'),
            name='b',
            borrow=True
        )
        self.params = [self.W, self.b]
        self.weights = [self.W]

        v_W = theano.shared(
            value=initialize_tensor(W_shape, dtype=theano.config.floatX, dist='zeros'),
            name='v_W',
            borrow=True
        )
        v_b = theano.shared(
            value=initialize_tensor(b_shape, dtype=theano.config.floatX, dist='zeros'),
            name='v_W',
            borrow=True
        )
        self.v_W = v_W
        self.v_b = v_b
        self.momentum = [self.v_W, self.v_b]

        self.L1 = abs(self.W).sum()
        self.L2 = abs(self.W ** 2).sum()

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)




    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        Parameters
        ----------
        y: theano.tensor.TensorType
          corresponds to a vecotr that gievs for each example the correct label

        Notes
        -----
        Use mean rather than sum so the learning rate is less dependent on the
        batch size
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch; zero one
        loss over the size of the minibatch.

        Parameters
        ----------
        y: theano.tensor.TensorType
          corresponds to a vecotr that gievs for each example the correct label
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

###########################################################################
## mlp

class MLP(BaseLayer):

    def __init__(self, rng, input, n_in, n_hidden, n_out,
                 activation=maxout, with_dropout=True):

        self.input = input
        self.hidden_layer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation,
            with_dropout=with_dropout
        )
        self.hidden_layer2 = HiddenLayer(
            rng=rng,
            input=self.hidden_layer1.output,
            n_in=n_hidden,
            n_out=n_hidden,
            activation=activation,
            with_dropout=with_dropout
        )
        self.log_reg_layer = LogisticRegression(
            rng=rng,
            input=self.hidden_layer2.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.params = self.hidden_layer1.params + self.hidden_layer2.params + self.log_reg_layer.params
        self.weights = self.hidden_layer1.weights + self.hidden_layer2.weights + self.log_reg_layer.weights
        self.momentum = self.hidden_layer1.momentum + self.hidden_layer2.momentum + self.log_reg_layer.momentum

        self.L1 = self.hidden_layer1.L1 + self.hidden_layer2.L1 + self.log_reg_layer.L1
        self.L2 = self.hidden_layer1.L2 + self.hidden_layer2.L2 + self.log_reg_layer.L2

        self.negative_log_likelihood = self.log_reg_layer.negative_log_likelihood
        self.errors = self.log_reg_layer.errors
        self.y_pred = self.log_reg_layer.y_pred


###########################################################################
## Convolution with optional pooling

class ConvolutionLayer(BaseLayer):
    """
    A convolution layer with optional pooling using shared valriables
    """
    def __init__(self, rng, input,
                 feature_maps_in, feature_maps_out,
                 filter_shape, input_shape, batch_size,
                 pool_size=None, pool_ignore_border=True,
                 W=None, b=None, activation=relu,
                 with_dropout=True, dropout_rate=0.5):

        self.input = input

        i_shape = (None, feature_maps_in) + input_shape
        W_shape = (feature_maps_out, feature_maps_in) + filter_shape
        b_shape = (feature_maps_out,)

        if W is None:
            W = theano.shared(
                initialize_tensor(W_shape, scale=0.01, dtype=theano.config.floatX, rng=rng, dist='normal'),
                name='W'
            )
        if b is None:
            b = theano.shared(
                initialize_tensor(b_shape, scale=0.1, dtype=theano.config.floatX, dist='ones'),
                name='b'
            )

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.weights = [self.W]

        v_W = theano.shared(
            value=initialize_tensor(W_shape, dtype=theano.config.floatX, dist='zeros'),
            name='v_W',
            borrow=True
        )
        v_b = theano.shared(
            value=initialize_tensor(b_shape, dtype=theano.config.floatX, dist='zeros'),
            name='v_W',
            borrow=True
        )
        self.v_W = v_W
        self.v_b = v_b
        self.momentum = [self.v_W, self.v_b]

        conv_out = T.nnet.conv.conv2d(
            self.input,
            filters=self.W,
            filter_shape = W_shape,
            image_shape = i_shape
        )

        # pool first to improve performance
        pooled_out = conv_out
        if pool_size is not None:
            pooled_out = downsample.max_pool_2d(
                conv_out,
                pool_size,
                ignore_border=pool_ignore_border
            )

        lin_output = pooled_out + self.b.dimshuffle('x',0,'x','x')
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        if with_dropout:
            self.dropout_rate = dropout_rate
            self.output = dropout(rng, self.output, p=dropout_rate)

        self.L1 = abs(self.W).sum()
        self.L2 = abs(self.W ** 2).sum()

###########################################################################
## lenet

class LeNet(BaseLayer):
    def __init__(self, rng, input, batch_size,
                 image_shape, n_image_channels,
                 nkerns, filter_shapes, pool_sizes,
                 n_hidden, n_out, with_dropout=True):

        # reshapes the input
        self.input = input.reshape((-1, n_image_channels) + image_shape)
        self.n_layers = len(nkerns)
        self.input_shape = image_shape
        self.n_out = n_out

        # build graphs
        self.convolution_layers = []
        self.params = []
        self.weights = []
        self.momentum = []
        self.L1 = 0
        self.L2 = 0
        for i in xrange(self.n_layers):
            if i == 0:
                feature_maps_in=n_image_channels
                input=self.input
                input_shape=self.input_shape
            else:
                feature_maps_in=nkerns[i - 1]
                input=self.convolution_layers[i - 1].output
                input_shape = self.shape_reduction(input_shape, filter_shapes[i - 1], pool_sizes[i - 1])

            convolution = ConvolutionLayer(
                rng=rng,
                input=input,
                feature_maps_in=feature_maps_in,
                feature_maps_out=nkerns[i],
                filter_shape=filter_shapes[i],
                input_shape=input_shape,
                batch_size=batch_size,
                pool_size=pool_sizes[i],
                activation=relu,
                with_dropout=with_dropout
            )

            self.convolution_layers.append(convolution)
            self.params.extend(convolution.params)
            self.weights.extend(convolution.weights)
            self.momentum.extend(convolution.momentum)

            self.L1 = self.L1 + convolution.L1
            self.L2 = self.L2 + convolution.L2

        input_shape = self.shape_reduction(input_shape, filter_shapes[-1], pool_sizes[-1])
        self.mlp_layer = MLP(
            rng=rng,
            input=self.convolution_layers[-1].output.flatten(2),
            n_in=nkerns[-1] * reduce(np.multiply, input_shape),
            n_hidden=n_hidden,
            n_out=self.n_out,
            activation=maxout,
            with_dropout=with_dropout
        )

        self.params.extend(self.mlp_layer.params)
        self.weights.extend(self.mlp_layer.weights)
        self.momentum.extend(self.mlp_layer.momentum)

        self.L1 = self.L1 + self.mlp_layer.L1
        self.L2 = self.L2 + self.mlp_layer.L2

        self.negative_log_likelihood = self.mlp_layer.negative_log_likelihood
        self.errors = self.mlp_layer.errors
        self.y_pred = self.mlp_layer.y_pred


    def shape_reduction(self, input_shape, filter_shape, pool_size):
        rtn = [i-f+1 for i, f in zip(input_shape, filter_shape)]
        rtn = tuple(np.divide(rtn, pool_size))
        return rtn


    def scale_weights(self, scale=0.5):
        for v in self.weights:
            v.set_value(v.get_value(borrow=True) * scale, borrow=True)
