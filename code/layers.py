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
## base layer

def initialize_tensor(scale, shape, dtype=theano.config.floatX, rng=rng, dist='uniform'):
    if dist=='uniform':
        rtn = np.asarray(
            rng.uniform(
                low=-1.0 * scale,
                high=1.0 * scale,
                size=shape
            ),
        dtype=dtype
        )
    elif dist=='zero':
        rtn = np.zeros(
            shape,
            dtype=dtype
        )
    else:
        raise NotImplementedError()
    return rtn


class BaseLayer(object):
    def __init__(self):
        pass

    def save_params(self, path):
        with open(path, 'wb') as file:
            for v in self.params:
                cPickle.dump(v.get_value(borrow=True), file, -1)

    def load_params(self, path):
        with open(path) as file:
            for v in self.params:
                v.set_value(cPickle.load(file), borrow=True)

###########################################################################
## hidden

class HiddenLayer(BaseLayer):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            # suggested initial weights larger for sigmoid activiation
            W_scale = 24. if (activation == theano.tensor.nnet.sigmoid) else 6.
            W_scale = np.sqrt(W_scale / (n_in + n_out))
            W_shape = (n_in, n_out)
            W = theano.shared(
                value=initialize_tensor(W_scale, W_shape, theano.config.floatX, rng=rng, dist='uniform'),
                name='W',
                borrow=True
            )

        if b is None:
            b_shape = (n_out,)
            b = theano.shared(
                value=initialize_tensor(None, b_shape, theano.config.floatX, rng=rng, dist='zero'),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = b

        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            #else 1.7159 * activation( (2./3) * lin_output) if activation == T.tanh # transformation
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


###########################################################################
## autoencoders

class AutoEncoder(BaseLayer):
    def __init__(self, np_rng, input, th_rng=None,
                 n_visible=784, n_hidden=500,
                 corruption_level=0.3,
                 W=None, b_hid=None, b_vis=None):

        # cache hypter parameters
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if not th_rng:
            th_rng = RandomStreams(np_rng.randint(2 ** 30))
        self.th_rng = th_rng

        # initialize parameters
        if not W:
            W_scale = 4. * np.sqrt(6. / (n_hidden + n_visible))
            W_shape = (n_visible, n_hidden)
            W = theano.shared(
                value = initialize_tensor(W_scale, W_shape, rng=np_rng),
                name='W',
                borrow=True
            )
        if not b_vis:
            b_vis = theano.shared(
                value = initialize_tensor(0., n_visible, dist='zero'),
                borrow=True
            )
        if not b_hid:
            b_hid = theano.shared(
                value = initialize_tensor(0., n_hidden, dist='zero'),
                name='b',
                borrow=True
            )

        self.W = W
        self.W_prime = self.W.T
        self.b = b_hid
        self.b_prime = b_vis
        self.params = [self.W, self.b, self.b_prime]
        self.forward_params = [self.W, self.b]

        # initialize cost function
        self.input = input
        corrupted_input = self.corrupt_input(self.input, corruption_level)
        self.hidden_values = self.get_hidden_values(corrupted_input)
        self.y = self.get_reconstructed_input(self.hidden_values)
        self.cross_entropy = - T.sum(
            self.input * T.log(self.y) + (1 - self.input) * T.log(1-self.y),
            axis=1
        )
        self.cost = T.mean(self.cross_entropy)

        # for "prediction"
        self.output = self.get_hidden_values(self.input)
        self.y_pred = self.get_reconstructed_input(self.output)
        self.error = T.sum((self.y_pred - self.input) ** 2)

    def corrupt_input(self, input, corruption_level=0.3):
        return self.th_rng.binomial(
            size=input.shape,
            n=1,
            p=(1 - corruption_level),
            dtype=theano.config.floatX
        ) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)


class ContractiveAutoEncoder(BaseLayer):
    def __init__(self, np_rng, input, th_rng=None,
                 n_visible=784, n_hidden=500,
                 batch_size=1,
                 W=None, b_hid=None, b_vis=None):

        # cache hypter parameters
        self.batch_size=batch_size
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # initialize rngs
        if not th_rng:
            th_rng = RandomStreams(np_rng.randint(2 ** 30))
        self.th_rng = th_rng

        # initialize parameters
        if not W:
            W_scale = 4. * np.sqrt(6. / (n_hidden + n_visible))
            W_shape = (n_visible, n_hidden)
            W = theano.shared(
                value = initialize_tensor(W_scale, W_shape, rng=np_rng),
                name='W',
                borrow=True
            )
        if not b_vis:
            b_vis = theano.shared(
                value = initialize_tensor(0., n_visible, dist='zero'),
                borrow=True
            )
        if not b_hid:
            b_hid = theano.shared(
                value = initialize_tensor(0., n_hidden, dist='zero'),
                name='b',
                borrow=True
            )

        self.W = W
        self.W_prime = self.W.T
        self.b = b_hid
        self.b_prime = b_vis
        self.params = [self.W, self.b, self.b_prime]

        # initialize cost function
        self.input = input
        self.hidden_values = self.get_hidden_values(self.input)
        self.y = self.get_reconstructed_input(self.hidden_values)
        self.cross_entropy = - T.sum(
            self.input * T.log(self.y) + (1 - self.input) * T.log(1-self.y),
            axis=1
        )
        jacobian = self.get_jacobian(self.hidden_values, self.W)
        self.penalty = T.sum(jacobian ** 2) / self.batch_size
        self.cost = T.mean(self.cross_entropy) * T.mean(penalty)

        # for "prediction"
        self.output = self.get_hidden_values(self.input)
        self.y_pred = self.get_reconstructed_input(self.output)
        self.error = T.sum((self.y_pred - self.input) ** 2)

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_jacobian(self, hidden, W):
        return T.reshape(
            hidden * (1-hidden),
            (self.batch_size, 1, self.n_hidden)
        ) * T.reshape(
            W,
            (1, self.n_visible, self.n_hidden)
        )


###########################################################################
## stacked autoencoder

class StackedAutoEncoder(BaseLayer):
    def __init__(self, x, y, np_rng, th_rng=None,
                 n_inputs=784,
                 hidden_layer_sizes=[500,500],
                 corruption_levels=[0.1,0.1],
                 n_out=10):

        # cache hypter parameters
        self.n_inputs = n_inputs
        self.n_out = n_out
        self.hidden_layer_sizes = hidden_layer_sizes
        self.corruption_levels = corruption_levels
        self.n_layers = len(hidden_layer_sizes)

        # initialize rngs
        self.np_rng = np_rng
        if not th_rng:
            th_rng = RandomStreams(np_rng.randint(2 ** 30))
        self.th_rng = th_rng

        # build graphs
        self.input = x
        self.encoder_layers = []
        self.params = []
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = self.n_inputs
                layer_input = self.input
            else:
                input_size = hidden_layer_sizes[i - 1]
                layer_input = self.encoder_layers[-1].output

            encoder = AutoEncoder(
                np_rng=self.np_rng,
                input=layer_input,
                th_rng=self.th_rng,
                n_visible=input_size,
                n_hidden=self.hidden_layer_sizes[i],
                corruption_level=self.corruption_levels[i]
            )
            self.encoder_layers.append(encoder)
            self.params.extend(encoder.forward_params)

        self.logistic_layer = LogisticRegression(
            input=self.encoder_layers[-1].output,
            n_in=self.hidden_layer_sizes[-1],
            n_out=n_out
        )
        self.params.extend(self.logistic_layer.params)

        # cost and errors
        self.cost = self.logistic_layer.negative_log_likelihood
        self.errors = self.logistic_layer.errors


###########################################################################
## Max pooling

class MaxPooling(BaseLayer):
    def __init__(self, input, shape, ignore_border=True):
        self.input = input
        self.output = downsample.max_pool_2d(
            input,
            shape,
            ignore_border=ignore_border
        )

###########################################################################
## Logistic regression

class LogisticRegression(BaseLayer):
    def __init__(self, input, n_in, n_out):
        """
        Parameters
        ----------
        n_in: tuple
          the dimention of the input
        n_out: integer
          the number of classes
        """
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

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
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hidden_layer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        self.log_reg_layer = LogisticRegression(
            input=self.hidden_layer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.log_reg_layer.W).sum()
        self.L2 = (self.hidden_layer.W ** 2).sum() + (self.log_reg_layer.W ** 2).sum()
        self.negative_log_likelihood = self.log_reg_layer.negative_log_likelihood
        self.errors = self.log_reg_layer.errors

        self.params = self.hidden_layer.params + self.log_reg_layer.params
        self.input = input
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
                 W=None, b=None, activation=T.tanh):

        self.input = input

        W_shape = (feature_maps_out, feature_maps_in) + filter_shape
        b_shape = (feature_maps_out,)
        i_shape = (batch_size, feature_maps_in) + input_shape

        W_scale = 1.0/np.sqrt(feature_maps_in * reduce(np.multiply,filter_shape))
        b_scale = 0.5

        if W is None:
            W = theano.shared(
                initialize_tensor(W_scale, W_shape, rng=rng, dist='uniform'),
                name='W'
            )
        if b is None:
            b = theano.shared(
                initialize_tensor(b_scale, b_shape, rng=rng, dist='zero'),
                name='b'
            )

        self.W = W
        self.b = b

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
        self.params = [self.W, self.b]

###########################################################################
## lenet

class LeNet(BaseLayer):
    def __init__(self, rng, input, batch_size,
                 image_shape, n_image_channels,
                 nkerns, filter_shapes, pool_sizes,
                 n_hidden, n_out):

        # reshapes the input
        self.input = input.reshape((batch_size, n_image_channels) + image_size)
        self.n_layers = len(nkerns)
        self.input_shape = image_shape
        self.n_out = n_out

        # build graphs
        self.convolution_layers = []
        self.params = []
        self.L1 = 0
        self.l2 = 0
        for i in xrange(self.n_layers):
            if i == 0:
                feature_maps_in=n_image_channels
                input=self.input
                input_shape=self.input_shape
            else:
                feature_maps_in=nkerns[i]
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
                pool_size=pool_sizes[i]
            )

            self.convolution_layers.append(convolution)
            self.params.extend(convolution.params)
            self.L1 = self.L1 + abs(convolution.W).sum()
            self.L2 = self.L2 + abs(convolution.W ** 2).sum()

        input_shape = self.shape_reduction(input_shape, filter_shapes[-1], pool_sizes[-1])
        self.mlp_layer = MLP(
            rng=rng,
            input=self.convolution_layers[-1].output.flatten(2),
            n_in=nkerns[-1] * reduce(np.multiply, input_shape),
            n_hidden=n_hidden,
            n_out=self.n_out
        )

        self.L1 = self.L1 + self.mlp_layer.L1
        self.L2 = self.L2 + self.mlp_layer.L2
        self.negative_log_likelihood = self.mlp_layer.negative_log_likelihood
        self.errors = self.mlp_layer.errors

        self.params.extend(self.mlp_layer.params)
        self.y_pred = self.mlp_layer.y_pred


    def shape_reduction(self, input_shape, filter_shape, pool_size):
        rtn = [i-f+1 for i, f in zip(input_shape, filter_shape)]
        rtn = tuple(np.divide(rtn, pool_size))
        return rtn
