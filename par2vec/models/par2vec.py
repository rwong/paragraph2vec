"""
Starter code from
http://deeplearning.net/tutorial/mlp.html

Implementation based on tutorial above, modified
for paragraph vectors in

    "Distributed Representations of Sentences and Documents"

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from par2vec.models.logistic import LogisticRegression

class HiddenLayer(object):
    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: theano.tensor.dmatrix
        :param inputs: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.inputs = inputs
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            #W = theano.shared(value=W_values, name='W', borrow=True)
            W = theano.shared(value=W_values, borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            #b = theano.shared(value=b_values, name='b', borrow=True)
            b = theano.shared(value=b_values, borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(inputs, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class Par2Vec(object):
    """Multi-Layer Perceptron For Paragraph and Word Vectors

    Tranlates documents and word to vectors and feeds them
    forward to the next hidden layer.  Propogation continues
    until the top softmax layer is reached (LogisticRegression).
    """

    def __init__(self, rng, docf, doc_dim, word_dim, n_hidden):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type doc_dim: int
        :param doc_dim: document vector size

        :type word_dim: int
        :param word_dim: word vector size

        :type n_hidden: list[int]
        :param n_hidden: number of hidden units per layer
        """

        # CHANGE: Make two layers
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function

        hiddenLayers = [ ]
        for ndx, num_hidden in enumerate(n_hidden):
            num_in = n_in if ndx == 0 else n_hidden[ndx - 1]
            layer_inputs = inputs if ndx == 0 \
                                  else hiddenLayers[ndx - 1].output
            layer = HiddenLayer(
                rng=rng,
                inputs=layer_inputs,
                n_in=num_in,
                n_out=num_hidden,
                activation=T.tanh
            )
            hiddenLayers.append(layer)

        self.hiddenLayers = hiddenLayers

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            inputs=self.hiddenLayers[-1].output,
            n_in=n_hidden[-1],
            n_out=n_out
        )

        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayers[0].W).sum()
        for layer in self.hiddenLayers[1:]:
            self.L1 += abs(layer.W).sum()

        self.L1 += abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayers[0].W ** 2).sum()
        for layer in self.hiddenLayers[1:]:
            self.L2_sqr += (layer.W ** 2).sum()

        self.L2_sqr += (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the
        # layers it is made out of
        self.params = self.hiddenLayers[0].params
        for layer in self.hiddenLayers[1:]:
            self.params += layer.params

        self.params += self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.inputs = inputs
