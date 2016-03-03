#!/bin/bash

"""Imdb sentiment classifcation using MLP from Theano

Half of code from
https://github.com/piskvorky/gensim/blob/develop/
    docs/notebooks/doc2vec-IMDB.ipynb

Other half from theano tutorial
"""

import sys
import os.path

import numpy as np
import gensim
import par2vec.models.par2vec
from par2vec.models.par2vec import Doc2Vec, TaggedDocument
from collections import namedtuple, OrderedDict

import multiprocessing
import random
from random import shuffle

import numpy
import theano
import theano.tensor as T
from par2vec.models.mlp import MLP

# For timing
import timeit
from contextlib import contextmanager
from timeit import default_timer
import time, datetime

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

SentimentDocument = namedtuple( 'SentimentDocument',
                                'words tags split sentiment' )

def load_imdb_dataset(dpath):
    # Will hold all docs in original order
    alldocs = [ ]
    # [Pos, Neg, Pos, Neg, N/A, ..., N/A]
    sentiment_labels = [1.0, 0.0, 1.0, 0.0, None, None, None, None]
    with open('data/aclImdb/alldata-id.txt') as alldata:
        for line_no, line in enumerate(alldata):
            # First token of each line is line number
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            # `tags = [tokens[0]]` would also work at extra memory cost
            tags = [line_no]
            # 25k train, 25k test, 50k extra split into 2 sets of 25k each
            split = ['train', 'test', 'extra', 'extra'][line_no//25000]
            # [12.5K pos, 12.5K neg]*2 then unknown
            sentiment = sentiment_labels[line_no//12500]
            alldocs.append(SentimentDocument(words, tags, split, sentiment))

    return alldocs

def shared_dataset(data_xy, borrow=True):
    # Function body from theano tutorial

    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def mlp_clsf(datasets, config, batch_size=20):
    # Function body from theano tutorial

    """
    Stochastic gradient descent optimization for a multilayer perceptron

    :type datasets: 3-list/tuple
    :param datasets: contains training, validation, and test sets,
                    each as a 2-tuple of the data instances with
                    their labels

    :type batch_size: int
    :param batch_size: number of instances to examine at a time

    :type configs: dict
    :param configs: contains MLP configurations:
                    learning_rate: float, learning rate used
                    L1_reg: float, L1-norm's weight when added to cost
                    L2_reg: float, L2-norms' weight when added to cost
                    n_epochs: int, maximal number of epochs to run optimizer
                    n_in: int, number of input units
                    n_out: int, number of classes
                    n_hidden: list[int], number of hidden units per layer
    """

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    print('... configuration')
    for k, v in config.items():
        print('\t', k, ':', v)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # Typical configuration
    # learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
    # n_epochs=100, n_hidden=[100]
    learning_rate = config['learning_rate']
    L1_reg = config['L1_reg']
    L2_reg = config['L2_reg']
    n_epochs = config['n_epochs']
    n_in = config['n_in']
    n_out = config['n_out']
    n_hidden = config['n_hidden']

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        inputs=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # Early-stopping parameters

    # Look as this many examples regardless
    patience = train_set_x.get_value(borrow=True).shape[0] // 2
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    print('MLP onfiguration:')
    for k, v in config.items():
        print('\t', k, ':', v)

    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)),
           file=sys.stderr)

dpath = "data/aclImdb/alldata-id.txt"
# Load corpus
assert os.path.isfile(dpath), "alldata-id.txt unavailable"

alldocs = load_imdb_dataset(dpath)
train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
unsup_docs = [doc for doc in alldocs if doc.split == 'extra']
doc_list = alldocs[:]  # For reshuffling per pass

print( '%d docs: %d train-sentiment, %d test-sentiment, %d unsupervised' % \
     ( len(doc_list), len(train_docs), len(test_docs), len(unsup_docs) ))

# Setup training
# Modify this depending on machine
core_count = multiprocessing.cpu_count()
cores = core_count - 2 if core_count > 2 else 1

assert par2vec.models.par2vec.FAST_VERSION > -1, \
       "this will be painfully slow otherwise"
print( 'Using FAST_VERSION:', par2vec.models.par2vec.FAST_VERSION )

# Following models learn by hierarchical softmax
simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates
    #     paper's 10-word total window size
    Doc2Vec( dm=1, dm_concat=1, size=300, window=10,
             negative=0, hs=1, min_count=2, workers=cores ),
    # PV-DBOW
    #Doc2Vec(dm=0, size=100, negative=0, hs=1, min_count=2, workers=cores),
    # PV-DM w/average
    #Doc2Vec( dm=1, dm_mean=1, size=300, window=10, negative=0, hs=1,
    #         min_count=2, workers=cores ),
]

# Speed setup by sharing results of 1st model's vocabulary scan
# PV-DM/concat requires one special NULL word so it serves as template
all_train = alldocs
simple_models[0].build_vocab(all_train)
print('Preparing', simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print('Preparing', model)

# Train concatenation model only for now
models_by_name = OrderedDict( (str(model), model) for model
                              in simple_models )

alpha, min_alpha, passes = (0.025, 0.001, 15)
alpha_delta = (alpha - min_alpha) / passes

print('Paragraph vector config...')
print('    alpha:', alpha, '; min_alpha:', min_alpha, '; passes:', passes)

print("START %s" % datetime.datetime.now())

# Use infer_vector to obtain paragraph vector on test set
# If a word in the paragraph doesn't exist in the vocabulary
# during inference, the word is ignored.  Another approach
# uses both test and training set for training, but the paragraph
# vectors for the test set are still inferred by freezing the
# rest of the network.  The paragraph vector in the test set
# is ignored, and the inferred vector is used instead.  This
# will allow word vectors to be obtained for words that only
# exist in the test set.  During prediction time, consider
# shuffling the contexts if only one document can be inferred
# at a time.
for epoch in range(passes):
    shuffle(all_train)  # shuffling gets best results

    for name, train_model in models_by_name.items():
        # train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(all_train)
            duration = '%.7f' % elapsed()

    print( 'completed pass %i in %ss at alpha %f with %s' %
           (epoch + 1, duration, alpha, name) )
    alpha -= alpha_delta

print("END %s" % str(datetime.datetime.now()))

# Imdb sentiment classification

# Prepare data
infer_steps=5
infer_alpha=0.1
print('Inference config...')
print('    infer_steps:', infer_steps, '; infer_alpha:', infer_alpha)

par2vec_model = simple_models[0]
senti_train = [( par2vec_model.docvecs[doc.tags[0]], doc.sentiment )
                 for doc in train_docs ]
senti_test = []
for doc in test_docs:
    inferred_docvec = par2vec_model.infer_vector(doc.words,
                          steps=infer_steps, alpha=infer_alpha)
    senti_test.append(( inferred_docvec, doc.sentiment ))

shuffle(senti_test)
mlp_train = shared_dataset(zip(*senti_train))
mlp_valid = shared_dataset(zip(*senti_test[:len(senti_test) // 2]))
mlp_test = shared_dataset(zip(*senti_test[len(senti_test) // 2:]))

assert ( mlp_valid[0].get_value(borrow=True).shape[0] > 0 and
         mlp_test[0].get_value(borrow=True).shape[0] > 0), \
       'invalid valid/test set sizes'

mlp_configs = [
    { 'learning_rate' : 0.01,
      'L1_reg'        : 0.00,
      'L2_reg'        : 0.0001,
      'n_epochs'      : 100,
      'n_out'         : 2 }
]

# At the moment, batch_size must be smaller than min(valid, test) sizes
batch_size = 20
hidden2in_ratios = [ 1.25 ]

for config in mlp_configs:
    # Size of first instance in training set
    n_in = simple_models[0].vector_size
    n_hidden = list(map(lambda ratio: int(ratio*n_in), hidden2in_ratios))
    config.update([ ('n_in', n_in), ('n_hidden', n_hidden) ])
    mlp_clsf([ mlp_train, mlp_valid, mlp_test ], config, batch_size )
