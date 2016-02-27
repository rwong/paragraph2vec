"""
Starter code from
http://deeplearning.net/tutorial/logreg.html
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import stat

import numpy

import theano
import theano.tensor as T

__all__ = [
    'load_mnist',
    'load_imdb_raw',
]

def load_mnist(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def _read_imdb_data(directory):
    # Assumes file names are [number]_[number].[extension]
    dataset = [ ]
    files = os.listdir(directory)
    for doc in files:
        [ id_str, rating_str ] = doc.rsplit('.', 1)[0].split('_')
        with open(directory + '/' + doc, 'r') as opendoc:
            contents = opendoc.read()
            dataset.append([ int(id_str), int(rating_str), contents ])

    return dataset

def load_imdb_raw(datapath):
    # Expecting path to folder to have following structure
    # datapath
    #   |- test
    #     |- neg
    #     |- pos
    #   |- train
    #     |- neg
    #     |- pos
    #     |- unsup

    if (not os.path.exists(datapath)):
        sys.stderr.write("Path \"" + datapath + "\" does not exist...")
        return

    if (not stat.S_ISDIR(os.stat(datapath).st_mode)):
        sys.stderr.write("Path \"" + datapath + "\" not a directory...")
        return

    testdirs = [ 'test/neg', 'test/pos' ]
    traindirs = [ 'train/neg', 'train/pos' ]
    unsupdirs = [ 'train/unsup' ]
    for dirname in testdirs + traindirs + unsupdirs:
        if (not os.path.exists(datapath + '/' + dirname)):
            sys.stderr.write("Missing some test/train directories...")
            return

    # Returns a 3-tuple (test, train, unsupervised) where each item of the
    # 3-tuple is a list of lists. Each element of test and train is the
    # raw document data with the sentiment label, id from the file name
    # [id]_[rating].txt, it's movie rating, and the document contents.
    # The unsupervised training set will have a sentiment of -1 and rating
    # of 0.  Positive sentiment is 1; negative is 0.
    neg_test_entries = _read_imdb_data(datapath + '/' + testdirs[0])
    neg_test_entries = list(map(lambda ent: ent + [0], neg_test_entries))
    pos_test_entries = _read_imdb_data(datapath + '/' + testdirs[1])
    pos_test_entries = list(map(lambda ent: ent + [1], pos_test_entries))

    neg_train_entries = _read_imdb_data(datapath + '/' + traindirs[0])
    neg_train_entries = list(map(lambda ent: ent + [0], neg_train_entries))
    pos_train_entries = _read_imdb_data(datapath + '/' + traindirs[1])
    pos_train_entries = list(map(lambda ent: ent + [1], pos_train_entries))

    unsup_entries = _read_imdb_data(datapath + '/' + unsupdirs[0])
    unsup_entries = list(map(lambda ent: ent + [-1], unsup_entries))

    testset = neg_test_entries + pos_test_entries
    trainset = neg_train_entries + pos_train_entries
    return testset, trainset, unsup_entries
