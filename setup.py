#!/usr/bin/env python

"""Experimenting with gensim and theano."""

import os
import sys
import warnings
import io

if sys.version_info[:2] < (2, 6):
    raise Exception('This version needs Python 2.6 or later.')

import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

model_dir = os.path.join(os.path.dirname(__file__), 'par2vec', 'models')

extensions = [
    Extension('par2vec.models.word2vec_inner',
        ['./par2vec/models/word2vec_inner.pyx'],
        include_dirs=[model_dir,
                      numpy.get_include()]),
    Extension('par2vec.models.par2vec_inner',
        ['./par2vec/models/par2vec_inner.pyx'],
        include_dirs=[model_dir,
                      numpy.get_include()]),
]

setup(
    name='par2vec',
    version='0.0.1a',
    description='Experimenting with gensim',

    ext_modules=cythonize(extensions),
)
