# paragraph2vec
Experimenting with Le and Mikolov's Paragraph Vectors in
"Distributed Representations of Sentences and Documents"

Description
-----------
This is primarily a learning exercise to obtain paragraph vectors
using gensim with some additional stuff from Theano.

Setup
-----
If C compiler exists, then entering

    python setup.py build_ext --inplace

will build a fast version of the training functions.
Changes to the Cython files in par2vec/models will require
rebuilding of the libraries.

Requirements
------------
gensim >= 0.12.0
