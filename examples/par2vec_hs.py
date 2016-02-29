#!/bin/bash

"""Retrieved 2016-02-28 from
https://github.com/piskvorky/gensim/blob/develop/
    docs/notebooks/doc2vec-IMDB.ipynb

Example of gensim doc2vec usage on Imdb dataset
"""

import os.path

import numpy as np
import gensim
import gensim.models.doc2vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple, OrderedDict

import multiprocessing
import random
from random import shuffle

# For timing
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

# Load corpus
assert os.path.isfile("data/aclImdb/alldata-id.txt"), \
       "alldata-id.txt unavailable"

SentimentDocument = namedtuple( 'SentimentDocument',
                                'words tags split sentiment' )

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

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
unsup_docs = [doc for doc in alldocs if doc.split == 'extra']
doc_list = alldocs[:]  # for reshuffling per pass

print( '%d docs: %d train-sentiment, %d test-sentiment, %d unsupervised' % \
     ( len(doc_list), len(train_docs), len(test_docs), len(unsup_docs) ))

# Setup Doc2Vec training
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, \
       "this will be painfully slow otherwise"

# Following models learn by hierarchical softmax
simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates
    #     paper's 10-word total window size
    Doc2Vec( dm=1, dm_concat=1, size=100, window=10, negative=0,
             hs=1, min_count=2, workers=cores ),
    # PV-DBOW
    Doc2Vec(dm=0, size=100, negative=0, hs=1, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec( dm=1, dm_mean=1, size=100, window=10, negative=0, hs=1,
             min_count=2, workers=cores ),
]

# Speed setup by sharing results of 1st model's vocabulary scan
# PV-DM/concat requires one special NULL word so it serves as template
simple_models[0].build_vocab(alldocs)
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

# Train concatenation model only for now
models_by_name = OrderedDict( (str(model), model) for model
                              in simple_models[:1] )

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())


# Use infer_vector to obtain paragraph vector on test set
all_train = (train_docs + unsup_docs)
for epoch in range(passes):
    shuffle(all_train)  # shuffling gets best results

    for name, train_model in models_by_name.items():
        # train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(all_train)
            duration = '%.1f' % elapsed()

    print( 'completed pass %i at alpha %f with %s' %
           (epoch + 1, alpha, name) )
    alpha -= alpha_delta

print("END %s" % str(datetime.datetime.now()))

# Do close documents seem more related than distant ones?
# Pick random doc, re-run cell for more examples
doc_id = np.random.randint(simple_models[0].docvecs.count)
model = simple_models[0]
# Get *all* similar documents
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)
print(u'TARGET (%d): «%s»\n' % (doc_id, ' '.join(alldocs[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
doc_stats = [ ('MOST', 0), ('MEDIAN', len(sims)//2),
              ('LEAST', len(sims) - 1) ]
for label, index in doc_stats:
    print(u'%s %s: «%s»\n' % ( label, sims[index],
                               ' '.join(alldocs[sims[index][0]].words) ))

# Do word vectors show useful similarities?
word_models = simple_models[:1]
# pick a random word with a suitable number of occurences
while True:
    word = random.choice(word_models[0].index2word)
    if word_models[0].vocab[word].count > 10:
        break
# Or uncomment below line, to just pick a word from the relevant domain:
#word = 'comedy/drama'

similars_per_model = [
    str(model.most_similar(word, topn=20)).replace('), ','),\n')
    for model in word_models
]

print( "most similar words for '%s' (%d occurences)" %
       (word, simple_models[0].vocab[word].count) )
for model, similars in zip(word_models, similars_per_model):
    print(model)
    print(similars)
