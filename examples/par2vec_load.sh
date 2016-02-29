#!/bin/bash

# Retrieved 2016-02-28 from
# https://github.com/piskvorky/gensim/blob/develop/
#     docs/notebooks/doc2vec-IMDB.ipynb
#
# Adapted from Mikolov's example go.sh script:
# https://groups.google.com/d/msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ

# Load corpus

imdblink="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

if [ ! -f "aclImdb/alldata-id.txt" ]
then
    if [ ! -d "aclImdb" ]
    then
        if [ ! -f "aclImdb_v1.tar.gz" ]
        then
            wget --quiet ${imdblink}
        fi
        tar xf aclImdb_v1.tar.gz
    fi

    # This function will convert text to lowercase and will disconnect
    # punctuation and special symbols from words
    function normalize_text {
        awk '{print tolower($0);}' < $1 |
        sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
            -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' \
            -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
            -e 's/<bt>/ /g' -e 's/<br>/ /g' \
            -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
    }

    export LC_ALL=C
    for j in train/pos train/neg test/pos test/neg train/unsup; do
        rm temp
        for i in `ls aclImdb/$j`; do
            cat aclImdb/$j/$i >> temp;
            awk 'BEGIN{print;}' >> temp;
        done
        normalize_text temp
        mv temp-norm aclImdb/$j/norm.txt
    done
    mv aclImdb/train/pos/norm.txt aclImdb/train-pos.txt
    mv aclImdb/train/neg/norm.txt aclImdb/train-neg.txt
    mv aclImdb/test/pos/norm.txt aclImdb/test-pos.txt
    mv aclImdb/test/neg/norm.txt aclImdb/test-neg.txt
    mv aclImdb/train/unsup/norm.txt aclImdb/train-unsup.txt

    cat aclImdb/train-pos.txt aclImdb/train-neg.txt \
        aclImdb/test-pos.txt aclImdb/test-neg.txt \
        aclImdb/train-unsup.txt > aclImdb/alldata.txt
    awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < aclImdb/alldata.txt \
        > aclImdb/alldata-id.txt
fi
