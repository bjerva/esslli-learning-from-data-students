#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script reads a CSV file and extracts pre-defined features from it.
The features are saved in a scikit-learn-friendly manner.
'''

__author__ = "Johannes Bjerva, and Malvina Nissim"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.1"
__maintainer__ = "Johannes Bjerva"
__email__ = "j.bjerva@rug.nl"
__status__ = "early alpha"

import csv
import argparse
from collections import defaultdict

import numpy as np
from scipy.sparse import lil_matrix
#from nltk.stem.snowball import SnowballStemmer

def read_features_from_csv(fname):
    X = []
    y = []
    with open(fname, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=args.delimiter)
        header = csv_reader.next()
        label_index = header.index('label')

        feature_indices = []
        for feature in args.features:
            if feature in header:
                feature_indices.append(header.index(feature))
            else:
                print('Feature {0} not found in header.'.format(feature))

        types = get_column_types(header)
        for line in csv_reader:
            label, features = get_line_features(line, types, label_index, feature_indices)
            X.append(features)
            y.append(label)

    #import pdb; pdb.set_trace()
    return X, np.asarray(y, dtype=str)

def get_column_types(header):

    types = np.zeros((len(header), ), dtype=np.object)
    for idx, name in enumerate(header):
        if 'cat' in name:
            types[idx] = np.array # TODO
        else:
            types[idx] = np.int32

    return types

def get_line_features(line, feature_dtypes, label_index, feature_indices):
    '''
    Gets the features in a line.
    Must have the format (label, feature(s)).
    '''
    #TODO: Add error handling / messages
    # Could go wrong:
    # * Not all features defined
    # * Some features neet to be converted to categories
    # * Lemmatisation etc. for text
    label = line[label_index]
    #label = label_to_id[line[label_index]]

    features = []
    for idx, column in enumerate(line):
        if idx == label_index: continue
        if idx in feature_indices:
            sentence_features.append(cat_to_id[column+'idx'])
        elif feature_dtypes[idx] == np.array:
            sentence_features = []
            if args.nwords:
                for n in xrange(args.nwords):
                    ngrams = find_ngrams(column.split(), n+1)
                    sentence_features.extend([cat_to_id[' '.join(ngram)] for ngram in ngrams])

            if args.nchars:
                for n in xrange(args.nchars):
                    ngrams = find_ngrams(' '.join(column.split()), n+1)
                    sentence_features.extend([cat_to_id[' '.join(ngram)] for ngram in ngrams])

        features.extend(sentence_features)

    features = np.asarray(features)

    return label, features

def find_ngrams(sentence, n):
  return set(zip(*[sentence[idx:] for idx in range(n)]))

def preprocess(word):
    return word.strip()#stemmer.stem(word.strip())

def features_to_one_hot(X):
    '''Convert, e.g., word id features to one hot representation'''
    n_cats   = len(cat_to_id) + 1
    print('n_cats', n_cats)
    one_hot_X = np.zeros((len(X), n_cats), dtype=np.int32)
    # TODO: Fix for several cats
    for idx, sentence in enumerate(X):
        for cat_id in sentence:
            one_hot_X[idx, cat_id] = 1

    return one_hot_X

def save_features(X, y, fname):
    '''Save X and y to a compressed .npz file'''
    np.savez_compressed(fname, X=X, y=y)

parser = argparse.ArgumentParser()
parser.add_argument('--csv', help='feature csv filename', type=str, required=True)
parser.add_argument('--fname', help='filename to store features', type=str, default=None, required=True)
parser.add_argument('--nwords', type=int)
parser.add_argument('--nchars', type=int)
parser.add_argument('--features', nargs='+', default=[])
parser.add_argument('--dtype', help='datatype in file', type=str, default=None)  # TODO: Not implemented
parser.add_argument('--delimiter', help='csv delimiter', type=str, default=',')  # TODO: Not implemented
parser.add_argument('--lang', help='data language', type=str, default='english')
args = parser.parse_args()

label_to_id = defaultdict(lambda: len(label_to_id)+1)
cat_to_id = defaultdict(lambda: len(cat_to_id)+1)

if __name__ == '__main__':
    fname = args.csv[:-4] if not args.fname else args.fname

    #stemmer = SnowballStemmer(args.lang)
    print('reading features...')
    X, y = read_features_from_csv(args.csv)
    print('one hot encoding...')
    X    = features_to_one_hot(X)
    print('saving features...')
    save_features(X, y, fname)
