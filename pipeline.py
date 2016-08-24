#!/usr/bin/env python

'''
Extract features and learn from them, without saving in between.
'''

__author__ = "Johannes Bjerva, and Malvina Nissim"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.2"
__maintainer__ = "Johannes Bjerva"
__email__ = "j.bjerva@rug.nl"
__status__ = "early alpha"

from feature_extractor import *
from learn_from_data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='feature csv filename', type=str, required=True)
    parser.add_argument('--fname', help='filename to store features', type=str, default=None)
    parser.add_argument('--nwords', type=int)
    parser.add_argument('--nchars', type=int)
    parser.add_argument('--features', nargs='+', default=[])
    parser.add_argument('--dtype', help='datatype in file', type=str, default=None)  # TODO: Not implemented
    parser.add_argument('--delimiter', help='csv delimiter', type=str, default=',')  # TODO: Not implemented
    parser.add_argument('--lang', help='data language', type=str, default='english')

    parser.add_argument('--npz', help='feature npz filename', type=str)
    parser.add_argument('--algorithms', help='ml algorithms', nargs='+', required=True)
    args = parser.parse_args()

    print('reading features...')
    X, y = read_features_from_csv(args)
    print('one hot encoding...')
    X    = features_to_one_hot(X)
    #print('pruning...')
    #X    = prune_features(X)

    train_X, train_y, test_X, test_y = make_splits(X, y)
    baseline(train_y, test_y)
    classifiers = get_classifiers(args)

    for clf in classifiers:
        clf.fit(train_X, train_y)
        evaluate_classifier(clf, test_X, test_y)
