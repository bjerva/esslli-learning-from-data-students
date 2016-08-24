#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script reads a saved numpy array with features prepared for sklearn.
The features are then used to learn something from the data.
'''

__author__ = "Johannes Bjerva, and Malvina Nissim"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.2"
__maintainer__ = "Johannes Bjerva"
__email__ = "j.bjerva@rug.nl"
__status__ = "early alpha"

import argparse
import random
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC

def read_features(fname):
    with open(fname, 'rb') as in_f:
        loaded = np.load(in_f)
        return loaded['X'], loaded['y']

def make_splits(X, y):
    X = list(X) # TODO: Why does this have to be a list?
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split = int(len(y) * 0.8)
    train_X = X[:split]
    train_y = y[:split]
    test_X  = X[split:]
    test_y  = y[split:]

    return train_X, train_y, test_X, test_y

def baseline(train_y, test_y):
    most_common = Counter(train_y).most_common()[0][0]
    baseline = sum([1 for label in test_y if label == most_common]) / float(len(test_y))
    print('Most frequent label:\t{0}'.format(most_common))
    print('Baseline accuracy:\t{0}'.format(baseline))

def get_classifiers(args):
    classifiers = []
    if 'nb' in args.algorithms:
        classifiers.append(MultinomialNB())
    if 'dt' in args.algorithms:
        classifiers.append(DecisionTreeClassifier(
        random_state=0,
        criterion='entropy',
        min_samples_leaf=args.min_samples,
        max_leaf_nodes=args.max_nodes))
    if 'svm' in args.algorithms:
        classifiers.append(LinearSVC(max_iter=500,random_state=0))
    if 'knn' in args.algorithms:
        classifiers.append(KNeighborsClassifier(n_neighbors=5))

    return classifiers

def evaluate_classifier(clf, test_X, test_y, args):
    preds = clf.predict(test_X)
    accuracy = accuracy_score(preds, test_y)# sum(preds == test_y) / float(len(test_y))

    print('Accuracy: {0} ({1})'.format(accuracy, str(clf)))
    if args.cm or args.plot:
        show_confusion_matrix(test_y, preds, args)

def show_confusion_matrix(test_y, pred_y, args):
    cm = confusion_matrix(test_y, pred_y, labels=sorted(list(set(test_y))))

    if args.norm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)

    print('Confusion matrix')
    if args.cm:
        print(cm)
    if args.plot:
        from plotting import plot_confusion_matrix # Import here due to potential matplotlib issues
        plot_confusion_matrix(cm, test_y)

    print(classification_report(test_y, pred_y, sorted(list(set(test_y)))))

random.seed(1337)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', help='feature npz filename', type=str)
    parser.add_argument('--algorithms', help='ml algorithms', nargs='+', required=True)
    args = parser.parse_args()

    X, y = read_features(args.npz)
    train_X, train_y, test_X, test_y = make_splits(X, y)
    baseline(train_y, test_y)
    classifiers = get_classifiers()

    for clf in classifiers:
        clf.fit(train_X, train_y)
        evaluate_classifier(clf)
