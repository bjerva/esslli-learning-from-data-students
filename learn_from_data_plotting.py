#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script reads a saved numpy array with features prepared for sklearn.
The features are then used to learn something from the data.
'''

__author__ = "Johannes Bjerva, and Malvina Nissim"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.1"
__maintainer__ = "Johannes Bjerva"
__email__ = "j.bjerva@rug.nl"
__status__ = "early alpha"

import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


def read_features(fname):
    with open(fname, 'rb') as in_f:
        loaded = np.load(in_f)
        return loaded['X'], loaded['y']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(9)
    plt.xticks(tick_marks, sorted(list(set(test_y))), rotation=45)
    plt.yticks(tick_marks, sorted(list(set(test_y))))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def analysis(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=sorted(list(set(test_y))))

    if args.norm:
        cm = np.log(cm)#cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    np.set_printoptions(precision=2)
    print('Confusion matrix')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)

parser = argparse.ArgumentParser()
parser.add_argument('--npz', help='feature npz filename', type=str, required=True)
parser.add_argument('--algorithms', help='ml algorithms', nargs='+', required=True)
parser.add_argument('--norm', help='normalise conf matrix', action='store_true')
args = parser.parse_args()
random.seed(1337)
np.random.seed(1337)
if __name__ == '__main__':
    X, y = read_features(args.npz)
    X = list(X)

    combined = zip(X, y)
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split = int(len(y) * 0.7)

    train_X = X[:split]
    train_y = y[:split]
    test_X  = X[split:]
    test_y  = y[split:]

    most_common = Counter(train_y).most_common()[0][0]
    baseline = sum([1 for label in test_y if label == most_common]) / float(len(test_y))
    print('Most frequent label:\t{0}'.format(most_common))
    print('Baseline accuracy:\t{0}'.format(baseline))

    classifiers = []
    if 'nb' in args.algorithms:
        classifiers.append(MultinomialNB())
    if 'dt' in args.algorithms:
        classifiers.append(DecisionTreeClassifier(max_features='sqrt', random_state=0))
    if 'svm' in args.algorithms:
        classifiers.append(LinearSVC(max_iter=500,random_state=0))
    if 'knn' in args.algorithms:
        classifiers.append(KNeighborsClassifier(n_neighbors=5))

    for clf in classifiers:
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)
        accuracy = sum(preds == test_y) / float(len(test_y))

        print('Accuracy: {0} ({1})'.format(accuracy, str(clf)))
        analysis(test_y, preds)
