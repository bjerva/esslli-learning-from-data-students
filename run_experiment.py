#!/usr/bin/env python

'''
Extract features and learn from them, without saving in between.

Running example:

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
    parser.add_argument('--plot', help='Show plot', action='store_true')
    parser.add_argument('--cm', help='Show confusion matrix', action='store_true')
    parser.add_argument('--norm', help='Normalise confusion matrix', action='store_true')
    parser.add_argument('--min-samples', help='Min leaf samples in decision tree', type=int, default=1)
    parser.add_argument('--max-nodes', help='Max leaf nodes in decision tree', type=int, default=None)

    args = parser.parse_args()

    print('reading features...')
    X, y = read_features_from_csv(args)
    print('one hot encoding...')
    X, feature_ids = features_to_one_hot(X)
    feature_names = [feature_ids[idx] for idx in range(len(feature_ids))]


    train_X, train_y, test_X, test_y = make_splits(X, y)
    #test_X, test_y = train_X, train_y # XXX: Example
    baseline(train_y, test_y)
    classifiers = get_classifiers(args)

    for clf in classifiers:
        clf.fit(train_X, train_y)
        evaluate_classifier(clf, test_X, test_y, args)

    from sklearn import tree
    from sklearn.externals.six import StringIO
    class_names = []
    for label in train_y:
        if label not in class_names:
            class_names.append(label)

    with open("iris.dot", 'w', encoding='utf-8') as f:
        f = tree.export_graphviz(clf, class_names=sorted(class_names), feature_names=feature_names, out_file=f, filled=True, rounded=True,
                         special_characters=True)
