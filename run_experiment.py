#!/usr/bin/env python

'''
Extract features and learn from them, without saving in between.

Running example:
python run_experiment.py --csv data/trainset-sentiment-extra.csv --algorithms nb --nwords 1
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

# def plot_decision_tree():
#     from sklearn import tree
#     from sklearn.externals.six import StringIO
#     feature_names = [feature_ids[idx] for idx in range(len(feature_ids))]
#     class_names = []
#     for label in train_y:
#         if label not in class_names:
#             class_names.append(label)
#
#     with open("decision_tree.dot", 'w', encoding='utf-8') as f:
#         f = tree.export_graphviz(clf, class_names=sorted(class_names), feature_names=feature_names, out_file=f, filled=True, rounded=True,
#                          special_characters=True)

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
    parser.add_argument('--k', help='number of neighbours for k-NN', type=int, default=1)
    parser.add_argument('--max-train-size', help='maximum number of training instances to look at', type=int, default=None)

    args = parser.parse_args()

    print('reading features...')
    X, y = read_features_from_csv(args)
    print('one hot encoding...')
    X, feature_ids = features_to_one_hot(X)

    train_X, train_y, dev_X, dev_y, test_X, test_y = make_splits(X, y)
    if args.max_train_size:
        train_X = train_X[:args.max_train_size]
        train_y = train_y[:args.max_train_size]

    print('n train samples: {0}'.format(len(train_y)))
    baseline(train_y, dev_y)
    classifiers = get_classifiers(args)

    for clf in classifiers:
        clf.fit(train_X, train_y)
        print('Results on the train set:')
        evaluate_classifier(clf, train_X, train_y, args)
        print('\n\nResults on the test set:')
        evaluate_classifier(clf, dev_X, dev_y, args)

    #print('Test set:')
    #evaluate_classifier(clf, test_X, test_y, args)
