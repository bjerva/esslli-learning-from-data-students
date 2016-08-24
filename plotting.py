#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Plotting utilities.
'''

__author__ = "Johannes Bjerva, and Malvina Nissim"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.2"
__maintainer__ = "Johannes Bjerva"
__email__ = "j.bjerva@rug.nl"
__status__ = "early alpha"

import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, test_y, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(np.vstack((cm, np.zeros(cm.shape[0]))), interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, sorted(list(set(test_y))), rotation=45)
    plt.yticks(tick_marks, sorted(list(set(test_y))))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
