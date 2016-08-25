"""
==========
Libsvm GUI
==========
A simple graphical frontend for Libsvm mainly intended for didactic
purposes. You can create data points by point and click and visualize
the decision region induced by different kernels and parameter settings.
Requirements
------------
 - Tkinter
 - scikits.learn
 - matplotlib with TkAgg
"""
from __future__ import division

#!/usr/bin/env python
#
# Author: Peter Prettenhoer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import Tkinter as Tk
import sys
import numpy as np

np.random.seed(1337)

from sklearn import svm, tree, neighbors, naive_bayes

y_min, y_max = 0, 100
x_min, x_max = 0, 100


class Model(object):
    def __init__(self):
        self.observers = []
        self.surface = None
        self.data = []
        self.cls = None
        self.surface_type = 0

    def changed(self, event):
        for observer in self.observers:
            observer.update(event, self)

    def add_observer(self, observer):
        self.observers.append(observer)

    def set_surface(self, surface):
        self.surface = surface


class Controller(object):
    def __init__(self, model):
        self.model = model
        self.classifier = Tk.IntVar()
        self.kernel = Tk.IntVar()
        self.surface_type = Tk.IntVar()

    def classify(self):
        print("classifying data")
        train = np.array(self.model.data)
        X = train[:, :2]
        y = train[:, 2]

        if self.classifier.get() == 0:
            print('decision tree')
            clf = tree.DecisionTreeClassifier()
            clf.fit(X, y)
        elif self.classifier.get() == 1:
            print('svm')
            kernel_map = {0: "linear", 1: "rbf", 2: "poly"}
            if len(np.unique(y)) == 1:
                clf = svm.OneClassSVM(kernel=kernel_map[self.kernel.get()])
                clf.fit(X)
            else:
                clf = svm.SVC(kernel=kernel_map[self.kernel.get()])
                clf.fit(X, y)
        elif self.classifier.get() == 2:
            print('naive bayes')
            clf = naive_bayes.MultinomialNB()
            clf.fit(X, y)
        elif self.classifier.get() == 3:
            n = int(self.neighbors.get())
            if n >= len(X):
                n = min(2, len(X))
            print('knn', n)
            clf = neighbors.KNeighborsClassifier(n_neighbors=n)
            clf.fit(X, y)

        if hasattr(clf, 'score'):
            print("Accuracy:", clf.score(X, y) * 100)
        X1, X2, Z = self.decision_surface(clf)
        self.model.clf = clf
        self.model.set_surface((X1, X2, Z))
        self.model.surface_type = self.surface_type.get()
        self.model.changed("surface")

        sys.stdout.flush()

    def decision_surface(self, cls):
        delta = 1
        x = np.arange(x_min, x_max + delta, delta)
        y = np.arange(y_min, y_max + delta, delta)
        X1, X2 = np.meshgrid(x, y)
        Z = cls.predict(np.c_[X1.ravel(), X2.ravel()])
        Z = Z.reshape(X1.shape)
        return X1, X2, Z

    def clear_data(self):
        self.model.data = []
        self.model.changed("clear")

    def add_example(self, x, y, label):
        self.model.data.append((x, y, label))
        self.model.changed("example_added")


class View(object):
    def __init__(self, root, controller):
        f = Figure()
        ax = f.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas.mpl_connect('button_press_event', self.onclick)
        toolbar = NavigationToolbar2TkAgg(canvas, root)
        toolbar.update()
        self.controllbar = ControllBar(root, controller)
        self.f = f
        self.ax = ax
        self.canvas = canvas
        self.controller = controller
        self.hascolormaps = False
        self.contours = []
        self.c_labels = None

    def onclick(self, event):
        if event.xdata and event.ydata:
            if event.button == 1:
                self.controller.add_example(event.xdata, event.ydata, 1)
            elif event.button == 3:
                self.controller.add_example(event.xdata, event.ydata, -1)

    def update(self, event, model):
        if event == "example_added":
            x, y, l = model.data[-1]
            if l == 1:
                color = 'w'
            elif l == -1:
                color = 'k'
            self.ax.plot([x], [y], "%so" % color, scalex=0.0, scaley=0.0)

        if event == "clear":
            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.contours = []
            self.c_labels = None

        if event == "surface":
            self.plot_decision_surface(model.surface, model.surface_type)

        self.canvas.draw()

    def plot_decision_surface(self, surface, type):
        X1, X2, Z = surface

        if len(self.contours) > 0:
            for contour in self.contours:
                for lineset in contour.collections:
                    lineset.remove()
            self.contours = []

        if self.c_labels:
            for label in self.c_labels:
                label.remove()

        if type == 0:
            levels = [-1.0, 0.0, 1.0]
            linestyles = ['dashed', 'solid', 'dashed']
            colors = 'k'
            self.contours.append(self.ax.contour(X1, X2, Z, levels,
                                                 colors=colors,
                                                 linestyles=linestyles))
        elif type == 1:
            self.contours.append(self.ax.contourf(X1, X2, Z, 10,
                                             cmap=matplotlib.cm.bone,
                                             origin='lower',
                                             alpha=0.85))
            self.contours.append(self.ax.contour(X1, X2, Z, [0.0],
                                                 colors='k',
                                                 linestyles=['solid']))
        else:
            raise ValueError("surface type unknown")


class ControllBar:
    def __init__(self, root, controller):
        fm = Tk.Frame(root)
        classifier_group = Tk.Frame(fm)
        Tk.Radiobutton(classifier_group, text="Decision tree", variable=controller.classifier,
                       value=0).pack(anchor=Tk.W)
        Tk.Radiobutton(classifier_group, text="SVM", variable=controller.classifier,
                       value=1).pack(anchor=Tk.W)
        Tk.Radiobutton(classifier_group, text="Naive Bayes", variable=controller.classifier,
                       value=2).pack(anchor=Tk.W)
        Tk.Radiobutton(classifier_group, text="k-NN", variable=controller.classifier,
                       value=3).pack(anchor=Tk.W)
        classifier_group.pack(side=Tk.LEFT)

        kernel_group = Tk.Frame(fm)
        Tk.Radiobutton(kernel_group, text="Linear", variable=controller.kernel,
                       value=0).pack(anchor=Tk.W)
        Tk.Radiobutton(kernel_group, text="RBF", variable=controller.kernel,
                       value=1).pack(anchor=Tk.W)
        Tk.Radiobutton(kernel_group, text="Poly", variable=controller.kernel,
                       value=2).pack(anchor=Tk.W)
        kernel_group.pack(side=Tk.LEFT)

        valbox = Tk.Frame(fm)

        controller.neighbors = Tk.StringVar()
        controller.neighbors.set("5")
        g = Tk.Frame(valbox)
        Tk.Label(g, text='k-NN Neighbors:').pack(side=Tk.LEFT)
        Tk.Entry(g, width=6, textvariable=controller.neighbors).pack(side=Tk.LEFT)
        g.pack()
        valbox.pack(side=Tk.LEFT)

        cmap_group = Tk.Frame(fm)
        Tk.Radiobutton(cmap_group, text="Hyperplanes",
                       variable=controller.surface_type, value=0).pack(
            anchor=Tk.W)
        Tk.Radiobutton(cmap_group, text="Surface",
                       variable=controller.surface_type, value=1).pack(
            anchor=Tk.W)

        cmap_group.pack(side=Tk.LEFT)

        train_button = Tk.Button(fm, text='Train', command=controller.classify)
        train_button.pack()
        fm.pack(side=Tk.LEFT)
        Tk.Button(fm, text='Clear',
                  command=controller.clear_data).pack(side=Tk.LEFT)


def main(argv):
    root = Tk.Tk()
    model = Model()
    controller = Controller(model)
    root.wm_title("SVM")
    view = View(root, controller)
    model.add_observer(view)
    Tk.mainloop()

def run():
    main('x')

if __name__ == "__main__":
    main(sys.argv)
