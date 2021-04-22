import io
import warnings
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from os.path import exists
from contextlib import redirect_stdout


# Plotting Support Functions
def configure_plots():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.set(style="ticks", color_codes=True, font_scale=1.5)
        sns.set_palette(sns.color_palette())
        plt.rcParams['figure.figsize'] = [16, 9]


def pair_plot(data, labels, names):
    n, d = data.shape
    hues = np.unique(labels)
    marks = itertools.cycle(('o', 's', '^', '.', 'd', ',')[:min(len(hues),6)])
    
    plt.figure(figsize=(16,12))
    _, axs = plt.subplots(d, d, sharex='col', sharey='row')

    for row in range(d):
        cat = data[:, row]
        
        # rescale y axes
        axs[row, 0].set_ylim(min(cat),max(cat))
        
        # set row and column labels
        axs[d-1, row].set_xlabel(names[row])
        axs[row, 0].set_ylabel(names[row])
        
        for column in range(d):
            ax = axs[row, column]
            
            # remove spines from top and right sides
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if column > row:
                ax.axis('off')
                continue
            
            if row == column:
                by_hue = [cat[np.where(labels == hue)] for hue in hues]
                ax.get_shared_y_axes().remove(ax)
                ax.autoscale()
                ax.hist(by_hue, bins=30, stacked=True)
            else:
                for hue in hues:
                    ax.scatter(data[:, column][np.where(labels == hue)],
                               data[:, row][np.where(labels == hue)],
                               20, marker=next(marks))


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


 
# Example Model

class Model:
    def __init__(self):
        self._clf = KNeighborsClassifier(n_neighbors=3)
        
    def fit(self, X, y):
        self.model = self._clf.fit(X, y)
        return self.model
 
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

    def __str__(self):
        return '<Secret Model>'
    
    def __repr__(self):
        return self.__str__()
    
