import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import load_iris


def configure_plots():
    '''Configures plots by making some quality of life adjustments'''

    for _ in range(2):
        plt.rcParams['figure.figsize'] = [16, 9]
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['lines.linewidth'] = 2

def distance_measure(a, b):
    '''A measures a distance between point(s) a and b.'''

    return np.linalg.norm(a - b, axis=int(len(a.shape) > 1))

def plot_knn(X_train, X_test, y_train, y_test, k=3):
    from sklearn import neighbors
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    plt.rcParams['figure.figsize'] = [17, 10]

    #specify classifier
    clf = neighbors.KNeighborsClassifier(k)

    #fit our data
    clf.fit(X_train, y_train)

    # setosa, versicolor, virginica
    list_dark = ['#006d2c','#a63603','#08519c']
    cmap_dark = ListedColormap(list_dark)
    list_bold = ['#31a354','#e6550d','#3182bd']
    cmap_bold = ListedColormap(list_bold)
    list_light = ['#bae4b3','#fdbe85','#bdd7e7']
    cmap_light = ListedColormap(list_light)

    # calculate min, max and limits
    X = np.concatenate((X_train, X_test))
    h = 0.02
    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # predict class using data and kNN classifier
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap = cmap_dark)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % k)
    plt.xlabel("Sepal Width")
    plt.ylabel("Sepal Length")
    legend_elements = [
        Line2D([0], [0], marker='o', color=list_dark[0], label='setosa',
               markerfacecolor=list_light[0], markersize=10),
        Line2D([0], [0], marker='o', color=list_dark[1], label=' versicolor',
               markerfacecolor=list_light[1], markersize=10),
        Line2D([0], [0], marker='o', color=list_dark[2], label='virginica',
               markerfacecolor=list_light[2], markersize=10),
        Line2D([0], [0], linewidth=0, marker='*', color='gray', label='test point (with ground truth color)',
                markerfacecolor='gray', markersize=15),
    ]

    # Plot also the test points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='*', s=160, cmap=cmap_bold)

    plt.legend(handles = legend_elements)
    plt.show()
