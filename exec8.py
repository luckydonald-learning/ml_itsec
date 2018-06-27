import colorsys
from math import sqrt
from luckydonaldUtils.files.basics import mkdir_p
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  # if tkinter fails: $ sudo apt-get install python3-tk
from matplotlib import colors

from numpy.random import multivariate_normal, rand


def generate_data_set(ncenters):
    var = 0.3
    means = np.random.rand(ncenters, 2) * 6

    X = [generate_gaussian_data(means[i], var) for i in range(ncenters)]
    return np.concatenate(tuple(X))


def generate_gaussian_data(mean, var, n=100):
    v = [[var, 0], [0, var]]
    return multivariate_normal(mean, v, n)


def my_kmeans(Z, k, niterations=20):
    centroids = initialize(k)
    plot_clustering(Z, None, centroids, 0)
    for i in range(niterations):
        clusters = assignment_step(Z, centroids)
        centroids = update_step(Z, clusters, k)
        plot_clustering(Z, clusters, centroids, i+1)

    return centroids


def plot_clustering(Z, clusters, centroids, it):
    cluster_colors = []
    plt.figure()
    plt.autoscale(True)
    for i in range(len(centroids)):
        if clusters:
            X = np.array([Z[x] for x in range(len(Z)) if clusters[x] == i])
        else:
            X = np.array(Z)
        # end if
        if len(X) != 0 and clusters:
            p = plt.plot(X[:, 0], X[:, 1], 'x', label='cluster #{}'.format(i))
            c = p[0].get_color()
            c = colorsys.rgb_to_hls(*colors.to_rgb(c))  # https://stackoverflow.com/a/49601444/3423324
            c = colorsys.hls_to_rgb(c[0], 0.5 * (1 - c[1]), c[2])
            cluster_colors.append(c)
        else:
            if not clusters:
                plt.plot(X[:, 0], X[:, 1], 'x', color='black', label='initial cluster'.format(i))
            # end if
            cluster_colors.append(None)
        # end if
    # end for
    for i in range(len(centroids)):
        plt.autoscale(False)
        plt.plot(
            centroids[i][0], centroids[i][1], '+', label='centroid #{}'.format(i), color=cluster_colors[i], markersize=20,
            scaley=False, scalex=False
        )
    # end for
    # plt.legend(loc="lower right")
    mkdir_p('out')
    file = 'out/%d.png' % it
    plt.savefig(file)
    print('saved {!r}'.format(file))


def initialize(k):
    """ 
    Generates k random d-dimensional centroids.

    :param k:

    :rtype: array of array[float, float]
    """
    return rand(k, 2) * 8  # from 0 to 8
# end def


def squared_euclidean_distances(Z, c):
    """
    Distance of the datapoint Z[i] to the centroid c

    :param Z: array of points
    :type  Z: array of tuple[float, float]

    :param c: centroid
    :type  c: tuple[float, float]
    """
    n = len(Z)
    x1, y1 = c
    D = np.zeros((n, 1))
    for i in range(n):
        x0, y0 = Z[i]
        D[i] = _distance_between_points(x0, y0, x1, y1)
    # end for
    return D
# end def


def _distance_between_points(x0, y0, x1, y1):
    return sqrt(((x1 - x0) * (x1 - x0)) + ((y1 - y0) * (y1 - y0)))
# end def


def assignment_step(Z, centroids):
    clusters = []
    for i_n in range(len(Z)):
        x0, y0 = Z[i_n]
        best_distance = None
        best_centroid = -1
        for i_k in range(len(centroids)):
            x1, y1 = centroids[i_k]
            new_distance = _distance_between_points(x0, y0, x1, y1)
            if best_distance is None or best_distance > new_distance:
                best_distance = new_distance
                best_centroid = i_k
            # end if
        # end for
        clusters.append(best_centroid)
    # end for
    return clusters


def update_step(Z, clusters, k):
    """
    Updates the centroid by calculating the mean of the vectors of each cluster.
    
    :param Z: 
    :param clusters: 
    :param k:
    :return: 
    """
    # mean = {0: (0.123, 0.2323), 1: (4.54645, 4.123), 2: (23.3, 45.0)}
    sum = [np.array([0.0, 0.0]) for i in range(k)]  # key: cluster, value: tuple(x,y)
    count = [0 for i in range(k)]  # key: cluster, value: int
    for i in range(len(clusters)):
        cluster = clusters[i]
        value = Z[i]
        sum[cluster] += value
        count[cluster] += 1
    # end for
    mean = np.array(sum) / np.array([count, count]).T
    return mean
    


N = 20
k = 10

Z = generate_data_set(N)
my_kmeans(Z, k)
