import numpy as np
import matplotlib.pyplot as plt

from numpy.random import multivariate_normal


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

    for i in range(niterations):
        clusters = assignment_step(Z, centroids)
        centroids = update_step(Z, clusters, k)
        plot_clustering(Z, clusters, centroids, i)

    return centroids


def plot_clustering(Z, clusters, centroids, it):
    plt.figure()
    for i in range(len(centroids)):
        X = Z[clusters == i, :]
        plt.plot(X[:, 0], X[:, 1], 'x')
        plt.savefig('%d.png' % it)


### IMPLEMENT ME ####

def initialize(k):
    # INSERT CODE HERE
    pass


def squared_euclidean_distances(Z, c):
    # INSERT CODE HERE
    pass


def assignment_step(Z, centroids):
    # INSERT CODE HERE
    pass


def update_step(Z, clusters, k):
    # INSERT CODE HERE
    pass


N = 10
k = 3

Z = generate_data_set(N)
my_kmeans(Z, k)
