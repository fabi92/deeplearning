#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

class MinibatchKmeans(object):

    def __init__(self, data, t_dataToCentroids, n_clusters, dims, learning_rate, init_cluster_vals=None):
        self.dims = dims
        self.n_clusters = n_clusters
        self.data = data
        self.t_dataToCentroids = t_dataToCentroids
        self.learning_rate = learning_rate
        
        if init_cluster_vals is None:
            self.centroids = theano.shared(
                value=np.random.rand(self.n_clusters, self.dims[1]).astype(theano.config.floatX),
                name="centroids"
            ) ## (20,784)
        else:
            self.centroids = theano.shared(
                value=init_cluster_vals.astype(theano.config.floatX),
                name="centroids"
            )

        self.dataToCentroids = self.getCentroids() ## (20,)
        self.dist = self.distance()
        self.error = self.getError()


    def getCentroids(self):
        return ((self.centroids**2).sum(axis=1, keepdims=True) + \
                   (self.data**2).sum(axis=1, keepdims=True).T - \
                   2*T.dot(self.centroids, self.data.T)).argmin(axis=0)

    def distance(self):
        return T.dot(self.t_dataToCentroids.T, self.data) - \
                self.t_dataToCentroids.sum(0)[:, None] * self.centroids

    def getError(self):
        return abs(self.dist).sum() / self.dims[0]

    def get_updates(self):## (20,)
        new_centroids = self.centroids - self.learning_rate * T.grad(self.error, self.centroids) #+ self.learning_rate * self.dist
        return [(self.centroids, new_centroids)]
