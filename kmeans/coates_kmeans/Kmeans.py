#!/usr/bin/env python

import theano.tensor as T
import theano
import pylab
import numpy as np

class Kmeans(object):

    def __init__(self, data, n_inputs, n_dim, centers,
                                   e_zca=0.03, e_norm=10):
        self.centers = centers
        self.n_inputs = n_inputs
        self.n_dim = n_dim
        self.e_zca = e_zca
        self.e_norm = e_norm
        
        valueVec = np.random.randint(n_inputs, size=centers)
        self.centroids = theano.shared(
            value=data[valueVec],
            name='centroids',
            borrow=True
        )

        self.data = theano.shared(np.asarray(data,
            dtype=theano.config.floatX),
            name='data',
            borrow=True
        )

        self.iteration = theano.shared(0)
        self.data_whiten = self.preprocess()

    def get_updates(self):
        argmaxed = abs(T.dot(self.centroids, self.data_whiten.T)).T.argmax(axis=1)
        J = theano.tensor.extra_ops.to_one_hot(argmaxed, self.centers, dtype='int32')
        S = T.dot(self.centroids, self.data_whiten.T).T * J
        
        new_centroids = T.dot(S.T, self.data_whiten) + self.centroids
        new_centroids = new_centroids / T.sqrt(T.sum(new_centroids**2, axis=1, keepdims=True))
        return [(self.centroids, new_centroids)]

    def preprocess(self):
        self.data_normalized = (self.data - self.data.mean(axis=0)) / T.sqrt(T.var(self.data, axis=0) + self.e_norm)

        self.data_normalized = self.data_normalized - self.data_normalized.mean(axis=0)
        self.X = self.data_normalized 
        self.cov = T.dot(self.data_normalized.T, self.data_normalized.conj())/ (self.n_inputs-1)
        
        self.D, self.V = T.nlinalg.eig(self.cov)
        self.I_eZca = self.D * self.e_zca
        return T.dot(self.V * (1./T.sqrt(self.D + self.I_eZca)) * self.V.T, self.data_normalized.T).T

    

    def applyToData(self):
        return ((self.centroids**2).sum(axis=1, keepdims=True) + \
           (self.data_whiten**2).sum(axis=1, keepdims=True).T - 2 * \
            T.dot(self.centroids, self.data_whiten.T)).argmin(axis=0)