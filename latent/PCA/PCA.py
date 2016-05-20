#!/usr/bin/env python

import theano.tensor as T

class PCA(object):

    def __init__(self, data, components=None, threshold=None):
        self.data = data

        if components is None and threshold is None:
            print('Either Components Or Threshold Parameters Needs To Have A Value')
            exit(-1)

        self.components = components
        self.threshold = threshold
 
    def process(self):
        data = self.data - self.data.mean(axis=0)
        cov = T.dot(data.T, data.conj())/ (data.shape[0]-1)
        evals, evecs = T.nlinalg.eig(cov)
        
        if self.components is None and \
                self.threshold is not None:
            self.components = T.gt(evals, self.threshold).sum()

        key = T.argsort(evals)[::-1][:self.components]

        self.evals, self.evecs = evals[key], evecs[:, key]
        self.pca = T.dot(self.evecs.T, data.T).T
        return self.pca