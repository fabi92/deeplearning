#!/usr/bin/env python
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
import cPickle as pickle

import Cost

class LogisticRegressor(object):
    def __init__(self, input, labels, n_in=None, n_out=None, weights=None, bias=None):
        if n_in is not None and n_out is not None:
            weights = np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            )
            bias = value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            )

        self.weights = theano.shared(
            value=weights,
            name='weights',
            borrow=True
        )
        self.bias = theano.shared(
            value=bias,
            name='bias',
            borrow=True
        )

        self.input = input
        self.labels = labels
        self.predictedHotOne = T.nnet.softmax(T.dot(self.input, self.weights) + self.bias)
        self.predictions = T.argmax(self.predictedHotOne, axis=1)

        self.params = [self.weights, self.bias]
        self.cost = Cost.negative_log_likelihood(self.predictedHotOne, self.labels)
        self.missclassified = T.mean(T.neq(self.predictions, self.labels))

    def saveRegressor(self, path):
        with open(path + '/regressor.lgr', 'wb') as f:
            weights = self.weights.eval()
            bias = self.bias.eval()
            pickle.dump([weights, bias], f)

def loadParams(path):
    if not path.endswith('.lgr'):
        print('A File From Type lgr Needs To Be Provided')
        exit(-1)
    return pickle.load( open( path , "rb" ) )