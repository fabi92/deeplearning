#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import timeit
import theano.tensor as T
import theano
import numpy as np

class ReducedLayer(object):

    def __init__(self, input, rng, n_input, n_reduced, activation):
        self.input = input

        self.weights = theano.shared(
            value=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_input + n_reduced)),
                        high=np.sqrt(6. / (n_input + n_reduced)),
                        size=(n_input, n_reduced)
                    )
            ),
            name='weights',
            borrow=True
        )

        # initialize the biases b as a vector of n_out 0s
        self.bias = theano.shared(
            value=np.zeros(
                (n_reduced,),
                dtype=theano.config.floatX
            ),
            name='bias',
            borrow=True
        )

        self.params = [self.weights, self.bias]

        if activation == theano.tensor.nnet.sigmoid:
            self.weights *= 4
        self.activation = activation

        self.output = activation(T.dot(input, self.weights) + self.bias)


class ReconstructionLayer(object):

    def __init__(self, input, rng, n_reduced, n_reconstructed, activation):
        self.input = input

        self.weights = theano.shared(
            value=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_reduced + n_reconstructed)),
                        high=np.sqrt(6. / (n_reduced + n_reconstructed)),
                        size=(n_reduced, n_reconstructed)
                    )
            ),
            name='weights',
            borrow=True
        )
        
        # initialize the biases b as a vector of n_out 0s
        self.bias = theano.shared(
            value=np.zeros(
                (n_reconstructed,),
                dtype=theano.config.floatX
            ),
            name='bias',
            borrow=True
        )

        self.params = [self.weights, self.bias]

        if activation == theano.tensor.nnet.sigmoid:
            self.weights *= 4
        self.activation = activation

        self.output = activation(T.dot(input, self.weights) + self.bias)
        