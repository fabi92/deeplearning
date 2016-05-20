#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import timeit
import theano.tensor as T
import theano
import numpy as np

import Cost

class LogisticLayer(object):

    def __init__(self, input, rng, n_hidden, n_out):
        self.input = input

        self.weights = theano.shared(
            value=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_hidden + n_out)),
                        high=np.sqrt(6. / (n_hidden + n_out)),
                        size=(n_hidden, n_out)
                    )
            ),
            name='weights',
            borrow=True
        )

        # initialize the biases b as a vector of n_out 0s
        self.bias = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='bias',
            borrow=True
        )

        self.predictedHotOne = T.nnet.softmax(T.dot(input, self.weights) + self.bias)
        self.predictions = T.argmax(self.predictedHotOne, axis=1)
        self.params = [self.weights, self.bias]
        self.costId = 'neglog'

    def cost(self, groundTruth):
        return Cost.negative_log_likelihood(self.predictedHotOne, groundTruth)

    def missclassified(self, groundTruth):
        return T.mean(T.neq(self.predictions, groundTruth))

class HiddenLayer(object):

    def __init__(self, input, rng, n_in, n_hidden, activation, dropout, isTrain, p=0.5):
        self.input = input
        self.rng = rng
        self.srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        self.p = p
        self.isTrain = isTrain
        
        #########################
        ##### Weight inits ######
        #########################
        if activation == T.tanh or activation == T.nnet.sigmoid:
            weights = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_hidden)),
                    high=np.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)
                )
            )
            if activation == T.nnet.sigmoid:
                weights *= 4
        elif activation == T.nnet.relu:
            weights = np.asarray(
                rng.randn(n_in, n_hidden) * np.sqrt(2 / (n_in + n_hidden))
            )
        #########################
        #########################

        self.weights = theano.shared(
            value=weights,
            name='weights',
            borrow=True
        )

        self.activation = activation
        
        # initialize the biases b as a vector of n_out 0s
        self.bias = theano.shared(
            value=np.zeros(
                (n_hidden,),
                dtype=theano.config.floatX
            ),
            name='bias',
            borrow=True
        )

        activated = activation(T.dot(self.input, self.weights) + self.bias) 
        if not dropout:    
            self.output = activated

        train_output = self.drop(np.cast[theano.config.floatX](1./self.p) * activated)
        self.output = T.switch(T.neq(self.isTrain, 0), train_output, activated) 
        self.params = [self.weights, self.bias]

    def drop(self, input, p=0.5):          
        mask = self.srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
        return input * mask
        