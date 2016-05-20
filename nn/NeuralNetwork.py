#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import timeit
import numpy
import theano
import theano.tensor as thea
import cPickle as pickle

from Layers import HiddenLayer, LogisticLayer

class FFNN(object):

    def __init__(self, input, rng, n_hiddenlayer, n_in, n_hidden, n_out, activation, isTrain, dropout=False):
        self.input = input

        self.hiddenlayers = []
        self.params = []
        
        self.inputLayer = HiddenLayer(
            input=input,
            rng=rng,
            n_in=n_in,
            n_hidden=n_hidden,
            activation=activation,
            isTrain=isTrain,
            dropout=dropout
        )
        self.params = self.params + self.inputLayer.params

        for hl in range(n_hiddenlayer - 1):
            if hl == 0:
                ins = self.inputLayer.output
            else:
                ins = self.hiddenlayers[-1].output

            hiddenLayer = HiddenLayer(
                input=ins,
                rng=rng,
                n_in=n_hidden,
                n_hidden=n_hidden,
                activation=activation,
                isTrain=isTrain,
                dropout=dropout
            )
            self.params = self.params + hiddenLayer.params
            self.hiddenlayers.append(hiddenLayer)

        if len(self.hiddenlayers) > 0:
            inp = self.hiddenlayers[-1].output
        else:
            inp = self.inputLayer.output

        self.logisticLayer =  LogisticLayer(
            input=inp,
            rng=rng,
            n_hidden=n_hidden,
            n_out=n_out
        )  
        
        self.params = self.params + self.logisticLayer.params
        self.layers = [self.inputLayer] + self.hiddenlayers + [self.logisticLayer]

    def l1Regularisation(self):
        l1 = 0
        for layer in self.layers:
            l1 = l1 + abs(layer.weights).sum()
        return l1
 
    def l2Regularisation(self):
        l1 = 0
        for layer in self.layers:
            l1 = l1 + (layer.weights ** 2).sum()
        return l1

    def cost(self, groundTruth):
        return self.logisticLayer.cost(groundTruth)

    def missclassified(self, groundTruth):
        return self.logisticLayer.missclassified(groundTruth)

    def save(self, path, name):
        with open(path + '/' + name + '.ffnn', 'wb') as f:
            params = []
            for param in self.params:
                params = params + [param.eval()]
            pickle.dump(params, f)

def loadParams(path):
  return pickle.load( open( path , "rb" ) )
