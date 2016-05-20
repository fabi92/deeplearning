#!/usr/bin/python

from __future__ import print_function

import os
import sys
import timeit
import numpy
import theano
import theano.tensor as thea
import cPickle as pickle
import Cost

from Layers import ReducedLayer, ReconstructionLayer

class AutoEncoder(object):

    def __init__(self, input, rng, n_input, n_reduced, n_reconstructed, sparsity_param, beta=0.01, activation=theano.tensor.nnet.sigmoid):
        self.input = input
        self.sparsity_param = sparsity_param
        self.beta = beta

        self.reducedLayer = ReducedLayer(
            input=input,
            rng=rng,
            activation=activation,
            n_input=n_input,
            n_reduced=n_reduced
        )

        self.reconstructionLayer =  ReconstructionLayer(
            input=self.reducedLayer.output,
            rng=rng,
            activation=activation,
            n_reduced=n_reduced,
            n_reconstructed=n_reconstructed
        )  

        self.params = self.reducedLayer.params + self.reconstructionLayer.params
        self.reconstruction = self.reconstructionLayer.output

        self.cost = Cost.squared_error_loss(self.reducedLayer.input, self.reconstructionLayer.output)
        self.kul_leib = self.beta * Cost.kullback_leibler_divergence(self.sparsity_param, self.reducedLayer.output)


    def setInput(self, input):
        self.input = input
        self.reducedLayer.input = input

    def save(self, path, name):
        params = [self.reducedLayer.weights.eval(), self.reducedLayer.bias.eval(),
           self.reconstructionLayer.weights.eval(), self.reconstructionLayer.bias.eval()]
        pickle.dump( params, open( path + "/" + name + ".aec" , "wb" ) )

def loadParams(path):
  return pickle.load( open( path , "rb" ) )