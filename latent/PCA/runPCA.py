#!/usr/bin/env python

from __future__ import print_function
import sys
import argparse
import numpy as np
import theano
import theano.tensor as T
from PCA import PCA
import LoadData

def runPCA(data, elems=None, components=None, threshold=None):

    t_data = theano.shared(np.asarray(data,
        dtype=theano.config.floatX),
        name='data',
        borrow=True
    )

    if components is not None and threshold is not None:
        print('You Can'' Run PCA Using Threshold And Components')
        exit(-1)
 
    t_components = None
    t_threshold = None

    if components is not None:
        t_components = theano.shared(
            value=components,
            name='components',
            borrow=True
        )
    elif threshold is not None:
        t_threshold = theano.shared(
            value=threshold,
            name='components',
            borrow=True
        )

    idx = T.lvector('idx')
    m_data = T.matrix('data')

    pca = PCA(
        data=m_data,
        components=t_components,
        threshold=t_threshold
    )

    theanoPCA = theano.function(
        inputs=[idx],
        outputs=pca.process(),
        givens={
            m_data: t_data[idx]
        }
    )
    
    if elems is None:
        elems = np.arange(len(data), dtype='int64')
    return theanoPCA(elems)


if __name__ == '__main__':

    theano.config.exception_verbosity='high'
    parser = argparse.ArgumentParser(prog='Principal Component Analysis', conflict_handler='resolve',description = '''\
        This script should enable the user to PCA on the given data set''')
    parser.add_argument('-e', '--elements', type=str, default=None, help='Vector Containing All Elements Of The Data Which Should Be Transformed')
    parser.add_argument('-c', '--components', type=int, help='Number Of Components')
    parser.add_argument('-t', '--threshold', type=float, help='If Eigenvalue Of Component Is Underneath This Threshold It Will Be Cut Off')
    
    requiredNamed = parser.add_argument_group('Required Arguments')
    requiredNamed.add_argument('-p', '--path', type=str, required=True, help='Path To The Training Set')
    requiredNamed.add_argument('-d', '--dataset', type=str, choices=['mnist', 'cifar'])
   
    parsed = parser.parse_args()

    if parsed.dataset == 'mnist':
        print('Loading Mnist Data')
        (imageData, imageLabels) = LoadData.loadMNISTUnSplit(parsed.path, shared=False)
        print(imageData.shape)
    elif parsed.dataset == 'cifar':
        print('Loading Cifar Data')
        (imageData, imageLabels) = LoadData.loadCIFAR10UnSplit(parsed.path, shared=False)
    print('Loaded')
    imageData = imageData / 255.
    
    pcad = runPCA(imageData, parsed.elements, parsed.components, parsed.threshold)

    print(pcad.shape)
    print(pcad[0:3])