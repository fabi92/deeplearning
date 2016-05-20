#!/usr/bin/python

from __future__ import print_function

import os
import sys
import timeit

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import theano
import theano.tensor as T
import argparse

from itertools import product

from KulisKmeans import KulisKmeans
from LoadData import loadCIFAR10UnSplit_Color,loadCIFAR10UnSplit
from Plotter import arraysToImgs


def get_color_palette(dataset, n_images=25, penalty=2.1, threshold=1e-3):
    ## parses the provided parameters according to the command line input
    
    (imageData, imageLabels) = loadCIFAR10UnSplit_Color(dataset, shuffle=False, shared=False)
    selection = np.random.randint(imageData.shape[0], size=n_images)

    imageData = imageData[:n_images]

    ### we are only interested in the colors
    t_k = T.scalar('k')
    colors = imageData.reshape(imageData.shape[0] * imageData.shape[1] * imageData.shape[2], 3)
    penalty = penalty * (255.**2)

    kmeans = KulisKmeans(
        data=colors,
        dims=colors.shape, 
        cluster_penalty=penalty
    )

    getIndicators = theano.function(
        inputs=[],
        outputs=kmeans.indicators
    )

    
    clusters_old = None
    mys_old = None
    k_old = 1
    not_converged=True
    epoch=1
    while not_converged:
        print('################')
        print('####Epoch %d####' %(epoch))
        (clusters, k, mys) = kmeans.iterate()
        print('Number New Clusters = %d' %(k - k_old))
        if mys_old is not None:
            if k_old == k:
                dif = ((mys - mys_old) ** 2 ).sum(axis=1)
                if (dif < threshold).all():
                    not_converged = False
                    print('################')
                    print('Kmeans has converged')
                    print('Centroids Differences < threshold ' + str(threshold))
                    print('################')
                    break
        k_old = k
        mys_old = mys
        clusters_old = clusters
        epoch = epoch+1

    print('Result')
    print('n_clusters: %d' %(k))
    print('################')
    
    indicators = getIndicators()

    colorValues = mys[indicators].reshape(n_images,32, 32, 3)
    images = None
    for n in range(n_images):
        image = colorValues[n]
        image = np.concatenate((image, np.ones((32, 3, 3)), imageData[n]), axis=1)
        if images is None:
        	images = image
        else:
        	images = np.concatenate((images, np.ones((7, 32*2+3, 3)),image), axis=0)
    
    plt.imshow(images)
    plt.show()
    matplotlib.image.imsave('images/imgs.png', images)


if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    parser = argparse.ArgumentParser(prog='k Means', conflict_handler='resolve',description = '''\
        This script should enable the user to k Means as mentioned in the last bonus exercise''')
    parser.add_argument('-i', '--n_images', type=int, default=20, required=False, help='Number Of Images Which Get Selected From The CIFAR10 Data Set')
    parser.add_argument('-p', '--penalty', type=float, default=0.2, required=False, help='Value For Cluster Penalty')
    parser.add_argument('-t', '--threshold', type=float, default=1e-3, required=False, help='Criteria For Converging')
    requiredNamed = parser.add_argument_group('Required Arguments')
    requiredNamed.add_argument('-d', '--dataset', type=str, required=True, help='Path To The Training Set (CIFAR10)')
   
    parsed = parser.parse_args()
    get_color_palette(parsed.dataset, n_images=parsed.n_images, penalty=parsed.penalty, threshold=parsed.threshold)