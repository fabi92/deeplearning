#!/usr/bin/python

from __future__ import print_function

import os
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T
from itertools import product
import argparse
import scipy

from Kmeans import Kmeans
import LoadData
from Plotter import arraysToImgs


def apply_kmeans():
    ## parses the provided parameters according to the command line input
    parser = argparse.ArgumentParser(prog='k Means', conflict_handler='resolve',description = '''\
        This script should enable the user to k Means as mentioned in the papper by Adam Coates \
        and Andrew Y. Nh according to the input parameters
        ''')
    parser.add_argument('-p', '--pcaAndPlot', type=bool, required=False, default=False, help='PCA Reduce to 2')
    requiredNamed = parser.add_argument_group('Required Arguments')
    requiredNamed.add_argument('-d', '--dataset', type=str, required=True, help='Path To The Training Set (CIFAR10)')
    requiredNamed.add_argument('-c', '--centers', type=int, required=True, help='Number Of Centers')
   
    parsed = parser.parse_args()
    
    (imageData, imageLabels) = LoadData.loadCIFAR10UnSplit(parsed.dataset, shuffle=True, shared=False)
    print(imageData.shape)
    print('Resizing to 12x12')
    greyImg = np.zeros((len(imageData), 12 * 12), dtype='uint8')
    for imgId in range(len(imageData)):
        greyImg[imgId] = scipy.misc.imresize(imageData[imgId].reshape(32,32), [12, 12]).reshape(12*12)
    greyImg = greyImg/255.

    print('done')
    dims = greyImg.shape

    kmeans = Kmeans(
        data=greyImg,
        centers=parsed.centers,
        n_inputs=dims[0],
        n_dim=dims[1]
    )

    iterate = theano.function(
        inputs=[],
        updates=kmeans.get_updates(),
        outputs=kmeans.centroids
    )

    applyKmeans = theano.function(
       inputs=[],
       outputs=kmeans.applyToData()
    )
    
    data2centroid_new = None
    data2centroid_old = None

    print('############\nStart Updating\n############\n')
    iteration = 0
    done_looping = False
    threshold = 1e-5
    while not done_looping:
        centroids = iterate()
        iteration = iteration + 1
        data2centroid_new = applyKmeans()
        if not data2centroid_old is None:
            n_dataChangedCluster = (dims[0] - \
                (data2centroid_new == data2centroid_old).sum())
            print( "\tIteration: %d - %d Points changed their clusters" \
                   % (iteration, n_dataChangedCluster))
            if old_centroids is not None:
                dif = ((old_centroids - centroids) ** 2 ).sum(axis=1) ##euclidean dist
                if (dif < threshold).all():
                    print('Kmeans has converged')
                    print('Centroids Difference = ' + str(dif) + ' < threshold ' + str(threshold))
                    done_looping = True
        data2centroid_old = data2centroid_new
        old_centroids = centroids
    print('############\nConverged\n############\n')
    if parsed.pcaAndPlot:
        pcaAndPlot(greyImg[:250], data2centroid_new[:250], centroids)
    print('Plotting Centroids')
    rows = parsed.centers / 10
    arraysToImgs(rows=rows,colums=10,arr=centroids,
        path='results/centroids.png',out_shape=(12,12))

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    apply_kmeans()