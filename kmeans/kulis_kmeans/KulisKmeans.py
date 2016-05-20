#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

class KulisKmeans(object):

    def __init__(self, data, dims, cluster_penalty=2.1):
        
        self.dims = dims
        self.cluster_penalty = cluster_penalty

        self.n_clusters = theano.shared(
            value=np.int(1),
            name='n_clusters'
        )

        self.data = theano.shared(
             value=data,
             name="data"
        )    
        
        self.indicators = theano.shared(
            value=np.ones(self.dims[0], dtype='uint32'),
            name="indices"
        )

        mu_init = np.zeros((self.dims[0], 3))
        mu_init[1,:] = data.mean(axis=0)

        self.mu = theano.shared(
            value=mu_init,
            name="mu"
        )

        t_idx = T.iscalar('t_idx')
        t_vec = T.vector('t_vec')

        self.D_i_c = (self.euclidean_dist(self.data, self.mu)[:,:self.n_clusters])**2
        self.min_dic = T.min(self.D_i_c, axis=1) > self.cluster_penalty

        self.getDics = theano.function(
            inputs=[],
            outputs=self.min_dic
        )

        self.updateClusters = theano.function(
            inputs=[t_idx],
            updates=[ \
                (self.indicators, T.set_subtensor(self.indicators[t_idx], self.n_clusters)), \
                (self.mu, T.set_subtensor(self.mu[self.n_clusters], self.data[t_idx])), \
                (self.n_clusters, self.n_clusters+1)
            ]
        )

        self.updateIndicators = theano.function(
            inputs=[t_idx],
            updates=[ \
                (self.indicators, T.set_subtensor(self.indicators[t_idx], T.argmin(self.D_i_c[t_idx])))
            ]
        )
        
        self.getLkFromIdx = theano.function(
            inputs=[t_idx],
            outputs=self.getLk(t_idx) 
        )

        self.getMu = theano.function(
            inputs=[],
            outputs=self.mu
        )

        self.getNClusters = theano.function(
            inputs=[],
            outputs=self.n_clusters
        )

    def getLk(self, it):
        return self.data[T.eq(self.indicators, it).nonzero()[0]]
        
    def euclidean_dist(self, x1, x2):
        return T.sqrt(((x1**2).sum(axis=1, keepdims=True) + (x2**2).sum(axis=1, keepdims=True).T - \
                   2*T.dot(x1, x2.T)))

    def iterate(self):
        dic = self.getDics() 
        for idx in range(dic.shape[0]):
            if dic[idx] == 1:
                self.updateClusters(idx) 
            else:
                self.updateIndicators(idx)
        lk = []
        k = self.getNClusters()
        for k_i in range(k):
            lk_x = self.getLkFromIdx(k_i)
            lk.append(lk_x)
            if lk_x.shape[0] > 0:
                self.mu = T.set_subtensor(self.mu[k_i], T.mean(lk_x, axis=0))
        return (np.array(lk), k, self.getMu()[:k,:])