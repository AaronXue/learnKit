__author__ = 'AaronXue'


from src.cluster.ClusterBase import ClusterBase
import numpy as np
from src.base.base import *
from src.math.distribution import *


class EM(ClusterBase):

    def e_step(self):
        for i in range(get_matrix_row_num(self._sample)):
            xsum=0
            for j in range(K):
                xsum+=pi[j]*multi_gaussian(sampleA[i],cluster_means[j],cov[j])
            for j in range(K):
                Z[i][j]=pi[j]*multi_gaussian(sampleA[i],cluster_means[j],cov[j])/xsum
        return Z

    def m_step(self):
        for j in range(K):
            nj=0
            for i in range(sampleN):
                nj+=Z[i][j]
            pi[j]=nj/sampleN
            tmean=numpy.zeros((1,D))
            for i in range(sampleN):
                tmean+=Z[i][j]*sampleA[i]
            cluster_means[j]=tmean/nj
            tcov=numpy.mat(numpy.zeros((D,D)))
            for i in range(sampleN):
                t1=(sampleA[i]-cluster_means[j]).T
                tcov+=Z[i][j]*t1*t1.T
            cov[j]=tcov/nj
        return pi,cluster_means,cov

    def CalcJ(sampleA,sampleN,cluster_means,Z,pi,K,cov):
        jsum=0
        for i in range(sampleN):
            for j in range(K):
                jsum+=Z[i][j]*(pi[j]*lib.base.MultiGaussian(sampleA[i],cluster_means[j],cov[j]))
        return jsum