__author__ = 'AaronXue'


from abc import ABCMeta, abstractmethod
from src.base.base import *
import numpy as np

class ClusterBase(object):

    __metaclass__ = ABCMeta

    def __init__(self, sample, k):

        self._sample = get_numpy_mat(sample).T
        self._k = k
        self._cluster_means = None

    def _init_cluster_means(self):

        return np.mat([self._sample.getA()[i] for i in np.random.permutation(get_matrix_row_num(self._sample))[0:self._k]])

    def get_k(self):

        return self._k

    def get_sample(self):

        return self._sample.getA()

    def get_result(self):

        return self._result







