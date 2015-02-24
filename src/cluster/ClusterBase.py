__author__ = 'AaronXue'


from abc import ABCMeta, abstractmethod
from src.base.base import *
import numpy as np

class ClusterBase(object):

    __metaclass__ = ABCMeta

    def __init__(self, sample, k):

        self._sample = sample
        self._k = k
        self._cluster = None

    def _init_cluster_means(self):

        return np.mat([self._sample.getA()[i] for i in np.random.choice(get_matrix_col_num(self._sample), self._k)])









