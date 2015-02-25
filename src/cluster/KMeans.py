__author__ = 'AaronXue'


from src.cluster.ClusterBase import ClusterBase
import numpy as np
import math
from src.base.base import *

CONVERGE = 0.001

class KMeans(ClusterBase):

    def __init__(self, sample, k, iter=0):

        super().__init__(sample, k)
        self._z = None
        self._iter = iter


    def step_one(self):

        self._z = np.zeros((get_matrix_row_num(self._sample), self._k))
        for i, sample_value in enumerate(self._sample):
            min = float("inf")
            t = None
            for j, cluster_value in enumerate(self._cluster_means):
                if np.linalg.norm(sample_value - cluster_value) < min:
                    min = np.linalg.norm(sample_value - cluster_value)
                    t = j
            self._z[i,t] = 1

    def step_two(self):
        n = get_matrix_row_num(self._sample)
        for j in range(self._k):
            self._cluster_means[j] = np.sum([self._z[i][j]*self._sample[i] for i in range(n)], axis=0) / np.sum([self._z[i][j] for i in range(n)])

    def cluster(self):

        self._cluster_means = self._init_cluster_means()
        prev_cluster_means = self._cluster_means
        if self._iter == 0:
            while True:
                self.step_one()
                self.step_two()
                if np.linalg.norm(prev_cluster_means - self._cluster_means) < CONVERGE:
                    break
        else:
            for i in range(self._iter):
                self.step_one()
                self.step_two()


        return self._get_result()


    def _get_result(self):

        n = get_matrix_row_num(self._sample)
        result = np.zeros(n)
        for i in range(n):
            for j in range(self._k):
                if self._z[i][j] == 1:
                    result[i] = j
        return result.getA1()




