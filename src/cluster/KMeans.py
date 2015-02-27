__author__ = 'AaronXue'


from src.cluster.ClusterBase import ClusterBase
import numpy as np
import math
from src.base.base import *

CONVERGE = 0.00001

class KMeans(ClusterBase):

    def step_one(self):

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
        prev_cluster_means = self._cluster_means.copy()
        it_num = 0
        if self._it == 0:
            while True:
                self.step_one()
                self.step_two()
                if np.linalg.norm(prev_cluster_means - self._cluster_means) < CONVERGE:
                    print("iteration: ", it_num)
                    break
                prev_cluster_means = self._cluster_means.copy()
                it_num += 1
        else:
            for i in range(self._it):
                self.step_one()
                self.step_two()
                it_num += 1
            print("iteration: ", it_num)
        return self._z.tolist()




