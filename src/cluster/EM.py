__author__ = 'AaronXue'


from src.cluster.ClusterBase import ClusterBase
import numpy as np
from src.base.base import *
from src.math.distribution import *

CONVERGE = 0.0001

class EM(ClusterBase):

    def __init__(self, sample, k, it=0, var=15):

        super().__init__(sample, k, it)
        self._pi = None
        self._cov = None
        self._var = var

    def e_step(self):

        for i in range(get_matrix_row_num(self._sample)):
            x_sum = 0
            for j in range(self._k):
                x_sum += self._pi[j]*multi_gaussian(self._sample[i], self._cluster_means[j], self._cov[j])
            for j in range(self._k):
                self._z[i][j] = self._pi[j]*multi_gaussian(self._sample[i], self._cluster_means[j], self._cov[j]) / x_sum

    def m_step(self):

        n = get_matrix_row_num(self._sample)
        d = get_matrix_col_num(self._sample)
        for j in range(self._k):
            cluster_sum = np.sum([self._z[i][j] for i in range(n)])
            self._pi[j] = cluster_sum / n
            self._cluster_means[j] = np.sum([self._z[i][j]*self._sample[i] for i in range(n)], axis=0) / cluster_sum
            sum_mat = np.zeros((d, d))
            for i in range(n):
                t = (self._sample[i]-self._cluster_means[j]).T
                sum_mat += self._z[i][j]*t*t.T
            self._cov[j] = sum_mat / cluster_sum

    def calc_j(self):

        n = get_matrix_row_num(self._sample)
        j_val = 0
        for i in range(n):
            for j in range(self._k):
                j_val += self._z[i][j]*((self._pi[j]*multi_gaussian(self._sample[i], self._cluster_means[j], self._cov[j])))
        return j_val


    def _init_once(self):

        d = get_matrix_col_num(self._sample)
        # init assignment
        self._cluster_means = self._init_cluster_means()
        for i, sample_value in enumerate(self._sample):
            min = float("inf")
            t = None
            for j, cluster_value in enumerate(self._cluster_means):
                if np.linalg.norm(sample_value - cluster_value) < min:
                    min = np.linalg.norm(sample_value - cluster_value)
                    t = j
            self._z[i,t] = 1
        # init pi
        self._pi = np.zeros(self._k)
        # init covariance
        self._cov = np.zeros((self._k, d, d))
        for i in range(self._k):
            for j in range(d):
                self._cov[i][j][j] = self._var
        # first m step
        self.m_step()
        # first calculate J
        return self.calc_j()

    def cluster(self):

        prev_j = self._init_once()
        it_num = 0
        if self._it == 0:
            while True:
                self.e_step()
                self.m_step()
                print(prev_j)
                curr_j = self.calc_j()
                if abs(prev_j - curr_j) < CONVERGE:
                    print("iteration: ", it_num)
                    break
                prev_j = curr_j
                it_num += 1
        else:
            for i in range(self._it):
                self.e_step()
                self.m_step()
                it_num += 1
            print("iteration: ", it_num)
        return self._z.tolist()


