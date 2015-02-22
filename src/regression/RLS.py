__author__ = 'AaronXue'

import numpy as np
from src.base.base import *
from src.regression.RegressionBase import RegressionBase


class RLS(RegressionBase):

    def __init__(self, sample_x, sample_y, k=1, lam=0.5):
        super().__init__(sample_x, sample_y, k)
        self._lam = lam

    def calculate(self):

        phi = feature_transformation(self._sample_x,self._k,get_matrix_col_num(self._sample_x))
        self._theta = (phi*phi.T+np.identity(self._k)*self._lam).I*phi*self._sample_y
        return self

