__author__ = "AaronXue"

from src.base.base import *
from src.regression.RegressionBase import RegressionBase


class LS(RegressionBase):

    def calculate(self):

        phi = feature_transformation(self._sample_x,self._k,get_matrix_col_num(self._sample_x))
        self._theta = (phi*phi.T).I*phi*self._sample_y
        return self





