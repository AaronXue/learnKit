__author__ = "AaronXue"

from src.base.base import *


class LS(object):
    """

    """
    def __init__(self, sample_x, sample_y, k=1):
        """

        :param sample_x:
        :param sample_y:
        :param k:
        :return:
        """
        self._sample_x = get_numpy_mat(sample_x)
        self._sample_y = get_numpy_mat(sample_y)
        self._k = k + 1
        self._theta = None
        self._predict_y = None
        self._poly_x = None

    def get_sample_x(self):
        return self._sample_x.getA1()

    def get_sample_y(self):
        return self._sample_y.getA1()

    def get_poly_x(self):
        return self._poly_x.getA1()

    def get_predict_y(self):
        return self._predict_y.getA1()

    def calculate(self):
        """

        :return:
        """
        print()
        phi = feature_transformation(self._sample_x,self._k,get_matrix_col_num(self._sample_x))
        self._theta = (phi*phi.T).I*phi*self._sample_y
        return self

    def predict(self, poly_x):
        """

        :param poly_x:
        :return:
        """
        if self._theta is None:
            raise Exception("Theta is None.")
        else:
            self._poly_x = get_numpy_mat(poly_x)
            phix = feature_transformation(self._poly_x, self._k,get_matrix_col_num(self._poly_x))
            self._predict_y = phix.T*self._theta
            return self._predict_y.getA()



