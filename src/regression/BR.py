__author__ = 'AaronXue'


import numpy as np
from src.base.base import *
from src.regression.RegressionBase import RegressionBase


class BR(RegressionBase):

    def __init__(self, sample_x, sample_y, k=1, theta_devi_pre=5, noise_var=5):
        super().__init__(sample_x, sample_y, k)
        self._theta_devi_pre = theta_devi_pre
        self._noise_var = noise_var
        self._theta_cov_post = None
        self._pred_diag_var = None

    def calculate(self):

        #CREATE PHI
        phi = feature_transformation(self._sample_x, self._k, get_matrix_col_num(self._sample_x))
        #POSTERIOR
        k_identity = np.identity(self._k)
        theta_e_pre = 0
        theta_cov_pre = k_identity*self._theta_devi_pre
        theta_cov_post = (k_identity*(1/self._theta_devi_pre)+phi*phi.T*(1/self._noise_var)).I
        theta_e_post = theta_cov_post*(1/self._noise_var)*phi*self._sample_y
        self._theta = theta_e_post
        self._theta_cov_post = theta_cov_post
        return self

    def predict(self, poly_x):

        if self._theta is None:
            raise Exception("Theta is None.")
        else:
            self._poly_x = get_numpy_mat(poly_x)
            phix = feature_transformation(self._poly_x, self._k, get_matrix_col_num(self._poly_x))
            pred_var = phix.T*self._theta_cov_post*phix
            pred_diag_var = np.sqrt(pred_var.diagonal())
            self._pred_diag_var = pred_diag_var
            self._predict_y = phix.T*self._theta
            return self._predict_y.getA()

    def is_distribution(self):

        return True

    def get_deri(self):

        return self._pred_diag_var.getA1()

