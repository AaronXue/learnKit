__author__ = 'AaronXue'


import numpy as np
from scipy import optimize
from src.base.base import *
from src.regression.RegressionBase import RegressionBase


class RR(RegressionBase):

    def calculate(self):

        #CREATE PHI
        phi = feature_transformation(self._sample_x, self._k, get_matrix_col_num(self._sample_x))
        #CREATE A_MATRIX
        k = self._k
        n = get_matrix_col_num(self._sample_x)
        a_matrix = np.zeros((2*n, k+n))
        a_matrix[0:n,0:k] = (-1)*phi.T
        a_matrix[n:n*2,0:k] = phi.T
        a_matrix[0:n,k:k+n] = (-1)*np.mat(np.identity(n))
        a_matrix[n:n*2,k:k+n] = (-1)*np.mat(np.identity(n))
        acons=a_matrix.tolist()
        #CREATE CONST
        fvec = np.zeros((k+n,1))
        fvec[k:k+n,0:1] = np.ones((n,1))
        bvec = np.zeros((2*n,1))
        bvec[0:n] = (-1)*self._sample_y
        bvec[n:n*2] = self._sample_y
        bcons = bvec.T.tolist()[0]
        objective = lambda x:np.dot(fvec.T, x)
        constraints = ({'type':'ineq','fun':lambda x:bcons - np.dot(acons, x)})
        theta_x = np.array([10 for i in range(n+k)])
        #MINIMIZE
        result_cons = optimize.minimize(objective, theta_x, method='SLSQP', constraints=constraints)
        #CALCULATE THETA
        theta = np.mat(result_cons['x'][0:k])
        self._theta = theta.T
        return self

