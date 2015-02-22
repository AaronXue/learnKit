__author__ = 'AaronXue'


import numpy as np
from scipy import optimize
from src.base.base import *
from src.regression.RegressionBase import RegressionBase


class LASSO(RegressionBase):

    def __init__(self, sample_x, sample_y, k=1, lam=0.5):
        super().__init__(sample_x, sample_y, k)
        self._lam = lam

    def calculate(self):

        #CREATE PHI
        phi = feature_transformation(self._sample_x,self._k,get_matrix_col_num(self._sample_x))
        #CREATE HESSIAN MATRIX
        k=self._k
        hessian=np.zeros((k*2,k*2))
        hessian[0:k,0:k]=phi*phi.T
        hessian[0:k,k:k*2]=(-1)*phi*phi.T
        hessian[k:k*2,0:k]=(-1)*phi*phi.T
        hessian[k:k*2,k:k*2]=phi*phi.T
        #CREATE CONSTRAINTS
        fvec=np.zeros((k*2,1))
        fvec[0:k,0:1]=phi*self._sample_y
        fvec[k:k*2,0:1]=(-1)*phi*self._sample_y
        fvec=np.ones((k*2,1))*self._lam-fvec
        cons=np.identity(k*2)
        theta_x=np.array([3 for i in range(k*2)])
        objective=lambda x:0.5*np.dot(x.T,np.dot(hessian,x))+np.dot(fvec.T,x)
        constraints=({'type':'ineq','fun':lambda x:np.dot(cons,x)})
        #MINIMIZE
        result_cons=optimize.minimize(objective, theta_x, method='SLSQP', constraints=constraints)
        #CALCULATE THETA
        theta=np.mat(result_cons['x'][0:k]-result_cons['x'][k:k*2])
        self._theta=theta.T
        return self