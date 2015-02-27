__author__ = 'AaronXue'


import math
import numpy as np
from src.base.base import *

def multi_gaussian(x, mean, cov):
    d = get_matrix_col_num(x)
    t = (x-mean).T
    cov_mat = np.mat(cov)
    return float(1/(np.power(2*math.pi,d/2)*np.power(np.linalg.det(cov),1/2))*np.exp((-1)/2*t.T*cov_mat.I*t))