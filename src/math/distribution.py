__author__ = 'AaronXue'


import math
import numpy as np

def MultiGaussian(x,mean,cov):
    d=len(x.getA1())
    t=(x-mean).T
    tcov=np.mat(cov)
    return float(1/(np.power(2*math.pi,d/2)*np.linalg.det(cov))*np.exp((-1)/2*t.T*tcov.I*t))