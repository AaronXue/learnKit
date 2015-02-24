__author__ = 'AaronXue'


import numpy as np


def L1_norm(x, y):
    return np.sum(np.abs(x-y))


def L2_norm(x, y):
    return np.sum(np.power(x-y,2))
