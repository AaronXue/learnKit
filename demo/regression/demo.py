__author__ = 'AaronXue'

from src.base import base
from src.regression.LS import LS

def ls_regression_demo():
    sample_x_addr = "data/polydata_data_sampx.txt"
    sample_y_addr = "data/polydata_data_sampy.txt"
    poly_x_addr = "data/polydata_data_polyx.txt"

    sample_x = base.read_matrix(sample_x_addr)
    sample_y = base.read_matrix(sample_y_addr)
    poly_x = base.read_matrix(poly_x_addr)

    ls = LS(sample_x,sample_y,5)
    ls.calculate().predict(poly_x)
    base.plot_regression("least-square regression", ls)



ls_regression_demo()




