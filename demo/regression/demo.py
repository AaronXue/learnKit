__author__ = 'AaronXue'


from src.base import base
from src.regression.LS import LS
from src.regression.RLS import RLS
from src.regression.LASSO import LASSO
from src.regression.RR import RR
from src.regression.BR import BR


def demo_read_data():

    sample_x_addr = "data/polydata_data_sampx.txt"
    sample_y_addr = "data/polydata_data_sampy.txt"
    poly_x_addr = "data/polydata_data_polyx.txt"

    sample_x = base.read_matrix(sample_x_addr)
    sample_y = base.read_matrix(sample_y_addr)
    poly_x = base.read_matrix(poly_x_addr)
    return sample_x, sample_y, poly_x


def ls_regression_demo():

    (sample_x, sample_y, poly_x) = demo_read_data()  #read data
    regression = LS(sample_x,sample_y,5)
    regression.calculate().predict(poly_x)
    base.plot_regression("least-square regression", regression)


def rls_regression_demo():

    (sample_x, sample_y, poly_x) = demo_read_data()  #read data
    regression = RLS(sample_x,sample_y,5)
    regression.calculate().predict(poly_x)
    base.plot_regression("regularized least-square regression", regression)


def lasso_regression_demo():

    (sample_x, sample_y, poly_x) = demo_read_data()  #read data
    regression = LASSO(sample_x,sample_y,5)
    regression.calculate().predict(poly_x)
    base.plot_regression("L1-regularized least-square regression", regression)


def rr_regression_demo():

    (sample_x, sample_y, poly_x) = demo_read_data()  #read data
    regression = RR(sample_x,sample_y,5)
    regression.calculate().predict(poly_x)
    base.plot_regression("robust regression", regression)


def br_regression_demo():

    (sample_x, sample_y, poly_x) = demo_read_data()  #read data
    regression = BR(sample_x,sample_y,5)
    regression.calculate().predict(poly_x)
    base.plot_regression("Bayesian regression", regression)


br_regression_demo()






