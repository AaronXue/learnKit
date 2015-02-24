__author__ = 'AaronXue'


from src.base import base


sample_addr = "data/cluster_data_dataA_X.txt"

sample = base.read_matrix(sample_addr)
sample = sample.T


