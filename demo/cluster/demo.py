__author__ = 'AaronXue'


from src.base import base
from src.cluster.KMeans import KMeans
from src.cluster.EM import EM

sample_addr = "data/cluster_data_dataA_X.txt"

sample = base.read_matrix(sample_addr)

cluster = EM(sample, 4, 5)
result = cluster.cluster()
print(result)


