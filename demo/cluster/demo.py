__author__ = 'AaronXue'


from src.base import base
from src.cluster.KMeans import KMeans

sample_addr = "data/cluster_data_dataA_X.txt"

sample = base.read_matrix(sample_addr)

cluster = KMeans(sample, 3)
result = cluster.cluster()
base.plot_cluster("K Means", cluster, result)


