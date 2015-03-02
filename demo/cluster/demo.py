__author__ = 'AaronXue'


from src.base import base
from src.cluster.KMeans import KMeans
from src.cluster.EM import EM

sample_addr = "data/cluster_data_dataA_X.txt"

sample = base.read_matrix(sample_addr)

cluster = EM(sample, 4)
result = cluster.cluster()
plot_list = [0]*len(result)
for i in range(len(result)):
    max_num = max(result[i])
    plot_list[i] = result[i].index(max_num)
print(plot_list)
base.plot_cluster("cluster", cluster, plot_list)



