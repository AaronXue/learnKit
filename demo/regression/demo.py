__author__ = 'AaronXue'

from src.base import base
from src.regression.LS import LS

samplexAddr="data/polydata_data_sampx.txt"
sampleyAddr="data/polydata_data_sampy.txt"
polyxAddr="data/polydata_data_polyx.txt"

samplex=base.ReadMatrix(samplexAddr)
sampley=base.ReadMatrix(sampleyAddr)
polyx=base.ReadMatrix(polyxAddr)

ls=LS(samplex,sampley,1)
ls.calculate()
print(ls.predict(polyx))
base.PlotRegression("least-square regression",ls)








