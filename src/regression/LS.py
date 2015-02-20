from src.base.base import *
import numpy


class LS:
    def __init__(self,samplex,sampley,k=1):
        self.samplex=samplex
        self.sampley=sampley
        self.k=k

    def calculate(self):
        #create Phi
        Phi = GenerateFeatureTransformation(samplex,k,GetMatrixColNum(samplex))
        #parameter estimate
        self.theta = (Phi*Phi.T).I*Phi*sampley

    def predict(self,polyx):

