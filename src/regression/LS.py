from src.base.base import *
import numpy


class LS:
    def __init__(self,samplex,sampley,k=1):
        self._samplex = GetNumpyMat(samplex)
        self._sampley = GetNumpyMat(sampley)
        self._k = k + 1
        self._theta = None
        self._predicty = None

    def calculate(self):
        #create Phi
        Phi = GenerateFeatureTransformation(self._samplex,self._k,GetMatrixColNum(self._samplex))
        #parameter estimate
        self._theta = (Phi*Phi.T).I*Phi*self._sampley

    def predict(self,polyx):
        if self._theta is None:
            print("Call calculate() before predict().")
            raise Exception("Theta is None.")
        else:
            self._polyx = GetNumpyMat(polyx)
            PhiX = GenerateFeatureTransformation(self._polyx,self._k,GetMatrixColNum(self._polyx))
            self._predicty = PhiX.T*self._theta
            return self._predicty.getA()



