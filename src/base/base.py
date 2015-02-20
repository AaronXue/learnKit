__author__ = 'XUE Yang'

import numpy as np
# import random
import matplotlib.pyplot as plt
import random


def GetMatrixColNum(matrix):
    return matrix.shape[1]


def GetMatrixRowNum(matrix):
    return matrix.shape[0]


def ReadMatrix (str):
    fin = open(str, 'r')
    dataList = []
    for line in fin:
        dataList.append(line.split())
    fin.close()
    return dataList


def GetNumpyMat(list):
    return np.mat(list,dtype=np.float32)


def GenerateFeatureTransformation(vector,K,N):
    Phi=np.mat(np.zeros((K,N)))
    for i in range(K):
        Phi[i][0:N]=np.power(vector,i)
    return Phi


def PlotRegression(title,LS):
    plt.plot(LS._polyx.getA1(),LS._predicty.getA1(),"b-",label="Estimated function")
    plt.plot(LS._samplex.getA1(),LS._sampley.getA1(),"ro",label="Samples")
    plt.title(title)
    plt.legend()
    plt.show()

def PlotCount(str1,type,polyy,funcPrediction,funcPre_d=None):
    plt.plot(funcPrediction,"bo-",label="Estimated counts")
    if funcPre_d!=None:
        plt.plot(funcPrediction+funcPre_d,"b-",label="Standard deviation")
        plt.plot(funcPrediction-funcPre_d,"b-")
    plt.plot(polyy,"go-",label="True counts")
    plt.title(str1)
    plt.legend(loc=2)
    plt.show()

def PlotFunction2(SAMPLE_NUM,MSEaverage_LS,MSEaverage_RLS,MSEaverage_LASSO,MSEaverage_RR,MSEaverage_BR,str):
    plt.plot(SAMPLE_NUM,MSEaverage_LS,"bo-",label="LS")
    plt.plot(SAMPLE_NUM,MSEaverage_RLS,"go-",label="RLS")
    plt.plot(SAMPLE_NUM,MSEaverage_LASSO,"ro-",label="LASSO")
    plt.plot(SAMPLE_NUM,MSEaverage_RR,"co-",label="RR")
    plt.plot(SAMPLE_NUM,MSEaverage_BR,"mo-",label="BR")
    plt.title(str)
    plt.legend()
    plt.show()

def SubsetCalculation(samplex,sampley,polyx,polyy,polyN,K,SAMPLE_NUM,LOOP_NUM,calculate):
    MSEaverage=[0 for i in range(len(SAMPLE_NUM))]
    for num in range(LOOP_NUM):
        MSElist=[]
        for i in SAMPLE_NUM:
            sublistx=list(samplex.getA1())
            sublisty=list(sampley.getA1())
            for j in range(50-i):
                k=random.randrange(len(sublistx))
                del sublistx[k]
                del sublisty[k]
            subsampleN=len(sublistx)
            subsamplex=np.mat(sublistx)
            subsampley=np.mat(sublisty).T
            MSE_LS=calculate(subsamplex,subsampley,polyx,polyy,subsampleN,polyN,K)
            MSElist.append(MSE_LS)
        for i in range(len(MSEaverage)):
            MSEaverage[i]+=MSElist[i]
    for i in range(len(MSEaverage)):
        MSEaverage[i]/=LOOP_NUM
    return MSEaverage

def OutlierCalculation(samplex,sampley,polyx,polyy,polyN,K,SAMPLE_NUM,LOOP_NUM,OUTLIER_NUM,calculate):
    MSEaverage=[0 for i in range(len(SAMPLE_NUM))]
    for num in range(LOOP_NUM):
        MSElist=[]
        for i in SAMPLE_NUM:
            sublistx=list(samplex.getA1())
            sublisty=list(sampley.getA1())
            for j in range(i):
                k=random.randrange(len(sublisty))
                sublisty[k]+=OUTLIER_NUM
            sampleN=len(sublistx)
            subsamplex=np.mat(sublistx)
            subsampley=np.mat(sublisty).T
            MSE=calculate(subsamplex,subsampley,polyx,polyy,sampleN,polyN,K)
            MSElist.append(MSE)
        for i in range(len(MSEaverage)):
            MSEaverage[i]+=MSElist[i]
    for i in range(len(MSEaverage)):
        MSEaverage[i]/=LOOP_NUM
    return MSEaverage


def HighOrderCalculation(samplex,sampley,polyx,polyy,sampleN,polyN,K_ORDER_NUM,calculate):
    MSElist=[]
    for K in K_ORDER_NUM:
        MSE=calculate(samplex,sampley,polyx,polyy,sampleN,polyN,K)
        MSElist.append(MSE)
    return MSElist


def WriteFile(list,str):
    f=open(str,'a')
    f.write(' '.join(format(x,".10f") for x in list))
    f.write('\n')
    f.close()
