__author__ = 'XUE Yang'

import numpy
# import random
import matplotlib.pyplot as plt
import random


def ReadMatrix (Str):
    fin = open(Str, 'r')
    tList = []
    for line in fin:
        tList.append(line.split())
    tMatrix=numpy.mat(tList,dtype=numpy.float32)
    fin.close()
    return tMatrix


def ReadPointNum(str):
    fin=open(str,'r')
    tnum=len(fin.readline().split())
    fin.close()
    return tnum


def ReadData(samplex_addr,sampley_addr,polyx_addr,polyy_addr):
    samplex=ReadMatrix(samplex_addr)
    sampley=ReadMatrix(sampley_addr)
    polyx=ReadMatrix(polyx_addr)
    polyy=ReadMatrix(polyy_addr)
    sampleN=ReadPointNum(samplex_addr)
    polyN=ReadPointNum(polyx_addr)
    return samplex,sampley,polyx,polyy,sampleN,polyN
'''
def GenerateFeatureTransformation(vector,K,N):
    Phi=numpy.mat(numpy.zeros((K,N)))
    for i in range(K):
        Phi[i][0:N]=numpy.power(vector,i)
    return Phi
'''


def GenerateFeatureTransformation(vector,K,N):
    Phi=numpy.mat(numpy.zeros((K,N)))
    Phi[0:K/3,0:N]=vector.T
    #Phi[K/4:K*2/4,0:N]=numpy.reciprocal(vector.T)
    Phi[K/3:K*2/3,0:N]=numpy.power(vector.T,2)
    Phi[K*2/3:K,0:N]=numpy.exp(numpy.power(vector.T,2))
    return Phi


def PlotFunction(str1,type,polyx,polyy,funcPrediction,samplex,sampley,funcPre_d=None):
    plt.plot(polyx,funcPrediction,"bo-",label="Estimated function")
    if funcPre_d!=None:
        plt.plot(polyx,funcPrediction+funcPre_d,"b-",label="Standard deviation")
        plt.plot(polyx,funcPrediction-funcPre_d,"b-")
    plt.plot(polyx,polyy,"go-",label="True function")
    plt.plot(samplex,sampley,"ro",label="Samples")
    plt.title(str1)
    plt.legend()
    # plt.savefig("c:\\figures\\"+str1+str(int(random.random()*1000))+'.png',dpi=200)
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
            subsamplex=numpy.mat(sublistx)
            subsampley=numpy.mat(sublisty).T
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
            subsamplex=numpy.mat(sublistx)
            subsampley=numpy.mat(sublisty).T
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
