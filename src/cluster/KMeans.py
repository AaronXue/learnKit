__author__ = 'AaronXue'


from src.cluster.ClusterBase import ClusterBase
import numpy as np
from src.math import norm

D=4
VAR=5

class KMeans(ClusterBase):

    def StepOne(sampleA,sampleN,cluster_means,K):
        Z=np.zeros((sampleN,K))
        for i,valueX in enumerate(sampleA.getA()):
            min=float("Inf")
            for j,valueM in enumerate(cluster_means.getA()):
                if norm.L2_norm(valueX,valueM)<min:
                    min=norm.L2_norm(valueX,valueM)
                    t=j
            Z[i,t]=1
        return cluster_means,Z

    def StepTwo(sampleA,sampleN,cluster_means,Z,K):
        for j in range(K):
            cluster_means[j]=np.sum([Z[i][j]*sampleA[i] for i in range(sampleN)],axis=0)/sum([Z[i][j] for i in range(sampleN)])
        return cluster_means









def CalcJ(sampleA,sampleN,cluster_means,Z,K,cov):
    jsum=0
    for i in range(sampleN):
        for j in range(K):
            jsum+=Z[i][j]*math.log(1/K*lib.base.MultiGaussian(sampleA[i],cluster_means[j],cov))
    return jsum


def calculate(sampleA,sampleN,K,iter):
    cluster_means=lib.base.ClusterInit(sampleA,sampleN,K)
    print("init")
    cov=numpy.identity(D)*VAR
    J0=None
    J=None
    '''
    while True:
        cluster_means,Z=StepOne(sampleA,sampleN,cluster_means,K)
        cluster_means=StepTwo(sampleA,sampleN,cluster_means,Z,K)
        J=CalcJ(sampleA,sampleN,cluster_means,Z,K,cov)
        print(J)
        if lib.base.ConvergeJ(J0,J):
            break
        J0=J
    '''

    for i in range(iter):
        cluster_means,Z=StepOne(sampleA,sampleN,cluster_means,K)
        cluster_means=StepTwo(sampleA,sampleN,cluster_means,Z,K)
        J=CalcJ(sampleA,sampleN,cluster_means,Z,K,cov)
        print(i)

    rows=sampleA.shape[0]
    result=numpy.zeros((1,rows))
    '''
    for j in range(K):
        sampleCluster=[]
        for i in range(sampleN):
            if Z[i][j]==1:
                t=sampleA.getA()[i]
                sampleCluster.append(t)
        samplePlot=numpy.array(sampleCluster)
        plt.plot(samplePlot[:,0],samplePlot[:,1],"o",label="Cluster "+str(j))
    plt.title("K means K="+str(K))
    plt.legend(loc=3)
    plt.show()

    return J
    '''
    for j in range(K):
        for i in range(sampleN):
            if Z[i][j]==1:
                result[0][i]=j


    return result
