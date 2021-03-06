__author__ = 'XUE Yang'

import numpy as np
import matplotlib.pyplot as plt
import random


def get_matrix_row_num(matrix):
    return matrix.shape[0]


def get_matrix_col_num(matrix):
    return matrix.shape[1]


def read_matrix (str):
    fin = open(str, 'r')
    data_list = []
    for line in fin:
        data_list.append(line.split())
    fin.close()
    return data_list


def get_numpy_mat(list):
    return np.mat(list,dtype = np.float32)


def feature_transformation(vector, k, n):
    phi=np.mat(np.zeros((k, n)))
    for i in range(k):
        phi[i][0:n] = np.power(vector, i)
    return phi


def plot_regression(title, Regression):
    plt.plot(Regression.get_poly_x(), Regression.get_predict_y(), "bo-", label="Estimated function")
    plt.plot(Regression.get_sample_x(), Regression.get_sample_y(), "ro", label="Samples")
    if Regression.is_distribution():
        plt.plot(Regression.get_poly_x(), Regression.get_predict_y()-Regression.get_deri(), "b-",label="Standard deviation")
        plt.plot(Regression.get_poly_x(), Regression.get_predict_y()+Regression.get_deri(), "b-")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_cluster(title, Cluster, result):
    sample = Cluster.get_sample()
    dimension = len(sample[0])
    if dimension != 2:
        raise Exception("Only support 2-d plot.")
    for j in range(Cluster.get_k()):
        plot_list = [i for i in range(len(result)) if j == result[i]]
        plot_sample_x= [sample[i][0] for i in plot_list]
        plot_sample_y= [sample[i][1] for i in plot_list]
        plt.plot(plot_sample_x, plot_sample_y, "o", label="Cluster "+str(j))
    plt.title(title)
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
