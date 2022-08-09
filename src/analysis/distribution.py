import argparse
import numpy as np
from scipy.stats import chi2, norm
import os
import sys
import pickle as pkl
sys.path.append(os.getcwd().rstrip("/").rstrip("analysis"))
from autoencoder import normalize
from matplotlib import pyplot as plt
from tqdm import tqdm
from data_processing import DATA_PATHS

"""
implementation of the multidimensional jarque-berat-test, as proposed
in: https://www.uni-muenster.de/Stochastik/paulsen/Abschlussarbeiten/Diplomarbeiten/Opetz.pdf
"""
def jarque_berat(d:np.array) -> float:
    normal = normalize(d)
    skew = multi_skew_normalized(normal)
    print(skew)
    cort = multi_cort_normalized(normal)
    print(cort)
    n = d.shape[0]
    m = d.shape[1]
    b1 = n/6*skew
    b2 = np.sqrt((n+3)*(n+5)/(8*m*(m+2)*(n-3)*(n-m-1)*(n-m+1)))*((n+1)*cort-m*(m+2)*(n-1))
    jb = b1+b2**2
    df = m*(m+1)*(m+2)/6+1
    print('df',df)
    print("b1,b2:")
    if b1 > df-1:
        p1 = 1-chi2.cdf(b1,df-1)+chi2.cdf(2*(df-1)-b1,df-1)
    else:
        p1 = chi2.cdf(b1,df-1)+1-chi2.cdf(2*(df-1)-b1,df-1)
    print('skew:',skew,'statistic:',b1,'\t: p-value:',p1)
    print('kurt:',cort,'statistic:',b2,'\t: p-value:',1-2*np.abs(.5-norm.cdf(b2)))
    if jb > df:
        stat = 1-chi2.cdf(jb,df) + chi2.cdf(2*df-jb,df)
    else:
        stat = chi2.cdf(jb,df) + 1-chi2.cdf(2*df-jb,df)
    # print("cdf:", stat)

    return stat, jb

"""
only applicable for centered data with unit covariance
"""
def multi_skew_normalized(d):
    if d.shape[0]>10000:
        skew = 0
        for row in tqdm(d):
            skew_r = np.matmul(row,d.T).flatten()
            skew_r = np.power(skew_r,3)
            skew += np.sum(skew_r)
    else:
        skew = np.matmul(d,d.T).flatten()
        skew = np.power(skew,3)
        skew = np.sum(skew)
    skew = skew/(d.shape[0]**2)
    return skew

"""
only applicable for centered data with unit covariance
"""
def multi_cort_normalized(d):
    cort = np.sum(np.linalg.norm(d,axis=1).flatten()**4)/d.shape[0]
    return cort


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", type=str, default="NCVR", help="dataset")
    parser.add_argument("-dim", type=str, default="256", help="dimension")
    parser.add_argument("-layout", type=str, default="s", help="encoder layout")
    args = parser.parse_args()
    ds = args.ds
    dim = args.dim
    layout = 'shallow' if args.layout=='s' else ''

    # vals = []
    # for _ in range(200):
    # d = np.random.normal(0,1,(10000,256))
    #    val = jarque_berat(d)
    #     vals.append(val[0])
    #     print(val)
    # # plt.hist(vals,bins=[0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1])
    # plt.hist(vals,bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    # plt.show()
    path = './configs/configurations_'+ds+'_single/config'+dim+layout+'/__encoded_a.pkl'
    print(path)
    with open(path,'rb') as f:
         d = pkl.load(f)[1]
    # d = d[:10000,]
    print(d.shape)
    print(jarque_berat(d))
