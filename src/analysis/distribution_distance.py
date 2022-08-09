import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
import pickle as pkl
from matplotlib import pyplot as plt
#from scipy.stats import norm, stats
from scipy.stats import shapiro
from scipy import stats
sys.path.append(os.getcwd().rstrip("/").rstrip("analysis"))
from autoencoder import normalize, cumulative_normal

def matrix_root(m: np.array, inverse:bool=False) -> np.array:
    eig, base = np.linalg.eig(m)
    if not all(e > 0 for e in eig):
        raise ValueError("Matrix is not positive definite")
    if not inverse:
        root_matrix = np.matmul(np.matmul(base, np.diag(np.sqrt([e for e in eig]))), base.T)
    else:
        root_matrix = np.matmul(np.matmul(base,np.diag(np.sqrt([1/e for e in eig]))), base.T)
    return root_matrix

def normalize(data,mean_cov=None,inverse=False,return_mean_cov=False):
    if mean_cov is not None:
        mean, cov = mean_cov
    else:
        mean, cov = (np.mean(data, axis=0), np.cov(data, rowvar=False))
    if not inverse:
        data = data-mean
        t = matrix_root(cov,inverse=True)
        data = np.matmul(data,t)
    else:
        t = matrix_root(cov, inverse=False)
        data = np.matmul(data, t)
        data = data+mean
    if return_mean_cov:
        return data, (mean, cov)
    else:
        return data

def cumulative_normal(x):
    return stats.norm(0, 1).cdf(x)

def assign_buckets(ds,n_bucks):
    ret = []
    exp = ds.shape[0]/n_bucks
    for dim in tqdm(range(ds.shape[1])):
        bucks = [0 for _ in range(n_bucks)]
        for rec in ds:
            # print(rec[dim])
            ix = int(rec[dim]*n_bucks)
            if ix >= len(bucks):
                print(ix)
                ix = len(bucks)-1
            elif ix < 0:
                print(ix)
                ix = 0
            bucks[ix] += 1
        bucks = [np.abs(x/exp-1) for x in bucks]
        ret.append(bucks)
    ret = np.array(ret)
    return ret

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
    #     d = np.random.normal(0,1,(10000,256))
    #        val = jarque_berat(d)
    #         vals.append(val[0])
    #         print(val)
    #     plt.hist(vals,bins=[0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1])
    #     plt.hist(vals,bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    #     plt.show()
    dss = ['NCVR_3', 'ohio']
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    deviation = ""
    for ds in dss:
        deviation = ""
        hist_intersection = ""
        shapiro_res = ""
        avg = 0
        for dim in [64, 128, 256]:
            sm = 0
            for layout in ['shallow', '']:
                path = '../configs/configurations_'+ds+'/config'+str(dim)+layout+'/__encoded_a.pkl'
                print(path)
                with open(path,'rb') as f:
                     d = pkl.load(f)[1]
                d = normalize(d)

                min_p = 1
                max_p = 0
                avg_p = 0
                min_stat = 1
                max_stat = 0
                avg_stat = 0
                min_counts = None
                min_hists = None
                max_counts = None
                max_hists = None
                min_d = 0
                max_d = 0
                min_his = 1
                max_his = 0
                for i in range(0, dim):
                    sample = np.random.choice(d[:, i], 1000, replace=False)
                    #sample = np.random.normal(0, 1, (d.shape[0]))
                    stat, p = shapiro(sample)
                    min_p = min(min_p, p)
                    min_stat = min(min_stat, stat)
                    max_p = max(max_p, p)
                    min_stat = max(max_stat, stat)
                    avg_p += p
                    avg_stat += stat
                    normal = np.random.normal(0, 1, (d.shape[0] * 100))
                    #normal = np.mean(normal, axis=1)
                    mini = np.min(d[:, i])
                    maxi = np.max(d[:, i])
                    mini_2 = np.min(normal)
                    maxi_2 = np.max(normal)
                    width = (max(maxi_2, maxi) - min(mini, mini_2))/1000
                    #print(width)
                    bins = [(min(mini, mini_2) + (i+1)*width) for i in range(1000)]

                    #pdf *= d.shape[0]

                    #np.diff(np.asarray(bins))
                    #print(bins)
                    #print(len(bins))
                    normal_hist = np.histogram(normal, bins='auto', range=(-5, 5),
                                               density=True)
                    bins = len(normal_hist[1])
                    actual_hist = np.histogram(d[:, i], bins=bins-1, range=(-5, 5), density=True)
                    #print(sum(actual_hist[0]*np.diff(actual_hist[1])))
                    actual_counts = actual_hist[0] * np.diff(actual_hist[1]) * d.shape[0]
                    normal_counts = normal_hist[0] * np.diff(normal_hist[1]) * d.shape[0]
                    actual_hist = (actual_counts, actual_hist[1])
                    normal_hist = (normal_counts, normal_hist[1])
                    #print(normal_hist)
                    his_inter_dim = np.sum(np.minimum(actual_hist[0], normal_hist[0])) / d.shape[0]
                    sm += np.sum(np.minimum(actual_hist[0], normal_hist[0]))/d.shape[0]
                    #min_his = min(his_inter_dim, min_his)
                    #max_his = max(his_inter_dim, max_his)
                    if (min_his > his_inter_dim) and ds == 'NCVR_2' and dim == 256 and layout == 'shallow':
                        min_his = his_inter_dim
                        min_hists = actual_hist[1]
                        min_counts = actual_counts
                        min_d = i
                    if (max_his < his_inter_dim) and ds == 'NCVR_2' and dim == 256 and layout == 'shallow':
                        max_his = his_inter_dim
                        max_hists = actual_hist[1]
                        max_counts = actual_counts
                        max_d = i
                if ds == 'NCVR_2' and dim == 256 and layout == 'shallow':
                    plt.hist(normal_hist[1][:-1], normal_hist[1], weights=normal_counts, color='red',
                     alpha=0.5)
                    plt.hist(min_hists[:-1], min_hists, weights=min_counts, color='blue', alpha=0.5)
                    #plt.title(str(p))
                    plt.savefig("hist_{}{}_{}.pdf".format(dim, 's', min_d))
                    print(min_his)
                    plt.show()
                    plt.hist(normal_hist[1][:-1], normal_hist[1], weights=normal_counts, color='red',
                             alpha=0.5)
                    plt.hist(max_hists[:-1], max_hists, weights=max_counts, color='blue', alpha=0.5)
                    # plt.title(str(p))
                    plt.savefig("hist_{}{}_{}.pdf".format(dim, 's', max_d))
                    print(max_his)
                    #plt.hist(d[:, i], bins=100, alpha=0.5)
                    plt.show()
                print(len(normal_counts))
                sm /= dim
                d = cumulative_normal(d)
                b = assign_buckets(d, 10000)
                deviation += str(np.mean(b))+"\t"
                hist_intersection += str(sm) + "\t"
                #print('data set\tdim\tlayout\tmin_p\tmax_p\tavg_p\tmin_stat\tmax_stat\tavg_stat\n')
                shapiro_res += (ds+"\t"+str(dim)+"\t"+layout+"\t"+str(min_p)+"\t"+str(max_p)+"\t"+str(avg_p/dim)+"\t"
                      + str(min_stat)+"\t"+str(max_stat)+"\t"+str(avg_stat/dim)+"\n")
        print(ds)
        print()
        print(deviation)
        print()
        print(hist_intersection)
        print()
        print('data set\tdim\tlayout\tmin_p\tmax_p\tavg_p\tmin_stat\tmax_stat\tavg_stat')
        print(shapiro_res)
