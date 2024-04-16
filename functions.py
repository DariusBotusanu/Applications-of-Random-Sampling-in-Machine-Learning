
import numpy as np

from scipy.stats import randint
from statistics import NormalDist



def bootstrap_samples(sampling_distribution, X, B):
    X_star = np.empty(shape=(B,len(X)))
    for i in range(B):
        X_star[i] = X[sampling_distribution.rvs(size=len(X))]
    return X_star

def plug_in_estimate(X, statistic):
    return statistic(X)

def se_corr(X,Y,B,sampling_distribution):
    T_star_arr = np.empty(shape=(B))
    for i in range(B):
        idx = sampling_distribution.rvs(size=len(X))
        T_star_arr[i] = np.corrcoef(X[idx], Y[idx])[0][1]
    return np.sqrt((1/(B-1))*np.sum((T_star_arr-T_star_arr.mean()*np.ones(B))**2))

def bootstrap_corrcoef_std(sampling_distribution,X,Y,B):
    T_star = np.empty(shape=(B))
    se_corr_arr = np.empty(shape=(B))
    X_star = np.empty(shape=(B,len(X)))
    Y_star = np.empty(shape=(B,len(Y)))
    for i in range(B):
        idx = sampling_distribution.rvs(size=len(X))
        X_star_i = np.array(X[idx])
        Y_star_i = np.array(Y[idx])
        X_star[i] = X_star_i
        Y_star[i] = Y_star_i
        se_corr_arr[i] = se_corr(X_star_i,Y_star_i,200,randint(low = 0, high=len(X_star_i)))
        T_star_i = np.corrcoef(X_star_i, Y_star_i)[0][1]
        T_star[i] = T_star_i
    return se_corr_arr, T_star

def compute_z_star(plug_in_est, T_star, std_arr):
    z = np.empty(len(T_star))
    for i, T in enumerate(T_star):
        z[i] = (T-plug_in_est)/std_arr[i]
    return z

def get_bootstrap_percentile(Z_star, alpha, B):
    idx = int(np.floor(alpha*B))
    return Z_star[idx]

def shape(estimate, up, low):
    return (up-estimate)/(estimate-low)

def bootstrap_samples(sampling_distribution, X, B):
    X_star = np.empty(shape=(B,len(X)))
    for i in range(B):
        X_star[i] = X[sampling_distribution.rvs(size=len(X))]
    return X_star

def percentile_interval(bootstrapped_statistics, alpha):
    return bootstrapped_statistics[int(len(bootstrapped_statistics)*alpha)//1], bootstrapped_statistics[int(len(bootstrapped_statistics)*(1-alpha))//1]

def jacknife_bivariate_samples(X,Y):
    arr = []
    for i in range(len(X)-1):
        arr.append(list(zip(np.append(X[:i][:], X[i+1:][:]), np.append(Y[:i][:], Y[i+1:][:]))))
    arr.append(list(zip(X[:-1][:],Y[:-1][:])))
    return np.array(arr)

def jacknife_samples(X):
    arr = np.empty((len(X), len(X)-1))
    for i in range(len(X)-1):
        arr[i] = np.append(X[:i][:], X[i+1:][:])
    arr[-1] = X[:-1][:]
    return arr

def acceleration(shanked_statistics):
    n = len(shanked_statistics)
    shanked_mean = shanked_statistics.mean()*np.ones(n)
    denominator = np.sum((shanked_mean-shanked_statistics)**3)
    nominator = 6*np.sum((shanked_mean-shanked_statistics)**2)**(3/2)
    return denominator/nominator

def bias_correction(boostrapped_stats, plug_in_estimation, B):
    logic_arr = sorted(boostrapped_stats) < plug_in_estimation
    return NormalDist(mu=0, sigma=1).inv_cdf(np.sum(logic_arr)/B)


def BCa_alphas(z_0, acc, alpha):
    standard_normal = NormalDist(mu=0, sigma=1)
    alpha_one = standard_normal.cdf(z_0+(z_0+standard_normal.inv_cdf(alpha/2))/(1-acc*(z_0+standard_normal.inv_cdf(alpha/2))))
    alpha_two = standard_normal.cdf(z_0+(z_0+standard_normal.inv_cdf(1-alpha/2))/(1-acc*(z_0+standard_normal.inv_cdf(1-alpha/2))))
    return alpha_one, alpha_two


def BCa_confidence_interval(bootstrapped_stats, alpha_one, alpha_two):
    B = len(bootstrapped_stats)
    return bootstrapped_stats[int(B*alpha_one//1)], bootstrapped_stats[int(B*alpha_two//1)]

def bootstrapped_statistics(x_stars, statistic):
    thetas = np.empty(len(x_stars))
    for i,x_star in enumerate(x_stars):
        thetas[i] = statistic(x_star)
    return thetas

if __name__=='main':
    print("Running functions.py")