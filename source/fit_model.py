import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy import optimize
from residuals import *
import datetime


# global ranges for parameters
p_lower = np.log10(np.array([1e-7,  np.log(2)/1000.0, 1.0e-7,  1.e-5]))   # values for LFQ
p_upper = np.log10(np.array([500.0, np.log(2)/0.1,    1.0e-1,  10.0]))

# options for fitting
nfev = 2000
jac = '2-point'     # was tested against 3-point - no increase in performance


def sample_lhs(lower, upper, samples=5):
    """
    :param lower/upper: limits for sampling
    :param samples: number of samples
    :return: latin hypercube sample within provided bounds
    """
    # get latin hypercube sample on interval 0,1
    norm = lhs(len(lower), samples=samples)
    # scale normalized parameter sample to provided ranges
    return (upper - lower) * norm + lower


def fit_stationary(t, protein, p_error, plot=False):
    pp = np.ma.array(protein, mask=np.isnan(protein))
    pe = np.ma.array(p_error, mask=np.isnan(protein))
    # calculate weighted mean
    weighted_mean = np.ma.average(pp, weights=1 / pe ** 2)
    residuals = (np.array(pp) - weighted_mean) / np.array(pe)
    # plotting - optional
    if plot:
        print np.sum(residuals ** 2), weighted_mean
        plt.errorbar(t, protein, yerr=p_error)
        plt.axhline(weighted_mean)
        plt.xlim([-1, 21])
        plt.show()
    residuals[np.isnan(residuals)] = 0
    return residuals, [weighted_mean]


def fit(func, m, t, protein, p_error, mRNA, samples=5, plot=False, p_old=None):
    # latin hypercube sampling (or passed parameters)
    if p_old is None:
        p_ini_matrix = sample_lhs(p_lower[:m], p_upper[:m], samples=samples)
    else:
        #p_ini_matrix = sample_lhs(p_lower[:m], p_upper[:m], samples=samples - 1)
        p_ini_matrix = [np.log10(p_old)]
    # fitting
    v_matrix = np.zeros(samples)
    r_matrix = np.zeros([samples, len(t)])
    p_matrix = np.zeros([samples, m])
    for idx, p_ini in enumerate(p_ini_matrix):
        result = optimize.least_squares(
            func,
            p_ini,
            bounds=(p_lower[:m], p_upper[:m]),
            max_nfev=nfev,
            jac=jac,
            args=(t, protein, p_error, mRNA))
        v_matrix[idx] = sum(result.fun ** 2)
        r_matrix[idx, :] = result.fun
        p_matrix[idx, :] = result.x
    # sort results and choose best fit
    ranking = v_matrix.argsort()
    v_matrix = v_matrix[ranking]
    r_matrix = r_matrix[ranking]
    p_matrix = p_matrix[ranking]
    v_hat = v_matrix[0]
    r_hat = r_matrix[0]
    p_hat = 10 ** p_matrix[0]
    # plotting - optional
    if plot:
        print v_hat, p_hat
        print v_matrix
        plt.errorbar(t, protein, yerr=p_error)
        plt.xlim([-1, 21])
        plt.show()
    return r_hat, p_hat


def fit_fixed(func, m, t, protein, p_error, mRNA, i, p, samples=5, plot=False):
    # create temporary ranges (small range for p(i))
    p_lower_temp = np.array(p_lower)
    p_upper_temp = np.array(p_upper)
    p_lower_temp[i] = np.log10(p[i]) - 1e-12
    p_upper_temp[i] = np.log10(p[i]) + 1e-12
    # latin hypercube sampling NOTE: currently deactivated, only local fit carried out
    #p_ini_matrix = sample_lhs(p_lower_temp[:m], p_upper_temp[:m], samples=samples)
    #p_ini_matrix = np.concatenate([[np.log10(p)], p_ini_matrix])
    p_ini_matrix = [np.log10(p)]
    # fitting
    v_matrix = np.zeros(1)
    r_matrix = np.zeros([1, len(t)])
    p_matrix = np.zeros([1, m])
    for idx, p_ini in enumerate(p_ini_matrix):
        result = optimize.least_squares(
            func,
            p_ini,
            bounds=(p_lower_temp[:m], p_upper_temp[:m]),
            max_nfev=nfev,
            jac=jac,
            args=(t, protein, p_error, mRNA))
        v_matrix[idx] = sum(result.fun ** 2)
        r_matrix[idx, :] = result.fun
        p_matrix[idx, :] = result.x
    # sort results and choose best fit
    ranking = v_matrix.argsort()
    v_matrix = v_matrix[ranking]
    r_matrix = r_matrix[ranking]
    p_matrix = p_matrix[ranking]
    v_hat = v_matrix[0]
    r_hat = r_matrix[0]
    p_hat = 10 ** p_matrix[0]
    # plotting - optional
    if plot:
        print v_hat, p_hat, r_hat
        print v_matrix
        plt.errorbar(t, protein, yerr=p_error)
        plt.xlim([-1, 21])
        plt.show()
    return v_hat, p_hat
