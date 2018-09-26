import numpy as np
from scipy.stats import trim_mean
from source.fit_model import fit, fit_stationary
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.stattools import durbin_watson


test = np.array([4, 12, 36, 20, 8])


def maximum_entropy(protein):
    # sort data and store original index
    index = protein.argsort()
    p_sorted = protein[index]
    # get intermediate points
    intermediates = (p_sorted[1:] + p_sorted[:-1]) / 2.0
    # compute lower and upper limit from trimmed mean of differences
    deviations = abs(protein[1:] - protein[:-1])
    lower = p_sorted[0] - trim_mean(deviations, 0.1)
    upper = p_sorted[-1] + trim_mean(deviations, 0.1)
    # compute interval means
    means = [0.25 * p_sorted[idx - 1] + 0.5 * p_sorted[idx] + 0.25 * p_sorted[idx + 1] for idx in range(1, len(protein) - 1)]
    m_lower = 0.75 * p_sorted[0] + 0.25 * p_sorted[1]
    m_upper = 0.75 * p_sorted[-1] + 0.25 * p_sorted[-2]
    means = np.insert(means, 0, m_lower)
    means = np.append(means, m_upper)
    # TODO: calculate sample quantiles?

    print index
    print p_sorted
    print intermediates
    print deviations
    print lower, upper
    print means


def par_boot(func_solve, func_fit, m, p, p_error, t, mRNA, res_old):
    n = 1000
    chi2_vec = np.zeros(n)
    dw_vec = np.zeros(n)
    if func_solve == 'stationary':
        # get solution to fitted model
        y_model = [p[0] for _ in res_old]
        # carry out n bootstraps
        for idx in range(n):
            # resample
            y_boot = y_model + np.array([np.random.normal(0, ps) for ps in p_error])
            # fit
            r, p_out = fit_stationary(t, y_boot, p_error)
            chi2_vec[idx] = np.sum(r**2)
            dw_vec[idx] = durbin_watson(r)
    else:
        # get solution to fitted model
        y_model = func_solve(np.log10(p), t, mRNA)
        # carry out n bootstraps
        for idx in range(n):
            # resample
            y_boot = y_model + np.array([np.random.normal(0, ps) for ps in p_error])
            # fit
            r, p_out = fit(func_fit, m, t, y_boot, p_error, mRNA, samples=1, plot=False, p_old=p)
            chi2_vec[idx] = np.sum(r**2)
            dw_vec[idx] = durbin_watson(r)
    # plotting to check distribution
    plot = False
    if plot:
        plt.hist(chi2_vec, bins=int(np.sqrt(n)))
        plt.show()
    # get p-value for chi2 test
    chi2_old = np.sum(res_old**2)
    chi2_ecdf = ECDF(chi2_vec)
    chi2_p = 1 - chi2_ecdf(chi2_old)                    # right sided test for chi2
    # get p-value for dw-test
    dw_old = durbin_watson(res_old)
    dw_ecdf = ECDF(dw_vec)
    dw_p = dw_ecdf(dw_old)                              # left sided test durbin-watson
    return chi2_p, dw_p

