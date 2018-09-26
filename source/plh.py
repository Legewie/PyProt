# importing
import numpy as np
import matplotlib.pyplot as plt
from source.fit_model import fit_fixed
from source.residuals import residuals_model0

# global ranges for parameters
p_lower = np.log10(np.array([1e-7,  np.log(2)/1000.0, 1.0e-7,  1.e-5]))   # values for LFQ
p_upper = np.log10(np.array([500.0, np.log(2)/0.1,    1.0e-1,  10.0]))

parameters = ['y0', 'degradation', 'production', 'delay']


def profile_likelihood_model0(p_hat, t, protein, p_error, mRNA, plot=False):
    # original v value
    v_hat = np.sum(residuals_model0(p_hat, t, protein, p_error, mRNA) ** 2)
    # max estimated increase in v for step
    delta = 0.1 * 3.841
    # minimum step size
    min_step = 1e-5

    # plh for increasing value of p(i)
    p_opt = np.array(p_hat)
    p_i = float(p_hat[0])
    dp_i = 0.1 * p_i
    v_opt = float(v_hat)
    v_diff = 0
    v_increasing = []
    p_increasing = []
    while np.log10(p_i) < p_upper[0] and v_diff < 3.841:
        # calculate 'estimated' increase in v based on step
        p_next = np.array(p_opt)
        p_next[0] = p_i + dp_i
        v_next = sum(residuals_model0(p_next, t, protein, p_error, mRNA) ** 2)
        # adapt step size
        if v_next - v_opt > delta:
            # decrease
            while v_next - v_opt > delta and np.log10(p_next[0]) < p_upper[0] and dp_i > min_step:
                dp_i /= 2.0
                p_next[0] = p_i + dp_i
                v_next = sum(residuals_model0(p_next, t, protein, p_error, mRNA) ** 2)
        else:
            # increase
            while v_next - v_opt < delta and np.log10(p_next[0]) < p_upper[0]:
                dp_i *= 2.0
                p_next[0] = p_i + dp_i
                v_next = sum(residuals_model0(p_next, t, protein, p_error, mRNA) ** 2)
        # take the step
        p_i = p_i + dp_i
        p_next[0] = p_i
        # check if step is still inside bounds
        if np.log10(p_i) < p_upper[0]:
            # calc new v
            v_opt = sum(residuals_model0(p_next, t, protein, p_error, mRNA) ** 2)
            p_opt = np.array(p_next)
        else:
            v_opt = np.nan
            p_opt = [np.nan]
        # store fitted values
        v_increasing.append(v_opt)
        p_increasing.append(p_opt)
        # update stopping criterion
        v_diff = v_opt - v_hat

    # plh for decreasing value of p(i)
    p_opt = np.array(p_hat)
    p_i = float(p_hat[0])
    dp_i = 0.1 * p_i
    v_opt = float(v_hat)
    v_diff = 0
    v_decreasing = []
    p_decreasing = []
    while np.log10(p_i) > p_lower[0] and v_diff < 3.841:
        # calculate 'estimated' increase in v based on step
        p_next = np.array(p_opt)
        p_next[0] = p_i - dp_i
        v_next = sum(residuals_model0(p_next, t, protein, p_error, mRNA) ** 2)
        # adapt step size
        if v_next - v_opt > delta:
            # decrease
            while v_next - v_opt > delta and np.log10(p_next[0]) > p_lower[0] and dp_i > min_step:
                dp_i /= 2.0
                p_next[0] = p_i - dp_i
                v_next = sum(residuals_model0(p_next, t, protein, p_error, mRNA) ** 2)
        else:
            # increase
            while v_next - v_opt < delta and np.log10(p_next[0]) > p_lower[0]:
                dp_i *= 2.0
                p_next[0] = p_i - dp_i
                v_next = sum(residuals_model0(p_next, t, protein, p_error, mRNA) ** 2)
        # take the step
        p_i = p_i - dp_i
        p_next[0] = p_i
        # check if step is still inside bounds
        if np.log10(p_i) > p_lower[0]:
            # calc new v
            v_opt = sum(residuals_model0(p_next, t, protein, p_error, mRNA) ** 2)
            p_opt = np.array(p_next)
        else:
            v_opt = np.nan
            p_opt = [np.nan]
        # store fitted values
        v_decreasing.append(v_opt)
        p_decreasing.append(p_opt)
        # update stopping criterion
        v_diff = v_opt - v_hat

    # patch results of increasing and decreasing together
    v_decreasing.reverse()
    v_final = np.concatenate([v_decreasing, [v_hat], v_increasing])
    p_decreasing.reverse()
    p_final = np.concatenate([p_decreasing, [p_hat], p_increasing])

    if plot:
        print 'upper limit reached in steps ', len(v_increasing)
        print 'lower limit reached in steps ', len(v_decreasing)
        print 'lower limit: ', p_final[0, 0]
        print 'upper limit: ', p_final[-1, 0]
        plt.plot(p_final[:, 0], v_final, marker='o')
        plt.plot(p_hat[0], v_hat, marker='o', color='red')
        plt.axhline(v_hat + 3.841, color='red', ls='--')
        plt.show()

    # return limits
    return p_final[0, 0], p_final[-1, 0]


def profile_likelihood(func, p_hat, t, protein, p_error, mRNA, i, plot=False):
    # deal with some numerical issues (p_hat value is outside of limits due to precision)
    for idx in range(len(p_hat)):
        if np.log10(p_hat[idx]) > p_upper[idx]:
            p_hat[idx] = 10 ** p_upper[idx]
    verbose = 0
    if verbose > 0: print '\nstaring plh for model ', func, ' parameter ', i
    # number of parameters
    m = len(p_hat)
    # original v value
    v_hat = np.sum(func(np.log10(p_hat), t, protein, p_error, mRNA) ** 2)
    # max estimated increase in v for step
    delta = 0.05 * 3.841
    # minimum step size
    min_step = 1e-5
    if verbose > 0:
        print 'v_hat: ', v_hat
        print 'p_hat: ', p_hat

    # plh for increasing value of p(i)
    p_opt = np.array(p_hat)
    p_i = float(p_hat[i])
    dp_i = 0.1 * p_i
    v_opt = float(v_hat)
    v_diff = 0
    v_increasing = []
    p_increasing = []
    while np.log10(p_i) < p_upper[i] and v_diff < 3.841:
        # calculate 'estimated' increase in v based on step
        p_next = np.array(p_opt)
        p_next[i] = p_i + dp_i
        v_next = sum(func(np.log10(p_next), t, protein, p_error, mRNA) ** 2)
        # adapt step size
        if v_next - v_opt > delta:
            # decrease
            while v_next - v_opt > delta and np.log10(p_next[i]) < p_upper[i] and dp_i > min_step:
                dp_i /= 2.0
                p_next[i] = p_i + dp_i
                v_next = sum(func(np.log10(p_next), t, protein, p_error, mRNA) ** 2)
        else:
            # increase
            while v_next - v_opt < delta and np.log10(p_next[i]) < p_upper[i]:
                dp_i *= 2.0
                p_next[i] = p_i + dp_i
                v_next = sum(func(np.log10(p_next), t, protein, p_error, mRNA) ** 2)
        # take the step
        p_i = p_i + dp_i
        if verbose > 0: print 'increasing p to ', p_i
        # check if step is still inside bounds
        if np.log10(p_i) < p_upper[i] - 1e-10:
            # fit model parameters (except p(i))
            v_opt, p_opt = fit_fixed(func, m, t, protein, p_error, mRNA, i, p_next)
        else:
            v_opt = np.nan
            p_opt = np.zeros(m) * np.nan
        # store fitted values
        v_increasing.append(v_opt)
        p_increasing.append(p_opt)
        # update stopping criterion
        v_diff = v_opt - v_hat

    # plh for decreasing value of p(i)
    p_opt = np.array(p_hat)
    p_i = float(p_hat[i])
    dp_i = 0.1 * p_i
    v_opt = float(v_hat)
    v_diff = 0
    v_decreasing = []
    p_decreasing = []
    while np.log10(p_i) > p_lower[i] and v_diff < 3.841:
        # calculate 'estimated' increase in v based on step
        p_next = np.array(p_opt)
        p_next[i] = p_i - dp_i

        v_next = sum(func(np.log10(p_next), t, protein, p_error, mRNA) ** 2)
        if verbose > 1: print 'estimated increase in v:, ', v_next
        # adapt step size
        if v_next - v_opt > delta:
            # decrease
            while v_next - v_opt > delta and np.log10(p_next[i]) > p_lower[i] and dp_i > min_step:
                dp_i /= 2.0
                if verbose > 1: print '  decreasing step to ', dp_i
                p_next[i] = p_i - dp_i
                v_next = sum(func(np.log10(p_next), t, protein, p_error, mRNA) ** 2)
        else:
            # increase
            while v_next - v_opt < delta and np.log10(p_next[i]) > p_lower[i]:
                dp_i *= 2.0
                if verbose > 1: print '  increasing step to ', dp_i
                p_next[i] = p_i - dp_i
                v_next = sum(func(np.log10(p_next), t, protein, p_error, mRNA) ** 2)
        # take the step
        p_i = p_i - dp_i
        if verbose > 0: print 'decreasing p to ', p_next[i]
        # check if step is still inside bounds (plus some tolerance)
        if np.log10(p_i) > p_lower[i] + 1e-10:
            # fit model parameters (except p(i))
            v_opt, p_opt = fit_fixed(func, m, t, protein, p_error, mRNA, i, p_next)
        else:
            v_opt = np.nan
            p_opt = np.zeros(m) * np.nan
        # store fitted values
        v_decreasing.append(v_opt)
        p_decreasing.append(p_opt)
        # update stopping criterion
        v_diff = v_opt - v_hat

    # patch results of increasing and decreasing together
    v_decreasing.reverse()
    v_final = np.concatenate([v_decreasing, [v_hat], v_increasing])
    p_decreasing.reverse()
    # need to be fixed if estimated parameter coincides wth lower boundary
    if np.log10(p_hat[i]) == p_lower[i]:
        p_decreasing = np.array([p_lower[:len(p_hat)]])
    if np.log10(p_hat[i]) == p_upper[i]:
        p_increasing = np.array([p_upper[:len(p_hat)]])
    p_final = np.concatenate([p_decreasing, [p_hat], p_increasing])

    if plot:
        print 'plh for model ', func, ' parameter ', i
        print 'upper limit reached in steps ', len(v_increasing)
        print 'lower limit reached in steps ', len(v_decreasing)
        print 'lower limit: ', p_final[0, i]
        print 'upper limit: ', p_final[-1, i]
        plt.plot(p_final[:, i], v_final, marker='o', label='profile likelihood')
        plt.plot(p_hat[i], v_hat, marker='o', color='red', label='best fit')
        plt.axhline(v_hat + 3.841, color='red', ls='--', label='chi2-threshold')
        plt.xlabel(parameters[i])
        plt.ylabel('chi2-value')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    # return limits
    return p_final[0, i], p_final[-1, i]