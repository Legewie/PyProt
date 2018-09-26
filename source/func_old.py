__author__ = 'bleep'




def fit_production_leastsq(protein, p_error, mRNA, samples=5, plot=False):
    # number of parameters
    n = 3
    # ranges for fitting
    ranges = [(lower, upper) for lower, upper in zip(p_lower[:n], p_upper[:n])]
    # latin hypercube sampling
    p_ini_matrix = sample_lhs(p_lower[:n], p_upper[:n], samples=samples)
    # fitting
    results = np.zeros((samples, n + 1))
    for idx, p_ini in enumerate(p_ini_matrix):
        p1, cov, infodict, e_msg, success = optimize.leastsq(
            residuals_modelC,
            p_ini,
            args=(protein, p_error, mRNA),
            full_output = True)
        results[idx, 0] = np.sum(residuals_modelC(p1, protein, p_error, mRNA) ** 2)
        results[idx, 1:] = p1
    # sort results and choose best fit
    results = results[results[:, 0].argsort()]
    v_hat = results[0, 0]
    p_hat = results[0, 1:]
    # plotting - optional
    if plot:
        print v_hat, p_hat
        plt.errorbar(t, protein, yerr=p_error)
        for result in results:
            y_model = solve_modelC(result[1:], mRNA)
            plt.plot(t, y_model)
        plt.xlim([-1, 21])
        plt.show()
    return results


def fit_production_lmfit(protein, p_error, mRNA, samples=5, plot=False):
    # number of parameters
    n = 3
    names = ['y0', 'l', 'a']
    # ranges for fitting
    ranges = [(lower, upper) for lower, upper in zip(p_lower[:n], p_upper[:n])]
    # latin hypercube sampling
    p_ini_matrix = sample_lhs(p_lower[:n], p_upper[:n], samples=samples)
    # fitting
    results = np.zeros((samples, n + 1))
    for idx, p_ini in enumerate(p_ini_matrix):
        # create parameters object
        p_obj = Parameters()
        for idx2 in range(n):
            p_obj.add(names[idx2], value=p_ini[idx2], min=p_lower[idx2], max=p_upper[idx2], vary=True)
        # fitting
        fit = Minimizer(residuals_modelC, p_obj, fcn_args=(protein, p_error, mRNA))
        fit.leastsq()
        results[idx, 0] = fit.chisqr
        results[idx, 1:] = [fit.params['y0'], fit.params['l'], fit.params['a']]
    # sort results and choose best fit
    results = results[results[:, 0].argsort()]
    v_hat = results[0, 0]
    p_hat = results[0, 1:]
    # plotting - optional
    if plot:
        print v_hat, p_hat
        plt.errorbar(t, protein, yerr=p_error)
        for result in results:
            #y_model = solve_modelC(result[1:], mRNA)
            #plt.plot(t, y_model)
            pass
        plt.xlim([-1, 21])
        plt.show()
    return results


def fit_production_alpso(protein, p_error, mRNA, samples=5, plot=False):
    # number of parameters
    n = 3
    names = ['y0', 'l', 'a']
    # initialization of the optimisation problem
    problem = pyOpt.Optimization('test', pyOpt_modelC)
    problem.addObj('f')
    # add variables to problem
    for idx in range(n):
        problem.addVar(names[idx],
                       'c',
                       lower=p_lower[idx],
                       upper=p_upper[idx],
                       value=np.random.rand(1) * (p_upper[idx] - p_lower[idx]) + p_lower[idx])
    optimizer = pyOpt.MIDACO()
    kwargs = {'y_data': protein, 'y_error': p_error, 'u': mRNA}
    [fstr, xstr, inform] = optimizer(problem, **kwargs)
    print fstr
    return fstr[0]



def pyOpt_modelC(parameters, **kwargs):
    # get arguments
    y_data = kwargs['y_data']
    y_error = kwargs['y_error']
    u = kwargs['u']
    # solve model
    y_model = solve_modelC(parameters, u)
    # calculate residuals
    residuals = np.sum((y_model - y_data) ** 2 / y_error ** 2)
    return residuals, [], 0




def fit_production_leastsq(protein, p_error, mRNA, samples=5, plot=False):
    start = datetime.datetime.now()

    # number of parameters
    n = 3
    # ranges for fitting
    ranges = [(lower, upper) for lower, upper in zip(p_lower[:n], p_upper[:n])]
    # latin hypercube sampling
    p_ini_matrix = sample_lhs(p_lower[:n], p_upper[:n], samples=samples)
    # fitting
    results = np.zeros((samples, n + 1))
    for idx, p_ini in enumerate(p_ini_matrix):
        p1, cov, infodict, e_msg, success = optimize.leastsq(
            residuals_modelC2,
            p_ini,
            args=(protein, p_error, mRNA),
            full_output=True)
        results[idx, 0] = np.sum(residuals_modelC2(p1, protein, p_error, mRNA) ** 2)
        results[idx, 1:] = p1
    # sort results and choose best fit
    results = results[results[:, 0].argsort()]
    v_hat = results[0, 0]
    p_hat = results[0, 1:]
    # plotting - optional
    if plot:
        print v_hat, p_hat
        plt.errorbar(t, protein, yerr=p_error)
        for result in results:
            y_model = solve_modelC(result[1:], mRNA)
            plt.plot(t, y_model)
        plt.xlim([-1, 21])
        plt.show()
    end = datetime.datetime.now()
    print end - start
    return results



def fit_production_mini(protein, p_error, mRNA, samples=5, plot=False):
    start = datetime.datetime.now()
    # number of parameters
    n = 3
    # ranges for fitting
    ranges = [(lower, upper) for lower, upper in zip(p_lower[:n], p_upper[:n])]
    # latin hypercube sampling
    p_ini_matrix = sample_lhs(p_lower[:n], p_upper[:n], samples=samples)
    # fitting
    results = np.zeros((samples, n + 1))
    for idx, p_ini in enumerate(p_ini_matrix):
        result = optimize.minimize(residuals_modelC, p_ini, method=method, options=options, args=(protein, p_error, mRNA), jac=False, bounds=ranges)
        results[idx, 0] = result.fun
        results[idx, 1:] = result.x
    # sort results and choose best fit
    results = results[results[:, 0].argsort()]
    v_hat = results[0, 0]
    p_hat = results[0, 1:]
    # plotting - optional
    if plot:
        print v_hat, p_hat
        plt.errorbar(t, protein, yerr=p_error)
        for result in results:
            y_model = solve_modelC(result[1:], mRNA)
            plt.plot(t, y_model)
        plt.xlim([-1, 21])
        plt.show()
    end = datetime.datetime.now()
    print end - start
    return results