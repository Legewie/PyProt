import numpy as np
from bisect import bisect_right
import matplotlib.pyplot as plt
from scipy import integrate

def solve_modelA(parameters, t, u):
    parameters = 10 ** parameters
    # annotate parameters
    y0 = parameters[0]
    l = parameters[1]
    # solve model
    y_model = y0 * np.exp(-l * t)
    return y_model


def solve_modelC(parameters, t, u):
    parameters = 10 ** parameters
    # annotate parameters
    y0 = parameters[0]
    l = parameters[1]
    a = parameters[2]
    y_model = np.zeros(len(t))
    # solve model
    y_model[0] = y0
    for idx in range(len(t) - 1):
        dt = t[idx + 1] - t[idx]
        m = (u[idx + 1] - u[idx]) / dt
        b = u[idx]
        c = y_model[idx] - a * b / l + a * m / l ** 2
        y_model[idx + 1] = a * b / l - a * m / l ** 2 + a * m * dt / l + c * np.exp(-l * dt)
    return y_model

def solve_modelC_ODE(t, y_in, parameters, u_t, u):
    parameters = 10 ** parameters
    # annotate parameters
    y0 = parameters[0]
    l = parameters[1]
    a = parameters[2]
    # solve_model
    mRNA = np.interp(t, u_t, u)
    return a * mRNA - l * y_in

def num_solve(parameters, t, u):
    ode = integrate.ode(solve_modelC_ODE)
    ode.set_integrator('lsoda', nsteps=1000)
    ode.set_initial_value(10 ** parameters[0], t=0.0)
    ode.set_f_params(parameters, t, u)
    # obtain the numerical solution
    t_out = [0.0]
    g_out = [10 ** parameters[0]]
    # get minimal time step
    dt = 1.0 #min(experiment.target_obj.y_t[1:] - experiment.target_obj.y_t[:-1]) FIXME
    while ode.successful() and ode.t < t[-1]:
        ode.integrate(ode.t + dt)
        t_out.append(ode.t)
        g_out.append(ode.y[0])
    # return only g_out with requested indices
    return [g_out[yt] for yt in t]



def solve_modelD(parameters, t, u):
    parameters = 10 ** parameters
    # get values for derivatives of u
    derivatives = (u[1:] - u[:-1]) / (t[1:] - t[:-1])
    # annotate parameters
    y0 = parameters[0]
    l = parameters[1]
    a = parameters[2]
    tau = parameters[3]
    # error checking (FIXME: I am not sure why tau can become NaN, it must be inside plh)
    if np.isnan(tau):
        tau = 0.0
    y_model = np.zeros(len(t))
    # solve first stage of model
    t_phase_1 = np.array([time for time in t if time < tau] + [tau])
    y_temp_1 = y0 * np.exp(-l * t_phase_1)
    # y(tau) part of y only if tau in t
    if tau in t:
        y_phase_1 = y_temp_1
    else:
        y_phase_1 = y_temp_1[:-1]
    # assign first part of solution
    y_model[:len(y_phase_1)] = y_phase_1
    # prepare stage 2 (production stage)
    t_phase_2 = np.array([time for time in t if time > tau])
    y_phase_2 = np.zeros(len(t_phase_2))
    # find time-point right of tau
    idx = bisect_right(t, tau)
    # calculate value for m and b on fist interval
    dt = t[idx] - tau
    m = derivatives[idx - 1]
    b = u[idx - 1] + m * (tau - t[idx - 1])
    c = y_temp_1[-1] - a * b / l + a * m / l ** 2
    y_phase_2[0] = a * b / l - a * m / l ** 2 + a * m * dt / l + c * np.exp(-l * dt)
    # solve the remaining part of stage 2 as in modelC
    offset = len(y_phase_1)
    for idx in range(len(y_phase_2) - 1):
        dt = t_phase_2[idx + 1] - t_phase_2[idx]
        m = derivatives[idx + offset]
        b = u[idx + offset]
        c = y_phase_2[idx] - a * b / l + a * m / l ** 2
        y_phase_2[idx + 1] = a * b / l - a * m / l ** 2 + a * m * dt / l + c * np.exp(-l * dt)
    y_model[len(y_phase_1):] = y_phase_2
    return y_model


def details(parameters, protein, p_error, mRNA, t, plot=False):
    # solve model
    y_model = solve_modelC(np.log10(parameters), t, mRNA)
    # calculate in and out
    protein_in = parameters[2] * mRNA
    protein_out = parameters[1] * y_model
    # plotting
    if plot:
        fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True)
        # protein
        ax1.errorbar(t, protein, yerr=p_error, color='black', lw=1, label='protein')
        ax1.plot(t, y_model, color='#ffa17c', lw=2, label='model')
        ax1.set_ylabel('protein [normalized LFQ]')
        #ax1.legend(loc=2, bbox_to_anchor=(0.05, 1.5))

        # mRNA
        ax2 = ax1.twinx()
        ax2.plot(t, mRNA, ls='--', color='gray', marker='o', label='mRNA')
        ax2.set_ylabel('mRNA [RPKM]')
        ax2.set_xlabel('time [h]')
        #ax2.legend(loc=4)
        plt.xlim(-1, 21)
        plt.grid()
        ax3.fill_between(t, protein_in, 0, color='red', lw=0, label='protein production', alpha=0.25)
        ax3.fill_between(t, protein_out, 0, color='blue', lw=0, label='protein degradation', alpha=0.25)
        ax3.set_xlabel('time [h]')
        ax3.set_ylabel('reaction velocity \n [normalized LFQ $s^{-1}$]')
        ax3.legend(loc=0)
        ax3.grid()
        #plt.tight_layout()
        plt.show()
    return sum(protein_in), sum(protein_out), sum(y_model)


################# ModelX equations

def rhs_modelX_ODE(t, y_in, parameters, u_t, u):
    # annotate parameters
    l = 10 ** parameters[1]
    a = 10 ** parameters[2]
    h = parameters[3]
    k = parameters[4]
    # interpolate mRNA
    mRNA = np.interp(t, u_t, u)
    # return RHS
    return a * ((t + 1) ** h / ((t + 1) ** h + k)) * mRNA - l * y_in


def solve_modelX(parameters, t, u):
    ode = integrate.ode(rhs_modelX_ODE)
    ode.set_integrator('lsoda', nsteps=1000)
    ode.set_initial_value(10 ** parameters[0], t=0.0)
    ode.set_f_params(parameters, t, u)
    # obtain the numerical solution
    t_out = [0.0]
    g_out = [10 ** parameters[0]]
    dt = 1.0
    while ode.successful() and ode.t < t[-1]:
        ode.integrate(ode.t + dt)
        t_out.append(ode.t)
        g_out.append(ode.y[0])
    # return only g_out with requested indices
    return [g_out[yt] for yt in t]



def rhs_modelX2_ODE(t, y_in, parameters, u_t, u, u2):
    # annotate parameters
    l = 10 ** parameters[1]
    a = 10 ** parameters[2]
    h = parameters[3]
    k = parameters[4]
    # interpolate mRNA
    mRNA = np.interp(t, u_t, u)
    rbp = np.interp(t, u_t, u2)

    # return RHS
    return a * (rbp ** h / (rbp ** h + k)) * mRNA - l * y_in


def solve_modelX2(parameters, t, u, u2):
    ode = integrate.ode(rhs_modelX2_ODE)
    ode.set_integrator('lsoda', nsteps=1000)
    ode.set_initial_value(10 ** parameters[0], t=0.0)
    ode.set_f_params(parameters, t, u, u2)
    # obtain the numerical solution
    t_out = [0.0]
    g_out = [10 ** parameters[0]]
    dt = 1.0
    while ode.successful() and ode.t < t[-1]:
        ode.integrate(ode.t + dt)
        t_out.append(ode.t)
        g_out.append(ode.y[0])
    # return only g_out with requested indices
    return [g_out[yt] for yt in t]