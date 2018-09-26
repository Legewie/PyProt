import numpy as np
from solve_model import *
import matplotlib.pyplot as plt


def residuals_model0(parameters, t, y_data, y_error, u):
    # solve model
    y0 = parameters[0]
    y_model = np.ones(len(y_data)) * y0
    # calculate residuals
    residuals = np.array((y_model - y_data) / y_error, dtype=float)
    residuals[np.isnan(residuals)] = 0
    return residuals


def residuals_modelA(parameters, t, y_data, y_error, u):
    # solve model
    y_model = solve_modelA(parameters, t, u)
    # calculate residuals
    residuals = np.array((y_model - y_data) / y_error, dtype=float)
    residuals[np.isnan(residuals)] = 0
    return residuals


def residuals_modelC(parameters, t, y_data, y_error, u):
    # solve model
    y_model = solve_modelC(parameters, t, u)
    # calculate residuals
    residuals = np.array((y_model - y_data) / y_error, dtype=float)
    residuals[np.isnan(residuals)] = 0
    return residuals


def residuals_modelD(parameters, t, y_data, y_error, u):
    # solve model
    y_model = solve_modelD(parameters, t, u)
    # calculate residuals
    residuals = np.array((y_model - y_data) / y_error, dtype=float)
    residuals[np.isnan(residuals)] = 0
    return residuals
