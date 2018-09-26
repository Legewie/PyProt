import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from source.solve_model import *

t = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20])
pt = ['p_00h', 'p_01h', 'p_02h', 'p_03h', 'p_04h', 'p_05h', 'p_06h', 'p_08h', 'p_10h', 'p_12h', 'p_14h', 'p_16h',
      'p_18h', 'p_20h']
mt = ['m_00h', 'm_01h', 'm_02h', 'm_03h', 'm_04h', 'm_05h', 'm_06h', 'm_08h', 'm_10h', 'm_12h', 'm_14h', 'm_16h',
      'm_18h', 'm_20h']
ps = ['ps_00h', 'ps_01h', 'ps_02h', 'ps_03h', 'ps_04h', 'ps_05h', 'ps_06h', 'ps_08h', 'ps_10h', 'ps_12h', 'ps_14h',
      'ps_16h', 'ps_18h', 'ps_20h']
ms = ['ms_00h', 'ms_01h', 'ms_02h', 'ms_03h', 'ms_04h', 'ms_05h', 'ms_06h', 'ms_08h', 'ms_10h', 'ms_12h', 'ms_14h',
      'ms_16h', 'ms_18h', 'ms_20h']
models = ['modelA', 'modelC', 'modelD']
funcs = [solve_modelA, solve_modelC, solve_modelD]
parms = ['y0', 'degradation', 'production', 'delay']
model_colors = ['#6289cc', '#1f4c99', '#ffa17c', '#cc452f', '#dbdbdb']

def plot_tc(FBgn, title):
    # load data
    data = pd.read_csv('~/ofc/projects/nuria/0.0_data/matlab_input_mean_em3.csv', index_col=0)
    # load results
    df = pd.read_csv('/home/bleep/ofc/projects/nuria/1.0_fitting/pyProt/results/result_full_5_10_25_chi2dwBS_LRT_corrected_1_final', index_col=0)
    # get entry for protein
    entry = df[df.Gene == FBgn].iloc[0]
    # get model and function for gene
    model = str(entry.model_final)
    func = funcs[models.index(model)]
    # get model parameters
    parameters = []
    count = models.index(model) + 2
    for idx in range(count):
        parameters.append(entry[parms[idx]])
    parameters = np.array(parameters)
    # get protein, p_error, mRNA and y_model
    protein = np.array(data[data.Gene == FBgn].iloc[0][pt])
    p_error = np.array(data[data.Gene == FBgn].iloc[0][ps])
    mRNA = np.array(data[data.Gene == FBgn].iloc[0][mt])
    y_model = func(np.log10(parameters), t, mRNA)
    ax1 = plt.gca()
    # plot protein
    ax1.errorbar(t, protein, yerr=p_error, marker='o', color='black', label='protein')
    ax1.plot(t, y_model, color=model_colors[count - 1], lw=2)
    ax1.set_ylabel('protein', size=20, labelpad=15)
    ax1.legend(loc=2, prop={'size': 20})
    # plot mRNA
    ax2 = ax1.twinx()
    ax2.plot(t, mRNA, ls='--', color='gray', marker='o', label='mRNA', alpha=0.75)
    ax2.set_ylabel('mRNA', size=20, labelpad=15)
    ax2.tick_params(labelsize=15)
    ax1.set_xlabel('time [h]', size=20, labelpad=15)
    ax1.tick_params(labelsize=15)
    ylim = ax2.get_ylim()
    ax2.set_ylim(ylim[0] * 0.8, ylim[1] * 1.2)
    ax2.legend(loc=4, prop={'size': 20})
    plt.xlim(-1, 21)
    plt.grid()
    plt.title(title + '\n', fontsize=20)
    plt.tight_layout()
    plt.show()