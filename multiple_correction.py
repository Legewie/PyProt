import pandas as pd
import numpy as np
from source.m_eval import model_evaluation
from statsmodels.stats.multitest import multipletests
from source.plh import *
from source.fit_model import *


models = ['model0', 'modelA', 'modelC', 'modelD', 'modelX']
funcs = [np.nan, residuals_modelA, residuals_modelC, residuals_modelD, np.nan]
parameter_numbers = [1, 2, 3, 4, 0]
parameter_names = ['y0', 'degradation', 'production', 'delay']

full = True
if full:
    t = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20])
    pt = ['p_00h', 'p_01h', 'p_02h', 'p_03h', 'p_04h', 'p_05h', 'p_06h', 'p_08h', 'p_10h', 'p_12h', 'p_14h', 'p_16h',
          'p_18h', 'p_20h']
    mt = ['m_00h', 'm_01h', 'm_02h', 'm_03h', 'm_04h', 'm_05h', 'm_06h', 'm_08h', 'm_10h', 'm_12h', 'm_14h', 'm_16h',
          'm_18h', 'm_20h']
    ps = ['ps_00h', 'ps_01h', 'ps_02h', 'ps_03h', 'ps_04h', 'ps_05h', 'ps_06h', 'ps_08h', 'ps_10h', 'ps_12h', 'ps_14h',
          'ps_16h', 'ps_18h', 'ps_20h']
    ms = ['ms_00h', 'ms_01h', 'ms_02h', 'ms_03h', 'ms_04h', 'ms_05h', 'ms_06h', 'ms_08h', 'ms_10h', 'ms_12h', 'ms_14h',
          'ms_16h', 'ms_18h', 'ms_20h']
else:
    t = np.array([3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20])
    pt = ['p_03h', 'p_04h', 'p_05h', 'p_06h', 'p_08h', 'p_10h', 'p_12h', 'p_14h', 'p_16h', 'p_18h', 'p_20h']
    mt = ['m_03h', 'm_04h', 'm_05h', 'm_06h', 'm_08h', 'm_10h', 'm_12h', 'm_14h', 'm_16h', 'm_18h', 'm_20h']
    ps = ['ps_03h', 'ps_04h', 'ps_05h', 'ps_06h', 'ps_08h', 'ps_10h', 'ps_12h', 'ps_14h', 'ps_16h', 'ps_18h', 'ps_20h']


def multitest_correct(results_file, data_file=None, out_file=None):
    # get results from fitting with bootstrap
    df_bs = pd.read_csv(results_file, index_col=0)

    if data_file is not None:
        # get data  (AFAIK only necessary for PLH)
        data = pd.read_csv(data_file, index_col=0)
        data.set_index('ProteinGroup', inplace=True)

    # carry out multiple testing correction
    df_bs['chi2q0'] = multipletests(df_bs['chi2p0'], method='fdr_bh')[1]
    df_bs['chi2qA'] = multipletests(df_bs['chi2pA'], method='fdr_bh')[1]
    df_bs['chi2qC'] = multipletests(df_bs['chi2pC'], method='fdr_bh')[1]
    df_bs['chi2qD'] = multipletests(df_bs['chi2pD'], method='fdr_bh')[1]
    df_bs['dw_q0'] = multipletests(df_bs['dw_p0'], method='fdr_bh')[1]
    df_bs['dw_qA'] = multipletests(df_bs['dw_pA'], method='fdr_bh')[1]
    df_bs['dw_qC'] = multipletests(df_bs['dw_pC'], method='fdr_bh')[1]
    df_bs['dw_qD'] = multipletests(df_bs['dw_pD'], method='fdr_bh')[1]

    for idx, row in df_bs.iterrows():
        print idx
        # get value for r (reject null-hypothesis)
        r0 = (row.chi2q0 < 0.05) or (row.dw_q0 < 0.05)
        rA = (row.chi2qA < 0.05) or (row.dw_qA < 0.05)
        rC = (row.chi2qC < 0.05) or (row.dw_qC < 0.05)
        rD = (row.chi2qD < 0.05) or (row.dw_qD < 0.05)
        # carry out model selection
        model, v_new = model_evaluation(r0, rA, rC, rD, row.v0, row.vA, row.vC, row.vD, verbose=0)
        # carry out correction step
        if model in ['model0', 'modelA'] and rC == False and row.pC_production > 2e-5:
            model_corrected = 'modelC'
            v = row.vC
        else:
            model_corrected = model
            v = v_new
        df_bs.loc[idx, 'model_final'] = model_corrected
        df_bs.loc[idx, 'v_final'] = v
        # assign final parameters
        p0 = [row.m0_y0]
        pA = [row.mA_y0, row.mA_degradation]
        pC = [row.mC_y0, row.mC_degradation, row.mC_production]
        pD = [row.mD_y0, row.mD_degradation, row.mD_production, row.mD_delay]
        p_hat = [p0, pA, pC, pD, np.nan][models.index(model_corrected)]
        # final parameters
        if model_corrected in models[:-1]:
            df_bs.loc[idx, 'y0']          = p_hat[0]
        if model_corrected in models[1:-1]:
            df_bs.loc[idx, 'degradation'] = p_hat[1]
        if model_corrected in models[2:-1]:
            df_bs.loc[idx, 'production']  = p_hat[2]
        if model_corrected in models[3:-1]:
            df_bs.loc[idx, 'delay']       = p_hat[3]
        # profile likelihood (TODO: write separate function with plh after multitesting)
        profile = False
        if profile:
            protein = np.array(data.ix[idx][pt], dtype=float)
            p_error = np.array(data.ix[idx][ps], dtype=float)
            mRNA = np.array(data.ix[idx][mt], dtype=float)

            if model_corrected == 'model0':
                y0_lower, y0_upper = profile_likelihood_model0(p_hat, t, protein, p_error, mRNA, plot=False)
                df_bs.loc[idx, 'y0_lower'] = y0_lower
                df_bs.loc[idx, 'y0_upper'] = y0_upper
            elif model_corrected in ['modelA', 'modelC', 'modelD']:
                func = funcs[models.index(model_corrected)]
                m = parameter_numbers[models.index(model_corrected)]
                for m_idx in range(m):
                    p_name = parameter_names[m_idx]
                    c_lower, c_upper = profile_likelihood(func, np.array(p_hat), t, protein, p_error, mRNA, m_idx, plot=False)
                    df_bs.loc[idx, p_name + '_lower'] = c_lower
                    df_bs.loc[idx, p_name + '_upper'] = c_upper
    # output
    if out_file is not None:
        df_bs.to_csv(out_file)
    else:
        return df_bs
