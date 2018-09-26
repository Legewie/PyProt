import datetime
import argparse
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import chi2
from source.m_eval import model_evaluation
from source.fit_model import *
from source.plh import profile_likelihood, profile_likelihood_model0
from source.bootstrap import par_boot


###################################### Input
start = datetime.datetime.now()

# important variables
models = ['model0', 'modelA', 'modelC', 'modelD', 'modelX']
funcs = [np.nan, residuals_modelA, residuals_modelC, residuals_modelD, np.nan]
parameter_numbers = [1, 2, 3, 4, 0]
parameter_names = ['y0', 'degradation', 'production', 'delay']
dw = pd.read_csv('dw_values_005')   # table with durbin watson threshold values

# settings (full or postMZT data, datafile, outfile, bootstrap or explicit stats)
full = True
in_file = './data/drosophila_full_lfq.csv'
out_file = './results/drosophila_full_lfq_out'
test = 'bootstrap'                              # explicit or bootstrap

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

# get the data
data = pd.read_csv(in_file, index_col=0)

# prepare output table
cols = ['Gene', 'n', 'model', 'model_corrected',
            'v0', 'chi2p0', 'dw0', 'r0', 'm0_y0',
            'vA', 'chi2pA', 'dwA', 'rA', 'mA_y0', 'mA_degradation',
            'vC', 'chi2pC', 'dwC', 'rC', 'mC_y0', 'mC_degradation', 'mC_production',
            'vD', 'chi2pD', 'dwD', 'rD', 'mD_y0', 'mD_degradation', 'mD_production', 'mD_delay',
            'mC_in', 'mC_out', 'mC_total',
            'v',
            'y0', 'y0_lower', 'y0_upper',
            'degradation', 'degradation_lower', 'degradation_upper',
            'production', 'production_lower', 'production_upper',
            'delay', 'delay_lower', 'delay_upper']
df_out = pd.DataFrame(np.zeros([len(data), len(cols)]) * np.nan, index=data.ProteinGroup, columns=cols)



####################################################### main loop
count = 0
m_count = 0
for idx, row in data.iterrows():
    count += 1
    print count / float(len(data))
    # get protein name, mRNA expression, protein expression and error
    protein_id = row.ProteinGroup
    mRNA = np.array(row[mt], dtype=float)
    protein = np.array(row[pt], dtype=float)
    p_error = np.array(row[ps], dtype=float)
    n = sum(~np.isnan(protein))

    # fit stationary model
    res0, p0 = fit_stationary(t, protein, p_error, plot=False)
    # fit degradation model
    resA, pA = fit(residuals_modelA, 2, t, protein, p_error, mRNA, samples=5, plot=False)
    # fit production model
    resC, pC = fit(residuals_modelC, 3, t, protein, p_error, mRNA, samples=10, plot=False)
    # fit delayed model
    resD, pD = fit(residuals_modelD, 4, t, protein, p_error, mRNA, samples=25, plot=False)

    # calculate v
    v0 = sum(res0 ** 2)
    vA = sum(resA ** 2)
    vC = sum(resC ** 2)
    vD = sum(resD ** 2)
    # calculate dw
    dw0 = durbin_watson(res0)
    dwA = durbin_watson(resA)
    dwC = durbin_watson(resC)
    dwD = durbin_watson(resD)
    # p-values based on explicit testing
    chi2p0 = 1 - chi2.cdf(v0, n - 1)
    chi2pA = 1 - chi2.cdf(vA, n - 2)
    chi2pC = 1 - chi2.cdf(vC, n - 3)
    chi2pD = 1 - chi2.cdf(vD, n - 4)
    # explicit testing
    # rejection because of chi2 test
    r0_ex = chi2p0 < 0.05
    rA_ex = chi2pA < 0.05
    rC_ex = chi2pC < 0.05
    rD_ex = chi2pD < 0.05
    # get lower limit for dw-test (explicit)
    dL0 = float(dw[(dw.n == n) & (dw.m == 2)].dL)
    dLA = float(dw[(dw.n == n) & (dw.m == 2)].dL)
    dLC = float(dw[(dw.n == n) & (dw.m == 3)].dL)
    if n == 7:
        dLD = float(dw[(dw.n == n) & (dw.m == 3)].dL)
    else:
        dLD = float(dw[(dw.n == n) & (dw.m == 4)].dL)
    # reject based on DW test
    r0_ex = r0_ex or (dw0 < dL0)
    rA_ex = rA_ex or (dwA < dLA)
    rC_ex = rC_ex or (dwC < dLC)
    rD_ex = rD_ex or (dwD < dLD)

    # bootstrap testing
    if test == 'bootstrap':
        # carry out bootstrap
        chi2p0_bs, dw_p0 = par_boot('stationary', None, 1, p0, p_error, t, mRNA, res0)
        chi2pA_bs, dw_pA = par_boot(solve_modelA, residuals_modelA, 2, pA, p_error, t, mRNA, resA)
        chi2pC_bs, dw_pC = par_boot(solve_modelC, residuals_modelC, 3, pC, p_error, t, mRNA, resC)
        chi2pD_bs, dw_pD = par_boot(solve_modelD, residuals_modelD, 4, pD, p_error, t, mRNA, resD)
        # reject based on chi2 test
        r0_bs = chi2p0_bs < 0.05
        rA_bs = chi2pA_bs < 0.05
        rC_bs = chi2pC_bs < 0.05
        rD_bs = chi2pD_bs < 0.05
        # reject based on DW test
        r0_bs = r0_bs or (dw_p0 < 0.05)
        rA_bs = rA_bs or (dw_pA < 0.05)
        rC_bs = rC_bs or (dw_pC < 0.05)
        rD_bs = rD_bs or (dw_pD < 0.05)
        model_bs, v_bs = model_evaluation(r0_bs, rA_bs, rC_bs, rD_bs, v0, vA, vC, vD, verbose=0)

        # carry out correction step (bootstrap)
        if model_bs in ['model0', 'modelA'] and rC_bs == False and pC[2] > 2e-5:
            model_corrected_bs = 'modelC'
            v = vC
        else:
            model_corrected_bs = model_bs
            v = v_bs

    # carry out model selection
    model_ex, v_ex = model_evaluation(r0_ex, rA_ex, rC_ex, rD_ex, v0, vA, vC, vD, verbose=0)

    # carry out correction step (explicit)
    if model_ex in ['model0', 'modelA'] and rC_ex == False and pC[2] > 2e-5:
        model_corrected_ex = 'modelC'
        v = vC
    else:
        model_corrected_ex = model_ex
        v = v_ex

    if test == 'explicit':
        model = model_ex
        model_corrected = model_corrected_ex
    elif test == 'bootstrap':
        model = model_bs
        model_corrected = model_corrected_bs

    # assign final parameters
    p_hat = [p0, pA, pC, pD, np.nan][models.index(model_corrected)]

    # gross production and degradation
    sum_in, sum_out, sum_total = details(pC, protein, p_error, mRNA, t, plot=False)

    # plot all four models for comparison
    plot = False
    if plot:
        print 'protein:, ', protein_id
        #print r0, rA, rC, rD, v0, vA, vC, vD
        print 'final model: ', model_corrected
        print 'p_hat: ', p_hat
        print 'v_hat: ', v
        plt.figure(figsize=(10, 7))
        ax1 = plt.gca()
        # protein
        ax1.errorbar(t, protein, yerr=p_error, marker='o', color='black', label='protein')
        #ax1.plot([0, 20], [p0[0], p0[0]], color='#6289cc', lw=2, label='stationary')
        #ax1.plot(t, solve_modelA(np.log10(pA), t, mRNA), color='#1f4c99', label='degradation')
        ax1.plot(t, solve_modelC(np.log10(pC), t, mRNA), color='#ffa17c', lw=2)#, label='production')
        #print solve_modelC(np.log10(pC), t, mRNA)
        #print num_solve(np.log10(pC), t, mRNA)
        #ax1.plot(t, solve_modelD(np.log10(pD), t, mRNA), color='#cc452f')#, label='delay')
        ax1.set_ylabel('protein [normalized LFQ]', size=15, labelpad=20)
        ax1.legend(loc=3)
        # mRNA
        ax2 = ax1.twinx()
        ax2.plot(t, mRNA, ls='--', color='gray', marker='o', label='mRNA')
        ax2.set_ylabel('mRNA [RPKM]', size=15, labelpad=20)
        ax1.set_xlabel('time [h]', size=15)
        ax2.set_ylim(0, None)
        ax2.legend(loc=2)
        ax1.tick_params(labelsize=13)
        ax2.tick_params(labelsize=13)
        plt.xlim(-1, 21)
        plt.tight_layout()
        plt.grid()
        plt.show()

    # write results to dataframe (except confidence intervals)
    df_out.loc[protein_id, 'Gene']            = row.Gene
    df_out.loc[protein_id, 'n']               = n
    df_out.loc[protein_id, 'model']           = model
    df_out.loc[protein_id, 'model_corrected'] = model_corrected
    df_out.loc[protein_id, 'v0']              = v0
    df_out.loc[protein_id, 'chi2p0']          = chi2p0
    df_out.loc[protein_id, 'dw0']             = dw0
    df_out.loc[protein_id, 'r0_ex']           = r0_ex
    df_out.loc[protein_id, 'm0_y0']           = p0[0]
    df_out.loc[protein_id, 'vA']              = vA
    df_out.loc[protein_id, 'chi2pA']          = chi2pA
    df_out.loc[protein_id, 'dwA']             = dwA
    df_out.loc[protein_id, 'rA_ex']           = rA_ex
    df_out.loc[protein_id, 'mA_y0']           = pA[0]
    df_out.loc[protein_id, 'mA_degradation']  = pA[1]
    df_out.loc[protein_id, 'vC']              = vC
    df_out.loc[protein_id, 'chi2pC']          = chi2pC
    df_out.loc[protein_id, 'dwC']             = dwC
    df_out.loc[protein_id, 'rC_ex']           = rC_ex
    df_out.loc[protein_id, 'mC_y0']           = pC[0]
    df_out.loc[protein_id, 'mC_degradation']  = pC[1]
    df_out.loc[protein_id, 'mC_production']   = pC[2]
    df_out.loc[protein_id, 'vD']              = vD
    df_out.loc[protein_id, 'chi2pD']          = chi2pD
    df_out.loc[protein_id, 'dwD']             = dwD
    df_out.loc[protein_id, 'rD_ex']           = rD_ex
    df_out.loc[protein_id, 'mD_y0']           = pD[0]
    df_out.loc[protein_id, 'mD_degradation']  = pD[1]
    df_out.loc[protein_id, 'mD_production']   = pD[2]
    df_out.loc[protein_id, 'mD_delay']        = pD[3]
    df_out.loc[protein_id, 'mC_in']           = sum_in
    df_out.loc[protein_id, 'mC_out']          = sum_out
    df_out.loc[protein_id, 'mC_total']        = sum_total
    df_out.loc[protein_id, 'v']               = v
    # bootstrap values
    if test == 'bootstrap':
        df_out.loc[protein_id, 'chi2p0_bs']   = chi2p0_bs
        df_out.loc[protein_id, 'dw_p0']       = dw_p0
        df_out.loc[protein_id, 'r0_bs']       = r0_bs
        df_out.loc[protein_id, 'chi2pA_bs']   = chi2pA_bs
        df_out.loc[protein_id, 'dw_pA']       = dw_pA
        df_out.loc[protein_id, 'rA_bs']       = rA_bs
        df_out.loc[protein_id, 'chi2pC_bs']   = chi2pC_bs
        df_out.loc[protein_id, 'dw_pC']       = dw_pC
        df_out.loc[protein_id, 'rC_bs']       = rC_bs
        df_out.loc[protein_id, 'chi2pD_bs']   = chi2pD_bs
        df_out.loc[protein_id, 'dw_pD']       = dw_pD
        df_out.loc[protein_id, 'rD_bs']       = rD_bs
        df_out.loc[protein_id, 'pC_production'] = pC[2]

    # final parameters
    if model_corrected in models[:-1]:
        df_out.loc[protein_id, 'y0']          = p_hat[0]
    if model_corrected in models[1:-1]:
        df_out.loc[protein_id, 'degradation'] = p_hat[1]
    if model_corrected in models[2:-1]:
        df_out.loc[protein_id, 'production']  = p_hat[2]
    if model_corrected in models[3:-1]:
        df_out.loc[protein_id, 'delay']       = p_hat[3]

    # profile likelihood
    profile = False
    if profile:
        if model_corrected == 'model0':
            y0_lower, y0_upper = profile_likelihood_model0(p_hat, t, protein, p_error, mRNA, plot=False)
            df_out.loc[protein_id, 'y0_lower'] = y0_lower
            df_out.loc[protein_id, 'y0_upper'] = y0_upper
        elif model_corrected in ['modelA', 'modelC', 'modelD']:
            func = funcs[models.index(model_corrected)]
            m = parameter_numbers[models.index(model_corrected)]
            for m_idx in range(m):
                p_name = parameter_names[m_idx]
                c_lower, c_upper = profile_likelihood(func, p_hat, t, protein, p_error, mRNA, m_idx, plot=False)
                df_out.loc[protein_id, p_name + '_lower'] = c_lower
                df_out.loc[protein_id, p_name + '_upper'] = c_upper

    # save
    df_out.to_csv(out_file)
    # print remaining time
    loop_end = datetime.datetime.now()
    print 'remaining time:', (loop_end - start) / count * (len(data) - count)

# print total duration
end = datetime.datetime.now()
print end - start
