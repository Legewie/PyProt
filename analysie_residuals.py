import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm, pearsonr
import pylab
import scipy.stats as stats

pt = ['p_00h', 'p_01h', 'p_02h', 'p_03h', 'p_04h', 'p_05h', 'p_06h', 'p_08h', 'p_10h', 'p_12h', 'p_14h', 'p_16h',
      'p_18h', 'p_20h']

# get data (for masking residuals)
data = pd.read_csv('~/ofc/projects/nuria/0.0_data/matlab_input_mean_em3.csv', index_col=0)
data.set_index('ProteinGroup', inplace=True)
mask = ((data[pt] > 0) * 1).replace(0, np.nan)

# get results (to subselect non-rejected models)
results = pd.read_csv('./results/result_full_5_10_25_chi2dwBS_LRT_corrected_1', index_col=0)

# get dataframe with residuals
df0 = pd.read_csv('residuals_lfq_r0')
dfA = pd.read_csv('residuals_lfq_rA')
dfC = pd.read_csv('residuals_lfq_rC')
dfD = pd.read_csv('residuals_lfq_rD')

# slice out nan from upper part (fixme: take into account next time)
df0 = df0.iloc[3761:]
dfA = dfA.iloc[3761:]
dfC = dfC.iloc[3761:]
dfD = dfD.iloc[3761:]

# set new index
df0.index = mask.index
dfA.index = mask.index
dfC.index = mask.index
dfD.index = mask.index

dfA = dfA * mask
dfC = dfC * mask
dfD = dfD * mask


# testing by shapiro wilk for normality of residuals (NOTE: highly inaccurate for n < 14 datapoints)
count0 = 0
df0_sub = df0.ix[results[~results.r0_bs].index]
for idx, row in df0_sub.iterrows():
    x = np.array(row[pt], dtype=float)
    W, p = shapiro(x[~np.isnan(x)])
    if p < 0.05:
        count0 += 1

countA = 0
dfA_sub = dfA.ix[results[~results.rA_bs].index]
for idx, row in dfA_sub.iterrows():
    x = np.array(row[pt], dtype=float)
    W, p = shapiro(x[~np.isnan(x)])
    if p < 0.05:
        countA += 1

countC = 0
dfC_sub = dfC.ix[results[~results.rC_bs].index]
for idx, row in dfC_sub.iterrows():
    x = np.array(row[pt], dtype=float)
    W, p = shapiro(x[~np.isnan(x)])
    if p < 0.05:
        countC += 1

countD = 0
dfD_sub = dfD.ix[results[~results.rD_bs].index]
for idx, row in dfD_sub.iterrows():
    x = np.array(row[pt], dtype=float)
    W, p = shapiro(x[~np.isnan(x)])
    if p < 0.05:
        countD += 1


model = 'delay'
df = dfD
r = 'rD_bs'
df_sub = df.ix[results[~results[r]].index]

for t in pt[:]:
    print t
    # get residuals, clean, normalize and sort
    x = np.array(df_sub[t])
    x = x[~np.isnan(x)]
    x = (x - x.mean()) / x.std()
    x.sort()
    # histogram of residuals
    plt.hist(x, bins=int(np.sqrt(len(x))), normed=True)
    pdf = norm.pdf(x, 0, 1)
    plt.plot(x, pdf, 'k', linewidth=2, label='fit')
    plt.grid()
    #plt.show()
    plt.savefig('./residuals_analysis/model_' + model + '_' + t + '_hist.png')
    plt.clf()
    # qq-plot
    y = norm.ppf(np.linspace(1e-3, 1 - 1e-3, len(x)))
    plt.scatter(y, x, alpha=0.5)
    print pearsonr(x, y)
    plt.plot([-4, 4], [-4, 4], color='red')
    plt.xlabel('theoretical quantiles')
    plt.ylabel('residuals')
    plt.grid()
    #plt.show()
    plt.savefig('./residuals_analysis/model_' + model + '_' + t + '_qq.png')
    plt.clf()



x = np.reshape(np.array(df_sub[pt]), (1, len(df_sub)*14))[0]
x = x[~np.isnan(x)]
x = (x - x.mean()) / x.std()
x.sort()
# histogram of residuals
plt.hist(x, bins=int(np.sqrt(len(x))), normed=True)
pdf = norm.pdf(x, 0, 1)
plt.plot(x, pdf, 'k', linewidth=1, label='normal distribution', color='red')
plt.xlabel('normalized residuals')
plt.ylabel('frequency')
plt.legend(loc='best')
plt.grid()
plt.show()
# qq-plot
y = norm.ppf(np.linspace(1e-3, 1 - 1e-3, len(x)))
plt.scatter(y, x, alpha=0.5)
print pearsonr(x, y)
plt.plot([-4, 4], [-4, 4], color='red')
plt.xlabel('theoretical quantiles')
plt.ylabel('normalized residuals')
plt.grid()
plt.show()