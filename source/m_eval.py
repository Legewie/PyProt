import numpy as np
import pandas as pd

def model_evaluation(r0, rA, rC, rD, v0, vA, vC, vD, verbose=0):
    # assign the base model
    if r0 == 0:
        base = 'model0'
    elif rA == 0:
        base = 'modelA'
    elif rC == 0:
        base = 'modelC'
    elif rD == 0:
        return 'modelD', vD
    else:
        return 'modelX', np.nan
    if verbose > 0: print base, 'is most simple possible model'
    # first round of testing: test model0 against others
    if base == 'model0':
        # is model A possible and also preffered ?
        if (rA == 0) and (v0 - vA > 3.841):
            if verbose > 0: print 'modelA is possible and significantly better'
            base = 'modelA'
        # is model C possible and also preffered ?
        elif (rC == 0) and (v0 - vC > 5.991):
            if verbose > 0: print 'modelC is possible and significantly better'
            base = 'modelC'
        # is model D possible and also preffered ?
        elif (rD == 0) and (v0 - vD > 7.851):
            if verbose > 0: print 'modelD is possible and significantly better'
            return 'modelD', vD
        else:
            if verbose > 0: print 'No model other than 0 is possible or significantly better'
            return 'model0', v0
    # second round of testing: test modelA against C and D
    if base == 'modelA':
        # is model C possible and also preffered ?
        if (rC == 0) and (vA - vC > 3.841):
            if verbose > 0: print 'modelC is possible and significantly better'
            base = 'modelC'
        # is model D possible and also preffered ?
        elif (rD == 0) and (vA - vD > 5.991):
            if verbose > 0: print 'modelD is possible and significantly better'
            return 'modelD', vD
        else:
            return 'modelA', vA
    # third/final round of testing: test modelC against D
    if base == 'modelC':
        # is model D possible and also preffered ?
        if (rD == 0) and (vC - vD > 3.841):
            if verbose > 0: print 'modelD is possible and significantly better'
            return 'modelD', vD
        else:
            return 'modelC', vC


'''
# BIC testing
r_vec = 1 - np.array([r0, rA, rC, rD]).astype(float)
v_vec = np.array([v0, vA, vC, vD])

if sum(r_vec) == 0:
    model_BIC = 'modelX'
    v = np.nan
else:
    r_vec[r_vec == 0] = np.inf
    BIC = np.array([v0 + np.log(n) * 1, vA + np.log(n) * 2, vC + np.log(n) * 3, vD + np.log(n) * 4])
    model_BIC = models[(BIC * r_vec).argmin()]
    v = v_vec[(BIC * r_vec).argmin()]
model = model_BIC
'''
