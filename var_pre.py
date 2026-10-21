"""Fit the indicator VAR(1) on 1947-1988 data only (var_fit_pre.npz), used by the pre-test Monte Carlo."""
import numpy as np
from ceiling2 import fit_var
S = np.load('streams.npz', allow_pickle=True); d = S['dates'].astype(str)
i1 = int(np.where(d == '1989-01')[0][0]); A, c, Sig = fit_var(S['Z'][:i1])
np.savez('var_fit_pre.npz', A=A, c=c, Sig=Sig); print('saved var_fit_pre.npz')
