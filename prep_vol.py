"""Second application: forecasting monthly log realized variance of the S&P 500 (Welch-Goyal svar).
Target y_t = log svar_t. Experts use information dated t-1 and are estimated by expanding-window OLS;
forecasts are out of sample from 1947:01. Regime indicators and their standardization match prep.py."""
import numpy as np, pandas as pd

d = pd.read_csv('PredictorData1926-2020.csv')
d['date'] = pd.to_datetime(d.yyyymm.astype(int).astype(str), format='%Y%m'); d = d.set_index('date')
r = d.CRSP_SPvw - d.Rfree
lv = np.log(d.svar)

X = pd.DataFrame(index=d.index)                       # regressors dated t (used for t+1)
X['lag1'] = lv
X['avg3'] = lv.rolling(3).mean()
X['avg12'] = lv.rolling(12).mean()
X['dfy'] = d.BAA - d.AAA
X['ret'] = r
X['negret'] = np.minimum(r, 0)
Z = pd.DataFrame(index=d.index)
Z['lsvar'] = lv; Z['dfy'] = d.BAA - d.AAA; Z['tms'] = d.lty - d.tbl; Z['infl'] = d.infl.shift(1); Z['ntis'] = d.ntis

df = pd.concat([lv.rename('y'), X.shift(1), Z.shift(1).add_prefix('z_')], axis=1).dropna()
dates = df.index; T = len(df); y = df.y.values
t0 = np.searchsorted(dates, pd.Timestamp('1947-01-01'))
specs = {'PM': None, 'RW': 'rw', 'AR1': ['lag1'], 'HAR': ['lag1', 'avg3', 'avg12'], 'Avg12': ['avg12'],
         'AR1+DFY': ['lag1', 'dfy'], 'AR1+Lev': ['lag1', 'ret', 'negret']}
names = list(specs)
F = np.full((T, len(names)), np.nan)
for t in range(t0, T):
    for k, (nm, sp) in enumerate(specs.items()):
        if sp is None: F[t, k] = y[:t].mean()
        elif sp == 'rw': F[t, k] = df.lag1.values[t]
        else:
            A = np.column_stack([np.ones(t), df[sp].values[:t]])
            b = np.linalg.lstsq(A, y[:t], rcond=None)[0]
            F[t, k] = b[0] + df[sp].values[t] @ b[1:]
Zraw = df[[f'z_{c}' for c in Z.columns]].values; Zs = np.full_like(Zraw, np.nan)
for t in range(t0, T):
    mu, sd = Zraw[:t].mean(0), Zraw[:t].std(0); Zs[t] = np.clip((Zraw[t] - mu) / sd, -4, 4)
# common scale: EWMA of squared AR(1) errors known at t-1 (the prevailing mean is a poor scale for a persistent target)
lam = 0.94; s2 = np.full(T, np.nan); s2[t0] = np.var(np.diff(y[:t0]))
k1 = names.index('AR1')
for t in range(t0 + 1, T): s2[t] = lam * s2[t - 1] + (1 - lam) * (y[t - 1] - F[t - 1, k1]) ** 2
np.savez('streams_vol.npz', F=F[t0:], y=y[t0:], Z=Zs[t0:], s=np.sqrt(s2[t0:]),
         dates=dates[t0:].strftime('%Y-%m').values.astype(str), names=np.array(names),
         znames=np.array(list(Z.columns)), rf=d.Rfree.reindex(dates).values[t0:])
e = y[t0:, None] - F[t0:]; E = list(dates[t0:].strftime('%Y-%m')).index('1957-01')
mse = (e[E:] ** 2).mean(0); ew = ((y[t0:] - F[t0:].mean(1))[E:] ** 2).mean()
print(dates[t0].date(), dates[-1].date(), T - t0)
print('MSE/EW 1957-2020:', dict(zip(names, np.round(mse / ew, 4))))
print('R2 vs PM:', dict(zip(names, np.round(100 * (1 - mse / mse[0]), 1))))
