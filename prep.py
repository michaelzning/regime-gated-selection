"""Build monthly data (Welch-Goyal 1926:12-2020:12) and pseudo-out-of-sample expert forecasts."""
import numpy as np, pandas as pd

d = pd.read_csv('PredictorData1926-2020.csv')
d['date'] = pd.to_datetime(d.yyyymm.astype(int).astype(str), format='%Y%m')
d = d.set_index('date')
r = d.CRSP_SPvw - d.Rfree                      # excess return realized over month t

# predictors observed at end of month t (used to forecast t+1)
X = pd.DataFrame(index=d.index)
X['value'] = np.log(d.D12) - np.log(d.Index)               # log dividend-price ratio
X['carry'] = d.D12 / d.Index - d.tbl                        # dividend yield minus T-bill yield
X['mom'] = r.rolling(12).sum()                              # past 12-month excess return
X['rev'] = r                                                # past 1-month excess return
# regime indicators observed at end of month t
Z = pd.DataFrame(index=d.index)
Z['lsvar'] = np.log(d.svar)                                 # log realized variance (daily returns)
Z['dfy'] = d.BAA - d.AAA                                    # default spread
Z['tms'] = d.lty - d.tbl                                    # term spread
Z['infl'] = d.infl.shift(1)                                 # CPI with one-month release lag
Z['ntis'] = d.ntis                                          # net equity issuance

# align: row t holds target r_t and information dated t-1
Xl, Zl = X.shift(1), Z.shift(1)
df = pd.concat([r.rename('r'), Xl, Zl.add_prefix('z_')], axis=1).dropna()
dates = df.index
T = len(df)
names = ['PM', 'Value', 'Carry', 'Momentum', 'Reversal']
xcols = ['value', 'carry', 'mom', 'rev']
t0 = np.searchsorted(dates, pd.Timestamp('1947-01-01'))     # first OOS expert forecast

F = np.full((T, 5), np.nan)
y = df.r.values
for t in range(t0, T):
    F[t, 0] = y[:t].mean()                                  # prevailing mean
    for k, c in enumerate(xcols):
        xx = df[c].values
        A = np.column_stack([np.ones(t), xx[:t]])
        b = np.linalg.lstsq(A, y[:t], rcond=None)[0]        # expanding-window OLS through t-1
        F[t, k + 1] = b[0] + b[1] * xx[t]

# regime features standardized with expanding moments (no look-ahead)
Zraw = df[[f'z_{c}' for c in Z.columns]].values
Zs = np.full_like(Zraw, np.nan)
for t in range(t0, T):
    mu, sd = Zraw[:t].mean(0), Zraw[:t].std(0)
    Zs[t] = np.clip((Zraw[t] - mu) / sd, -4, 4)

# common time-varying scale: EWMA of squared prevailing-mean errors, known at t-1
lam = 0.94
s2 = np.full(T, np.nan); s2[t0] = y[:t0].var()
for t in range(t0 + 1, T):
    s2[t] = lam * s2[t - 1] + (1 - lam) * (y[t - 1] - F[t - 1, 0]) ** 2
np.savez('streams.npz', F=F[t0:], y=y[t0:], Z=Zs[t0:], s=np.sqrt(s2[t0:]),
         dates=dates[t0:].strftime('%Y-%m').values.astype(str), names=np.array(names),
         znames=np.array(list(Z.columns)), rf=d.Rfree.reindex(dates).values[t0:])
print('streams', dates[t0].date(), dates[-1].date(), T - t0, 'months')
e = y[t0:, None] - F[t0:]
print('OOS MSE x1e4 by expert:', dict(zip(names, np.round((e ** 2).mean(0) * 1e4, 3))))
