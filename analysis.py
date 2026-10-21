import numpy as np, json
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

S = np.load('streams.npz', allow_pickle=True)
F, y, Z, rf = S['F'], S['y'], S['Z'], S['rf']
dates = S['dates'].astype(str); names = list(S['names']); zn = list(S['znames'])
T, N = F.shape
ix = lambda d: int(np.where(dates == d)[0][0])
E0 = ix('1957-01')
L = lambda f: np.load(f'{f}.npz')['W']
Wd = {'DMA': L('W_dma'), 'MS no z': L('W_msnoz'),
      'gate0': L('W_gate_pen0'), 'gate0.5': L('W_gate_p0.5'), 'gate2': L('W_gate'), 'gate8': L('W_gate_p8'),
      'filt0': L('W_filt_pen0'), 'filt0.5': L('W_filt_p0.5'), 'filt2': L('W_filt'), 'filt8': L('W_filt_p8'),
      'mlp0': L('W_mlp_p0'), 'mlp2': L('W_mlp_mlp2')}
e2 = (y[:, None] - F) ** 2

def nw_var(x, L=6):
    x = x - x.mean(); n = len(x); v = (x ** 2).mean()
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * (x[l:] * x[:-l]).mean()
    return v / n

def metrics(f, sl):
    yy, ff, pm, ew = y[sl], f[sl], F[sl, 0], F[sl].mean(1)
    e, ep, ee = yy - ff, yy - pm, yy - ew
    r2 = 1 - (e ** 2).sum() / (ep ** 2).sum()
    cw = ep ** 2 - (e ** 2 - (pm - ff) ** 2)
    p_cw = float(1 - stats.norm.cdf(cw.mean() / np.sqrt(nw_var(cw)))) if np.std(cw) > 0 else None
    d = ee ** 2 - e ** 2
    p_dm = float(2 * (1 - stats.norm.cdf(abs(d.mean() / np.sqrt(nw_var(d)))))) if np.std(d) > 0 else None
    idx = np.arange(T)[sl]; var = np.array([y[t - 120:t].var() for t in idx])
    def cer(fv):
        w = np.clip(fv / (3 * var), 0, 1.5); rp = rf[sl] + w * yy
        return 1200 * (rp.mean() - 1.5 * rp.var())
    return dict(R2os=100 * r2, p_cw=p_cw, mse_ew=(e ** 2).mean() / (ee ** 2).mean(), p_dm=p_dm,
                dCER=cer(ff) - cer(pm))

periods = {'1957--2020': slice(E0, T), '1957--1989': slice(E0, ix('1990-01')),
           '1990--2020': slice(ix('1990-01'), T), '2010--2020': slice(ix('2010-01'), T)}
fc = {'Prevailing mean': F[:, 0], 'Equal weights': F.mean(1)}
for k in range(1, N): fc[names[k]] = F[:, k]
best = np.full(T, np.nan)
for t in range(E0, T): best[t] = F[t, np.argmin(e2[t - 60:t].mean(0))]
fc['Best trailing 60m'] = best
for k, W in Wd.items(): fc[k] = (W * F).sum(1)
fc['gate0 argmax'] = F[np.arange(T), np.argmax(np.nan_to_num(Wd['gate0']), 1)]
fc['gate2 argmax'] = F[np.arange(T), np.argmax(np.nan_to_num(Wd['gate2']), 1)]
m = fc['filt2']
fc['filt2 nearest'] = F[np.arange(T), np.argmin(np.nan_to_num(np.abs(F - m[:, None]), nan=9.0), 1)]
R = {p: {k: metrics(f, sl) for k, f in fc.items()} for p, sl in periods.items()}

sl = slice(E0, T)
def wstats(W):
    w = W[sl]; ent = -(w * np.log(np.clip(w, 1e-300, 1))).sum(1).mean()
    sw = (np.argmax(w, 1)[1:] != np.argmax(w, 1)[:-1]).mean() * 12
    return dict(entropy=float(ent / np.log(N)), maxw=float(w.max(1).mean()), switches_py=float(sw),
                meanw=[float(v) for v in w.mean(0)])
Wst = {k: wstats(W) for k, W in Wd.items()}

# Giacomini-White CPA tests
def cpa(d, X):
    h = X * d[:, None]; n = len(d); hb = h.mean(0); hc = h - hb; Om = hc.T @ hc / n
    for l in range(1, 7):
        G = hc[l:].T @ hc[:-l] / n; Om += (1 - l / 7) * (G + G.T)
    st = n * hb @ np.linalg.solve(Om, hb); return float(1 - stats.chi2.cdf(st, X.shape[1]))
one = np.ones((T - E0, 1)); Xall = np.column_stack([one, Z[sl]])
CPA = {}
for k in range(1, N):
    d = e2[sl, 0] - e2[sl, k]
    CPA[names[k]] = {'all': cpa(d, Xall), **{zn[j]: cpa(d, np.column_stack([one, Z[sl, j]])) for j in range(Z.shape[1])}}
ewe = (y - F.mean(1)) ** 2
CPA_ew = {names[k]: cpa(ewe[sl] - e2[sl, k], Xall) for k in range(N)}

# Hindsight upper bound: best expert per tercile of each indicator, chosen with full-sample hindsight
HB = {}
for j, zname in enumerate(zn):
    zz = Z[sl, j]; q = np.quantile(zz, [1 / 3, 2 / 3]); g = np.digitize(zz, q)
    sse = 0.0; pick = []
    for b in range(3):
        mk = g == b; k = int(np.argmin(e2[sl][mk].sum(0))); pick.append(names[k]); sse += e2[sl][mk, k].sum()
    HB[zname] = dict(ratio=float(sse / ewe[sl].sum()), picks=pick)
dec = np.array([int(d[:3]) for d in dates[sl]]); sse = 0.0
for dd in np.unique(dec):
    mk = dec == dd; sse += e2[sl][mk].sum(0).min()
HB['decade'] = dict(ratio=float(sse / ewe[sl].sum()))
HB['best single'] = dict(ratio=float(e2[sl].sum(0).min() / ewe[sl].sum()))

# Expert losses by volatility tercile (relative to PM)
lv = Z[sl, zn.index('lsvar')]; q = np.quantile(lv, [1 / 3, 2 / 3]); g = np.digitize(lv, q)
VOL = {lab: [float(100 * (1 - e2[sl][g == b, k].sum() / e2[sl][g == b, 0].sum())) for k in range(N)]
       for b, lab in enumerate(['low', 'mid', 'high'])}

json.dump(dict(R=R, W=Wst, CPA=CPA, CPA_ew=CPA_ew, HB=HB, VOL=VOL), open('paper_results.json', 'w'), indent=1)

# Figures
import matplotlib.dates as mdates, datetime as dt
dd = [dt.datetime.strptime(x, '%Y-%m') for x in dates[sl]]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for k, lab, st in [('Equal weights', 'Equal weights', '-'), ('gate2', 'Static gate, shrunk', '--'),
                   ('filt2', 'Filtered selector, shrunk', '-.'), ('gate0', 'Static gate, unshrunk', ':'),
                   ('mlp0', 'MLP gate, unshrunk', (0, (5, 2, 1, 2)))]:
    c = np.cumsum((y[sl] - F[sl, 0]) ** 2 - (y[sl] - fc[k][sl]) ** 2) * 100
    ax.plot(dd, c, linestyle=st, lw=1.2, label=lab, color='k' if k == 'Equal weights' else None)
ax.axhline(0, color='grey', lw=0.6); ax.set_ylabel('Cumulative SSE(PM) $-$ SSE(model), $\\times 100$')
ax.legend(fontsize=7, frameon=False, ncol=2); fig.tight_layout(); fig.savefig('fig_cumsse.pdf')

fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.9), sharey=True)
for ax, k, ttl in [(axs[0], 'gate0', 'Unshrunk static gate'), (axs[1], 'gate2', 'Shrunk static gate')]:
    ax.stackplot(dd, Wd[k][sl].T, labels=names, colors=['0.85', '0.65', '0.45', '0.3', '0.15'])
    ax.set_title(ttl, fontsize=9); ax.set_ylim(0, 1)
axs[0].set_ylabel('Weight'); axs[1].legend(fontsize=6, loc='upper right', frameon=True)
fig.tight_layout(); fig.savefig('fig_weights.pdf')
print('ok')
