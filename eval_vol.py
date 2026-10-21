import numpy as np, json
from scipy import stats
from ceiling2 import fit_var, sim_var, rotations, diagnostics
S = np.load('streams_vol.npz', allow_pickle=True)
F, y, Z = S['F'], S['y'], S['Z']; d = S['dates'].astype(str); names = list(S['names'])
T, N = F.shape; ix = lambda s: int(np.where(d == s)[0][0]); E0 = ix('1957-01')
def nwv(x, L=6):
    x = x - x.mean(); n = len(x); v = (x**2).mean()
    for l in range(1, L+1): v += 2*(1-l/(L+1))*(x[l:]*x[:-l]).mean()
    return v/n
def met(f, sl, bench):
    e2 = (y[sl]-f[sl])**2; ew2 = (y[sl]-F[sl].mean(1))**2; b2 = (y[sl]-bench[sl])**2
    dd = ew2 - e2; db = b2 - e2
    p = lambda x: float(2*(1-stats.norm.cdf(abs(x.mean()/np.sqrt(nwv(x)))))) if x.std() > 0 else None
    return dict(mse_ew=e2.mean()/ew2.mean(), p_ew=p(dd), mse_best=e2.mean()/b2.mean(), p_best=p(db))
e2 = (y[:, None]-F)**2
fc = {'EW': F.mean(1)}
best = np.full(T, np.nan)
for t in range(E0, T): best[t] = F[t, np.argmin(e2[t-60:t].mean(0))]
fc['best60'] = best
for k in ['gate_vol_p2', 'gate_vol_p0', 'filt_vol_p2']:
    W = np.load(f'W_{k}.npz')['W']; fc[k] = (W*F).sum(1)
    w = W[E0:]; print(k, 'mean w', np.round(w.mean(0), 2), 'maxw %.3f' % w.max(1).mean(), 'top chg/yr %.1f' % ((np.argmax(w,1)[1:]!=np.argmax(w,1)[:-1]).mean()*12))
    if k == 'gate_vol_p2':
        fc['gate_vol_p2_nearest'] = F[np.arange(T), np.argmin(np.nan_to_num(np.abs(F - fc[k][:, None]), nan=99), 1)]
for k in range(N): fc[names[k]] = F[:, k]
HAR = F[:, names.index('HAR')]
per = {'1957-2020': slice(E0, T), '1957-1989': slice(E0, ix('1990-01')), '1990-2020': slice(ix('1990-01'), T), '2010-2020': slice(ix('2010-01'), T)}
R = {p: {k: met(f, sl, HAR) for k, f in fc.items()} for p, sl in per.items()}
for k in fc:
    m = R['1957-2020'][k]
    print(f"{k:22s} MSE/EW {m['mse_ew']:.4f} p {m['p_ew']!s:.5}  MSE/HAR {m['mse_best']:.4f} p {m['p_best']!s:.5} | sub MSE/EW", [round(R[p][k]['mse_ew'], 3) for p in list(per)[1:]], 'MSE/HAR', [round(R[p][k]['mse_best'], 3) for p in list(per)[1:]])
v = np.load('var_fit.npz'); rng = np.random.default_rng(93); out = {}
i1 = ix('1989-01')
for lab, a, b, A in [('full', E0, T, (v['A'], v['c'], v['Sig'])), ('pre1989', E0, i1, fit_var(Z[:i1]))]:
    Fs, ys, Zs = F[a:b], y[a:b], Z[a:b]
    PV = sim_var(*A, len(ys), 2000, rng); PR = rotations(Zs, 2000, rng)
    r = dict(var=diagnostics(Fs, ys, Zs, PV), rot=diagnostics(Fs, ys, Zs, PR, qs=(3,)))
    r['disp'] = float(((Fs-Fs.mean(1, keepdims=True))**2).max(1).mean()/((ys-Fs.mean(1))**2).mean())
    out[lab] = r
    print(lab, 'disp %.3f' % r['disp'], {q: (round(r['var'][q]['min_actual'], 4), round(r['var'][q]['placebo_min_mean'], 4), round(r['var'][q]['p_fw'], 3)) for q in (2, 3, 5, 10)}, 'rot3 FW %.3f' % r['rot'][3]['p_fw'])
    print('   q3 actual', np.round(r['var'][3]['actual'], 4), 'p', np.round(r['var'][3]['p'], 3), 'placebo mean', np.round(r['var'][3]['placebo_mean'], 4))
json.dump(dict(R=R, diag=out), open('vol_results.json', 'w'), indent=1, default=float)
