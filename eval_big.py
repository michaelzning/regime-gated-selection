import numpy as np, json
from scipy import stats
from ceiling2 import fit_var, sim_var, rotations, diagnostics
S = np.load('streams_big.npz', allow_pickle=True)
F, y, Z, rf = S['F'], S['y'], S['Z'], S['rf']; d = S['dates'].astype(str); names = list(S['names'])
T, N = F.shape; ix = lambda s: int(np.where(d == s)[0][0]); E0 = ix('1957-01')
def nwv(x, L=6):
    x = x - x.mean(); n = len(x); v = (x**2).mean()
    for l in range(1, L+1): v += 2*(1-l/(L+1))*(x[l:]*x[:-l]).mean()
    return v/n
def met(f, sl):
    yy, ff, pm, ew = y[sl], f[sl], F[sl, 0], F[sl].mean(1)
    e, ep, ee = yy-ff, yy-pm, yy-ew
    cw = ep**2 - (e**2 - (pm-ff)**2); dd = ee**2 - e**2
    idx = np.arange(T)[sl]; var = np.array([y[t-120:t].var() for t in idx])
    def cer(fv):
        w = np.clip(fv/(3*var), 0, 1.5); rp = rf[sl] + w*yy; return 1200*(rp.mean()-1.5*rp.var())
    return dict(R2=100*(1-(e**2).sum()/(ep**2).sum()),
                pcw=float(1-stats.norm.cdf(cw.mean()/np.sqrt(nwv(cw)))) if cw.std()>0 else None,
                mse_ew=(e**2).mean()/(ee**2).mean(),
                pdm=float(2*(1-stats.norm.cdf(abs(dd.mean()/np.sqrt(nwv(dd)))))) if dd.std()>0 else None,
                dcer=cer(ff)-cer(pm))
e2 = (y[:, None]-F)**2
fc = {'EW': F.mean(1)}
best = np.full(T, np.nan)
for t in range(E0, T): best[t] = F[t, np.argmin(e2[t-60:t].mean(0))]
fc['best60'] = best
# DMA
from jax.scipy.special import gammaln
def logt(e, sc, nu=5.0):
    sc = sc*np.sqrt((nu-2)/nu)
    return float(gammaln((nu+1)/2)-gammaln(nu/2)) - 0.5*np.log(nu*np.pi) - np.log(sc) - (nu+1)/2*np.log1p((e/sc)**2/nu)
lt = logt(y[:, None]-F, S['s'][:, None]); post = np.full(N, 1/N); Wd = np.zeros((T, N))
for t in range(T):
    pr = post**0.99; pr /= pr.sum(); Wd[t] = pr; p = pr*np.exp(lt[t]-lt[t].max()); post = p/p.sum()
fc['DMA'] = (Wd*F).sum(1)
for k in ['gate_big_p2', 'gate_big_p0', 'filt_big_p2']:
    W = np.load(f'W_{k}.npz')['W']; fc[k] = (W*F).sum(1)
    w = W[E0:]; print(k, 'maxw %.3f' % w.max(1).mean(), 'entropy %.3f' % (-(w*np.log(np.clip(w,1e-300,1))).sum(1).mean()/np.log(N)),
                    'top changes/yr %.1f' % ((np.argmax(w,1)[1:]!=np.argmax(w,1)[:-1]).mean()*12))
for k in range(1, N): fc[names[k]] = F[:, k]
per = {'1957-2020': slice(E0, T), '1957-1989': slice(E0, ix('1990-01')), '1990-2020': slice(ix('1990-01'), T), '2010-2020': slice(ix('2010-01'), T)}
R = {p: {k: met(f, sl) for k, f in fc.items()} for p, sl in per.items()}
for k in fc:
    m = R['1957-2020'][k]
    print(f"{k:14s} R2 {m['R2']:6.2f} pcw {m['pcw']!s:.5} mse/ew {m['mse_ew']:.4f} pdm {m['pdm']!s:.5} dcer {m['dcer']:5.2f} | sub", [round(R[p][k]['R2'],2) for p in list(per)[1:]])
ind = [R['1957-2020'][n]['R2'] for n in names[1:]]
print('individual experts R2 range', round(min(ind),2), round(max(ind),2), 'mse/ew range', round(min(R['1957-2020'][n]['mse_ew'] for n in names[1:]),4), round(max(R['1957-2020'][n]['mse_ew'] for n in names[1:]),4))
# diagnostics
v = np.load('var_fit.npz'); rng = np.random.default_rng(91); out = {}
i1 = ix('1989-01')
for lab, a, b, A in [('full', E0, T, (v['A'], v['c'], v['Sig'])), ('pre1989', E0, i1, fit_var(Z[:i1]))]:
    Fs, ys, Zs = F[a:b], y[a:b], Z[a:b]
    PV = sim_var(*A, len(ys), 2000, rng); PR = rotations(Zs, 2000, rng)
    r = dict(var=diagnostics(Fs, ys, Zs, PV), rot=diagnostics(Fs, ys, Zs, PR, qs=(3,)))
    r['disp'] = float(((Fs-Fs.mean(1, keepdims=True))**2).max(1).mean()/((ys-Fs.mean(1))**2).mean())
    out[lab] = r
    print(lab, 'disp %.4f' % r['disp'], {q: (round(r['var'][q]['min_actual'],4), round(r['var'][q]['placebo_min_mean'],4), round(r['var'][q]['p_fw'],3)) for q in (2,3,5,10)},
          'rot3 FW %.3f' % r['rot'][3]['p_fw'])
    print('   q3 actual', np.round(r['var'][3]['actual'], 4), 'p', np.round(r['var'][3]['p'], 3))
json.dump(dict(R=R, diag=out), open('big_results.json', 'w'), indent=1, default=float)
