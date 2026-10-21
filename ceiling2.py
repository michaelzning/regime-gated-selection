"""Hindsight-ceiling diagnostics with placebo indicators that preserve the joint law of the real indicators.
VAR placebo: Gaussian VAR(1) fitted to the indicator vector. Rotation placebo: circular shift of the real
indicator matrix by a random offset of at least 60 months (preserves its exact law, breaks alignment with losses).
p-values use (1 + #{placebo <= actual}) / (B + 1)."""
import numpy as np, json, sys
from scipy.optimize import minimize
S = np.load('streams.npz', allow_pickle=True)
F0, y0, Z0 = S['F'], S['y'], S['Z']; d = S['dates'].astype(str); zn = list(S['znames'])

def fit_var(Z):
    X, Y = Z[:-1], Z[1:]; X1 = np.column_stack([np.ones(len(X)), X])
    B = np.linalg.lstsq(X1, Y, rcond=None)[0]; U = Y - X1 @ B
    return B[1:].T, B[0], np.cov(U.T)

def sim_var(A, c, Sig, n, R, rng):
    L = np.linalg.cholesky(Sig); mu = np.linalg.solve(np.eye(len(c)) - A, c)
    X = np.empty((R, n, len(c))); x = np.tile(mu, (R, 1))
    for t in range(-200, n):
        x = c + x @ A.T + rng.standard_normal((R, len(c))) @ L.T
        if t >= 0: X[:, t] = x
    return X

def rotations(Z, R, rng, minshift=60):
    n = len(Z); k = rng.integers(minshift, n - minshift, R)
    return np.stack([np.roll(Z, s, axis=0) for s in k])

def cell_ids(z, q):
    return np.searchsorted(np.quantile(z, np.arange(1, q) / q), z, side='right')

def sel_ceiling(e2, ewss, z, q):
    g = cell_ids(z, q); return sum(e2[g == b].sum(0).min() for b in range(q)) / ewss

def comb_ceiling(F, y, ewss, z, q):
    g = cell_ids(z, q); tot = 0.0; N = F.shape[1]
    for b in range(q):
        Fb, yb = F[g == b], y[g == b]
        r = minimize(lambda w: ((yb - Fb @ w) ** 2).sum(), np.full(N, 1 / N), method='SLSQP',
                     bounds=[(0, 1)] * N, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
        tot += r.fun
    return tot / ewss

def diagnostics(F, y, Z, PZ, qs=(2, 3, 5, 10), comb=False, PZc=None):
    e2 = (y[:, None] - F) ** 2; ewss = ((y - F.mean(1)) ** 2).sum(); dz = Z.shape[1]; B = PZ.shape[0]
    out = {}
    for q in qs:
        act = np.array([sel_ceiling(e2, ewss, Z[:, j], q) for j in range(dz)])
        pl = np.array([[sel_ceiling(e2, ewss, PZ[r, :, j], q) for j in range(dz)] for r in range(B)])
        out[q] = dict(actual=act.tolist(), placebo_mean=pl.mean(0).tolist(),
                      p=[float((1 + (pl[:, j] <= act[j]).sum()) / (B + 1)) for j in range(dz)],
                      min_actual=float(act.min()), placebo_min_mean=float(pl.min(1).mean()),
                      p_fw=float((1 + (pl.min(1) <= act.min()).sum()) / (B + 1)))
    if comb:
        Bc = PZc.shape[0]
        act = np.array([comb_ceiling(F, y, ewss, Z[:, j], 3) for j in range(dz)])
        pl = np.array([[comb_ceiling(F, y, ewss, PZc[r, :, j], 3) for j in range(dz)] for r in range(Bc)])
        out['comb3'] = dict(actual=act.tolist(), placebo_mean=pl.mean(0).tolist(),
                            p=[float((1 + (pl[:, j] <= act[j]).sum()) / (Bc + 1)) for j in range(dz)],
                            p_fw=float((1 + (pl.min(1) <= act.min()).sum()) / (Bc + 1)))
    return out

if __name__ == '__main__':
    rng = np.random.default_rng(31)
    A, c, Sig = fit_var(Z0)                                   # joint law of the indicators, 1947-2020
    np.savez('var_fit.npz', A=A, c=c, Sig=Sig)
    print('VAR(1) eigenvalues', np.round(np.abs(np.linalg.eigvals(A)), 3))
    res = {}
    for lab, a, b in [('full', '1957-01', None), ('pre1989', '1957-01', '1989-01')]:
        i0 = int(np.where(d == a)[0][0]); i1 = len(d) if b is None else int(np.where(d == b)[0][0])
        F, y, Z = F0[i0:i1], y0[i0:i1], Z0[i0:i1]; n = len(y)
        PV = sim_var(A, c, Sig, n, 2000, rng); PR = rotations(Z, 2000, rng)
        res[lab] = dict(var=diagnostics(F, y, Z, PV, comb=(lab == 'full'), PZc=PV[:300]),
                        rot=diagnostics(F, y, Z, PR, qs=(3,)))
        ewss = ((y - F.mean(1)) ** 2); D2 = ((F - F.mean(1, keepdims=True)) ** 2).max(1)
        res[lab]['dispersion_bound'] = float(D2.mean() / ewss.mean())
        v = res[lab]
        print(lab, 'disp bound %.4f' % v['dispersion_bound'])
        for q in (2, 3, 5, 10):
            w = v['var'][q]; print('  VAR q=%d act' % q, np.round(w['actual'], 4), 'p', np.round(w['p'], 3),
                                   'min %.4f plmin %.4f FW %.3f' % (w['min_actual'], w['placebo_min_mean'], w['p_fw']))
        w = v['rot'][3]; print('  ROT q=3 p', np.round(w['p'], 3), 'FW %.3f' % w['p_fw'], 'plmin %.4f' % w['placebo_min_mean'])
        if 'comb3' in v['var']:
            w = v['var']['comb3']; print('  comb3', np.round(w['actual'], 4), 'p', np.round(w['p'], 3), 'FW %.3f' % w['p_fw'])
    json.dump(res, open('ceiling2.json', 'w'), indent=1)
