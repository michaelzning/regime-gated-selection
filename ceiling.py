import numpy as np, json
from scipy.optimize import minimize
S = np.load('streams.npz', allow_pickle=True)
F, y, Z = S['F'], S['y'], S['Z']; d = S['dates'].astype(str); zn = list(S['znames'])
E0 = int(np.where(d == '1957-01')[0][0])
F, y, Z = F[E0:], y[E0:], Z[E0:]; n, N = F.shape; dz = Z.shape[1]
e2 = (y[:, None] - F) ** 2; ew = ((y - F.mean(1)) ** 2).sum()
rho = np.array([np.corrcoef(Z[1:, j], Z[:-1, j])[0, 1] for j in range(dz)])

def sel_ceiling(z, k):
    g = np.searchsorted(np.quantile(z, np.arange(1, k) / k), z, side='right')
    return sum(e2[g == b].sum(0).min() for b in range(k)) / ew

def comb_ceiling(z, k):
    g = np.searchsorted(np.quantile(z, np.arange(1, k) / k), z, side='right'); tot = 0.0
    for b in range(k):
        Fb, yb = F[g == b], y[g == b]
        obj = lambda w: ((yb - Fb @ w) ** 2).sum()
        r = minimize(obj, np.full(N, 1 / N), method='SLSQP', bounds=[(0, 1)] * N,
                     constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
        tot += r.fun
    return tot / ew

def placebos(R, seed):
    rng = np.random.default_rng(seed); X = np.empty((R, n, dz)); u = rng.standard_normal((R, n, dz))
    X[:, 0] = u[:, 0]
    for t in range(1, n): X[:, t] = rho * X[:, t - 1] + np.sqrt(1 - rho ** 2) * u[:, t]
    return X

out = {'rho': rho.tolist()}
P = placebos(2000, 21)
for k in [2, 3, 5, 10]:
    act = np.array([sel_ceiling(Z[:, j], k) for j in range(dz)])
    pl = np.array([[sel_ceiling(P[r, :, j], k) for j in range(dz)] for r in range(2000)])
    p_each = [(pl[:, j] <= act[j]).mean() for j in range(dz)]
    p_fw = (pl.min(1) <= act.min()).mean()                       # family-wise: best of five indicators
    out[f'sel_k{k}'] = dict(actual=act.tolist(), placebo_mean=pl.mean(0).tolist(),
                            p=p_each, best=zn[int(act.argmin())], p_familywise=float(p_fw))
    print(k, 'actual', np.round(act, 4), 'placebo mean', np.round(pl.mean(0), 4), 'p', np.round(p_each, 3), 'FW p', p_fw)
P2 = P[:300]
act = np.array([comb_ceiling(Z[:, j], 3) for j in range(dz)])
pl = np.array([[comb_ceiling(P2[r, :, j], 3) for j in range(dz)] for r in range(300)])
out['comb_k3'] = dict(actual=act.tolist(), placebo_mean=pl.mean(0).tolist(),
                      p=[float((pl[:, j] <= act[j]).mean()) for j in range(dz)],
                      p_familywise=float((pl.min(1) <= act.min()).mean()))
print('comb k3 actual', np.round(act, 4), 'placebo', np.round(pl.mean(0), 4), out['comb_k3']['p'], out['comb_k3']['p_familywise'])
json.dump(out, open('ceiling.json', 'w'), indent=1)
