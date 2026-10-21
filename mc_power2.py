"""Semi-synthetic Monte Carlo on the real expert streams and indicators.
True conditional mean m_t = sum_k softmax(beta * z_{t-1})_k f_{k,t}: expert k is favored when indicator k is high.
y*_t = m_t + block-permuted equal-weight residuals. beta = 0 is the null under which equal weights are optimal."""
import sys, json, time, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.optimize import minimize
from scipy import stats
jax.config.update('jax_enable_x64', True)
S = np.load('streams.npz', allow_pickle=True)
F, y, Z, s = S['F'], S['y'], S['Z'], S['s']; d = S['dates'].astype(str)
T, N = F.shape; dz = Z.shape[1]
E0 = int(np.where(d == '1957-01')[0][0]); SPLIT = int(np.where(d == '1989-01')[0][0])
resid = y - F.mean(1)
rho = np.array([np.corrcoef(Z[E0 + 1:, j], Z[E0:-1, j])[0, 1] for j in range(dz)])

def logt(e, sc, nu=5.0):
    sc = sc * jnp.sqrt((nu - 2) / nu)
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi) - jnp.log(sc) - (nu + 1) / 2 * jnp.log1p((e / sc) ** 2 / nu)

def weights(th, Zj):
    a = jnp.concatenate([jnp.zeros(1), th[:N - 1]])
    G = jnp.concatenate([jnp.zeros((1, dz)), th[N - 1:N - 1 + (N - 1) * dz].reshape(N - 1, dz)])
    return jax.nn.softmax(a[None, :] + Zj @ G.T, axis=1)

def nll(th, Fj, yj, Zj, sj, pen):
    W = weights(th, Zj); c = jnp.exp(th[-1])
    ld = logt(yj[:, None] - Fj, c * sj[:, None])
    lm = jax.scipy.special.logsumexp(jnp.log(W) + ld, axis=1)
    return -lm.sum() + pen * jnp.sum(th[:-1] ** 2)
vg = jax.jit(jax.value_and_grad(nll))

def fit_gate(ys, pen):
    tr = slice(0, SPLIT)
    args = (jnp.asarray(F[tr]), jnp.asarray(ys[tr]), jnp.asarray(Z[tr]), jnp.asarray(s[tr]), pen)
    f = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x), *args))
    th = minimize(f, np.zeros((N - 1) * (dz + 1) + 1), jac=True, method='L-BFGS-B').x
    W = np.asarray(weights(jnp.asarray(th), jnp.asarray(Z)))
    return (W * F).sum(1)

def gw_p(dd, X):
    h = X * dd[:, None]; n = len(dd); hb = h.mean(0); hc = h - hb; Om = hc.T @ hc / n
    for l in range(1, 7):
        G = hc[l:].T @ hc[:-l] / n; Om += (1 - l / 7) * (G + G.T)
    return 1 - stats.chi2.cdf(n * hb @ np.linalg.solve(Om, hb), X.shape[1])

def ceilings(e2, ewss, Zm):
    out = []
    for j in range(Zm.shape[1]):
        z = Zm[:, j]; g = np.searchsorted(np.quantile(z, [1 / 3, 2 / 3]), z, side='right')
        out.append(sum(e2[g == b].sum(0).min() for b in range(3)) / ewss)
    return np.array(out)

def placebo_Z_old(R, n, rng):
    X = np.empty((R, n, dz)); u = rng.standard_normal((R, n, dz)); X[:, 0] = u[:, 0]
    for t in range(1, n): X[:, t] = rho * X[:, t - 1] + np.sqrt(1 - rho ** 2) * u[:, t]
    return X

from ceiling2 import sim_var
import os
_v = np.load(os.environ.get('VARFILE', 'var_fit.npz'))
def placebo_Z(R, n, rng):
    return sim_var(_v['A'], _v['c'], _v['Sig'], n, R, rng)

if __name__ == '__main__':
    beta = float(sys.argv[1]); reps = int(sys.argv[2]); seed = int(sys.argv[3]); disp = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    fbar = F.mean(1, keepdims=True); F[:] = fbar + disp * (F - fbar)
    resid = y - F.mean(1)
    rng = np.random.default_rng(seed); res = []
    PZ = placebo_Z(300, SPLIT - E0, rng)
    for r in range(reps):
        logits = beta * np.column_stack([Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3], Z[:, 4]])
        Wt = np.exp(logits - logits.max(1, keepdims=True)); Wt /= Wt.sum(1, keepdims=True)
        m = (Wt * F).sum(1)
        nb = T // 12 + 1; perm = rng.permutation(nb)
        eps = np.concatenate([resid[12 * b:12 * b + 12] for b in perm])[:T]
        ys = m + eps
        ev = slice(E0, SPLIT); e2 = (ys[ev, None] - F[ev]) ** 2; ewe = (ys[ev] - F[ev].mean(1)) ** 2
        oracle = ((ys[ev] - m[ev]) ** 2).sum() / ewe.sum()
        X = np.column_stack([np.ones(SPLIT - E0), Z[ev]])
        cpa_min = min(gw_p(ewe - e2[:, k], X) for k in range(N))          # any expert vs equal weights
        act = ceilings(e2, ewe.sum(), Z[ev]).min()
        pl = np.array([ceilings(e2, ewe.sum(), PZ[q]).min() for q in range(PZ.shape[0])])
        p_ceil = (1 + (pl <= act).sum()) / (len(pl) + 1)
        te = slice(SPLIT, T); ewt = ((ys[te] - F[te].mean(1)) ** 2).sum()
        g2 = ((ys[te] - fit_gate(ys, 2.0)[te]) ** 2).sum() / ewt
        g0 = ((ys[te] - fit_gate(ys, 0.0)[te]) ** 2).sum() / ewt
        orc_te = ((ys[te] - m[te]) ** 2).sum() / ewt
        res.append(dict(oracle=oracle, oracle_test=orc_te, cpa_reject=float(cpa_min < 0.05 / N),
                        ceil_reject=float(p_ceil < 0.05), ceil_gain=1 - act, gate2=g2, gate0=g0, p_ceil=p_ceil))
    json.dump(res, open(f'{os.environ.get('MCPREFIX','mc2')}_b{beta}_d{disp}_s{seed}.json', 'w'))
    a = {k: np.mean([x[k] for x in res]) for k in res[0]}
    print(beta, disp, {k: round(v, 4) for k, v in a.items()})
