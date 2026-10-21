"""Null calibration of the placebo-gate test as implemented.
Each replication draws a 'real' indicator path from a known process that is independent of the outcomes and
expert forecasts: a VAR(1) with the coefficients fitted to the actual indicators but Student-t(5) innovations,
so that the Gaussian placebo model is misspecified. The complete procedure is then applied: a VAR(1) is
re-estimated on the simulated path, B Gaussian placebo paths are drawn from the estimate, the primary linear
gate (lambda = 2) is trained on the 'real' path and on each placebo, and the rank p-value is recorded."""
import sys, json, numpy as np
from placebo_batch import r2_gate, T
from ceiling2 import fit_var, sim_var

def t_var_path(A, c, Sig, n, rng, nu=5.0, burn=200):
    L = np.linalg.cholesky(Sig); mu = np.linalg.solve(np.eye(len(c)) - A, c); x = mu.copy(); out = np.empty((n, len(c)))
    scale = np.sqrt((nu - 2) / nu)
    for t in range(-burn, n):
        e = rng.standard_t(nu, len(c)) * scale
        x = c + A @ x + L @ e
        if t >= 0: out[t] = x
    return out

if __name__ == '__main__':
    start, stop, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    v = np.load('var_fit.npz'); res = []
    for r in range(start, stop):
        rng = np.random.default_rng(777_000 + r)
        z = t_var_path(v['A'], v['c'], v['Sig'], T, rng)
        A, c, Sig = fit_var(z)
        P = sim_var(A, c, Sig, T, B, rng)
        real = r2_gate(z); pl = np.array([r2_gate(P[b]) for b in range(B)])
        res.append(dict(real=float(real), p=float((1 + (pl >= real).sum()) / (B + 1)), pl_mean=float(pl.mean())))
    json.dump(res, open(f'size_{start}_{stop}_B{B}.json', 'w'))
    p = np.array([x['p'] for x in res]); print(start, stop, 'reject@5%% %.3f reject@10%% %.3f' % ((p <= 0.05).mean(), (p <= 0.10).mean()))
