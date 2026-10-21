import os, numpy as np, jax.numpy as jnp
from scipy.optimize import minimize
import placebo_batch as pb
def r2_from(th0, Z):
    Zj = jnp.asarray(Z); th = th0.copy(); fc = np.full(pb.T, np.nan)
    for t in range(pb.EVAL0, pb.T, 12):
        mask = jnp.asarray((np.arange(pb.T) < t).astype(float))
        f = lambda x: tuple(np.asarray(v, float) for v in pb.vg(jnp.asarray(x), Zj, mask))
        th = minimize(f, th, jac=True, method='L-BFGS-B', options={'maxiter': 500}).x
        W = np.asarray(pb.wj(jnp.asarray(th), Zj)); fc[t:t+12] = (W[t:t+12] * pb.F[t:t+12]).sum(1)
    sl = slice(pb.EVAL0, pb.T)
    return 100 * (1 - ((pb.y[sl] - fc[sl])**2).sum() / ((pb.y[sl] - pb.F[sl, 0])**2).sum())
rng = np.random.default_rng(5); out = []
for k in range(20):
    th0 = rng.normal(0, 1.0, pb.P); th0[-1] = rng.normal(0, 0.5)
    out.append(r2_from(th0, pb.Z0))
out = np.array(out); print('PEN', pb.PEN, 'R2 across 20 random starts: min %.4f max %.4f sd %.5f' % (out.min(), out.max(), out.std()))
