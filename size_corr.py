"""Size of the placebo-gate test under a null that preserves dependence between the indicators and the
forecasting system. The real indicators and the real expert forecasts are kept, so the indicators remain
correlated with the experts and their disagreement, but the outcome is replaced by r*_t = fbar_t + e_t with e_t
the equal-weight residuals permuted in 12-month blocks. Equal weighting is then optimal and the indicators
carry no information about which expert is more accurate. The primary gate (lambda = 2) is trained on the real
indicators and on B VAR placebos for each replication."""
import sys, json, numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import placebo_batch as pb
from ceiling2 import sim_var

vg = jax.jit(jax.value_and_grad(lambda th, Z, mask, yv: obj_y(th, Z, mask, yv)))
def obj_y(th, Z, mask, yv):
    W, a, G = pb.weights(th, Z)
    ld = pb.logt(yv[:, None] - pb.Fj, jnp.exp(th[-1]) * pb.sj[:, None])
    lm = jax.scipy.special.logsumexp(jnp.log(W) + ld, axis=1)
    return -(mask * lm).sum() + pb.PEN * (jnp.sum(a ** 2) + jnp.sum(G ** 2))

def r2(Z, yv):
    Zj, yj = jnp.asarray(Z), jnp.asarray(yv); th = np.zeros(pb.P); fc = np.full(pb.T, np.nan)
    for t in range(pb.EVAL0, pb.T, 12):
        mask = jnp.asarray((np.arange(pb.T) < t).astype(float))
        f = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x), Zj, mask, yj))
        th = minimize(f, th, jac=True, method='L-BFGS-B', options={'maxiter': 500}).x
        W = np.asarray(pb.wj(jnp.asarray(th), Zj)); fc[t:t + 12] = (W[t:t + 12] * pb.F[t:t + 12]).sum(1)
    sl = slice(pb.EVAL0, pb.T)
    return 100 * (1 - ((yv[sl] - fc[sl]) ** 2).sum() / ((yv[sl] - pb.F[sl, 0]) ** 2).sum())

if __name__ == '__main__':
    start, stop, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    v = np.load('var_fit.npz'); fbar = pb.F.mean(1); resid = pb.y - fbar; res = []
    for r in range(start, stop):
        rng = np.random.default_rng(555_000 + r)
        nb = pb.T // 12 + 1; perm = rng.permutation(nb)
        eps = np.concatenate([resid[12 * b:12 * b + 12] for b in perm])[:pb.T]
        ys = fbar + eps
        P = sim_var(v['A'], v['c'], v['Sig'], pb.T, B, rng)
        real = r2(pb.Z0, ys); pl = np.array([r2(P[b], ys) for b in range(B)])
        res.append(dict(real=float(real), p=float((1 + (pl >= real).sum()) / (B + 1)), pl_mean=float(pl.mean())))
    json.dump(res, open(f'sizecorr_{start}_{stop}_B{B}.json', 'w'))
    p = np.array([x['p'] for x in res]); print(start, stop, 'reject@5%% %.3f @10%% %.3f' % ((p <= 0.05).mean(), (p <= 0.10).mean()),
                                             'real-placebo %.3f' % np.mean([x['real'] - x['pl_mean'] for x in res]))
