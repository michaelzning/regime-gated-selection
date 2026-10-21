"""Placebo-gate test for the static linear gate, batched.
The static gate's weights are softmax(a + G z_{t-1}) regardless of past errors, so its prequential likelihood
is a sum over months; an expanding estimation window is implemented with a mask, so the objective is compiled
once and reused for every annual refit and every placebo draw. Estimation choices match select3.py
(Student-t densities, common scale, ridge penalty, warm starts, annual refits from 1957)."""
import sys, os, json, time, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)
S = np.load(os.environ.get('STREAMS', 'streams.npz'), allow_pickle=True)
F, y, Z0, s = S['F'], S['y'], S['Z'], S['s']; dates = S['dates'].astype(str)
T, N = F.shape; dz = Z0.shape[1]; NU = 5.0; PEN = float(os.environ.get('PEN', '2.0'))
EVAL0 = int(np.where(dates == '1957-01')[0][0])
Fj, yj, sj = jnp.asarray(F), jnp.asarray(y), jnp.asarray(s)
P = (N - 1) * (dz + 1) + 2

def logt(e, sc):
    sc = sc * jnp.sqrt((NU - 2) / NU)
    return (gammaln((NU + 1) / 2) - gammaln(NU / 2) - 0.5 * jnp.log(NU * jnp.pi) - jnp.log(sc)
            - (NU + 1) / 2 * jnp.log1p((e / sc) ** 2 / NU))

def weights(th, Z):
    a = jnp.concatenate([jnp.zeros(1), th[:N - 1]])
    G = jnp.concatenate([jnp.zeros((1, dz)), th[N - 1:N - 1 + (N - 1) * dz].reshape(N - 1, dz)])
    return jax.nn.softmax(a[None, :] + Z @ G.T, axis=1), a, G

def obj(th, Z, mask):
    W, a, G = weights(th, Z)
    ld = logt(yj[:, None] - Fj, jnp.exp(th[-1]) * sj[:, None])
    lm = logsumexp(jnp.log(W) + ld, axis=1)
    return -(mask * lm).sum() + PEN * (jnp.sum(a ** 2) + jnp.sum(G ** 2))
vg = jax.jit(jax.value_and_grad(obj))
wj = jax.jit(lambda th, Z: weights(th, Z)[0])

def r2_gate(Z):
    Zj = jnp.asarray(Z); th = np.zeros(P); fc = np.full(T, np.nan)
    for t in range(EVAL0, T, 12):
        mask = jnp.asarray((np.arange(T) < t).astype(float))
        f = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x), Zj, mask))
        th = minimize(f, th, jac=True, method='L-BFGS-B', options={'maxiter': 500}).x
        W = np.asarray(wj(jnp.asarray(th), Zj)); fc[t:t + 12] = (W[t:t + 12] * F[t:t + 12]).sum(1)
    sl = slice(EVAL0, T)
    if os.environ.get('STAT') == 'mse_ew': return ((y[sl] - fc[sl]) ** 2).sum() / ((y[sl] - F[sl].mean(1)) ** 2).sum()
    return 100 * (1 - ((y[sl] - fc[sl]) ** 2).sum() / ((y[sl] - F[sl, 0]) ** 2).sum())

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'const':
        print('constant-weight gate (indicators set to zero): %.4f' % r2_gate(np.zeros_like(Z0))); sys.exit()
    if mode == 'real':
        t0 = time.time(); print('real gate R2 %.4f' % r2_gate(Z0), '%.1fs' % (time.time() - t0)); sys.exit()
    kind, start, stop = mode, int(sys.argv[2]), int(sys.argv[3])
    from ceiling2 import sim_var
    v = np.load('var_fit.npz'); out = []
    for b in range(start, stop):
        rng = np.random.default_rng(10_000 + b + (0 if kind == 'var' else 50_000))
        if kind == 'var': Zp = sim_var(v['A'], v['c'], v['Sig'], T, 1, rng)[0]
        else: Zp = np.roll(Z0, int(rng.integers(60, T - 60)), axis=0)
        out.append(r2_gate(Zp))
    fn = f"pb{os.environ.get('TAG','')}_{kind}_p{PEN:g}_{start}_{stop}.json"; json.dump(out, open(fn, 'w'))
    print(kind, start, stop, 'mean %.3f' % np.mean(out))
