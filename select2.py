"""Selectors on the expert streams. Re-estimated every 12 months on an expanding window."""
import sys, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)

S = np.load('streams.npz', allow_pickle=True)
F, y, Z, s = S['F'], S['y'], S['Z'], S['s']
dates = S['dates']; T, N = F.shape; dz = Z.shape[1]
import os
NU, RHO = 5.0, 0.01
PEN = float(os.environ.get('PEN', '2.0'))
H = 8
SUF = os.environ.get('SUF', '')
EVAL0 = int(np.where(dates == '1957-01')[0][0])

def logt(e, sc):
    sc = sc * jnp.sqrt((NU - 2) / NU)
    return (gammaln((NU + 1) / 2) - gammaln(NU / 2) - 0.5 * jnp.log(NU * jnp.pi) - jnp.log(sc)
            - (NU + 1) / 2 * jnp.log1p((e / sc) ** 2 / NU))

def mlp_logits(th, z):
    i = 0
    W1 = th[i:i + H * dz].reshape(H, dz); i += H * dz
    b1 = th[i:i + H]; i += H
    W2 = th[i:i + N * H].reshape(N, H); i += N * H
    return W2 @ jnp.tanh(W1 @ z + b1)

def unpack(th, kind):
    if kind == 'mlp':
        a = jnp.concatenate([jnp.zeros(1), th[-(N + 1):-2]])
        return a, None, 0.0, th[-1]
    a = jnp.concatenate([jnp.zeros(1), th[:N - 1]])
    G = jnp.zeros((N, dz))
    if kind in ('gate', 'filt'):
        G = jnp.concatenate([jnp.zeros((1, dz)), th[N - 1:N - 1 + (N - 1) * dz].reshape(N - 1, dz)])
    kap = th[-2] if kind in ('filt', 'msnoz') else 0.0
    return a, G, kap, th[-1]

def npar(kind):
    if kind == 'mlp': return H * dz + H + N * H + (N - 1) + 2
    return (N - 1) + ((N - 1) * dz if kind in ('gate', 'filt') else 0) + 1 + 1

def run(th, kind, F, y, Z, s):
    a, G, kap, logc = unpack(th, kind); c = jnp.exp(logc)
    rho = RHO if kind in ('filt', 'msnoz') else 0.0
    eye = jnp.eye(N)
    def step(post, inp):
        f, yt, z, st = inp
        g = a + (mlp_logits(th, z) if kind == 'mlp' else G @ z)
        L = g[None, :] + kap * eye
        Q = (1 - rho) * jax.nn.softmax(L, axis=1) + rho / N
        prior = post @ Q
        ld = logt(yt - f, c * st)
        lm = jax.scipy.special.logsumexp(jnp.log(prior) + ld)
        post = jnp.exp(jnp.log(prior) + ld - lm)
        return post, (prior, lm)
    _, (P, lm) = jax.lax.scan(step, jnp.full(N, 1.0 / N), (F, y, Z, s))
    return P, lm

def make_obj(kind, n):
    Fj, yj, Zj, sj = map(jnp.asarray, (F[:n], y[:n], Z[:n], s[:n]))
    def obj(th):
        _, lm = run(th, kind, Fj, yj, Zj, sj)
        a, G, _, _ = unpack(th, kind)
        if kind == 'mlp':
            return -lm.sum() + PEN * jnp.sum(th[:-2] ** 2)
        return -lm.sum() + PEN * (jnp.sum(a ** 2) + jnp.sum(G ** 2))
    return jax.jit(jax.value_and_grad(obj))

def fit(kind, n, th0):
    vg = make_obj(kind, n)
    f = lambda x: tuple(np.asarray(v, dtype=float) for v in vg(jnp.asarray(x)))
    b = [(None, None)] * npar(kind)
    if kind in ('filt', 'msnoz'): b[-2] = (0.0, 8.0)
    return minimize(f, th0, jac=True, method='L-BFGS-B', bounds=b, options={'maxiter': 500}).x

def rolling(kind):
    W = np.full((T, N), np.nan); thetas = []
    th = np.zeros(npar(kind)); th[-1] = 0.0
    if kind == 'mlp': th[:-2] = 0.1 * np.random.default_rng(0).standard_normal(npar(kind) - 2)
    if kind in ('filt', 'msnoz'): th[-2] = 2.0
    runj = jax.jit(lambda th, F, y, Z, s: run(th, kind, F, y, Z, s)[0])
    for t in range(EVAL0, T, 12):
        th = fit(kind, t, th)
        thetas.append((dates[t], th.copy()))
        P = np.asarray(runj(jnp.asarray(th), *map(jnp.asarray, (F, y, Z, s))))
        W[t:t + 12] = P[t:t + 12]          # weights for the next 12 months with frozen parameters
    return W, thetas

if __name__ == '__main__':
    kind = sys.argv[1]
    if kind == 'dma':
        W = np.full((T, N), np.nan); post = np.full(N, 1 / N)
        lt = np.asarray(logt(jnp.asarray(y[:, None] - F), jnp.asarray(s[:, None])))
        for t in range(T):
            pr = post ** 0.99; pr /= pr.sum(); W[t] = pr
            p = pr * np.exp(lt[t] - lt[t].max()); post = p / p.sum()
        np.savez(f'W_{kind}{SUF}.npz', W=W)
    else:
        W, th = rolling(kind)
        np.savez(f'W_{kind}{SUF}.npz', W=W, tdates=np.array([d for d, _ in th]),
                 thetas=np.array([x for _, x in th]))
        print(kind, 'last theta', np.round(th[-1][1], 3))
    print(kind, 'done')
