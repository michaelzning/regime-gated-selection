"""Monte Carlo designs in which the true selection mechanism differs from the gate's softmax-linear form.
m_t = (1-g) * fbar_t + g * f_{k(t),t}, where k(t) is chosen by
  thresh:   terciles of realized variance (z_1): low -> Value, mid -> Carry, high -> Momentum
  latent:   a hidden Markov chain over the four factor experts, persistence 0.97, whose transitions are
            tilted toward expert j when indicator j is high (eta = 1), so the indicators are only partly informative
  interact: the signs of realized variance and the default spread jointly: (+,+) Value, (+,-) Carry, (-,+) Momentum, (-,-) Reversal
g = 0 is the null (equal weights optimal). Pre-test on 1957-1988, gate trained 1947-1988, evaluated 1989-2020."""
import sys, os, json, numpy as np
import mc_power2 as mc

def selection(design, rng):
    T, Z = mc.T, mc.Z
    if design == 'thresh':
        z = Z[:, 0]; q = np.nanquantile(z[mc.E0:mc.SPLIT], [1 / 3, 2 / 3])
        return np.choose(np.searchsorted(q, z, side='right'), [1, 2, 3])
    if design == 'interact':
        a, b = Z[:, 0] > 0, Z[:, 1] > 0
        return np.where(a & b, 1, np.where(a & ~b, 2, np.where(~a & b, 3, 4)))
    if design == 'latent':
        k = np.empty(T, int); k[0] = rng.integers(1, 5)
        for t in range(1, T):
            logit = 1.0 * Z[t - 1, :4] + np.log(0.97 / 0.01) * (np.arange(1, 5) == k[t - 1])
            p = np.exp(logit - logit.max()); p /= p.sum(); k[t] = 1 + rng.choice(4, p=p)
        return k

if __name__ == '__main__':
    design, g, disp, reps, seed = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    F = mc.F; fbar = F.mean(1, keepdims=True); F[:] = fbar + disp * (F - fbar)
    rng = np.random.default_rng(seed); PZ = mc.placebo_Z(300, mc.SPLIT - mc.E0, rng); res = []
    T, E0, SPLIT, N = mc.T, mc.E0, mc.SPLIT, mc.N
    for r in range(reps):
        k = selection(design, rng)
        m = (1 - g) * F.mean(1) + g * F[np.arange(T), k]
        nb = T // 12 + 1; perm = rng.permutation(nb)
        eps = np.concatenate([mc.resid[12 * b:12 * b + 12] for b in perm])[:T]
        ys = m + eps
        ev = slice(E0, SPLIT); e2 = (ys[ev, None] - F[ev]) ** 2; ewe = (ys[ev] - F[ev].mean(1)) ** 2
        X = np.column_stack([np.ones(SPLIT - E0), mc.Z[ev]])
        cpa_min = min(mc.gw_p(ewe - e2[:, j], X) for j in range(N))
        act = mc.ceilings(e2, ewe.sum(), mc.Z[ev]).min()
        pl = np.array([mc.ceilings(e2, ewe.sum(), PZ[q]).min() for q in range(PZ.shape[0])])
        p_ceil = (1 + (pl <= act).sum()) / (len(pl) + 1)
        te = slice(SPLIT, T); ewt = ((ys[te] - F[te].mean(1)) ** 2).sum()
        g2 = ((ys[te] - mc.fit_gate(ys, 2.0)[te]) ** 2).sum() / ewt
        orc = ((ys[te] - m[te]) ** 2).sum() / ewt
        res.append(dict(oracle_test=orc, cpa_reject=float(cpa_min < 0.05 / N), ceil_gain=1 - act, p_ceil=p_ceil, gate2=g2))
    json.dump(res, open(f"{os.environ.get('MISPREFIX','mcmis')}_{design}_g{g}_d{disp}.json", 'w'))
    rej = np.array([(x['ceil_gain'] > 0) and (x['p_ceil'] < 0.05) for x in res]); g2 = 1 - np.array([x['gate2'] for x in res])
    print(design, g, disp, 'oracle %.2f%%' % (100 * (1 - np.mean([x['oracle_test'] for x in res]))),
          'cpa %.2f' % np.mean([x['cpa_reject'] for x in res]), 'rule %.3f' % rej.mean(),
          'gate %.2f%%' % (100 * g2.mean()), 'gate|rej %s' % (round(100 * g2[rej].mean(), 2) if rej.any() else '-'),
          'gate|not %s' % (round(100 * g2[~rej].mean(), 2) if (~rej).any() else '-'),
          'beats|rej %s' % (round((g2[rej] > 0).mean(), 2) if rej.any() else '-'), 'beats|not %s' % (round((g2[~rej] > 0).mean(), 2) if (~rej).any() else '-'))
