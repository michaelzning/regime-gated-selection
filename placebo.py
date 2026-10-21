import numpy as np, json
S = np.load('streams.npz', allow_pickle=True); F, y, Z = S['F'], S['y'], S['Z']
dates = S['dates'].astype(str); E0 = int(np.where(dates == '1957-01')[0][0])
e2 = ((y[:, None] - F) ** 2)[E0:]; ew = ((y - F.mean(1)) ** 2)[E0:]; n = len(ew)
def ceil(zz):
    q = np.quantile(zz, [1/3, 2/3]); g = np.digitize(zz, q)
    return sum(e2[g == b].sum(0).min() for b in range(3)) / ew.sum()
rng = np.random.default_rng(7); P = []
for _ in range(2000):
    u = rng.standard_normal(n); x = np.empty(n); x[0] = u[0]
    for t in range(1, n): x[t] = 0.98 * x[t-1] + np.sqrt(1 - 0.98**2) * u[t]
    P.append(ceil(x))
P = np.array(P); zn = list(S['znames'])
out = {z: dict(actual=float(ceil(Z[E0:, j])), p=float((P <= ceil(Z[E0:, j])).mean())) for j, z in enumerate(zn)}
out['placebo'] = dict(mean=float(P.mean()), q05=float(np.quantile(P, .05)), q50=float(np.median(P)))
print(json.dumps(out, indent=1)); json.dump(out, open('placebo.json', 'w'))
# sampling s.e. of MSE ratio of equal weights relative to PM
d = ((y - F[:, 0]) ** 2)[E0:] - ew
x = d - d.mean(); v = (x**2).mean() + sum(2*(1-l/7)*(x[l:]*x[:-l]).mean() for l in range(1, 7))
print('se of (MSE_PM-MSE_EW)/MSE_EW in %:', 100*np.sqrt(v/n)/ew.mean())
