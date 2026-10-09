import numpy as np


def paired(a, b, strata=None, n_boot=5000, seed=0):
    """a, b: per-cell metric arrays (same order). Returns mean diff (a-b), 95% CI (stratified cell bootstrap),
    two-sided sign-flip p, n improved (a<b), dz."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    if strata is None:
        strata = np.zeros(d.size, int)
    strata = np.asarray(strata)
    boots = np.empty(n_boot)
    groups = [np.nonzero(strata == s)[0] for s in np.unique(strata)]
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        boots[i] = d[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_boot, d.size))
    null = np.abs((flips * d).mean(1))
    p = (np.sum(null >= obs - 1e-15) + 1) / (n_boot + 1)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    return dict(diff=d.mean(), lo=lo, hi=hi, p=p, improved=int(np.sum(d < 0)), n=d.size, dz=dz)


def holm(ps):
    ps = np.asarray(ps, float); m = ps.size
    order = np.argsort(ps); adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj
