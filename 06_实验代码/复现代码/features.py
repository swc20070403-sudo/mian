"""Handcrafted pulse features following Appendix A of the manuscript (143 waveform + 7 context)."""
import numpy as np
import pywt

S1, S2, S3 = slice(1, 31), slice(31, 41), slice(41, 101)


def _hist_entropy(x, bins=16):
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        lo, hi = x[i].min(), x[i].max()
        if hi - lo < 1e-15:
            out[i] = 0.0
            continue
        c, _ = np.histogram(x[i], bins=bins, range=(lo, hi))
        p = c[c > 0] / c.sum()
        out[i] = -(p * np.log(p)).sum() / np.log(bins)
    return out


def _skew_kurt(x):
    n = x.shape[1]
    d = x - x.mean(1, keepdims=True)
    m2 = (d ** 2).mean(1); m3 = (d ** 3).mean(1); m4 = (d ** 4).mean(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        g1 = m3 / m2 ** 1.5
        g2 = m4 / m2 ** 2 - 3
        G1 = np.sqrt(n * (n - 1)) / (n - 2) * g1
        G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
    return G1, G2


def _linfit(x):
    n = x.shape[1]
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    xm = x.mean(1, keepdims=True)
    a = ((x - xm) * tc).sum(1) / (tc ** 2).sum()
    b = xm[:, 0] - a * t.mean()
    res = x - (b[:, None] + a[:, None] * t)
    sst = ((x - xm) ** 2).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1 - (res ** 2).sum(1) / sst
    return a, b, r2


def _zero_cross(y):
    out = np.empty(y.shape[0])
    for i in range(y.shape[0]):
        z = y[i][y[i] != 0]
        out[i] = np.sum(z[1:] * z[:-1] < 0) if z.size > 1 else 0
    return out


def segment_ops(x, sample_std=False):
    """15 operators of Eqs. (A2)-(A6) applied to window x (n_records, n)."""
    n = x.shape[1]
    mean = x.mean(1)
    std = x.std(1, ddof=1 if sample_std else 0)
    rms = np.sqrt((x ** 2).mean(1))
    energy = (x ** 2).sum(1)
    rng = x.max(1) - x.min(1)
    q75, q25 = np.percentile(x, [75, 25], axis=1)
    iqr = q75 - q25
    med = np.median(x, 1)
    mad = np.median(np.abs(x - med[:, None]), 1)
    auc = ((x[:, 1:] + x[:, :-1]) / 2).sum(1)
    G1, G2 = _skew_kurt(x)
    H = _hist_entropy(x)
    a, b, r2 = _linfit(x)
    tv = np.abs(np.diff(x, axis=1)).sum(1)
    return [mean, std, rms, energy, rng, iqr, mad, auc, G1, G2, H, a, b, r2, tv]


SEG_NAMES = ['mean', 'std', 'rms', 'energy', 'range', 'iqr', 'mad', 'auc', 'skew', 'kurt',
             'hent', 'slope', 'intercept', 'r2', 'tv']


def diff_ops(d):
    """11 operators on a difference sequence."""
    mean = d.mean(1); std = d.std(1); mn = d.min(1); mx = d.max(1)
    med = np.median(d, 1); absmean = np.abs(d).mean(1)
    G1, G2 = _skew_kurt(d)
    rms = np.sqrt((d ** 2).mean(1))
    zc = _zero_cross(d)
    tv = np.abs(np.diff(d, axis=1)).sum(1)
    return [mean, std, mn, mx, med, absmean, G1, G2, rms, zc, tv]


DIFF_NAMES = ['mean', 'std', 'min', 'max', 'median', 'absmean', 'skew', 'kurt', 'rms', 'zc', 'tv']


def _apen_sampen(x, m=2, rfac=0.2):
    N = x.size
    r = rfac * x.std()
    if r <= 0:
        return 0.0, 0.0

    def emb(mm):
        return np.lib.stride_tricks.sliding_window_view(x, mm)

    def phi(mm):
        e = emb(mm)
        d = np.abs(e[:, None, :] - e[None, :, :]).max(-1)
        C = (d <= r).mean(1)
        return np.log(C).mean()

    apen = phi(m) - phi(m + 1)
    # sample entropy: exclude self matches, same number of templates N-m
    e_m = emb(m)[:N - m]
    e_m1 = emb(m + 1)
    dB = np.abs(e_m[:, None, :] - e_m[None, :, :]).max(-1)
    dA = np.abs(e_m1[:, None, :] - e_m1[None, :, :]).max(-1)
    iu = np.triu_indices(N - m, 1)
    B = (dB[iu] <= r).sum()
    A = (dA[iu] <= r).sum()
    sampen = -np.log(A / B) if A > 0 and B > 0 else np.nan
    return apen, sampen


def _perm_entropy(x, order=3):
    e = np.lib.stride_tricks.sliding_window_view(x, order)
    pat = np.argsort(e, axis=1, kind='stable')
    codes = pat[:, 0] * 9 + pat[:, 1] * 3 + pat[:, 2]
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return -(p * np.log(p)).sum() / np.log(6)


def _hurst(x, lags=(1, 2, 4, 8, 16)):
    t = []
    for l in lags:
        t.append(np.sqrt(np.mean((x[l:] - x[:-l]) ** 2)))
    t = np.array(t)
    if np.any(t <= 0):
        return np.nan
    return np.polyfit(np.log(lags), np.log(t), 1)[0]


def waveform_features(V):
    """V: (n, 101) absolute voltage. Returns (n, 143) array and names."""
    V = np.asarray(V, float)
    n = V.shape[0]
    feats, names = [], []
    # --- segment statistics (45)
    for sname, sl in zip(['s1', 's2', 's3'], [S1, S2, S3]):
        for nm, f in zip(SEG_NAMES, segment_ops(V[:, sl])):
            feats.append(f); names.append(f'seg_{sname}_{nm}')
    # --- global waveform (24)
    g = segment_ops(V, sample_std=True)
    gsel = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]  # 12 operators
    for i in gsel:
        feats.append(g[i]); names.append(f'glob_{SEG_NAMES[i]}')
    for t in [0, 30, 40, 100]:
        feats.append(V[:, t]); names.append(f'glob_V{t}')
    feats.append(V[:, 100] - V[:, 0]); names.append('glob_net')
    for q in [5, 10, 25, 50, 75, 90, 95]:
        feats.append(np.percentile(V, q, axis=1)); names.append(f'glob_q{q}')
    # --- difference dynamics (22)
    d1 = np.diff(V, axis=1); d2 = np.diff(d1, axis=1)
    for dn, d in [('d1', d1), ('d2', d2)]:
        for nm, f in zip(DIFF_NAMES, diff_ops(d)):
            feats.append(f); names.append(f'diff_{dn}_{nm}')
    # --- transient boundary (18)
    pairs = [(1, 0), (31, 30), (41, 40), (2, 0), (32, 30), (42, 40), (30, 1), (40, 31), (100, 41),
             (30, 0), (40, 0), (100, 0), (40, 30), (100, 40), (50, 41)]
    for a, b in pairs:
        feats.append(V[:, a] - V[:, b]); names.append(f'tb_V{a}-V{b}')
    seg = V[:, 41:101]
    sgn = np.sign(V[:, 100] - V[:, 41])[:, None]
    p = sgn * (seg - V[:, 41:42])
    tot = np.abs(V[:, 100] - V[:, 41])
    for alpha in [0.5, 0.632, 0.9]:
        thr = alpha * tot
        out = np.full(n, np.nan)
        for i in range(n):
            idx = np.nonzero(p[i] >= thr[i])[0]
            if idx.size == 0:
                continue
            k = idx[0]
            if k == 0:
                out[i] = 0.0
            else:
                y0, y1 = p[i, k - 1], p[i, k]
                out[i] = (k - 1) + (thr[i] - y0) / (y1 - y0) if y1 != y0 else k
        feats.append(out); names.append(f'tb_t{int(alpha * 100)}')
    # --- frequency & wavelet (24)
    N = 101
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    x = (V - V.mean(1, keepdims=True)) * w
    X = np.fft.rfft(x, axis=1)  # 51 bins
    P = np.abs(X) ** 2
    fk = np.arange(X.shape[1]) / N
    for k in range(1, 9):
        feats.append(2 * np.abs(X[:, k]) / w.sum()); names.append(f'fft_A{k}')
    Pn = P[:, 1:51]
    tot = Pn.sum(1)
    for lo, hi, nm in [(1, 3, 'B1'), (4, 7, 'B2'), (8, 15, 'B3'), (16, 50, 'B4')]:
        feats.append(P[:, lo:hi + 1].sum(1) / tot); names.append(f'fft_{nm}')
    kst = 1 + Pn.argmax(1)
    feats.append(fk[kst]); names.append('fft_fdom')
    feats.append(2 * np.abs(X[np.arange(n), kst]) / w.sum()); names.append('fft_Adom')
    f1 = fk[1:51]
    C = (Pn * f1).sum(1) / tot
    Ssp = np.sqrt((Pn * (f1 - C[:, None]) ** 2).sum(1) / tot)
    with np.errstate(divide='ignore'):
        F = np.exp(np.log(np.maximum(Pn, 1e-300)).mean(1)) / Pn.mean(1)
    pk = Pn / tot[:, None]
    Hf = -(pk * np.log(np.maximum(pk, 1e-300))).sum(1) / np.log(50)
    cs = np.cumsum(Pn, 1)
    f85 = f1[(cs >= 0.85 * tot[:, None]).argmax(1)]
    for f, nm in [(C, 'centroid'), (Ssp, 'spread'), (F, 'flatness'), (Hf, 'entropy'), (f85, 'roll85')]:
        feats.append(f); names.append(f'fft_{nm}')
    coeffs = pywt.wavedec(V, 'db2', level=4, mode='symmetric', axis=1)
    en = np.stack([(c ** 2).sum(1) for c in coeffs], 1)
    en = en / en.sum(1, keepdims=True)
    for j, nm in enumerate(['cA4', 'cD4', 'cD3', 'cD2', 'cD1']):
        feats.append(en[:, j]); names.append(f'wav_{nm}')
    # --- nonlinear complexity (10)
    for lag in [1, 5, 10]:
        a = V[:, :-lag]; b = V[:, lag:]
        ac = a - a.mean(1, keepdims=True); bc = b - b.mean(1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            feats.append((ac * bc).sum(1) / np.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1)))
        names.append(f'nl_acf{lag}')
    act = V.var(1)
    mob = np.sqrt(d1.var(1) / act)
    mob_d = np.sqrt(d2.var(1) / d1.var(1))
    feats += [act, mob, mob_d / mob]; names += ['nl_hj_act', 'nl_hj_mob', 'nl_hj_com']
    ap = np.empty(n); se = np.empty(n); pe = np.empty(n); hu = np.empty(n)
    for i in range(n):
        ap[i], se[i] = _apen_sampen(V[i])
        pe[i] = _perm_entropy(V[i])
        hu[i] = _hurst(V[i])
    feats += [ap, se, pe, hu]; names += ['nl_apen', 'nl_sampen', 'nl_permen', 'nl_hurst']
    Fm = np.stack(feats, 1)
    Fm[~np.isfinite(Fm)] = np.nan
    assert Fm.shape[1] == 143, Fm.shape
    return Fm, names
