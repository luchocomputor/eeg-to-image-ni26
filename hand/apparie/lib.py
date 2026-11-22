import numpy as np, torch
from torch.nn import functional as F
from scipy.stats import betabinom
import os; S = os.path.dirname(os.path.abspath(__file__)) + "/"
torch.set_num_threads(6)
D = torch.load(S + "../../outputs/preds_h1500/_apparie_cache.pt")
preds = {r: F.normalize(p.float(), dim=-1) for r, p in D["preds"].items()}
Gh, true, subj, stim = D["Gh"].float(), D["true"].numpy(), D["subj"].numpy(), D["stim"].numpy()
M = Gh.shape[0]
key = subj * 100000 + stim
uk, gid_all = np.unique(key, return_inverse=True)
qsubj = uk // 100000
qtrue = np.zeros(len(uk), np.int64); qtrue[gid_all] = true
W40 = betabinom.cdf(4, 40000 - 1, np.arange(M) + 1, M - 1 - np.arange(M) + 1)   # P(top-5 à 40k | rang r sur 1500)
rng0 = np.random.default_rng(0)
# groupes k=2 : paires de répétitions au sein d'un (sujet, image)
o = rng0.permutation(len(key)); o = o[np.argsort(key[o], kind="stable")]
pos = np.zeros(len(key), np.int64)
ks = key[o]; st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
gid2 = np.full(len(key), -1); n2 = 0; q2 = []
for a, b in zip(st, np.r_[st[1:], len(ks)]):
    m = (b - a) // 2
    if m:
        gid2[o[a:a + 2 * m]] = n2 + np.arange(2 * m) // 2
        q2 += [gid_all[o[a]]] * m
        n2 += m
q2 = np.array(q2)


def ranks(P, mode=0):
    """mode 0 : toutes répétitions par (sujet, image) ; 2 : paires ; 1 : par essai. P non normalisé accepté
    (la plateforme moyenne les sorties brutes)."""
    if mode == 1:
        agg, t = P, true
    elif mode == 0:
        agg = torch.zeros(len(uk), P.shape[1]).index_add_(0, torch.from_numpy(gid_all), P)
        t = qtrue
    else:
        g = np.where(gid2 < 0, n2, gid2)
        agg = torch.zeros(n2 + 1, P.shape[1]).index_add_(0, torch.from_numpy(g), P)[:n2]
        t = qtrue[q2]
    agg = F.normalize(agg, dim=-1)
    out = np.empty(len(agg), np.int64)
    tt = torch.from_numpy(t)
    for a in range(0, len(agg), 4096):
        sc = agg[a:a + 4096] @ Gh.T
        out[a:a + 4096] = (sc > sc.gather(1, tt[a:a + 4096, None])).sum(1).numpy()
    return out


def comb(ws, norm=True):
    P = sum(w * preds[r] for r, w in ws.items() if w)
    return F.normalize(P, dim=-1) if norm else P


def mets(r):
    return dict(s40=W40[r], top5=(r < 5).astype(float), top20=(r < 20).astype(float), mrr=1 / (r + 1.0), lr=-np.log2(r + 1.0))


BOOT = np.random.default_rng(1).integers(0, len(uk), (2000, len(uk)))


def report(name, r, ref=None, mode=0):
    m = mets(r)
    s = f"{name:44s} 40k {m['s40'].mean()*1e3:.4f}e-3  top5 {m['top5'].mean():.4f}  top20 {m['top20'].mean():.4f}  mrr {m['mrr'].mean():.4f}  -log2r {m['lr'].mean():.3f}"
    if ref is not None and mode == 0:
        mr = mets(ref)
        for k in ("s40", "top5", "mrr", "lr"):
            d = m[k] - mr[k]
            b = d[BOOT].mean(1)
            base = abs(mr[k].mean())
            s += f" | d{k} {100*d.mean()/base:+.1f}% [{100*np.percentile(b,5)/base:+.1f},{100*np.percentile(b,95)/base:+.1f}]"
    print(s, flush=True)
    return m
