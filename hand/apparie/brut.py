"""Garde-fou « norme libre » (20/11) : sur le proxy fidèle (4 sujets exclus x 1 500 images exclues, galerie homogène),
compare en APPARIÉ l'agrégation des sorties BRUTES (ce que fait la plateforme) à l'agrégation des mêmes sorties
normalisées par essai, pour un ou plusieurs fichiers preds_proxy.pt (sorties brutes d'un modèle proxy, ordre des essais
de train_v3.py). Bootstrap sur les 6 000 requêtes (sujet, image), IC à 90 %.

  python -I hand/apparie/brut.py chemin/preds_proxy.pt [autre.pt ...] [--json sortie.json] [--gammas 0.5,1,1.5,2]
Plusieurs fichiers : chaque modèle seul, puis leur somme (bruts ramenés à une norme moyenne de 1).
Dernière ligne : VERDICT brut_meilleur | neutre | brut_pire, jugé à 2 RÉPÉTITIONS (régime du scellé : 11 sujets,
au plus ~2 présentations par image), bootstrap sur les paires, top-5 et MRR.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import betabinom
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
FAST = ROOT / "hand" / "fast"
torch.set_num_threads(4)
args = sys.argv[1:]
out_json = args[args.index("--json") + 1] if "--json" in args else None
gammas = [float(v) for v in args[args.index("--gammas") + 1].split(",")] if "--gammas" in args else [0.5, 1.5, 2.0]
files = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--json", "--gammas"))]

ix = np.load(FAST / "index.npz")
recs = json.load(open(FAST / "recs.json"))
G = F.normalize(torch.from_numpy(np.load(FAST / "targets.npy")).float(), dim=-1)
rec_base = np.concatenate([[0], np.cumsum([r["n"] for r in recs])]).astype(np.int64)
i0 = np.round(ix["start"] * 120).astype(np.int64)
ok = (i0 >= 0) & (i0 + 120 <= (rec_base[1:] - rec_base[:-1])[ix["rec"]])
cfg = json.load(open(ROOT / "outputs/v2/reve_c/config.json"))          # même découpage proxy pour tous les runs (seed 0)
ph = np.where(ok & np.isin(ix["subj"], cfg["held"]) & np.isin(ix["stim"], cfg["held_img"]))[0]
stim, subj = ix["stim"][ph], ix["subj"][ph].astype(np.int64)
gal = np.unique(stim)
col = np.full(len(G), -1); col[gal] = np.arange(len(gal))
Gh, M = G[torch.from_numpy(gal)], len(gal)
true = col[stim]
key = subj * 100000 + stim
uk, gid = np.unique(key, return_inverse=True)
nq = len(uk)
qtrue = np.zeros(nq, np.int64); qtrue[gid] = true
W40 = betabinom.cdf(4, 40000 - 1, np.arange(M) + 1, M - 1 - np.arange(M) + 1)
BOOT = np.random.default_rng(1).integers(0, nq, (2000, nq))
o = np.random.default_rng(0).permutation(len(key)); o = o[np.argsort(key[o], kind="stable")]
ks = key[o]; st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
g2 = np.full(len(key), -1); n2 = 0; q2 = []
for a_, b_ in zip(st, np.r_[st[1:], len(ks)]):
    m_ = (b_ - a_) // 2
    if m_:
        g2[o[a_:a_ + 2 * m_]] = n2 + np.arange(2 * m_) // 2
        q2 += [gid[o[a_]]] * m_
        n2 += m_
q2 = np.array(q2)
BOOT2 = np.random.default_rng(2).integers(0, n2, (1000, n2))


def rank_of(Q, t):
    Q = F.normalize(Q, dim=-1)
    r = np.empty(len(Q), np.int64)
    tt = torch.as_tensor(t)
    for a_ in range(0, len(Q), 4096):
        sc = Q[a_:a_ + 4096] @ Gh.T
        r[a_:a_ + 4096] = (sc > sc.gather(1, tt[a_:a_ + 4096, None])).sum(1).numpy()
    return r


def ranks(P, k):
    if k == 1:
        return rank_of(P, true)
    if k == 0:
        return rank_of(torch.zeros(nq, P.shape[1]).index_add_(0, torch.as_tensor(gid), P), qtrue)
    m_ = g2 >= 0
    return rank_of(torch.zeros(n2, P.shape[1]).index_add_(0, torch.as_tensor(g2[m_]), P[m_]), qtrue[q2])


def delta(r, r0, boot=None):
    """Écart relatif apparié et IC à 90 % pour top-5, MRR, 40k (bootstrap sur les requêtes, ou sur les paires)."""
    out = {}
    boot = BOOT if boot is None else boot
    for name, f in (("top5", lambda x: (x < 5).astype(float)), ("mrr", lambda x: 1 / (x + 1.0)), ("s40", lambda x: W40[x])):
        d, base = f(r) - f(r0), f(r0).mean()
        b = d[boot].mean(1) / base
        out[name] = dict(val=float(f(r).mean()), ref=float(base), rel=float(d.mean() / base),
                         lo=float(np.percentile(b, 5)), hi=float(np.percentile(b, 95)))
    return out


def show(tag, P, U):
    res = {}
    r1 = ranks(U, 1)
    line = f"{tag:38s} par essai t5 {np.mean(r1 < 5):.4f} mrr {np.mean(1 / (r1 + 1.0)):.4f}"
    for k in (2, 0):
        r0, r = ranks(U, k), ranks(P, k)
        line += (f" | k={k or 'tous'} normalisé t5 {np.mean(r0 < 5):.4f} mrr {np.mean(1 / (r0 + 1.0)):.4f} -> brut t5 {np.mean(r < 5):.4f} "
                 f"mrr {np.mean(1 / (r + 1.0)):.4f}")
        if k == 2:
            d2 = delta(r, r0, BOOT2)
            line += " (" + ", ".join(f"{n} {100 * v['rel']:+.1f}% [{100 * v['lo']:+.1f},{100 * v['hi']:+.1f}]" for n, v in d2.items() if n != "s40") + ")"
        if k == 0:
            d = delta(r, r0)
            line += "".join(f" | {n} {100 * v['rel']:+.1f}% [{100 * v['lo']:+.1f},{100 * v['hi']:+.1f}]" for n, v in d.items())
            res = dict(d, k2=d2, per_trial_top5=float(np.mean(r1 < 5)), per_trial_mrr=float(np.mean(1 / (r1 + 1.0))))
    print(line, flush=True)
    return res


report, raws = {}, []
for f in files:
    P = torch.load(f, map_location="cpu").float()
    assert P.shape == (len(key), 1536), (f, P.shape, len(key))
    n = P.norm(dim=-1)
    c = n / n.mean()
    U = P / n[:, None].clamp_min(1e-12)
    raws.append(P / n.mean())
    name = Path(f).parent.name or Path(f).stem
    print(f"=== {name} : norme moyenne {n.mean():.3f}, cv {float(n.std() / n.mean()):.3f}")
    report[name] = show(name, P, U)
    report[name]["cv"] = float(n.std() / n.mean())
    for g in gammas:
        show(f"  norme^{g}", U * c[:, None] ** g, U)
if len(raws) > 1:
    S = sum(raws)
    report["somme"] = show("somme des bruts", S, F.normalize(sum(F.normalize(r, dim=-1) for r in raws), dim=-1))
main = report.get("somme") or report[next(iter(report))]
t5, mrr = main["k2"]["top5"], main["k2"]["mrr"]
if t5["hi"] < 0 and mrr["hi"] < 0:
    verdict = "brut_pire"
elif (t5["lo"] > 0 or mrr["lo"] > 0) and t5["rel"] > 0 and mrr["rel"] > 0:
    verdict = "brut_meilleur"
else:
    verdict = "neutre"
report["verdict"] = verdict
if out_json:
    json.dump(report, open(out_json, "w"), indent=1)
print("VERDICT", verdict)
