"""Second proxy, fidèle au scellé. FAQ Track 01 : la galerie scellée est « l'ensemble des images cibles
de la partition d'évaluation » (homogène), participants nouveaux, répétitions moyennées par participant. Le proxy81
mêlait 200 images stim_test (plus typiques) et 1 500 images jamais requêtées, ce qui récompense le biais vers les
images typiques et a sous-estimé l'ensemble REVE du ZIP 12 de 30 %.
Ici : requêtes = 4 sujets exclus x 1 500 images exclues (~4 répétitions), galerie = ces 1 500 images (toutes cibles),
top-5 extrapolé à 40 000 (bêta-binomiale) pour k = 1 (par essai), 2 et toutes les répétitions du sujet.
Calibration : gsoft seul (scellé 0,000313 par essai / 0,000475 agrégé) et analogue du ZIP 12 (0,000505 / 0,000826).
python hand/eval_h1500.py reve_b reve_c revef_c reve_d fast_aug_gsoft things_lag_gsoft
"""
import sys

__file__ = __file__.replace("eval_h1500.py", "eval_scelle_like.py")
exec(open(__file__).read().split("preds = {r: predict(r, p81)")[0])
ph = PROXIES["proxy"][0]
galh = np.unique(stim[ph])
colh = np.full(len(G), -1)
colh[galh] = np.arange(len(galh))
Gh, Mh = G[torch.from_numpy(galh)], len(galh)
CACHE_H = ROOT / "outputs" / "preds_h1500"
CACHE_H.mkdir(exist_ok=True)
preds = {}
for r in runs:
    f = CACHE_H / f"{r}.pt"
    if not f.exists():
        torch.save(F.normalize(predict(r, ph).float(), dim=-1).half(), f)
    preds[r] = torch.load(f).float()
print(f"requêtes {len(ph)} essais, galerie {Mh} images", flush=True)


def show(name, P, norm=True):
    if norm:
        P = F.normalize(P, dim=-1)
    out = []
    for k in (1, 2, 0):
        r = ranks(P, ph, k, colh, Gh)
        out += [betabinom.cdf(4, 40000 - 1, r + 1, Mh - 1 - r + 1).mean(), (r < 5).mean()]
    print(f"{name:60s} " + "   ".join(f"{a:.6f} {b:.4f}" for a, b in zip(out[::2], out[1::2])), flush=True)
    return out


def comb(ws):
    return sum(w * F.normalize(preds[r], dim=-1) for r, w in ws.items() if w and r in preds)


print(f"{'modèle':60s} 40k/top-5 1500 : k=1 (par essai)        k=2                    toutes répétitions")
base = show("fast_aug_gsoft  [scellé 0,000313 / 0,000475]", preds["fast_aug_gsoft"])
z12 = show("analogue ZIP 12 : reve_b+reve_c+2*revef_c+f+t  [scellé 0,000505 / 0,000826]",
           comb({"reve_b": 1, "reve_c": 1, "revef_c": 2, "fast_aug_gsoft": 1, "things_lag_gsoft": 1}))
print(f"rapports ZIP 12 / gsoft sur ce proxy : k=1 {z12[0] / base[0]:.2f} (scellé 1,61)   k=2 {z12[2] / base[2]:.2f}   "
      f"toutes {z12[4] / base[4]:.2f} (scellé agrégé 1,74)")
for r in runs:
    show(r, preds[r])
C = {
    "A reve_c+revef_c+0.5*(f+t)": {"reve_c": 1, "revef_c": 1, "fast_aug_gsoft": .5, "things_lag_gsoft": .5},
    "B reve_c+revef_c+reve_d+0.5*(f+t)": {"reve_c": 1, "revef_c": 1, "reve_d": 1, "fast_aug_gsoft": .5, "things_lag_gsoft": .5},
    "C reve_c+revef_c+reve_d+0.25*(f+t)": {"reve_c": 1, "revef_c": 1, "reve_d": 1, "fast_aug_gsoft": .25, "things_lag_gsoft": .25},
    "D reve_c+revef_c+reve_d+(f+t)": {"reve_c": 1, "revef_c": 1, "reve_d": 1, "fast_aug_gsoft": 1, "things_lag_gsoft": 1},
    "E reve_c+revef_c+reve_d": {"reve_c": 1, "revef_c": 1, "reve_d": 1},
    "F reve_b+reve_c+revef_c+reve_d+0.5*(f+t)": {"reve_b": 1, "reve_c": 1, "revef_c": 1, "reve_d": 1,
                                                 "fast_aug_gsoft": .5, "things_lag_gsoft": .5},
}
best, bv = None, -1
for n, ws in C.items():
    v = show(n, comb(ws))[4]
    if v > bv:
        best, bv = n, v
print(f"\ncentrage par lot (moyenne du lot de 512 essais mélangés retirée), sur {best}")
P = F.normalize(comb(C[best]), dim=-1)
o = np.random.default_rng(3).permutation(len(P))
for lam in (0.25, 0.5, 1.0):
    Q = P.clone()
    for a in range(0, len(o), 512):
        b = torch.from_numpy(o[a:a + 512])
        Q[b] = P[b] - lam * P[b].mean(0, keepdim=True)
    show(f"  centré lam={lam} (lots de 512)", Q)
print(f"\nvoisins d'entraînement (référence = images hors galerie évaluée), sur {best}")
ref = np.setdiff1d(np.arange(len(G)), np.union1d(galh, gal))
Gr = G[torch.from_numpy(ref)]
for K, tau in ((8, 0.01), (32, 0.03)):
    m = torch.empty_like(P)
    for a in range(0, len(P), 1024):
        v, i = (P[a:a + 1024] @ Gr.T).topk(K, dim=1)
        m[a:a + 1024] = (torch.softmax(v / tau, dim=1)[..., None] * Gr[i]).sum(1)
    for beta in (0.25, 0.5, 1.0):
        show(f"  K={K} tau={tau} beta={beta}", P + beta * F.normalize(m, dim=-1))
