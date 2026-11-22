"""Évaluation de référence : le proxy qui reproduit le score scellé de gsoft (sous 5 %).
Requêtes = 4 sujets exclus x 200 images stim_test (concepts absents de l'entraînement, comme le scellé).
1) galerie = 1 700 images jamais vues (1 500 exclues + 200 stim_test), top-5 estimé pour 40 000 images
   (bêta-binomiale sur le rang) : k = 1 ~ top-5 par essai, k = 2 ~ score scellé agrégé ;
2) galerie = les 200 images de concepts nouveaux seules : top-1 / top-5, k = 1, 2, 8.
Repères gsoft : 40k k=1 0,000303 (scellé 0,000313), k=2 0,000492 (scellé 0,000475).
python hand/eval_scelle_like.py fast_aug_gsoft things_gsoft ... [--ens] [--adabn 64]
  --ens : ajoute l'ensemble de tous ; --adabn N : AdaBN comme la soumission, lots de N essais tirés au hasard
  (sujets mêlés), BatchNorm sur les statistiques du lot, rien de conservé d'un lot à l'autre.
"""
import sys
from scipy.stats import betabinom

ENS = "--ens" in sys.argv
ADA = int(sys.argv[sys.argv.index("--adabn") + 1]) if "--adabn" in sys.argv else 0
if ADA:
    i = sys.argv.index("--adabn")
    del sys.argv[i:i + 2]
sys.argv = [a for a in sys.argv if a != "--ens"]
exec(open(__file__.replace("eval_scelle_like.py", "eval_g16k.py")).read().split("combos = ")[0])

if ADA:
    def predict(run, ids):
        net = build_net(cfgs[run]["arch"])
        net.load_state_dict(torch.load(ROOT / "outputs/v2" / run / "best.pt", map_location="cpu"))
        net.eval()
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train(True)
                m.momentum = 0.0
        o = np.random.default_rng(2).permutation(len(ids))
        starts = list(range(0, len(ids), ADA))
        if len(starts) > 1 and len(ids) - starts[-1] < ADA:      # pas de dernier lot minuscule
            starts.pop()
        P = torch.empty(len(ids), 1536, dtype=torch.float16)
        with torch.no_grad():
            for a, e in zip(starts, starts[1:] + [len(ids)]):
                b = o[a:e]
                g = gidx[ids[b]]
                x = torch.from_numpy(big[g[:, None] + np.arange(120)].astype(np.float32)).transpose(1, 2)
                x = (x - x[:, :, :24].mean(-1, keepdim=True)).clamp(-20, 20)
                if cfgs[run].get("car"):
                    x = x - x.mean(1, keepdim=True)
                if cfgs[run].get("zscore"):
                    x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-3)
                P[torch.from_numpy(b)] = F.normalize(net(x).float(), dim=-1).half()
        return P

ids_all = np.concatenate([PROXIES["proxy"][0], PROXIES["proxy81"][0]])
gal = np.unique(stim[ids_all])
M = len(gal)
col = np.full(len(G), -1)
col[gal] = np.arange(M)
p81 = PROXIES["proxy81"][0]
gal200 = np.unique(stim[p81])
col200 = np.full(len(G), -1)
col200[gal200] = np.arange(len(gal200))


def ranks(P, ids, k, colmap, Gsub):
    gid, n = k_groups(ids, k)
    gid = np.where(gid < 0, n, gid)
    agg = F.normalize(torch.zeros(n + 1, P.shape[1]).index_add_(0, torch.from_numpy(gid), P.float())[:n], dim=-1)
    true = np.zeros(n + 1, np.int64)
    true[gid] = colmap[stim[ids]]
    true = torch.from_numpy(true[:n])
    sc = agg @ Gsub.T
    return (sc > sc.gather(1, true[:, None])).sum(1).numpy()


Gu, G200 = G[torch.from_numpy(gal)], G[torch.from_numpy(gal200)]
preds = {r: predict(r, p81).float() for r in runs}
combos = [(r,) for r in runs] + ([tuple(runs)] if ENS and len(runs) > 1 else [])
print(f"{'modèle':40s} {'40k k=1':>9s} {'40k k=2':>9s}   200 concepts : top-1 / top-5 (k=1, 2, 8)")
for c in combos:
    P = F.normalize(sum(F.normalize(preds[r], dim=-1) for r in c), dim=-1)
    e = [betabinom.cdf(4, 40000 - 1, r + 1, M - 1 - r + 1).mean() for r in (ranks(P, p81, k, col, Gu) for k in (1, 2))]
    t = [(lambda r: ((r < 1).mean(), (r < 5).mean()))(ranks(P, p81, k, col200, G200)) for k in (1, 2, 8)]
    print(f"{'+'.join(c):40s} {e[0]:9.6f} {e[1]:9.6f}   " + "  ".join(f"{a:.4f}/{b:.4f}" for a, b in t), flush=True)
