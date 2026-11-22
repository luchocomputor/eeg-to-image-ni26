"""Grande galerie : modèles seuls et ensembles dans les conditions supposées du
scellé (galerie = les 16 740 images ; k essais moyennés par sujet-image ; mêmes paquets que train_v2).
Proxys : 4 sujets exclus x 1 500 images exclues (~4 rép.) et x 200 images stim_test (~81 rép.).
CPU (le GPU entraîne). python hand/eval_g16k.py fast_aug fast_plain fast_aug_gsoft
"""
import itertools, json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "hand" / "fast"
sys.path.insert(0, str(ROOT / "hand"))
from models_v2 import build_net  # noqa: E402

torch.set_num_threads(4)
runs = sys.argv[1:]
cfgs = {r: json.load(open(ROOT / "outputs/v2" / r / "config.json")) for r in runs}
held, held_img = cfgs[runs[0]]["held"], cfgs[runs[0]]["held_img"]
assert all(c["held"] == held and c["held_img"] == held_img for c in cfgs.values()), "proxys différents"
recs = json.load(open(FAST / "recs.json"))
ix = np.load(FAST / "index.npz")
G = F.normalize(torch.from_numpy(np.load(FAST / "targets.npy")).float(), dim=-1)
big = np.load(FAST / "signal_f16.npy", mmap_mode="r")
rec_base = np.concatenate([[0], np.cumsum([r["n"] for r in recs])]).astype(np.int64)
i0 = np.round(ix["start"] * 120).astype(np.int64)
gidx = rec_base[ix["rec"]] + i0
n_rec = rec_base[1:] - rec_base[:-1]
ok = (i0 >= 0) & (i0 + 120 <= n_rec[ix["rec"]])
stim, subj, split = ix["stim"], ix["subj"].astype(np.int64), ix["split"]
is_held, is_himg = np.isin(subj, held), np.isin(stim, held_img)
PROXIES = {"proxy": (np.where(ok & is_held & is_himg)[0], (1, 2)),
           "proxy81": (np.where(ok & is_held & (split == 2))[0], (1, 2, 0))}


def predict(run, ids):
    net = build_net(cfgs[run]["arch"])
    net.load_state_dict(torch.load(ROOT / "outputs/v2" / run / "best.pt", map_location="cpu"))
    dev = "mps" if cfgs[run]["arch"].startswith("reve") and torch.backends.mps.is_available() else "cpu"   # REVE : 69 M
    net = net.to(dev).eval()
    bs = 256 if dev == "mps" else 2048
    P = torch.empty(len(ids), 1536, dtype=torch.float16)
    with torch.no_grad():
        for j in range(0, len(ids), bs):
            g = gidx[ids[j:j + bs]]
            x = torch.from_numpy(big[g[:, None] + np.arange(120)].astype(np.float32)).transpose(1, 2)
            x = (x - x[:, :, :24].mean(-1, keepdim=True)).clamp(-20, 20)
            if cfgs[run].get("car"):
                x = x - x.mean(1, keepdim=True)
            if cfgs[run].get("zscore"):
                x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-3)
            P[j:j + bs] = F.normalize(net(x.to(dev)).float(), dim=-1).cpu().half()
    return P


def k_groups(ids, k):          # identique à train_v2.k_groups
    key = subj[ids] * 100000 + stim[ids]
    o = np.random.default_rng(0).permutation(len(ids))
    o = o[np.argsort(key[o], kind="stable")]
    ks = key[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    gid, n = np.full(len(ids), -1), 0
    for a, b in zip(st, np.r_[st[1:], len(ks)]):
        m = 1 if k == 0 else (b - a) // k
        if m:
            gid[o[a:b if k == 0 else a + m * k]] = n + (0 if k == 0 else np.arange(m * k) // k)
            n += m
    return gid, int(n)


def big_gal(P, ids, k):
    gid, n = k_groups(ids, k)
    gid = np.where(gid < 0, n, gid)
    agg = torch.zeros(n + 1, P.shape[1])
    for a in range(0, len(ids), 8192):
        agg.index_add_(0, torch.from_numpy(gid[a:a + 8192]), P[a:a + 8192].float())
    agg = F.normalize(agg[:n], dim=-1)
    true = np.zeros(n + 1, np.int64)
    true[gid] = stim[ids]
    true = torch.from_numpy(true[:n])
    r = torch.cat([(lambda sc: (sc > sc.gather(1, true[a:a + 4096, None])).sum(1))(agg[a:a + 4096] @ G.T)
                   for a in range(0, n, 4096)])
    return (r < 5).float().mean().item(), (r < 50).float().mean().item(), n


combos = [(r,) for r in runs] + [c for m in range(2, len(runs) + 1) for c in itertools.combinations(runs, m)]
rows = {c: {} for c in combos}
for pname, (ids, ks) in PROXIES.items():
    preds = {r: predict(r, ids) for r in runs}
    for c in combos:
        P = F.normalize(sum(preds[r].float() for r in c), dim=-1) if len(c) > 1 else preds[c[0]]
        for k in ks:
            rows[c][f"{pname} k={k or 'tous'}"] = big_gal(P, ids, k)
    del preds
cols = list(rows[combos[0]])
print("top-5 / top-50 dans la galerie 16 740 (hasard top-5 0,0003, top-50 0,0030) ; n = requêtes")
print(f"{'modèle(s)':44s}" + "".join(f"{c:>20s}" for c in cols))
for c in combos:
    print(f"{'+'.join(c):44s}" + "".join(f"{rows[c][k][0]:9.5f} /{rows[c][k][1]:8.5f} " for k in cols), flush=True)
print(f"{'n':44s}" + "".join(f"{rows[combos[0]][k][2]:>19d} " for k in cols))
