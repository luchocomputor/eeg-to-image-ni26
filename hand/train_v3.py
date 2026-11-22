"""Entraînement v3 (20/11) = train_v2 + --rawnorm (norme de sortie libre = confiance par essai, comme la ClipLoss
norm_kind=y de NeuralBench : la plateforme moyenne les sorties BRUTES, donc la norme pondère les répétitions),
--init (affinage depuis un point de contrôle), --stop_epoch, --eval_light, --save_preds. Sans ces options : identique à v2.

Entraînement v2 : encodeurs EEG -> DINOv2 évalués sur des SUJETS EXCLUS.

Pourquoi : l'EEGNet v1 fait 11x le hasard sur la validation officielle (mêmes sujets) mais
~1x sur le scellé (nouveaux sujets, nouvelles images). On mesure donc tout sur un proxy du
scellé : sujets jamais vus à l'entraînement x images de stim_test (jamais vues non plus),
prédictions moyennées par (sujet, image), galerie = toutes les images stim_test.

Données : fenêtres reconstruites depuis les memmaps du cache (hand/fast/, vérifié identique
au loader officiel), tenues en RAM en float16 : une passe ~ temps GPU seulement.

Usage : python hand/train_v2.py --name tsconv_aug --arch tsconv --epochs 12 [--holdout 4]
        --holdout 0 : entraîne sur TOUS les sujets et tous les splits (modèle final).
"""
import argparse, glob, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
FAST = Path(os.environ.get("NI_FAST", ROOT / "hand" / "fast"))        # Kaggle : NI_FAST, NI_OUT
CACHE = (ROOT / "tracks/image_decoding/data/neural_compet/cache/"
         "neuralset.extractors.neuro.EegExtractor._get_data,1/"
         "frequency=120,filter=(0.10,75),name=EegExtractor,scaler=RobustScaler,"
         "notch_filter=(50,60)-8c9c9e77")
T, NB, CLAMP = 120, 24, 20.0

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--arch", default="tsconv", choices=["eegnet", "tsconv", "atm", "fast", "fastatm", "fastwide", "reve", "reveg", "revef", "revel", "revelf"])
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--bs", type=int, default=1024)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--holdout", type=int, default=4, help="nb de sujets exclus (0 = modèle final)")
p.add_argument("--hold_img", type=int, default=1500, help="images stim_train exclues (proxy 4 rép.)")
p.add_argument("--mix", type=float, default=0.0, help="proba de moyenner avec un essai de la même image")
p.add_argument("--mixk", type=int, default=1,
               help="avec --mix : K > 1 = moyenne simple de l'essai et de k autres essais de la même image (tous sujets), "
                    "k tiré de 1 à K, même décalage temporel pour tous ; 1 = mélange convexe à deux d'origine")
p.add_argument("--shift", type=int, default=0, help="décalage temporel max (échantillons)")
p.add_argument("--chdrop", type=float, default=0.0)
p.add_argument("--noise", type=float, default=0.0)
p.add_argument("--zscore", action="store_true", help="z-score par canal et par époque")
p.add_argument("--car", action="store_true", help="référence moyenne commune par époque")
p.add_argument("--scale", type=float, default=0.0, help="facteur d'amplitude aléatoire 1 +- scale")
p.add_argument("--learn_temp", action="store_true")
p.add_argument("--sym", action="store_true", help="perte CLIP symétrique")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_steps", type=int, default=0, help="test rapide")
p.add_argument("--profile", action="store_true")
p.add_argument("--ema", type=float, default=0.0, help="moyenne mobile des poids (ex. 0.999) ; éval et sauvegarde sur l'EMA")
p.add_argument("--gneg", type=float, default=0.0,
               help="poids (0-1) d'une perte sur TOUTE la galerie des images d'entraînement (0 = CLIP seul)")
p.add_argument("--soft", type=float, default=0.0,
               help="cibles douces pour --gneg : température sur les similarités DINOv2 (0 = cibles dures)")
p.add_argument("--device", default="auto")
p.add_argument("--init_seed", type=int, default=0,
               help="autre initialisation et autre ordre des données, même découpage proxy (ensembles)")
p.add_argument("--things", default="",
               help="dossier(s) des fenêtres THINGS-EEG2 (things_sXX.npz, kaggle/things_prep.py), séparés par des virgules")
p.add_argument("--things_lag", default="0",
               help="THINGS : fenêtre avancée de n éch. pour caler les PEV sur Alljoined ; 'auto' = estimé par sujet")
p.add_argument("--amp", action="store_true", help="GPU : float16 (autocast dans le modèle) + GradScaler")
p.add_argument("--grad_ckpt", action="store_true", help="REVE : recalcul des activations (mémoire, ~+30 %% de temps)")
p.add_argument("--lr_backbone", type=float, default=0.0,
               help="taux du tronc pré-entraîné (REVE) ; 0 = même taux que la tête")
p.add_argument("--clip", type=float, default=0.0, help="écrêtage de la norme du gradient (0 = non)")
p.add_argument("--rawnorm", action="store_true",
               help="perte sur la sortie NON normalisée (norme = confiance apprise par essai) ; échelles globales apprises")
p.add_argument("--nolastnorm", action="store_true", help="réseaux convolutifs : retire la LayerNorm finale (norme libre)")
p.add_argument("--init", default="", help="point de contrôle initial (affinage)")
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--stop_epoch", type=int, default=0, help="arrêt après cette passe (1 = après la 1re), calendrier inchangé")
p.add_argument("--eval_light", action="store_true", help="évalue proxy et proxy81 seulement, sans variantes")
p.add_argument("--save_preds", action="store_true", help="enregistre les sorties brutes du proxy à chaque passe")
p.add_argument("--time_limit", type=float, default=0.0, help="secondes d'entraînement au plus ; au-delà : évaluation et arrêt")
p.add_argument("--epoch_frac", type=float, default=1.0, help="fraction des lots d'une passe (affinage court)")
p.add_argument("--keep_epochs", default="", help="passes (1 = première) dont on garde aussi le point de contrôle, ex. 3 -> ep3.pt")
p.add_argument("--t0", type=float, default=0.07, help="--rawnorm : température initiale visée (CLIP)")
p.add_argument("--tg0", type=float, default=0.03, help="--rawnorm : température initiale visée (galerie)")
args = p.parse_args()
OUT = Path(os.environ.get("NI_OUT", ROOT / "outputs" / "v2")) / args.name
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
log = open(OUT / "log.txt", "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n"); log.flush()


say(f"=== {time.strftime('%d/%m %H:%M')} {vars(args)}")

# ---------------------------------------------------------------- données
recs = json.load(open(FAST / "recs.json"))
ix = np.load(FAST / "index.npz")
targets = torch.from_numpy(np.load(FAST / "targets.npy"))
MODE, ORDER = "round", "base_then_clip"   # vérifié (check_windows.py) : exact hors égalités x.5
best_variant = f"{MODE}/{ORDER}"
t0 = time.time()
n_tot = sum(r["n"] for r in recs)
rec_base = np.concatenate([[0], np.cumsum([r["n"] for r in recs])]).astype(np.int64)
SIG = FAST / "signal_f16.npy"          # partagé entre processus via le cache disque de l'OS
if not SIG.exists():
    tmp = np.lib.format.open_memmap(FAST / "signal_f16.tmp.npy", "w+", np.float16, (n_tot, 32))
    for i, r in enumerate(recs):
        mm = np.memmap(CACHE / r["file"], np.float32, "r", offset=r["offset"], shape=(r["n"], r["c"]))
        tmp[rec_base[i]:rec_base[i + 1]] = mm
    tmp.flush(); del tmp
    (FAST / "signal_f16.tmp.npy").rename(SIG)
big = np.load(SIG, mmap_mode="r")
assert big.shape == (n_tot, 32)
say(f"signal : {big.nbytes / 1e9:.1f} Go (memmap) en {time.time() - t0:.0f} s ; découpe {MODE}/{ORDER}")

x = ix["start"] * 120.0
i0 = {"round": np.round(x), "floor": np.floor(x), "ceil": np.ceil(x)}[MODE].astype(np.int64)
gidx = rec_base[ix["rec"]] + i0                 # premier échantillon de chaque fenêtre
lo = rec_base[ix["rec"]]                        # bornes de l'enregistrement (décalages)
hi = rec_base[ix["rec"] + 1] - T
stim, subj, split = ix["stim"], ix["subj"].astype(np.int64), ix["split"]
AR = np.arange(T)

n_rec = rec_base[1:] - rec_base[:-1]
ok = (i0 >= 0) & (i0 + T <= n_rec[ix["rec"]])                # 355 fenêtres en bord exclues
subjects = np.unique(subj)
held = rng.choice(subjects, args.holdout, replace=False) if args.holdout else np.array([], int)
is_held = np.isin(subj, held)
train_imgs = np.unique(stim[split < 2])
held_img = rng.choice(train_imgs, args.hold_img, replace=False) if args.holdout else np.array([], int)
is_himg = np.isin(stim, held_img)
if args.init_seed:     # après le tirage du proxy : seuls l'initialisation et l'ordre changent
    torch.manual_seed(1000 + args.init_seed)
    rng = np.random.default_rng(1000 + args.init_seed)
if args.holdout:
    train_ids = np.where(ok & ~is_held & ~is_himg & (split < 2))[0]   # sujets et images vus
else:
    train_ids = np.where(ok)[0]                                        # tout (stim_test autorisé)
proxy_ids = np.where(ok & is_held & is_himg)[0]           # PROXY : nouveaux sujets x nouvelles images, 4 rép.
proxy81_ids = np.where(ok & is_held & (split == 2))[0]    # nouveaux sujets x 200 images stim_test, 81 rép.
insub_ids = np.where(ok & ~is_held & is_himg)[0]          # sujets vus x nouvelles images, 4 rép.
say(f"sujets {len(subjects)} ; exclus {sorted(held.tolist())} ; images exclues {len(held_img)} ; "
    f"train {len(train_ids)} ; proxy {len(proxy_ids)} ; proxy81 {len(proxy81_ids)} ; "
    f"même-sujets {len(insub_ids)}")

if args.things:   # THINGS-EEG2 (mêmes images) : fenêtres de W éch. (126 ou 144) ajoutées après le signal Alljoined
    files = sorted(f for d in args.things.split(",")
                   for f in glob.glob(os.path.join(d, "**", "things_s*.npz"), recursive=True))
    metas = [np.load(f) for f in files]
    n_w = [len(m["img"]) for m in metas]
    n_th = sum(n_w)
    X0 = metas[0]["X"]
    W = X0.shape[2]
    PADT = (W - T) // 2                                          # marge de chaque côté de la fenêtre officielle
    name2id = {n: i for i, n in enumerate(json.load(open(FAST / "stims.json"))["stims"])}
    comb = np.empty((n_tot + n_th * W, 32), np.float16)         # tenu en RAM (~ 3,6 + 7,7 Go)
    comb[:n_tot] = big
    # Recalage (18/11) : les PEV THINGS arrivent ~50 ms plus tôt dans la fenêtre que ceux d'Alljoined. Référence :
    # PEV moyen d'Alljoined (20 000 fenêtres, CAR, ligne de base), électrodes postérieures, 25-600 ms.
    POST = list(range(9, 26))      # P1 ... P2 dans l'ordre des 32 électrodes (CH32 de kaggle/things_prep.py)

    def erp_full(X):               # (n, 32, L) -> moyenne après CAR ; la ligne de base se retire ensuite
        X = X.astype(np.float32)
        return (X - X.mean(1, keepdims=True)).mean(0)

    s_aj = np.sort(np.random.default_rng(0).choice(np.where(ok)[0], 20000, replace=False))
    ref = erp_full(big[gidx[s_aj][:, None] + AR].transpose(0, 2, 1))
    ref = (ref - ref[:, :NB].mean(-1, keepdims=True))[POST, NB + 3:NB + 72].ravel()
    pos, th, lags = n_tot, {"stim": [], "subj": [], "split": []}, []
    for k, (f, m, n) in enumerate(zip(files, metas, n_w)):
        X = X0 if k == 0 else m["X"]
        comb[pos:pos + n * W] = X.transpose(0, 2, 1).reshape(-1, 32)
        pos += n * W
        full = erp_full(X[np.sort(np.random.default_rng(1).choice(n, min(n, 20000), replace=False))])
        cs = []
        for s0 in range(3, W - T - 2):                          # départs gardant la marge du décalage +-3
            e = full[:, s0:s0 + T] - full[:, s0:s0 + NB].mean(-1, keepdims=True)
            cs.append(np.corrcoef(ref, e[POST, NB + 3:NB + 72].ravel())[0, 1])
        est = PADT - (3 + int(np.argmax(cs)))
        lag = est if args.things_lag == "auto" else int(args.things_lag)
        assert abs(lag) <= PADT - 3, f"recalage {lag} hors marge {PADT - 3}"
        lags.append(lag)
        say(f"  {Path(f).stem} : recalage estimé {est:+d} éch. (r = {max(cs):.2f} ; sans recalage "
            f"r = {cs[PADT - 3]:.2f}) ; appliqué {lag:+d}")
        th["stim"].append(np.array([name2id[f"{i:05d}.jpg"] for i in m["img"]]))
        th["subj"].append(np.full(n, 1000 + int(Path(f).stem.split("_s")[1])))
        th["split"].append(m["split"].astype(split.dtype))
        del X
    del X0
    big = comb
    th = {k: np.concatenate(v) for k, v in th.items()}
    nom = n_tot + np.arange(n_th) * W + PADT - np.repeat(lags, n_w)     # début de la fenêtre de 120 éch. recalée
    n_aj = len(stim)
    gidx, lo, hi = np.concatenate([gidx, nom]), np.concatenate([lo, nom - 3]), np.concatenate([hi, nom + 3])
    stim, subj, split = (np.concatenate([stim, th["stim"]]), np.concatenate([subj, th["subj"]]),
                         np.concatenate([split, th["split"]]))
    th_ids = n_aj + np.arange(n_th)
    if args.holdout:     # proxy propre : ni les images exclues, ni les 200 concepts de stim_test
        th_ids = th_ids[(th["split"] < 2) & ~np.isin(th["stim"], held_img)]
    train_ids = np.concatenate([train_ids, th_ids])
    say(f"THINGS-EEG2 : {len(files)} fichiers, {n_th} fenêtres, {len(th_ids)} à l'entraînement ; "
        f"train total {len(train_ids)}")

# partenaires « même image » pour le mélange
order = train_ids[np.argsort(stim[train_ids], kind="stable")]
s_sorted = stim[order]
first = np.searchsorted(s_sorted, np.arange(targets.shape[0]))
count = np.searchsorted(s_sorted, np.arange(targets.shape[0]), side="right") - first


def get_x(ids, train=False, offset=0):
    g = gidx[ids] + offset
    if offset:
        g = np.clip(g, lo[ids], hi[ids])
    if train and args.shift:
        g = np.clip(g + rng.integers(-args.shift, args.shift + 1, len(ids)), lo[ids], hi[ids])
    xb = torch.from_numpy(big[g[:, None] + AR]).to(dev).float().transpose(1, 2)  # (B, C, T)
    if ORDER == "clip_then_base":
        xb = xb.clamp(-CLAMP, CLAMP)
    xb = xb - xb[:, :, :NB].mean(-1, keepdim=True)
    if ORDER == "base_then_clip":
        xb = xb.clamp(-CLAMP, CLAMP)
    return xb


MIX_SEEN = []


def augment(xb, ids):
    if args.mix and args.mixk > 1:
        # Moyenne d'essais de la même image (pari « nbm », 20/11 au soir) : l'essai et k partenaires tirés parmi tous les
        # essais d'entraînement de la même image (autres sujets et répétitions), k uniforme de 1 à mixk, avec la
        # probabilité --mix ; sinon l'essai seul. Le potentiel évoqué s'additionne, le bruit se moyenne : entrées à
        # meilleur rapport signal sur bruit et moins propres à un sujet. Un seul décalage temporel par essai, appliqué
        # aussi à ses partenaires (potentiels alignés). Formes fixes : lot x (mixk + 1) fenêtres, une seule lecture.
        s = stim[ids]
        B, K = len(ids), args.mixk
        k = rng.integers(1, K + 1, B)
        k[rng.random(B) >= args.mix] = 0
        part = order[first[s][:, None] + (rng.random((B, K)) * count[s][:, None]).astype(np.int64)]
        allid = np.concatenate([np.asarray(ids)[:, None], part], 1)
        sh = rng.integers(-args.shift, args.shift + 1, B) if args.shift else np.zeros(B, np.int64)
        g = np.clip(gidx[allid] + sh[:, None], lo[allid], hi[allid]).reshape(-1)
        x = torch.from_numpy(big[g[:, None] + AR]).to(dev).float().transpose(1, 2)
        if ORDER == "clip_then_base":
            x = x.clamp(-CLAMP, CLAMP)
        x = x - x[:, :, :NB].mean(-1, keepdim=True)
        if ORDER == "base_then_clip":
            x = x.clamp(-CLAMP, CLAMP)
        w = torch.from_numpy((np.arange(K + 1)[None, :] <= k[:, None]).astype(np.float32)).to(dev)
        xm = (x.reshape(B, K + 1, x.shape[1], x.shape[2]) * w[:, :, None, None]).sum(1) / w.sum(1)[:, None, None]
        if not MIX_SEEN:      # contrôle au premier lot : mêmes images, et bilan dans le journal
            MIX_SEEN.append(1)
            assert (stim[part] == s[:, None]).all() and xm.shape == xb.shape, "mélange : partenaires d'une autre image"
            say(f"mélange par moyenne d'essais : proba {args.mix}, 1 à {K} partenaires ; 1er lot : {int((k > 0).sum())}/{B} essais moyennés, "
                f"k moyen {k[k > 0].mean() if (k > 0).any() else 0:.1f}, autres sujets parmi les partenaires {float((subj[part] != subj[np.asarray(ids)][:, None]).mean()):.2f}, "
                f"écart-type seul {float(xm[torch.from_numpy(k == 0).to(dev)].std()) if (k == 0).any() else 0:.3f} / moyenné {float(xm[torch.from_numpy(k > 0).to(dev)].std()) if (k > 0).any() else 0:.3f}")
        xb = xm
    elif args.mix:   # formes fixes (taille du lot) : évite la fragmentation de la mémoire MPS
        s = stim[ids]
        part = order[first[s] + (rng.random(len(ids)) * count[s]).astype(np.int64)]
        lam = rng.uniform(0.3, 0.7, len(ids))
        lam[rng.random(len(ids)) >= args.mix] = 1.0
        lam = torch.from_numpy(lam).float().to(dev)[:, None, None]
        xb = lam * xb + (1 - lam) * get_x(part, train=True)
    if args.chdrop:
        xb = xb * (torch.rand(xb.shape[0], xb.shape[1], 1, device=dev) > args.chdrop)
    if args.noise:
        xb = xb + args.noise * torch.randn_like(xb)
    if args.scale:
        xb = xb * (1 + args.scale * (2 * torch.rand(xb.shape[0], 1, 1, device=dev) - 1))
    return xb


def prep(xb):
    if args.car:
        xb = xb - xb.mean(1, keepdim=True)
    if args.zscore:
        xb = (xb - xb.mean(-1, keepdim=True)) / (xb.std(-1, keepdim=True) + 1e-3)
    return xb


# ---------------------------------------------------------------- modèles
sys.path.insert(0, str(ROOT / "hand"))
sys.path.insert(0, str(Path(__file__).resolve().parent))     # Kaggle : models_v2.py à côté du script
from models_v2 import build_net  # noqa: E402

net = build_net(args.arch)
if args.nolastnorm and isinstance(getattr(net, "head", None), nn.Sequential) and isinstance(net.head[-1], nn.LayerNorm):
    net.head[-1] = nn.Identity()
if args.init:
    sd = torch.load(args.init, map_location="cpu")
    miss, unexp = net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=False)
    say(f"init {args.init} ; clés manquantes {list(miss)} ; inattendues {list(unexp)}")
net = net.to(dev)
if args.grad_ckpt:
    net.reve.transformer.ckpt = True
ema = None
if args.ema:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=dev), requires_grad=args.learn_temp)
logit_scale_g = nn.Parameter(torch.tensor(math.log(1 / 0.03), device=dev), requires_grad=args.learn_temp)
params = list(net.parameters()) + ([logit_scale] if args.learn_temp else [])
params += [logit_scale_g] if (args.gneg and args.learn_temp) else []
if args.lr_backbone and hasattr(net, "backbone_parameters"):     # tronc pré-entraîné : taux plus faible
    bb = {id(q) for q in net.backbone_parameters()}
    params = [{"params": [q for q in params if id(q) in bb], "lr": args.lr_backbone},
              {"params": [q for q in params if id(q) not in bb], "lr": args.lr}]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and dev == "cuda"))
steps_per_epoch = int(len(train_ids) // args.bs * args.epoch_frac)
total = args.epochs * steps_per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
say(f"{args.arch} : {sum(q.numel() for q in net.parameters()) / 1e6:.2f} M paramètres ; "
    f"{steps_per_epoch} lots/passe ; device {dev}")
tgt_n = F.normalize(targets, dim=-1).to(dev)
cand = np.unique(stim[train_ids])                     # galerie d'entraînement (images exclues absentes)
col = np.full(targets.shape[0], -1, np.int64)
col[cand] = np.arange(len(cand))
cand_t, col_t = tgt_n[torch.from_numpy(cand).to(dev)], torch.from_numpy(col).to(dev)


def gneg_loss(pred, ids):
    """Softmax sur toute la galerie d'entraînement (~15-17k images) au lieu des 1 024 du lot :
    le score scellé classe l'image vraie parmi ~16,7k candidates très proches (cos moyen 0,68)."""
    s = torch.from_numpy(stim[ids].astype(np.int64)).to(dev)
    lg = (pred * logit_scale_g.exp() if args.rawnorm else F.normalize(pred, dim=-1) * logit_scale_g.exp().clamp(max=100)) @ cand_t.T
    if args.soft:
        q = ((tgt_n[s] @ cand_t.T) / args.soft).softmax(1)
        return -(q * lg.log_softmax(1)).sum(1).mean()
    return F.cross_entropy(lg, col_t[s])


def clip_loss(pred, ids):
    s = torch.from_numpy(stim[ids].astype(np.int64)).to(dev)
    same = (s[:, None] == s[None, :]).float()                 # positifs multiples (même image)
    logits = (pred * logit_scale.exp() if args.rawnorm else F.normalize(pred, dim=-1) * logit_scale.exp().clamp(max=100)) @ tgt_n[s].T
    lp = logits.log_softmax(1)
    loss = -(torch.logsumexp(lp.masked_fill(same == 0, -1e4), 1)).mean()
    if args.sym:
        lp2 = logits.log_softmax(0)
        loss = 0.5 * (loss - torch.logsumexp(lp2.masked_fill(same == 0, -1e4), 0).mean())
    return loss


# ---------------------------------------------------------------- évaluation
@torch.no_grad()
def predict(ids, bs=1024, offset=0, reref=False):
    model = ema.module if ema is not None else net
    model.eval()
    out = []
    for j in range(0, len(ids), bs):
        xb = get_x(ids[j:j + bs], offset=offset)
        if reref:                       # autre référence : tout par rapport à Cz (canal 0)
            xb = xb - xb[:, :1]
        out.append(model(prep(xb)).float().cpu())
    out = torch.cat(out)
    net.train()
    return out


def retrieval(pred, ids, center=None):
    """Score scellé : moyenne par (sujet, image), normalisation, rang dans la galerie des images
    présentes. Renvoie top-1, top-5, rang percentile moyen (0,5 = hasard), taille de galerie."""
    pred = pred.clone()
    if center is not None:
        pred = pred - center
    s = stim[ids]
    gal, ginv = np.unique(s, return_inverse=True)
    key = subj[ids] * 100000 + ginv
    ukey, kinv = np.unique(key, return_inverse=True)
    agg = torch.zeros(len(ukey), pred.shape[1]).index_add_(0, torch.from_numpy(kinv), pred)
    true = torch.from_numpy(ginv[np.unique(kinv, return_index=True)[1]])
    sc = F.normalize(agg, dim=-1) @ F.normalize(targets[gal], dim=-1).T
    own = sc[torch.arange(len(sc)), true]
    rank = (sc > own[:, None]).sum(1)
    return dict(top1=(rank < 1).float().mean().item(), top5=(rank < 5).float().mean().item(),
                pct=1 - (rank.float() / (len(gal) - 1)).mean().item(), gallery=len(gal), queries=len(sc))


def batch_center(P, b):
    """Centrage par lot comme sur la plateforme : lots aléatoires de b époques mélangées."""
    perm = torch.from_numpy(np.random.default_rng(1).permutation(len(P)))
    Q = P.clone()
    for j in range(0, len(P), b):
        k = perm[j:j + b]
        Q[k] = P[k] - P[k].mean(0, keepdim=True)
    return Q


def k_groups(ids, k):
    """Paquets de k essais d'un même sujet-image (k = 0 : tous) -> numéro de paquet par essai (-1 : hors paquet)."""
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
    return gid, n


def big_gal(P, ids, k):
    """Conditions supposées du scellé (cf. calib_scelle.py) : galerie = les 16 740 images,
    requêtes = k essais moyennés par sujet-image (k = 1 : essai seul)."""
    gid, n = k_groups(ids, k)
    if n == 0:
        return dict(top5=0.0, top50=0.0, n=0)
    gid = np.where(gid < 0, n, gid)
    agg = F.normalize(torch.zeros(n + 1, P.shape[1]).index_add_(0, torch.from_numpy(gid), P)[:n], dim=-1)
    true = np.zeros(n + 1, np.int64)
    true[gid] = stim[ids]
    true = torch.from_numpy(true[:n]).to(dev)
    r = torch.cat([(lambda sc: (sc > sc.gather(1, true[a:a + 2048, None])).sum(1).cpu())(
        agg[a:a + 2048].to(dev) @ tgt_n.T) for a in range(0, n, 2048)])
    return dict(top5=(r < 5).float().mean().item(), top50=(r < 50).float().mean().item(), n=int(n))


def scelle_like(P81, ids81):
    """Proxy calibré sur le scellé (comme hand/eval_scelle_like.py, 19/11) : sujets exclus x 200 concepts nouveaux.
    40k : galerie des ~1 700 images jamais vues, top-5 extrapolé à 40 000 images (bêta-binomiale), k = 1, 2 ;
    200 : galerie des 200 concepts, top-1 / top-5, k = 1, 2, 8."""
    from scipy.stats import betabinom
    out = {}
    for name, g in (("40k", np.unique(stim[np.concatenate([proxy_ids, proxy81_ids])])), ("200", np.unique(stim[ids81]))):
        col = np.full(targets.shape[0], -1)
        col[g] = np.arange(len(g))
        Gs = tgt_n[torch.from_numpy(g).to(dev)]
        for k in ((1, 2) if name == "40k" else (1, 2, 8)):
            gid, n = k_groups(ids81, k)
            if n == 0:
                continue
            gid = np.where(gid < 0, n, gid)
            agg = F.normalize(torch.zeros(n + 1, P81.shape[1]).index_add_(0, torch.from_numpy(gid), P81)[:n], dim=-1)
            true = np.zeros(n + 1, np.int64)
            true[gid] = col[stim[ids81]]
            true = torch.from_numpy(true[:n]).to(dev)
            sc = agg.to(dev) @ Gs.T
            r = (sc > sc.gather(1, true[:, None])).sum(1).cpu().numpy()
            if name == "40k":
                out[f"40k_k{k}"] = float(betabinom.cdf(4, 40000 - 1, r + 1, len(g) - 1 - r + 1).mean())
            else:
                out[f"200_top1_k{k}"], out[f"200_top5_k{k}"] = float((r < 1).mean()), float((r < 5).mean())
    return out


def per_trial(P, ids):
    """Top-5 / top-1 par essai dans la galerie des images présentes (la norme ne compte pas)."""
    s = stim[ids]
    gal, ginv = np.unique(s, return_inverse=True)
    Gs = F.normalize(targets[gal], dim=-1)
    true = torch.from_numpy(ginv)
    r = torch.cat([(lambda sc: (sc > sc.gather(1, true[a:a + 4096, None])).sum(1))(F.normalize(P[a:a + 4096], dim=-1) @ Gs.T)
                   for a in range(0, len(P), 4096)])
    return dict(top1=(r < 1).float().mean().item(), top5=(r < 5).float().mean().item(),
                pct=1 - (r.float() / (len(gal) - 1)).mean().item())


def evaluate(tag):
    res = {}
    sets = (("proxy", proxy_ids), ("proxy81", proxy81_ids), ("meme_sujets", insub_ids))
    for name, ids in (sets[:2] if args.eval_light else sets):
        if len(ids) == 0:
            continue
        if args.max_steps:                         # test rapide
            ids = ids[::10]
        Praw = predict(ids)
        P = F.normalize(Praw, dim=-1)              # sortie normalisée par essai (ZIP 12 et 14)
        res[name] = retrieval(P, ids)
        res[name + "_brut"] = retrieval(Praw, ids)     # moyenne des sorties BRUTES (ce que fait la plateforme)
        res[name + "_essai"] = per_trial(P, ids)
        nr = Praw.norm(dim=-1)
        res[name + "_norme"] = dict(moy=nr.mean().item(), cv=(nr.std() / nr.mean()).item())
        if name == "proxy":
            for k in (1, 2):
                res[f"G16k_proxy_brut_k{k}"] = big_gal(Praw, ids, k)
            if args.save_preds and not args.max_steps:
                torch.save(Praw.half(), OUT / "preds_proxy.pt")
        if args.eval_light:
            if name == "proxy":
                for k in (1, 2):
                    res[f"G16k_{name}_k{k}"] = big_gal(P, ids, k)
            continue
        if name in ("proxy", "proxy81"):
            for k in ((1, 2) if name == "proxy" else (1, 2, 0)):
                res[f"G16k_{name}_k{k or 'tous'}"] = big_gal(P, ids, k)
        if name == "proxy81":
            res["scelle"] = scelle_like(P, ids)
        res[name + "_centre"] = retrieval(P, ids, center=P.mean(0))
        if name == "proxy":
            res["proxy_lot64"] = retrieval(batch_center(P, 64), ids)
            res["proxy_decal3"] = retrieval(F.normalize(predict(ids, offset=3), dim=-1), ids)
            res["proxy_refCz"] = retrieval(F.normalize(predict(ids, reref=True), dim=-1), ids)
    say(tag, json.dumps({k: {m: round(v, 5) for m, v in d.items()} for k, d in res.items()}))
    return res


# ---------------------------------------------------------------- entraînement
best, step = -1.0, 0
T_START = time.time()
if args.rawnorm:      # échelles globales initiales : température visée / norme moyenne de la sortie à l'initialisation
    with torch.no_grad():
        net.eval()
        n0 = net(prep(get_x(train_ids[:256]))).float().norm(dim=-1).mean().item()
        net.train()
    logit_scale.data.fill_(math.log(1 / args.t0 / n0))
    logit_scale_g.data.fill_(math.log(1 / args.tg0 / n0))
    say(f"rawnorm : norme moyenne initiale {n0:.2f} ; échelles initiales {logit_scale.exp().item():.3f} / {logit_scale_g.exp().item():.3f}")
json.dump(dict(vars(args), held=sorted(held.tolist()), held_img=sorted(held_img.tolist()),
               variant=best_variant),
          open(OUT / "config.json", "w"), indent=1)
for ep in range(args.epochs):
    perm = rng.permutation(train_ids)
    t0, losses = time.time(), []
    for b in range(steps_per_epoch):
        ids = perm[b * args.bs:(b + 1) * args.bs]
        if args.profile:
            torch.mps.synchronize(); ta = time.time()
        xb = get_x(ids, train=True)
        if args.profile:
            torch.mps.synchronize(); tb = time.time()
        xb = prep(augment(xb, ids))
        if args.profile:
            torch.mps.synchronize(); tc = time.time()
        out = net(xb)
        loss = clip_loss(out, ids)
        if args.gneg:
            loss = (1 - args.gneg) * loss + args.gneg * gneg_loss(out, ids)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], args.clip)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if ema is not None:
            ema.update_parameters(net)
        if dev == "mps" and step % 200 == 0:
            torch.mps.empty_cache()
        losses.append(loss.item())
        if args.profile:
            td = time.time()
            prof = locals().setdefault("prof", [])
            prof.append((tb - ta, tc - tb, td - tc))
            if len(prof) == 25:
                say("profil (s/pas) données, augmentations, calcul :", np.mean(prof[5:], 0).round(3))
        if b % 200 == 0:
            say(f"  ep {ep} lot {b}/{steps_per_epoch} loss {np.mean(losses[-200:]):.3f} "
                f"T {1 / logit_scale.exp().item():.3f} Tg {1 / logit_scale_g.exp().item():.3f} "
                + (f"norme {out.detach().float().norm(dim=-1).mean().item():.2f} cv {(out.detach().float().norm(dim=-1).std() / out.detach().float().norm(dim=-1).mean()).item():.3f} " if args.rawnorm else "")
                + f"({time.time() - t0:.0f} s)")
        if args.max_steps and step >= args.max_steps:
            break
        if args.time_limit and time.time() - T_START > args.time_limit:
            say(f"limite de temps atteinte au lot {b} de la passe {ep} : évaluation puis arrêt")
            break
    res = evaluate(f"PASSE {ep} loss {np.mean(losses):.3f} ({time.time() - t0:.0f} s)")
    if dev == "mps":
        torch.mps.empty_cache()
    final = ema.module if ema is not None else net
    torch.save(final.state_dict(), OUT / "last.pt")
    if str(ep + 1) in args.keep_epochs.split(","):
        torch.save(final.state_dict(), OUT / f"ep{ep + 1}.pt")
    crit = res["proxy_brut" if args.rawnorm else "proxy"]["pct"] if "proxy" in res else ep      # modèle final : la dernière passe
    if crit > best:
        best = crit
        torch.save(final.state_dict(), OUT / "best.pt")
        json.dump(dict(epoch=ep, **res), open(OUT / "best.json", "w"), indent=1)
        say("  -> meilleur (rang percentile proxy)")
    if args.max_steps and step >= args.max_steps:
        break
    if args.time_limit and time.time() - T_START > args.time_limit:
        break
    if args.stop_epoch and ep + 1 >= args.stop_epoch:
        say(f"arrêt demandé après la passe {ep + 1} (calendrier de {args.epochs} passes)")
        break
say("FINI")
