"""Vérifie une soumission v3 (20/11 ; modèles en norme libre, voir make_submission_v3.py) avant envoi :
python hand/test_submission_v3.py submissions/15_xxx. Mêmes contrôles que test_submission.py, plus : la norme de la
sortie varie d'un essai à l'autre (confiance), reste finie et strictement positive, et la sortie d'un essai ne
dépend pas des autres essais du lot quand aucun réseau convolutif n'est en AdaBN.

Vérifie une soumission v2 avant envoi : python hand/test_submission.py submissions/04_xxx

1. chargement par le solveur comme sur la plateforme (meta minimal, device cpu) ;
2. fenêtres OFFICIELLES (hand/fast/check_windows.npz, sorties du loader officiel) : sortie (B, 1536)
   finie ; égale à la prédiction directe des réseaux (même prétraitement qu'à l'entraînement) ;
3. canaux mélangés + canal en trop : même sortie ; canal absent : tourne ;
4. durée différente (121 échantillons) : tourne ; petit lot (< 32, pas de centrage par lot) : tourne.
"""
import importlib.util, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks" / "image_decoding"))
sys.path.insert(0, str(ROOT / "hand"))
sub = (ROOT / sys.argv[1]).resolve()
if "--models" in sys.argv:      # sous-ensemble de modèles (mémoire du Mac) : dossier temporaire de liens, même submission.py
    import os, shutil
    keep = [int(v) for v in sys.argv[sys.argv.index("--models") + 1].split(",")]
    part = sub.with_name(sub.name + "__part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir()
    c0 = json.load(open(sub / "config.json"))
    c0["models"] = [c0["models"][i] for i in keep]
    json.dump(c0, open(part / "config.json", "w"), indent=1)
    shutil.copy(sub / "submission.py", part / "submission.py")
    for mm in c0["models"]:
        os.link(sub / mm["file"], part / mm["file"])
        if mm.get("align"):
            shutil.copy(sub / mm["align"], part / mm["align"])
    if c0.get("gate"):
        shutil.copy(sub / c0["gate"]["file"], part / c0["gate"]["file"])
    if (c0.get("align") or {}).get("embed"):
        shutil.copy(sub / c0["align"]["embed"], part / c0["align"]["embed"])
    if (sub / "center.npy").exists():
        shutil.copy(sub / "center.npy", part / "center.npy")
    print("sous-ensemble", keep, [mm["run"] for mm in c0["models"]])
    sub = part
spec = importlib.util.spec_from_file_location("submission", sub / "submission.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
conf = json.load(open(sub / "config.json"))
names = conf["ch_names"]


def load(ch):
    s = m.Solver.__new__(m.Solver)
    return m.Solver.load_model(s, {"submission_dir": sub, "device": "cpu", "ch_names": ch})


cw = np.load(ROOT / "hand" / "fast" / "check_windows.npz")
X = torch.from_numpy(cw["X"][:512]).float()
ok = True

p1 = load(names).predict(X)
print("forme", tuple(p1.shape), "finie", bool(torch.isfinite(p1).all()))
ok &= p1.shape == (512, 1536) and bool(torch.isfinite(p1).all())

Xd = X
if conf.get("detrend"):
    t_ = torch.linspace(-1, 1, X.shape[-1])
    Xd = X - (X * t_).sum(-1, keepdim=True) / (t_ * t_).sum() * t_
    Xd = Xd - Xd[:, :, :24].mean(-1, keepdim=True)
    print(f"dérive retirée : écart moyen à l'entrée {float((Xd - X).abs().mean()):.3f}")
Wk = None
JS_LAMS = []
al_ = conf.get("align")
if al_ and len(X) >= int(al_.get("min", 32)):      # mêmes poids de voisinage que le solveur
    if al_.get("embed") and float(al_.get("tau", 0)) > 0:
        se = m.SubjectEmbed(n_cls=int(al_["n_cls"]), Fm=int(al_["Fm"]), d=int(al_["d"]))
        se.load_state_dict(torch.load(sub / al_["embed"], map_location="cpu"))
        se.eval()
        with torch.no_grad():
            e_ = se.embed(X - X.mean(1, keepdim=True))
        sim_ = e_ @ e_.T / float(al_["tau"])
        sim_.fill_diagonal_(-1e9)
        Wk = torch.softmax(sim_, dim=1)
    else:
        Wk = (1.0 - torch.eye(len(X))) / (len(X) - 1)
    print(f"alignement : tau {al_.get('tau')}, lam {al_.get('lam')}, K {al_.get('K')} ; poids maximal moyen {float(Wk.max(1).values.mean()):.3f}")
# prédiction directe : réseaux de outputs/v2 + même prétraitement que train_v2
from models_v2 import build_net  # noqa: E402
outs = []
for mm in conf["models"]:
    cfg = json.load(open(ROOT / "outputs" / "v2" / mm["run"] / "config.json"))
    net = build_net(mm["arch"])
    net.load_state_dict({k: v.float() if v.is_floating_point() else v
                         for k, v in torch.load(sub / mm["file"], map_location="cpu").items()})
    net.eval()
    xin = Xd
    if not str(mm["arch"]).startswith("reve") and float(conf.get("conv_zero", 0)) > 0:
        sd_ = xin.std(-1)
        xin = xin * (sd_ <= float(conf["conv_zero"]) * sd_.median(1, keepdim=True).values)[:, :, None]
    xin = xin - xin.mean(1, keepdim=True) if cfg.get("car") else xin
    if cfg.get("zscore"):
        xin = (xin - xin.mean(-1, keepdim=True)) / (xin.std(-1, keepdim=True) + 1e-3)
    if conf.get("adabn") and len(X) >= conf["adabn"]:      # AdaBN : BatchNorm sur les statistiques du lot
        for b in net.modules():
            if isinstance(b, torch.nn.modules.batchnorm._BatchNorm):
                b.train(True)
                b.momentum = 0.0
    with torch.no_grad():
        def _al(o_):
            if Wk is None or not mm.get("align"):
                return o_
            t_al = torch.load(sub / mm["align"], map_location="cpu")
            nr_ = o_.norm(dim=-1, keepdim=True).clamp_min(1e-12); u_ = o_ / nr_
            dl = Wk @ u_ - t_al["m"].float()
            lam_ = float(al_.get("lam", 1.0))
            if al_.get("js"):      # force adaptative : même formule que le solveur
                eps2_ = (1.0 - u_.mean(0).pow(2).sum()).clamp_min(0.0) * Wk.pow(2).sum(1).mean()
                lam_ = lam_ * float((1.0 - eps2_ / dl.pow(2).sum(1).mean().clamp_min(1e-12)).clamp(0.0, 1.0))
                JS_LAMS.append(round(lam_, 3))
            V_ = t_al["V"].float()
            if V_.shape[1] > 0:
                dl = (dl @ V_) @ V_.T
            return (u_ - lam_ * dl) * nr_
        if mm.get("raw"):
            o = _al(net(xin).float() * float(mm.get("scale", 1.0)))
            nr = o.norm(dim=-1, keepdim=True)
            lo, hi = conf.get("norm_clip", [0.05, 20.0])
            outs.append(o / nr.clamp_min(1e-12) * (nr.clamp(lo, hi) if float(conf.get("gamma", 1.0)) != 0.0 else 1.0))
            continue
        o = sum(F.normalize(net(m.EnsembleV2._shift(xin, s)), dim=-1) for s in conf.get("tta", [0]))
        outs.append(F.normalize(_al(F.normalize(o, dim=-1)), dim=-1))
if JS_LAMS:
    print(f"alignement à force adaptative, lot de {len(X)} : force par modèle {JS_LAMS}")
w = [float(mm.get("weight", 1.0)) for mm in conf["models"]]
israw = [bool(mm.get("raw")) for mm in conf["models"]]
if any(israw) and float(conf.get("gamma", 1.0)) != 0.0:
    wr = sum(wi for wi, r in zip(w, israw) if r)
    c = sum(wi * o.norm(dim=-1, keepdim=True) for wi, o, r in zip(w, outs, israw) if r) / wr
    g = float(conf.get("gamma", 1.0))
    if conf.get("combine", "rawsum") == "dirconf":
        ref = F.normalize(sum(wi * F.normalize(o, dim=-1) for wi, o in zip(w, outs)), dim=-1) * c.pow(g)
    else:
        ref = sum(wi * o for wi, o, r in zip(w, outs, israw) if r)
        if g != 1.0:
            ref = ref * c.pow(g - 1.0)
        old = [wi * o for wi, o, r in zip(w, outs, israw) if not r]
        if old:
            ref = ref + c.pow(g) * sum(old)
        ref = ref / sum(w)
    nr = p1.norm(dim=-1)
    print(f"norme de la sortie : moyenne {nr.mean():.3f}, cv {nr.std() / nr.mean():.3f}, min {nr.min():.3f}, max {nr.max():.3f}")
    ok &= bool((nr > 0).all()) and bool(torch.isfinite(nr).all()) and float(nr.std() / nr.mean()) > 0.01
else:
    ref = sum(wi * o for wi, o in zip(w, outs)) / sum(w)
    if conf.get("normalize") or any(israw):
        ref = F.normalize(ref, dim=-1)
if conf.get("gate"):
    from models_v2 import QualityGate
    gate = QualityGate()
    gate.load_state_dict(torch.load(sub / conf["gate"]["file"], map_location="cpu"))
    gate.eval()
    with torch.no_grad():
        q = torch.sigmoid(gate(Xd - Xd.mean(1, keepdim=True))).clamp_min(float(conf["gate"].get("floor", 0.02)))
    ref = ref * q.pow(float(conf["gate"].get("gamma", 1.0)))[:, None]
    print(f"porte : probabilité médiane {q.median():.3f}, p10 {q.quantile(0.1):.3f}, min {q.min():.3f}")
    ok &= bool(torch.isfinite(q).all())
if conf["center"] == "batch":
    ref = ref - ref.mean(0, keepdim=True)
elif conf["center"] == "fixed" and conf.get("has_center_vec"):
    ref = ref - torch.from_numpy(np.load(sub / "center.npy"))
d = (p1 - ref).abs().max().item()
print("écart soumission / réseau direct", f"{d:.2e}")
ok &= d < 1e-4 * max(1.0, float(ref.abs().max()))

perm = np.random.default_rng(0).permutation(32)
X2 = torch.cat([X[:, perm], torch.randn(len(X), 1, 120)], 1)
p2 = load([names[i] for i in perm] + ["T7"]).predict(X2)
d2 = (p1 - p2).abs().max().item()
print("canaux mélangés + canal en trop : écart", f"{d2:.2e}")
ok &= d2 < 1e-4

p3 = load(names[1:]).predict(X[:, 1:])
print("Cz absent : forme", tuple(p3.shape))
p4 = load(names).predict(F.interpolate(X, size=121, mode="linear", align_corners=True))
print("T = 121 : forme", tuple(p4.shape))
p5 = load(names).predict(X[:7])
print("lot de 7 : forme", tuple(p5.shape))
ok &= p3.shape == p4.shape == (512, 1536) and p5.shape == (7, 1536)

# aucun état conservé d'un appel à l'autre (FAQ) : deux appels identiques, statistiques BatchNorm intactes
mdl = load(names)
before = [b.running_mean.clone() for net, _, _ in mdl.nets for b in net.modules()
          if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
q1 = mdl.predict(X)
mdl.predict(X[256:] * 2)
q2 = mdl.predict(X)
after = [b.running_mean for net, _, _ in mdl.nets for b in net.modules()
         if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
d6 = (q1 - q2).abs().max().item()
same = all(torch.equal(u, v) for u, v in zip(before, after))
print("sans mémoire entre appels : écart", f"{d6:.2e}", "; statistiques BatchNorm intactes", same)
ok &= d6 < 1e-6 and same
# des essais non finis (NaN, inf) ne doivent ni donner une sortie non finie ni contaminer le reste du lot
Xn = X[:64].clone()
Xn[5] = float("nan")
Xn[9, 3, 40:] = float("inf")
p7 = torch.as_tensor(mdl.predict(Xn)).float()
fin7 = bool(torch.isfinite(p7).all()) and float(p7.norm(dim=-1).min()) > 0
print("essais non finis dans un lot de 64 : sortie finie", fin7)
ok &= fin7
if sub.name.endswith("__part"):
    import shutil
    shutil.rmtree(sub, ignore_errors=True)
print("OK" if ok else "ÉCHEC")
