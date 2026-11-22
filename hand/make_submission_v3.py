"""Emballe une soumission v3 (20/11) = make_submission.py + modèles en NORME LIBRE (config.json : rawnorm).
La plateforme moyenne les sorties BRUTES des répétitions d'un même (sujet, image) : la norme de la sortie est le
poids de l'essai. Un modèle « rawnorm » sort son vecteur brut, divisé par sa norme moyenne (constante par modèle,
mesurée ici sur des fenêtres d'entraînement) ; les modèles normalisés sortent une direction unitaire.
  --combine rawsum  : sortie = somme des bruts + c(x) x somme des directions normalisées, c(x) = norme relative
                      moyenne des modèles bruts sur cet essai (confiance de l'essai)
  --combine dirconf : sortie = c(x)^gamma x direction de l'ensemble (toutes les directions normalisées)
  --gamma g         : exposant de la confiance (1 = norme apprise ; 0 = sortie normalisée)
Sans modèle rawnorm : identique à make_submission.py.

Emballe une soumission v2 : un ou plusieurs modèles de outputs/v2/<run>/.

python hand/make_submission.py --out submissions/04_xxx --runs tsconv_all [atm_all ...]
       [--ckpt best|last] [--center none|batch|fixed] [--center_vec chemin.npy]

submission.py = architectures de hand/models_v2.py (copiées) + solveur. Prédiction :
moyenne des embeddings normalisés des modèles, puis centrage optionnel (moyenne du lot si le
lot a >= 32 époques, sinon vecteur fixe s'il est fourni). Canaux retrouvés par nom ; si
moins de la moitié des noms correspondent et que le nombre de canaux est le même, on garde
l'ordre reçu (même montage). Durée différente : rééchantillonnage linéaire.
"""
import argparse, json, shutil, zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--weights", nargs="*", type=float, default=None,
               help="poids de chaque modèle dans la moyenne des embeddings normalisés (défaut : 1)")
p.add_argument("--ckpt", default="best")
p.add_argument("--center", default="none", choices=["none", "batch", "fixed"])
p.add_argument("--center_vec", default=None)
p.add_argument("--half", action="store_true", help="poids en float16 dans le ZIP (REVE : 280 -> 140 Mo)")
p.add_argument("--normalize", action="store_true",
               help="sortie normalisée (L2) : la plateforme moyenne les sorties brutes sur les répétitions")
p.add_argument("--tta", default="0", help="décalages temporels moyennés au test, ex. -2,0,2 (défaut : aucun)")
p.add_argument("--nozip", action="store_true", help="dossier seulement (puis test, puis hand/zip_move.py)")
p.add_argument("--compute", default="", help="estimation du calcul d'entraînement, déclarée dans l'en-tête (règle 4)")
p.add_argument("--move", default="", help="runs (séparés par des virgules) dont le fichier de poids float16 est DÉPLACÉ dans la soumission (disque)")
p.add_argument("--gate", default="", help="porte de qualité par essai (state dict de models_v2.QualityGate) ; sortie x sigmoid(porte)^gate_gamma")
p.add_argument("--gate_gamma", type=float, default=2.0)
p.add_argument("--gate_floor", type=float, default=0.02, help="plancher de la probabilité de la porte")
p.add_argument("--detrend", action="store_true",
               help="retire la dérive linéaire de chaque canal (puis ligne de base) avant tous les réseaux : neutre sur données propres, protège des dérives lentes")
p.add_argument("--conv_zero", type=float, default=0.0,
               help="réseaux convolutifs : met à zéro, avant la référence moyenne, les canaux dont l'écart-type dépasse r fois la médiane des canaux de l'essai (0 = non)")
p.add_argument("--align", type=int, default=-1,
               help="alignement de la moyenne par lot des modèles REVE : -1 = non ; 0 = décalage complet ; K > 0 = décalage restreint aux K premières composantes principales des directions de référence")
p.add_argument("--align_lam", type=float, default=1.0)
p.add_argument("--align_tau", type=float, default=0.0,
               help="0 : décalage commun au lot (moyenne du lot) ; > 0 : décalage par essai, moyenne des autres essais du lot pondérée par la ressemblance des empreintes de sujet (softmax(cos / tau))")
p.add_argument("--subj_embed", default="", help="state dict de models_infer.SubjectEmbed (requis si --align_tau > 0)")
p.add_argument("--align_min", type=int, default=32, help="taille de lot minimale pour aligner")
p.add_argument("--align_js", action="store_true",
               help="force adaptative par lot et par modèle (rétrécissement de type James-Stein) : --align_lam fois la part de l'écart observé qui dépasse le bruit d'estimation de la moyenne")
p.add_argument("--n_align", type=int, default=1024, help="fenêtres d'entraînement pour la référence")
p.add_argument("--combine", default="rawsum", choices=["rawsum", "dirconf"])
p.add_argument("--gamma", type=float, default=1.0)
p.add_argument("--n_scale", type=int, default=512, help="fenêtres d'entraînement pour mesurer la norme moyenne des modèles bruts")
p.add_argument("--adabn", type=int, default=0,
               help="AdaBN : BatchNorm sur les statistiques du lot dès cette taille de lot (0 = non)")
a = p.parse_args()

out = ROOT / a.out
out.mkdir(parents=True, exist_ok=True)
fast = json.load(open(ROOT / "hand" / "fast" / "recs.json"))
ch_names = json.load(open(ROOT / "outputs" / "eegnet_clip_alljoined" / "meta.json"))["ch_names"]
models = []
for k, spec in enumerate(a.runs):
    run, _, ck = spec.partition("@")    # « revef_c@best » : point de contrôle propre à ce modèle
    cfg = json.load(open(ROOT / "outputs" / "v2" / run / "config.json"))
    src = ROOT / "outputs" / "v2" / run / f"{ck or a.ckpt}.pt"
    if not src.exists():    # proxy sans last.pt (ex. reve_d) : on prend best.pt, celui du proxy
        src = src.with_name("best.pt")
        print(f"{run} : pas de {a.ckpt}.pt, best.pt utilisé")
    if a.half:      # float16 ; recopié en float32 dans le réseau au chargement
        import os
        import torch
        sd = torch.load(src, map_location="cpu", weights_only=True)
        dst = out / f"net{k}.pt"
        if dst.exists():
            dst.unlink()
        if all(v.dtype != torch.float32 for v in sd.values()):     # déjà en float16 : déplacé ou lié, pas recopié
            if run in a.move.split(","):
                os.replace(src.resolve(), dst)
            else:
                try:
                    os.link(src.resolve(), dst)
                except OSError:
                    shutil.copy(src, dst)
        else:
            torch.save({n: (v.half() if v.is_floating_point() else v) for n, v in sd.items()}, dst)
        del sd
    else:
        shutil.copy(src, out / f"net{k}.pt")
    models.append(dict(file=f"net{k}.pt", arch=cfg["arch"], zscore=bool(cfg.get("zscore")),
                       car=bool(cfg.get("car")), run=run, weight=a.weights[k] if a.weights else 1.0,
                       raw=bool(cfg.get("rawnorm")), scale=1.0))
if any(m["raw"] for m in models) or a.align >= 0:
    # Sur des fenêtres d'entraînement : norme moyenne de chaque modèle brut (constante d'échelle) et, pour l'alignement
    # par lot, référence de chaque modèle REVE (moyenne de ses directions de sortie et leurs composantes principales).
    import sys
    import torch
    sys.path.insert(0, str(ROOT / "hand"))
    from models_v2 import build_net
    FAST = ROOT / "hand" / "fast"
    ix = np.load(FAST / "index.npz")
    rec_base = np.concatenate([[0], np.cumsum([r["n"] for r in fast])]).astype(np.int64)
    i0 = np.round(ix["start"] * 120).astype(np.int64)
    okw = (i0 >= 0) & (i0 + 120 <= (rec_base[1:] - rec_base[:-1])[ix["rec"]]) & (ix["split"] < 2)
    n_ref = max(a.n_scale, a.n_align if a.align >= 0 else 0)
    pick = np.sort(np.random.default_rng(0).choice(np.where(okw)[0], n_ref, replace=False))
    g = rec_base[ix["rec"][pick]] + i0[pick]
    big = np.load(FAST / "signal_f16.npy", mmap_mode="r")
    xw = torch.from_numpy(big[g[:, None] + np.arange(120)].astype(np.float32)).transpose(1, 2)
    xw = (xw - xw[:, :, :24].mean(-1, keepdim=True)).clamp(-20, 20)
    if a.detrend:
        t_ = torch.linspace(-1, 1, 120)
        xw = xw - (xw * t_).sum(-1, keepdim=True) / (t_ * t_).sum() * t_
        xw = xw - xw[:, :, :24].mean(-1, keepdim=True)
    for k, m in enumerate(models):
        isreve = str(m["arch"]).startswith("reve")
        if not (m["raw"] or (a.align >= 0 and isreve)):
            continue
        net = build_net(m["arch"])
        sd = torch.load(out / m["file"], map_location="cpu", weights_only=True)
        net.load_state_dict({n: v.float() if v.is_floating_point() else v for n, v in sd.items()})
        net.eval()
        xin = xw - xw.mean(1, keepdim=True) if m["car"] else xw
        if m["zscore"]:
            xin = (xin - xin.mean(-1, keepdim=True)) / (xin.std(-1, keepdim=True) + 1e-3)
        with torch.no_grad():
            o = torch.cat([net(xin[j:j + 256]).float() for j in range(0, len(xin), 256)])
        nr = o.norm(dim=-1)
        if m["raw"]:
            m["scale"] = float(1.0 / nr.mean())
            print(f"{m['run']} : norme moyenne {nr.mean():.3f} (cv {nr.std() / nr.mean():.3f}) -> échelle {m['scale']:.5f}")
        if a.align >= 0 and isreve:
            u = o / nr[:, None].clamp_min(1e-12)
            mref = u.mean(0)
            V = torch.pca_lowrank(u - mref, q=max(a.align, 1), niter=4)[2][:, :a.align] if a.align > 0 else torch.zeros(1536, 0)
            torch.save(dict(m=mref.half(), V=V.half()), out / f"align{k}.pt")
            m["align"] = f"align{k}.pt"
        del net, sd
conf = dict(ch_names=ch_names, n_times=120, models=models, center=a.center, has_center_vec=False,
            adabn=a.adabn, normalize=a.normalize, tta=[int(v) for v in a.tta.split(",")],
            combine=a.combine, gamma=a.gamma, norm_clip=[0.05, 20.0], detrend=bool(a.detrend), conv_zero=a.conv_zero,
            align=dict(K=a.align, lam=a.align_lam, min=a.align_min, tau=a.align_tau, js=bool(a.align_js)) if a.align >= 0 else None)
if a.align >= 0 and a.align_tau > 0:
    import torch
    esd = torch.load(a.subj_embed, map_location="cpu", weights_only=True)
    torch.save(esd, out / "subj.pt")
    conf["align"].update(embed="subj.pt", Fm=int(esd["temporal.weight"].shape[0]), d=int(esd["emb.weight"].shape[0]),
                         n_cls=int(esd["cls.weight"].shape[0]))
if a.gate:
    import torch
    gsd = torch.load(a.gate, map_location="cpu", weights_only=True)
    torch.save(gsd, out / "gate.pt")
    conf["gate"] = dict(file="gate.pt", gamma=a.gate_gamma, floor=a.gate_floor)
if a.center_vec:
    np.save(out / "center.npy", np.load(a.center_vec).astype(np.float32))
    conf["has_center_vec"] = True
json.dump(conf, open(out / "config.json", "w"), indent=1)

SOLVER = '''

# --- Solveur ---
import json

from torch.nn import functional as F

from benchmark_utils.base_solver import CompetSolver


class EnsembleV2:      # v3 : modèles en norme libre (conf["models"][i]["raw"]), voir make_submission_v3.py
    def __init__(self, d, conf, ch_names, device):
        self.device, self.conf = device, conf
        self.n_times = conf["n_times"]
        self.nets = []
        for m in conf["models"]:
            net = build_net(m["arch"], n_chans=len(conf["ch_names"]), n_times=self.n_times)
            net.load_state_dict(torch.load(d / m["file"], map_location="cpu", weights_only=True))
            self.nets.append((net.to(device).eval(), m.get("car", False), m["zscore"]))
        self.gate = None
        if conf.get("gate"):      # porte de qualité par essai : sous-pondère les essais sans réponse évoquée exploitable
            self.gate = QualityGate(n_chans=len(conf["ch_names"]), n_times=self.n_times)
            self.gate.load_state_dict(torch.load(d / conf["gate"]["file"], map_location="cpu", weights_only=True))
            self.gate = self.gate.to(device).eval()
        self.raw = [bool(m.get("raw")) for m in conf["models"]]
        self.isconv = [not str(m["arch"]).startswith("reve") for m in conf["models"]]
        self.subj = None
        al = conf.get("align") or {}
        if al.get("embed"):
            self.subj = SubjectEmbed(n_chans=len(conf["ch_names"]), n_cls=int(al["n_cls"]), Fm=int(al["Fm"]), d=int(al["d"]))
            self.subj.load_state_dict(torch.load(d / al["embed"], map_location="cpu", weights_only=True))
            self.subj = self.subj.to(device).eval()
        self.aref = []      # alignement par lot : référence (moyenne des directions, composantes principales) par modèle REVE
        for m in conf["models"]:
            r = None
            if conf.get("align") and m.get("align"):
                t = torch.load(d / m["align"], map_location="cpu", weights_only=True)
                r = (t["m"].float().to(device), t["V"].float().to(device))
            self.aref.append(r)
        self.nclip = [float(v) for v in conf.get("norm_clip", [0.05, 20.0])]
        self.scale = [float(m.get("scale", 1.0)) for m in conf["models"]]
        self.w = [float(m.get("weight", 1.0)) for m in conf["models"]]
        train_ch = conf["ch_names"]
        upper = [str(c).upper() for c in ch_names]
        idx = [upper.index(c.upper()) if c.upper() in upper else -1 for c in train_ch]
        found = sum(i >= 0 for i in idx)
        if found < len(train_ch) // 2 and len(ch_names) == len(train_ch):
            idx = list(range(len(train_ch)))
            print("[v2] noms de canaux non reconnus : ordre reçu conservé")
        self.idx = idx
        self.adabn = int(conf.get("adabn", 0))
        print(f"[v2] {sum(i >= 0 for i in idx)}/{len(train_ch)} canaux ; {len(self.nets)} modèle(s) ; "
              f"centrage {conf['center']} ; AdaBN {self.adabn or 'non'} ; décalages {conf.get('tta', [0])} ; "
              f"sortie normalisée {bool(conf.get('normalize'))}")
        self.center_vec = None
        if conf.get("has_center_vec"):
            import numpy as np
            self.center_vec = torch.from_numpy(np.load(d / "center.npy")).to(device)

    def _kernel(self, sel):
        """Poids (B, B) servant à estimer, pour chaque essai, la sortie moyenne des autres essais « du même
        participant » dans le lot : softmax de la ressemblance des empreintes de sujet (diagonale exclue), ou poids
        uniformes (moyenne du lot) si tau = 0. None si l'alignement est inactif ou le lot trop petit."""
        al = self.conf.get("align")
        B = sel.shape[0]
        if not al or B < int(al.get("min", 32)):
            return None
        if self.subj is None or float(al.get("tau", 0.0)) <= 0:
            return (1.0 - torch.eye(B, device=sel.device)) / (B - 1)
        e = self.subj.embed(sel - sel.mean(1, keepdim=True)).float()
        sim = e @ e.T / float(al["tau"])
        sim.fill_diagonal_(-1e9)
        return torch.softmax(sim, dim=1)

    def _align(self, o, ref, W):
        """Alignement dans le lot (statistiques du lot courant, rien de gardé) : pour chaque essai, la direction
        moyenne des sorties des essais voisins (poids W) est comparée à la moyenne de référence du modèle (mesurée
        sur les données d'entraînement) ; l'écart, restreint aux composantes principales si K > 0, est retiré de la
        direction de l'essai. La norme de chaque sortie est conservée."""
        if ref is None or W is None:
            return o
        mref, V = ref
        nr = o.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = o / nr
        dlt = W @ u - mref
        lam = float(self.conf["align"].get("lam", 1.0))
        if self.conf["align"].get("js"):
            # Force adaptative (rétrécissement de type James-Stein, par lot et par modèle) : l'écart observé contient
            # le décalage réel plus le bruit d'estimation de la moyenne, (1 - |moyenne des directions|^2) x somme des
            # poids^2. On ne retire que la part qui dépasse ce bruit : sans décalage réel, la correction s'éteint.
            eps2 = (1.0 - u.mean(0).pow(2).sum()).clamp_min(0.0) * W.pow(2).sum(1).mean()
            lam = lam * float((1.0 - eps2 / dlt.pow(2).sum(1).mean().clamp_min(1e-12)).clamp(0.0, 1.0))
        if V.shape[1] > 0:
            dlt = (dlt @ V) @ V.T
        return (u - lam * dlt) * nr

    def _run(self, net, xin):
        """AdaBN (adaptation sans étiquette autorisée par la FAQ : statistiques du lot courant) : les BatchNorm
        normalisent avec la moyenne et la variance du lot ; leurs statistiques d'entraînement sont restaurées
        après l'appel, rien n'est conservé d'un lot à l'autre. Petits lots : statistiques d'entraînement."""
        bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if not bns:              # sans BatchNorm (REVE) : par morceaux de 1 024, mémoire bornée
            return torch.cat([net(xin[j:j + 1024]) for j in range(0, len(xin), 1024)])
        if not self.adabn or xin.shape[0] < self.adabn:
            return net(xin)
        saved = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone(), m.momentum)
                 for m in bns]
        try:
            for m in bns:
                m.train(True)
                m.momentum = 0.0
            return net(xin)
        finally:
            for m, (rm, rv, nb, mo) in zip(bns, saved):
                m.running_mean.copy_(rm)
                m.running_var.copy_(rv)
                m.num_batches_tracked.copy_(nb)
                m.momentum = mo
                m.train(False)

    @staticmethod
    def _shift(x, s):
        """Décalage de s échantillons dans la fenêtre reçue, bord opposé recopié (ligne de base ou fin d'époque)."""
        if s == 0:
            return x
        if s > 0:
            return torch.cat([x[..., s:], x[..., -1:].expand(*x.shape[:-1], s)], -1)
        return torch.cat([x[..., :1].expand(*x.shape[:-1], -s), x[..., :s]], -1)

    @torch.no_grad()
    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)      # un essai non fini ne doit pas contaminer le lot (AdaBN, alignement)
        B, _, T = X.shape
        sel = torch.zeros(B, len(self.idx), T, dtype=torch.float32, device=self.device)
        for k, i in enumerate(self.idx):
            if i >= 0:
                sel[:, k] = X[:, i]
        if T != self.n_times:
            sel = F.interpolate(sel, size=self.n_times, mode="linear", align_corners=True)
        W = self._kernel(sel)             # voisinage par empreinte de sujet, sur l'entrée telle que reçue
        if self.conf.get("detrend"):      # dérive linéaire par canal retirée, puis ligne de base (200 premières ms)
            t = torch.linspace(-1, 1, sel.shape[-1], device=sel.device)
            sel = sel - (sel * t).sum(-1, keepdim=True) / (t * t).sum() * t
            sel = sel - sel[:, :, :max(1, round(0.2 * sel.shape[-1]))].mean(-1, keepdim=True)
        outs = []
        for (net, car, z), raw, sc, isconv, aref in zip(self.nets, self.raw, self.scale, self.isconv, self.aref):
            xin = sel
            if isconv and float(self.conf.get("conv_zero", 0)) > 0:      # canaux aberrants (bruit fort, saut) mis à zéro :
                sd = xin.std(-1)                                         # sinon la référence moyenne les propage partout
                xin = xin * (sd <= float(self.conf["conv_zero"]) * sd.median(1, keepdim=True).values)[:, :, None]
            if car:
                xin = xin - xin.mean(1, keepdim=True)
            if z:
                xin = (xin - xin.mean(-1, keepdim=True)) / (xin.std(-1, keepdim=True) + 1e-3)
            if raw:      # norme libre : vecteur brut, ramené à une norme moyenne de 1 (constante du modèle) ;
                o = self._align(torch.nan_to_num(self._run(net, xin).float() * sc), aref, W)   # norme bornée : aucun essai ni
                nr = o.norm(dim=-1, keepdim=True)                            # modèle ne peut écraser les autres
                if float(self.conf.get("gamma", 1.0)) == 0.0:      # exposant nul : direction seule, comme un modèle normalisé
                    outs.append(o / nr.clamp_min(1e-12))
                else:
                    outs.append(o / nr.clamp_min(1e-12) * nr.clamp(self.nclip[0], self.nclip[1]))
                continue
            tta = self.conf.get("tta", [0])
            o = sum(F.normalize(self._run(net, self._shift(xin, s)).float(), dim=-1) for s in tta)
            outs.append(F.normalize(self._align(F.normalize(o, dim=-1), aref, W), dim=-1))
        if any(self.raw) and float(self.conf.get("gamma", 1.0)) != 0.0:
            # la plateforme moyenne les sorties brutes des répétitions : la norme rendue pondère l'essai
            wr = sum(w for w, r in zip(self.w, self.raw) if r)
            c = sum(w * o.norm(dim=-1, keepdim=True) for w, o, r in zip(self.w, outs, self.raw) if r) / wr
            g = float(self.conf.get("gamma", 1.0))
            if self.conf.get("combine", "rawsum") == "dirconf":
                p = F.normalize(sum(w * F.normalize(o, dim=-1) for w, o in zip(self.w, outs)), dim=-1) * c.pow(g)
            else:
                p = sum(w * o for w, o, r in zip(self.w, outs, self.raw) if r)
                if g != 1.0:
                    p = p * c.pow(g - 1.0)
                old = [w * o for w, o, r in zip(self.w, outs, self.raw) if not r]
                if old:
                    p = p + c.pow(g) * sum(old)
                p = p / sum(self.w)
        else:
            p = sum(w * o for w, o in zip(self.w, outs)) / sum(self.w)
            if self.conf.get("normalize") or any(self.raw):
                p = F.normalize(p, dim=-1)
        if self.gate is not None:
            g = self.conf["gate"]
            q = torch.sigmoid(self.gate(sel - sel.mean(1, keepdim=True)).float()).clamp_min(float(g.get("floor", 0.02)))
            p = p * q.pow(float(g.get("gamma", 1.0)))[:, None]
        c = self.conf["center"]
        if c == "batch" and B >= 32:
            p = p - p.mean(0, keepdim=True)
        elif c in ("batch", "fixed") and self.center_vec is not None:
            p = p - self.center_vec
        return p


class Solver(CompetSolver):

    name = "EEG-DINOv2-v2"

    def load_model(self, meta):
        d = meta["submission_dir"]
        conf = json.loads((d / "config.json").read_text())
        return EnsembleV2(d, conf, list(meta["ch_names"]), meta["device"])
'''
uses_things = any(json.load(open(ROOT / "outputs" / "v2" / r.partition("@")[0] / "config.json")).get("things") for r in a.runs)
data = ("Alljoined-1.6M et THINGS-EEG2 (données publiques recommandées par la FAQ de la piste ;\n"
        "32 électrodes Emotiv retrouvées par nom, fenêtres THINGS recalées dans le temps sur Alljoined)"
        if uses_things else "Alljoined-1.6M uniquement")
archs = sorted({m['arch'] for m in models})
reve = ("\nREVE : initialisé depuis les poids publics brain-bzh/reve-base (Hugging Face, pré-entraînés par leurs auteurs),\n"
        "puis affiné de bout en bout sur ces données." if any(x.startswith("reve") for x in archs) else "")
comp = f"\nCalcul d'entraînement (estimation) : {a.compute}." if a.compute else ""
header = (f'"""Encodeurs EEG -> DINOv2-giant ({", ".join(archs)} ; perte CLIP), entraînés hors plateforme sur\n'
          f'{data} (32 canaux Emotiv, 1 s à 120 Hz).{reve}{comp}\nVoir config.json."""\n')
src = (ROOT / "hand" / "models_v2.py").read_text() + "\n\n" + (ROOT / "hand" / "models_infer.py").read_text()
(out / "submission.py").write_text(header + src + SOLVER)
z = out.with_suffix(".zip")
if not a.nozip:
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for q in sorted(out.iterdir()):
            f.write(q, q.name)
    print(z, f"{z.stat().st_size / 1e6:.1f} Mo", [m["run"] for m in models], a.center)
else:
    print(out, "(sans ZIP)", [m["run"] for m in models], a.center)
