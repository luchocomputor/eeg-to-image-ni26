"""Samedi 21/11, pipeline automatique (20/11 au soir, direction du challenge). Voir samedi_v3.sh et le BILAN.

Attente des deux noyaux Kaggle -> journaux et santé de chaque run -> arbitrage par les proxys du jour -> sélection ->
téléchargement des SEULS poids retenus (float16) -> hand/make_submission_v3.py -> hand/test_submission_v3.py (par
sous-ensembles : mémoire du Mac) -> zip -> submissions/PRET_ZIP15.txt.
Ne pousse rien sur Kaggle, n'envoie rien sur Codabench, ne supprime que son dossier temporaire kaggle/_dl_samedi.
Relançable sans risque (les modèles déjà rangés dans outputs/v2 sont gardés).

Volets (lanceurs : kaggle/outils/gen_samedi_v3.sh) :
  pari A  final_nba_sNN  affinage doux et long (cibles dures dans le lot, norme libre, lots de 64, tronc 1e-4, 5 passes ;
                         points de contrôle des passes 3 et 5) .......................... proxy nba_p
  pari B  final_nbm_sNN  le même + moyenne d'essais de la même image (--mix 0.5 --mixk 6) . proxy nbm_p
  socle   rwft_<final>   conversion courte d'un final existant en norme libre ; à défaut, les anciens finals normalisés
Arbitrage (rang percentile agrégé du proxy du jour, sorties normalisées, galerie de 1 500 images, repère reve_d) : un
pari entre s'il n'est pas sous le repère de plus de la marge, prend double poids s'il le dépasse nettement, sinon reste
dehors ; sans proxy il entre à demi-poids. On construit et on propose TOUJOURS un ZIP (l'envoi est un tir gratuit : la
meilleure note reste acquise) : au pire le socle seul.

Réglages : kaggle/outils/plan_samedi.json.
  --deadline HH:MM   heure limite d'attente des noyaux (défaut : plan) ; ensuite on construit avec ce qui est là
  --max_reve N       nombre maximal de REVE-base dans le ZIP (défaut : plan)
  --no_wait          ne pas attendre : récupérer ce qui est fini et construire
  --no_fetch         ne rien demander à Kaggle (construire avec outputs/v2 tel quel)
  --plan             afficher la sélection et s'arrêter (aucun poids téléchargé, aucun fichier écrit)
  --gamma G          force l'exposant de la norme (0 = sorties normalisées comme les ZIP 12 et 14)
  --no_gate          sans porte de qualité
"""
import argparse, datetime, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2, KG = Path(os.environ.get("NI_V2", ROOT / "outputs" / "v2")), ROOT / "kaggle"     # NI_V2 : essais seulement
PY = str(ROOT / ".venv" / "bin" / "python")
PY310 = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
KAGGLE = "/Library/Frameworks/Python.framework/Versions/3.10/bin/kaggle"
TMP = Path(os.environ.get("NI_TMP", KG / "_dl_samedi"))     # seul dossier temporaire du script
PLAN = json.load(open(os.environ.get("NI_PLAN", KG / "outils" / "plan_samedi.json")))

p = argparse.ArgumentParser()
p.add_argument("--deadline", default=PLAN["deadline"])
p.add_argument("--max_reve", type=int, default=PLAN["max_reve"])
p.add_argument("--kernels", default=",".join(PLAN["kernels"]))
p.add_argument("--no_wait", action="store_true")
p.add_argument("--no_fetch", action="store_true")
p.add_argument("--plan", action="store_true")
p.add_argument("--gamma", type=float, default=None)
p.add_argument("--no_gate", action="store_true")
p.add_argument("--out", default="")
p.add_argument("--log", default="logs/test15.log")
p.add_argument("--pret", default="submissions/PRET_ZIP15.txt")
a = p.parse_args()
FIXE_S, PAR_REVE_S, MAX_S = 330, 290, 5500          # temps plateforme estimé (BILAN) ; limite 7 200 s
BEST = "ZIP 12 (0,000826)"


def say(*x):
    print(time.strftime("%d/%m %H:%M:%S"), *x, flush=True)


def run(cmd, **kw):
    try:
        return subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired as e:      # appel réseau figé (coupure, veille du Mac) : compté comme un échec
        so = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(e.cmd, 124, stdout=so, stderr=f"délai dépassé ({e.timeout:.0f} s)")


def scrub(s):
    return re.sub(r"KGAT_\S+", "<jeton>", s or "")


# ------------------------------------------------------------------ type, santé et courbe d'un run
def kind_of(cfg):
    final, ft = cfg.get("holdout") == 0, bool(cfg.get("init"))
    ep, frac = int(cfg.get("epochs", 0)), float(cfg.get("epoch_frac") or 1.0)
    if cfg.get("rawnorm") and int(cfg.get("bs", 512)) == 64 and not cfg.get("sym") and not ft:
        k = "nb"
    elif cfg.get("rawnorm") and ft and ep == 1 and frac < 1:
        k = "ft"
    else:
        k = "autre"
    return ("" if final else "proxy_") + k


def curve(log):
    """Par passe : perte, et (proxy) rang percentile et top-5 agrégés, sorties normalisées et brutes."""
    out = []
    for m in re.finditer(r"^PASSE (\d+) loss ([\d.]+) \([^)]*\) (\{.*\})\s*$", log, re.M):
        try:
            res = json.loads(m.group(3))
        except Exception:
            res = {}
        out.append(dict(ep=int(m.group(1)) + 1, loss=float(m.group(2)),
                        pct=res.get("proxy", {}).get("pct"), top5=res.get("proxy", {}).get("top5"),
                        pct_brut=res.get("proxy_brut", {}).get("pct"), top5_brut=res.get("proxy_brut", {}).get("top5"),
                        essai_top5=res.get("proxy_essai", {}).get("top5")))
    return out


def health(d):
    """(sain, raison, infos) d'après config.json et log.txt."""
    try:
        cfg = json.load(open(d / "config.json"))
        log = (d / "log.txt").read_text()
    except Exception as e:
        return False, f"journal ou config illisible ({e!r})", {}
    log = log[log.rfind("=== "):] if "=== " in log else log       # dernier lancement seulement
    cv = [float(v) for v in re.findall(r"norme [\d.]+ cv ([\d.]+)", log)]
    cur = curve(log)
    losses = [c["loss"] for c in cur]
    kind = kind_of(cfg)
    info = dict(final=cfg.get("holdout") == 0, epochs=int(cfg.get("epochs", 0)), arch=cfg.get("arch"), losses=losses,
                kind=kind, raw=bool(cfg.get("rawnorm")), cv=cv[-1] if cv else None, curve=cur,
                frac=float(cfg.get("epoch_frac") or 1.0), base=Path(str(cfg.get("init") or "")).parent.name)
    th = PLAN["health"].get(kind.replace("proxy_", ""))
    if th is None:
        return False, f"type non prévu ({kind})", info
    if "Traceback" in log or re.search(r"loss nan", log):
        return False, "erreur ou NaN dans le journal", info
    need = info["epochs"] - (1 if "limite de temps atteinte" in log and info["epochs"] >= 4 else 0)     # coupé proprement en fin de course
    if len(losses) < need or "FINI" not in log:
        return False, f"inachevé ({len(losses)}/{info['epochs']} passes)", info
    if losses[0] > th["first_max"]:
        return False, f"1re passe à {losses[0]:.3f} (> {th['first_max']})", info
    if losses[-1] > th["last_max"]:
        return False, f"dernière passe à {losses[-1]:.3f} (> {th['last_max']})", info
    if any(b > x_ + 0.02 for x_, b in zip(losses, losses[1:])):
        return False, f"perte qui remonte {losses}", info
    if info["cv"] is not None and not (th.get("cv_min", 0.0) <= info["cv"] <= th.get("cv_max", 9.0)):
        return False, f"dispersion de la norme anormale (cv {info['cv']:.3f})", info
    return True, "sain", info


# ------------------------------------------------------------------ Kaggle
def kernel_names(kdir):
    try:
        q = json.loads(re.search(r"^QUEUES = (.*)$", (KG / kdir / "runner.py").read_text(), re.M).group(1))
        names = [n for runs in q.values() for n, _ in runs]
    except Exception:
        names = []
    return names + [n + "_r" for n in names]


def status(kdir):
    slug = json.load(open(KG / kdir / "kernel-metadata.json"))["id"]
    r = run([KAGGLE, "kernels", "status", slug], timeout=180)
    s = scrub((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "")
    m = re.search(r"KernelWorkerStatus\.(\w+)", s)
    if m:
        for k in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
            if k in m.group(1).upper():
                return k
        return m.group(1).upper()
    return "? " + s[-120:]


MAN = {}          # run -> dict(kdir, ok, why, info, local, d, files)


def is_fake(d):
    """Faux run de répétition générale (journal commençant par « === faux ») : jamais utilisé hors --no_fetch."""
    try:
        return (d / "log.txt").read_text()[:40].lstrip().startswith("=== faux")
    except Exception:
        return False


def fetch_meta(kdir):
    """Journaux, config (et sorties brutes des proxys) de chaque run d'un noyau fini ; aucun poids."""
    meta = KG / kdir / "kernel-metadata.json"
    r = run([PY310, "-I", KG / "outils" / "samedi_fetch.py", "list", meta], timeout=300)
    if r.returncode:
        say(f"{kdir} : liste des sorties impossible :", scrub(r.stderr)[-300:])
        return None
    files = [f for f in r.stdout.split("\n") if f]
    names = sorted({f.split("/")[1] for f in files if f.startswith("outputs/") and f.count("/") == 2})
    say(f"{kdir} : {len(files)} fichiers, runs {names}")
    st = TMP / kdir
    for n in names:
        mine = [f.split("/")[2] for f in files if f.startswith(f"outputs/{n}/")]
        if (V2 / n).exists() and is_fake(V2 / n):
            say(f"  {n} : faux run d'essai trouvé dans outputs/v2 : supprimé, le vrai est récupéré")
            shutil.rmtree(V2 / n)
        if (V2 / n / "config.json").exists() and any((V2 / n).glob("*.pt")):
            ok, why, info = health(V2 / n)
            MAN[n] = dict(kdir=kdir, ok=ok, why=why + " (déjà dans outputs/v2)", info=info, local=True, d=V2 / n, files=mine)
            say(f"  {n} : déjà dans outputs/v2 ({why})")
            continue
        shutil.rmtree(st / "outputs" / n, ignore_errors=True)
        pats = f"outputs/{n}/config.json,outputs/{n}/log.txt,outputs/{n}/best.json"
        if "preds_proxy.pt" in mine:
            pats += f",outputs/{n}/preds_proxy.pt"
        run([PY310, "-I", KG / "outils" / "samedi_fetch.py", "get", meta, st, pats], timeout=900)
        d = st / "outputs" / n
        ok, why, info = health(d)
        MAN[n] = dict(kdir=kdir, ok=ok, why=why, info=info, local=False, d=d, files=mine)
        say(f"  {n} : {why} ; type {info.get('kind')} ; pertes {info.get('losses')} ; cv norme {info.get('cv')}")
        if not ok:
            rej = ROOT / "logs" / "samedi_rejetes"
            rej.mkdir(parents=True, exist_ok=True)
            if (d / "log.txt").exists():
                shutil.copy(d / "log.txt", rej / f"{n}.log.txt")
    return names


def fetch_weights(n, w):
    """Télécharge le point de contrôle w d'un run retenu, le passe en float16 et range le run dans outputs/v2."""
    e = MAN[n]
    if (V2 / n / w).exists():
        return True
    if e["local"]:
        return False
    meta, st = KG / e["kdir"] / "kernel-metadata.json", TMP / e["kdir"]
    f = e["d"] / w
    r = run([PY310, "-I", KG / "outils" / "samedi_fetch.py", "get", meta, st, f"outputs/{n}/{w}"], timeout=1800)
    if r.returncode or not f.exists() or f.stat().st_size < 50e6:
        say(f"  {n} : téléchargement de {w} raté :", scrub(r.stdout + r.stderr)[-300:])
        return False
    r = run([PY, "-I", KG / "outils" / "to_half.py", f])
    if r.returncode:
        say(f"  {n} : poids illisibles :", r.stderr[-300:])
        return False
    (V2 / n).mkdir(parents=True, exist_ok=True)
    for q in e["d"].iterdir():
        if q.name != "preds_proxy.pt":
            shutil.move(str(q), str(V2 / n / q.name))
    e.update(local=True, d=V2 / n)
    say(f"  {n} : {w} rangé dans outputs/v2 ({(V2 / n / w).stat().st_size / 1e6:.0f} Mo)")
    return True


def wait_and_fetch(kdirs):
    h, m = map(int, a.deadline.split(":"))
    deadline = datetime.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    pending, last = list(kdirs), {}
    say(f"attente des noyaux {pending} jusqu'à {deadline:%d/%m %H:%M}" + (" (sans attente)" if a.no_wait else ""))
    while pending:
        for k in list(pending):
            s = status(k)
            if s != last.get(k):
                say(f"{k} : {s}")
                last[k] = s
            if s in ("COMPLETE", "ERROR", "CANCEL"):
                if fetch_meta(k) is not None:
                    pending.remove(k)
        if not pending or a.no_wait or datetime.datetime.now() >= deadline:
            break
        time.sleep(120)
    if pending:
        say(f"NON RÉCUPÉRÉS (pas finis ou inaccessibles à l'heure limite) : {pending} ; on construit sans eux")
    return pending


# ------------------------------------------------------------------ arbitrage par les proxys du jour
def arbitrate():
    """Pour chaque pari : (verdict, passe retenue, texte). Verdicts : double, dedans, dehors, minoritaire (pas de mesure)."""
    out = {}
    for bet, c in PLAN["bets"].items():
        e = MAN.get(c["proxy"])
        if e is None and (V2 / c["proxy"] / "log.txt").exists() and (a.no_fetch or not is_fake(V2 / c["proxy"])):      # proxy déjà rangé (relance)
            ok_, why_, info_ = health(V2 / c["proxy"])
            e = dict(info=info_)
        cur = [x for x in (e["info"].get("curve") or []) if x.get("pct") is not None] if e else []
        cand = [x for x in cur if x["ep"] in c["epochs"]]
        if not e or not cand:
            out[bet] = (c["missing"], c["epochs"][-1], f"proxy {c['proxy']} absent ou inachevé : règle par défaut « {c['missing']} »")
            continue
        best = max(cand, key=lambda x: x["pct"])
        d = best["pct"] - c["ref"]
        verdict = "double" if d >= c["double"] else "dedans" if d >= -c["margin"] else "dehors"
        txt = (f"proxy {c['proxy']} : " + ", ".join(f"passe {x['ep']} rang pct {x['pct']:.4f} top-5 {x['top5']:.4f} (brut {x['pct_brut']:.4f})" for x in cur)
               + f" ; repère {c['ref']:.4f} ; meilleure passe gardée {best['ep']} ; écart {d:+.4f} -> {verdict}")
        out[bet] = (verdict, best["ep"], txt)
    for bet, (v, ep, txt) in out.items():
        say(f"arbitrage {bet} : {v} (passe {ep}) | {txt}")
    return out


def guard():
    """Sorties brutes contre normalisées sur les proxys du jour (hand/apparie/brut.py, régime à 2 répétitions)."""
    files = [e["d"] / "preds_proxy.pt" for n, e in MAN.items() if e["ok"] and (e["d"] / "preds_proxy.pt").exists()]
    if not files:
        say("garde-fou : aucune sortie brute de proxy du jour ; on s'en tient aux mesures du 20/11")
        return None, ""
    TMP.mkdir(parents=True, exist_ok=True)
    r = run([PY, "-I", "hand/apparie/brut.py", *files, "--json", TMP / "garde_fou.json"])
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    for l in lines:
        say("  " + l[:420])
    m = re.search(r"VERDICT (\w+)", r.stdout)
    if r.returncode or not m:
        say("garde-fou : mesure impossible :", r.stderr[-300:])
        return None, ""
    return m.group(1), next((l for l in reversed(lines) if "k=2" in l and "%" in l), "")[:320]


# ------------------------------------------------------------------ sélection
def select(arb):
    """Renvoie [(run, point de contrôle, poids, groupe)] et la liste des anciens finals normalisés qui complètent."""
    ok_final = {n: e for n, e in MAN.items() if e["ok"] and e["info"].get("final")}
    for n, e in MAN.items():
        if not e["ok"]:
            say(f"écarté : {n} ({e['why']})")
    n_max = min(a.max_reve, (MAX_S - FIXE_S) // PAR_REVE_S)
    chosen = []
    for bet, c in PLAN["bets"].items():                     # paris, dans l'ordre du plan
        v, ep, _ = arb.get(bet, ("dehors", c["epochs"][-1], ""))
        w = {"double": 3.0, "dedans": 2.0, "minoritaire": 1.0}.get(v, 0.0)      # le pari pèse plus que le socle (tir gratuit)
        if not w:
            continue
        names = sorted((n for n in ok_final if re.fullmatch(c["pat"], n)), key=lambda n: (ok_final[n]["info"]["losses"][-1], n))
        for n in names[:c["cap"]]:
            has = lambda f: f in MAN[n]["files"] or (V2 / n / f).exists()
            ck = f"ep{ep}" if ep != MAN[n]["info"]["epochs"] and has(f"ep{ep}.pt") else "last"
            chosen.append((n, ck, w, c["label"]))
    # socle : conversions courtes, dans l'ordre de priorité des modèles de base
    prio = PLAN["base_priority"]
    fts = sorted((n for n in ok_final if ok_final[n]["info"]["kind"] == "ft"),
                 key=lambda n: prio.index(MAN[n]["info"]["base"]) if MAN[n]["info"]["base"] in prio else 99)
    covered = set()
    for n in fts:
        if len(chosen) >= n_max:
            break
        chosen.append((n, "last", 1.0, "socle"))
        covered.add(MAN[n]["info"]["base"])
    chosen = chosen[:n_max]
    if not chosen:      # secours : aucun modèle du jour ; anciens finals + les deux modèles en norme libre du 20/11 (confiance)
        for spec in PLAN.get("secours_raw", []):
            n, _, ck = spec.partition("@")
            if (V2 / n / f"{ck or 'last'}.pt").exists():
                cfg = json.load(open(V2 / n / "config.json"))
                MAN[n] = dict(kdir="", ok=True, why="secours", local=True, d=V2 / n, files=[q.name for q in (V2 / n).iterdir()],
                              info=dict(kind="secours", arch=cfg["arch"], losses=[], cv=None, epochs=0, final=False, base=""))
                chosen.append((n, ck or "last", 1.0, "secours"))
        if chosen:
            say("SECOURS : aucun modèle du jour n'est sain ; ZIP = anciens finals + confiance des modèles en norme libre du 20/11")
    old = []
    for spec in PLAN.get("fill_old", []):                   # anciens finals normalisés : complètent jusqu'à fill_to
        if len(chosen) + len(old) >= min(PLAN.get("fill_to", 0), n_max):
            break
        n, _, ck = spec.partition("@")
        if n in covered:
            continue
        f = V2 / n / ((ck or "last") + ".pt")
        if f.exists() or f.with_name("best.pt").exists():
            old.append(spec)
    return chosen, old


# ------------------------------------------------------------------ construction
def build(chosen, old, missing, arb, verdict, guard_line):
    pret = ROOT / a.pret
    head = f"ZIP 15, généré par kaggle/outils/samedi_v3.py le {time.strftime('%d/%m à %H:%M')}"
    arb_lines = [f"arbitrage {b} : {v} (passe {ep}) ; {t}" for b, (v, ep, t) in arb.items()]
    if not chosen:
        lines = [head, "RIEN CONSTRUIT : aucun modèle du jour n'est sain. Secours : envoyer le ZIP de repli s'il existe "
                 f"(voir BILAN), sinon ne rien envoyer : la meilleure note ({BEST}) reste acquise.",
                 f"noyaux non récupérés : {', '.join(missing) or 'aucun'} ; détail dans logs/samedi.log"] + arb_lines
        say("\n" + "\n".join(lines))
        if not a.plan:
            pret.write_text("\n".join(lines) + "\n")
        return False
    gamma = PLAN["gamma"] if a.gamma is None else a.gamma
    if a.gamma is None and verdict == "brut_pire":
        gamma = 0.0
        say("garde-fou : à 2 répétitions, les sorties brutes des proxys du jour font PIRE que les normalisées -> gamma = 0")
    gate = None if a.no_gate or not PLAN.get("gate") else PLAN["gate"]
    if gate and not (ROOT / gate["file"]).exists():
        say(f"porte absente ({gate['file']}) : ZIP sans porte")
        gate = None
    share = PLAN["conv_share"]
    conv = [c for c in PLAN["conv"] if (V2 / c / "last.pt").exists() or (V2 / c / "best.pt").exists()] if share > 0 else []
    say(f"sélection : {len(chosen)} REVE du jour + {len(old)} anciens finals normalisés ; {len(conv)} conv ({100 * share:.0f} % du poids) ; "
        f"gamma {gamma} ; porte {'gamma %s' % gate['gamma'] if gate else 'non'} ; temps plateforme estimé {FIXE_S + PAR_REVE_S * (len(chosen) + len(old))} s")
    for n, ck, w, grp in chosen:
        i = MAN[n]["info"]
        say(f"  {n:28s} {grp:7s} {i['kind']:7s} {i['arch']:6s} @{ck:5s} poids {w} pertes {i.get('losses')} cv {i.get('cv')}")
    if a.plan:
        return None
    got = []                                        # poids des seuls retenus, paris d'abord ; arrêt si le disque manque
    for n, ck, w, grp in sorted(chosen, key=lambda c: c[3] == "socle"):
        need = 0.45e9 + 0.16e9 * (len(got) + len(old) + 1) + 1.0e9     # téléchargement + ZIP à venir + réserve (swap)
        if not (V2 / n / (ck + ".pt")).exists() and shutil.disk_usage(ROOT).free < need:
            say(f"disque : {shutil.disk_usage(ROOT).free / 1e9:.1f} Go libres, {n} et les suivants ne sont pas téléchargés")
            break
        if fetch_weights(n, ck + ".pt"):
            got.append((n, ck, w, grp))
    if len(got) < len(chosen):
        say(f"non retenus faute de poids : {sorted({c[0] for c in chosen} - {c[0] for c in got})}")
        covered = {MAN[c[0]]["info"].get("base") for c in got}
        for spec in PLAN.get("fill_old", []):       # les anciens finals (déjà sur le disque) complètent
            if len(got) + len(old) >= min(PLAN.get("fill_to", 0), a.max_reve):
                break
            nm, _, ck = spec.partition("@")
            if nm in covered or spec in old:
                continue
            f = V2 / nm / ((ck or "last") + ".pt")
            if f.exists() or f.with_name("best.pt").exists():
                old.append(spec)
    chosen = [c for c in chosen if c in got]
    if not chosen and not old:
        return build([], [], missing, arb, verdict, guard_line)
    n, nv = len(chosen) + len(old), len(conv)
    wsum = sum(c[2] for c in chosen) + len(old)
    wv = round(share / (1 - share) * wsum / max(nv, 1), 4)       # part des réseaux convolutifs dans le poids total
    specs = [f"{c[0]}@{c[1]}" for c in chosen] + old + conv
    weights = [c[2] for c in chosen] + [1.0] * len(old) + [wv] * nv
    est = FIXE_S + PAR_REVE_S * n
    compute = "environ 75 h GPU T4 (Kaggle) et 8 h GPU Apple (MPS) pour les modèles de cet ensemble, hors expériences"
    name = a.out or f"15_nba{sum(c[3] == 'pari A' for c in chosen)}_nbm{sum(c[3] == 'pari B' for c in chosen)}_rwft{sum(c[3] == 'socle' for c in chosen)}_old{len(old)}_ens{nv}"
    out, k = ROOT / "submissions" / name, 1
    while out.exists() or out.with_suffix(".zip").exists():
        k += 1
        out = ROOT / "submissions" / f"{name}_v{k}"
    cmd = [PY, "-I", "hand/make_submission_v3.py", "--out", out.relative_to(ROOT), "--runs", *specs, "--weights", *weights,
           "--ckpt", "last", "--half", "--adabn", 32, "--nozip", "--compute", compute,
           "--combine", PLAN["combine"], "--gamma", gamma, "--move", ",".join(c[0] for c in chosen if c[3] != "secours")]
    if gate:
        cmd += ["--gate", gate["file"], "--gate_gamma", gate["gamma"], "--gate_floor", gate.get("floor", 0.02)]
    al = PLAN.get("align")
    if al and float(al.get("tau", 0)) > 0 and not (ROOT / al.get("embed", "")).exists():
        say(f"empreinte de sujet absente ({al.get('embed')}) : ZIP sans alignement")
        al = None
    if al:
        cmd += ["--align", al.get("K", 0), "--align_lam", al.get("lam", 1.0), "--align_tau", al.get("tau", 0.0), "--align_min", al.get("min", 32)]
        if al.get("js"):
            cmd += ["--align_js"]
        if float(al.get("tau", 0)) > 0:
            cmd += ["--subj_embed", al["embed"]]
    if PLAN.get("detrend"):
        cmd += ["--detrend"]
    if PLAN.get("conv_zero"):
        cmd += ["--conv_zero", PLAN["conv_zero"]]
    r = run(cmd)
    say("make_submission_v3 :", (r.stdout.strip().splitlines() or [""])[-1], r.stderr[-400:] if r.returncode else "")
    for l in r.stdout.splitlines():
        if "norme moyenne" in l:
            say("  " + l)
    lines = [head]
    ok = r.returncode == 0
    last = "(test non lancé : make_submission_v3 a échoué)"
    if ok:      # test par sous-ensembles de 5 modèles au plus (mémoire du Mac), le 1er mêlant REVE du jour, anciens et conv
        t0 = time.time()
        idx = list(range(len(specs)))
        first = sorted(set(idx[:2] + idx[len(chosen):len(chosen) + 1] + idx[n:n + 2]))
        rest = [i for i in idx if i not in first]
        chunks = [first] + [rest[j:j + 5] for j in range(0, len(rest), 5)]
        logs, lasts = [], []
        for ch in chunks:
            r = run([PY, "-I", "hand/test_submission_v3.py", out.relative_to(ROOT), "--models", ",".join(map(str, ch))])
            logs.append(r.stdout + ("\n[stderr]\n" + r.stderr if r.returncode else ""))
            tl = [l for l in r.stdout.splitlines() if l.strip()]
            lasts.append(tl[-1] if tl else "(aucune sortie)")
            if r.returncode or lasts[-1] != "OK":
                break
        (ROOT / a.log).write_text("\n".join(logs))
        ok = len(lasts) == len(chunks) and all(l == "OK" for l in lasts)
        last = "OK" if ok else lasts[-1]
        shutil.rmtree(out.with_name(out.name + "__part"), ignore_errors=True)
        say(f"test_submission_v3 ({time.time() - t0:.0f} s, {len(chunks)} sous-ensembles) : « {last} » ; journal {a.log}")
    if ok:
        r = run([PY, "-I", "hand/zip_move.py", out.relative_to(ROOT)])
        z = out.with_suffix(".zip")
        ok = r.returncode == 0 and z.exists()
        say("zip_move :", (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1])
    if ok:
        r = run(["unzip", "-tq", z])
        ok = r.returncode == 0
        lines += [f"PRÊT : {z}" if ok else f"ÉCHEC : archive corrompue {z}",
                  f"taille : {z.stat().st_size / 1e6:.0f} Mo ; archive vérifiée (unzip -t) : {'oui' if ok else 'NON'}",
                  f"dernière ligne du test ({a.log}) : {last}"]
    else:
        lines += [f"ÉCHEC : pas de ZIP 15 utilisable. Secours : relancer « kaggle/outils/samedi_v3.sh --no_wait --max_reve 10 », "
                  f"sinon ne rien envoyer : la meilleure note ({BEST}) reste acquise.",
                  f"dernière ligne du test ({a.log}) : {last}", f"dossier laissé pour diagnostic : {out}"]
    lines += [f"contenu : {len(chosen)} REVE-base en norme libre (" + ", ".join(f"{g} x {sum(c[3] == g for c in chosen)}" for g in ("pari A", "pari B", "socle", "secours")) + ") ; "
              f"{len(old)} anciens finals normalisés ; {nv} réseaux convolutifs (poids {wv}, {100 * share:.0f} % du total)",
              f"sortie : non normalisée, exposant de la norme {gamma}, combinaison {PLAN['combine']}" + (f", porte de qualité (exposant {gate['gamma']})" if gate else ", sans porte")
              + ((f", alignement des REVE sur la moyenne du lot (force {'adaptative, au plus ' if al.get('js') else ''}{al.get('lam')})" if float(al.get("tau", 0)) <= 0
                  else f", alignement par sujet dans le lot (tau {al.get('tau')}, lam {al.get('lam')})") if al else "")
              + (", dérive linéaire retirée" if PLAN.get("detrend") else "") + (f", canaux aberrants à zéro pour les conv (x {PLAN['conv_zero']})" if PLAN.get("conv_zero") else ""),
              *arb_lines,
              f"garde-fou (sorties brutes contre normalisées, proxys du jour, 2 répétitions) : {verdict or 'non mesuré'} {guard_line}",
              f"temps plateforme estimé : {est} s (limite 7 200 s)",
              f"noyaux non récupérés : {', '.join(missing) or 'aucun'}",
              "modèles : " + " ".join(f"{s}x{w}" for s, w in zip(specs, weights)),
              "À envoyer par Louis sur Codabench (Sealed Final Phase), ZIP tel quel, avant 14 h."]
    pret.write_text("\n".join(lines) + "\n")
    say("\n" + "\n".join(lines))
    return ok


if __name__ == "__main__":
    kdirs = [k for k in a.kernels.split(",") if k]
    say(f"=== samedi_v3.py : noyaux {kdirs}, max {a.max_reve} REVE, libre {shutil.disk_usage(ROOT).free / 1e9:.1f} Go")
    missing = []
    if not a.no_fetch:
        try:
            missing = wait_and_fetch(kdirs)
        except Exception as e:                      # quoi qu'il arrive côté Kaggle, on construit avec ce qui est là
            say("récupération interrompue :", scrub(repr(e)))
            missing = kdirs
    else:                                           # outputs/v2 tel quel : runs des lanceurs déjà rangés
        for k in kdirs:
            for n in kernel_names(k):
                if (V2 / n / "config.json").exists():
                    ok, why, info = health(V2 / n)
                    MAN[n] = dict(kdir=k, ok=ok, why=why, info=info, local=True, d=V2 / n, files=[q.name for q in (V2 / n).iterdir()])
    arb = arbitrate()
    verdict, guard_line = guard()
    chosen, old = select(arb)
    ok = build(chosen, old, missing, arb, verdict, guard_line)
    if not a.plan:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(0 if ok or a.plan else 1)
