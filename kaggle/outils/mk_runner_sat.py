"""Lanceurs de samedi 21/11, dérivé de mk_runner_seq.py : chaque GPU enchaîne ses runs.
Ajouts : (1) les données sont copiées une fois sur le disque local (/tmp) AVANT tout entraînement et ce temps n'est pas
compté dans le plafond (le 9e tour est mort d'un démarrage lent : lot 200 à 5 848 s) ; (2) durée estimée par run selon
final / proxy et nombre de passes ; (3) plafond dur de session (Kaggle coupe à 12 h et perd les sorties) ;
(4) best.pt des finals supprimé à la fin (copie de last.pt) ; (5) essai à sec sans GPU (NI_DRY=1) ;
(6) « --stop_after K » dans les arguments d'un run (option du lanceur, pas de train_v2) : le run garde son calendrier
(--epochs) mais est arrêté dès que le point de contrôle de la K-ième passe est écrit, comme revef_c@best (passe 2 sur 3).

python3 mk_runner_sat.py <dir> <slug> <heures_plafond> "<doc>" "0|nom|args" "0|nom2|args" "1|nom3|args" ...
(le premier champ est le GPU ; l'ordre des runs d'un même GPU est l'ordre d'exécution ; heures_plafond se compte
après la copie des données)
"""
import json, os, sys

os.chdir(os.path.expanduser("~/Downloads/neural-interfaces26/kaggle"))
d, slug, hours, doc = sys.argv[1:5]
queues = {}
for spec in sys.argv[5:]:
    gpu, name, args = spec.split("|", 2)
    queues.setdefault(int(gpu), []).append((name, args))
os.makedirs(d, exist_ok=True)
m = json.load(open("train_reve_r3/kernel-metadata.json"))
m["id"] = m["id"].split("/")[0] + "/" + slug
m["title"] = slug
json.dump(m, open(f"{d}/kernel-metadata.json", "w"), indent=1)

TEMPLATE = r'''"""@@DOC@@ Chaque GPU enchaîne ses runs ; un run ne démarre que s'il peut finir avant le plafond.
Code REVE autonome (models_v2.py) : seuls les poids pré-entraînés viennent de Hugging Face (public).
Généré par mk_runner_sat.py."""
import glob, json, os, re, shutil, subprocess, sys, threading, time

QUEUES = @@QUEUES@@
COMMON = "--epochs 3 --bs 512 --amp --clip 1.0 --sym --learn_temp --mix 0.3 --shift 3 --chdrop 0.1 --noise 0.1 --gneg 0.5 --soft 0.02"
TIMEOUT = int(@@HOURS@@ * 3600)        # compté APRÈS la copie des données
HARD = int(11.6 * 3600)                # depuis le début du script : Kaggle coupe la session à 12 h
SEC_FINAL, SEC_PROXY, STARTUP = 3150, 2100, 700      # par passe : final 3 241 lots ; proxy 1 896 lots + évaluation
CHECK_LOT, CHECK_MAX = 1200, 7.82      # santé : perte au lot 1200 de la 1re passe (normale 7,72-7,78 ; décroché 7,91)
DRY = bool(os.environ.get("NI_DRY"))   # essai à sec, sans GPU : faux entraîneur, durées divisées par NI_DRY_SCALE
SCALE = float(os.environ.get("NI_DRY_SCALE", "1000")) if DRY else 1.0
WORK = os.environ.get("NI_WORK", "/kaggle/working")
POLL = 30 / SCALE if DRY else 30
T_SCRIPT = time.time()

if DRY:
    code, fast, ck, n_gpu = os.environ["NI_DRY_TRAINER"], "/tmp", "", int(os.environ.get("NI_DRY_GPUS", "2"))
else:
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download("brain-bzh/reve-base", "model.safetensors")
    code = glob.glob("/kaggle/input/**/train_v2.py", recursive=True)[0]
    fast = os.path.dirname(glob.glob("/kaggle/input/**/signal_f16.npy", recursive=True)[0])
    mv2 = os.path.join(os.path.dirname(code), "models_v2.py")
    assert '"revef"' in open(code).read() and 'head == "flat"' in open(mv2).read(), \
        "ancienne version du code montée : attendre la fin du traitement du jeu ni26-code"
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    print(r.stdout, flush=True)
    n_gpu = len([l for l in r.stdout.splitlines() if l.strip()])
    # --- données : copie séquentielle sur le disque local, une fois, avant tout entraînement ---
    tc = time.time()
    local = "/tmp/ni_fast"
    try:
        os.makedirs(local, exist_ok=True)
        for f in sorted(os.listdir(fast)):          # recs.json, index.npz, targets.npy, stims.json, signal_f16.npy...
            if os.path.isfile(os.path.join(fast, f)):
                shutil.copyfile(os.path.join(fast, f), os.path.join(local, f))
        assert os.path.getsize(os.path.join(local, "signal_f16.npy")) == os.path.getsize(os.path.join(fast, "signal_f16.npy"))
        fast = local
        print(f"données copiées dans {local} en {time.time() - tc:.0f} s", flush=True)
    except Exception as e:      # disque local plein, etc. : on lit au moins le signal une fois (cache disque chaud)
        print(f"copie locale impossible ({e!r}) : lecture du signal en place", flush=True)
        with open(os.path.join(fast, "signal_f16.npy"), "rb") as h:
            while h.read(1 << 26):
                pass
        print(f"signal lu en {time.time() - tc:.0f} s", flush=True)
print("code", code, "| alljoined", fast, "| reve", ck, "| GPU", n_gpu, flush=True)
if 0 < n_gpu < len(QUEUES):     # un seul GPU attribué : tout s'enchaîne sur le GPU 0, dans l'ordre des files
    print(f"ATTENTION : {n_gpu} GPU pour {len(QUEUES)} files, files fusionnées sur le GPU 0", flush=True)
    QUEUES = {"0": [r for g in sorted(QUEUES, key=int) for r in QUEUES[g]]}
os.makedirs(WORK, exist_ok=True)
t0 = time.time()                # le plafond TIMEOUT part d'ici (démarrage non compté)
LIMIT = min(t0 + TIMEOUT / SCALE, T_SCRIPT + HARD / SCALE)     # instant absolu de fin
procs, done, skipped, unstable, cut, failed = {}, [], [], [], [], []


def stop_after(args):
    m = re.findall(r"--stop_after (\d+)", args)
    return int(m[-1]) if m else 0


def need_s(args):
    full = COMMON + " " + args
    ep = stop_after(args) or int(re.findall(r"--epochs (\d+)", full)[-1])
    final = (re.findall(r"--holdout (\d+)", full) or ["4"])[-1] == "0"
    return (ep * (SEC_FINAL if final else SEC_PROXY) + STARTUP) / SCALE


def health(name):
    """None tant que le lot de contrôle n'est pas atteint, sinon la perte lue à ce lot."""
    try:
        m = re.search(rf"ep 0 lot {CHECK_LOT}/\d+ loss ([\d.]+)", open(f"{WORK}/{name}.out").read())
    except OSError:
        return None
    return float(m.group(1)) if m else None


def launch(gpu, name, extra):
    env = dict(os.environ, NI_FAST=fast, NI_OUT=f"{WORK}/outputs", NI_REVE=ck, CUDA_VISIBLE_DEVICES=str(gpu))
    extra = re.sub(r"--stop_after \d+", "", extra)      # option du lanceur
    cmd = [sys.executable, "-u", code, "--name", name, "--device", "cuda"] + COMMON.split() + extra.split()
    print(f"[GPU {gpu}] {time.strftime('%H:%M')} " + " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, env=env, stdout=open(f"{WORK}/{name}.out", "w"), stderr=subprocess.STDOUT)


def worker(gpu, runs):
    runs = list(runs)
    while runs:
        name, extra = runs.pop(0)
        need = need_s(extra)
        if time.time() + need > LIMIT:
            skipped.append(name)
            print(f"[GPU {gpu}] {name} non démarré : {need * SCALE:.0f} s nécessaires, "
                  f"{(LIMIT - time.time()) * SCALE:.0f} s restants", flush=True)
            continue
        p = procs[name] = launch(gpu, name, extra)
        checked, k_stop, stopped = False, stop_after(extra), False
        while p.poll() is None:
            time.sleep(POLL)
            if k_stop and not stopped:       # « -> meilleur » suit l'écriture de last.pt et best.pt à chaque passe d'un final
                try:
                    n_saved = open(f"{WORK}/{name}.out").read().count("-> meilleur")
                except OSError:
                    n_saved = 0
                if n_saved >= k_stop:
                    stopped = True
                    p.kill(); p.wait()
                    json.dump(dict(stop_after=k_stop), open(f"{WORK}/outputs/{name}/arret.json", "w"))
                    print(f"[GPU {gpu}] {time.strftime('%H:%M')} {name} arrêté après {k_stop} passes (voulu)", flush=True)
                    break
            loss = None if checked else health(name)
            if loss is not None:
                checked = True
                if loss > CHECK_MAX:
                    p.kill(); p.wait()
                    print(f"[GPU {gpu}] {name} INSTABLE (perte {loss:.3f} au lot {CHECK_LOT}) : arrêté", flush=True)
                    unstable.append(name)
                    shutil.rmtree(f"{WORK}/outputs/{name}", ignore_errors=True)
                    seed = int((re.findall(r"--init_seed (\d+)", extra) or ["0"])[-1]) + 100
                    retry = re.sub(r"--init_seed \d+", "", extra) + f" --init_seed {seed}"
                    if not name.endswith("_r"):          # une seule relance par run
                        runs.insert(0, (name + "_r", retry))     # en tête de file, si le temps le permet
        if name in unstable or name in cut:
            continue
        if p.returncode == 0 or stopped:
            done.append(name)
            print(f"[GPU {gpu}] {time.strftime('%H:%M')} {name} fini", flush=True)
        else:
            failed.append(name)
            print(f"[GPU {gpu}] {time.strftime('%H:%M')} {name} ÉCHEC (code {p.returncode}), voir {name}.out", flush=True)


threads = [threading.Thread(target=worker, args=(int(g), r), daemon=True) for g, r in QUEUES.items()]
for t in threads:
    t.start()
while any(t.is_alive() for t in threads) and time.time() < LIMIT:
    time.sleep(POLL)
for name, p in list(procs.items()):
    if p.poll() is None:
        cut.append(name)
        p.kill(); p.wait()
        print(f"plafond atteint : {name} arrêté (inachevé, à ne pas utiliser)", flush=True)
for t in threads:
    t.join(timeout=5 * POLL)
for name in procs:              # final : best.pt est une copie de last.pt (sortie deux fois moins lourde)
    cfg, b, l = (f"{WORK}/outputs/{name}/{f}" for f in ("config.json", "best.pt", "last.pt"))
    try:
        if os.path.exists(cfg) and json.load(open(cfg)).get("holdout") == 0 and os.path.exists(b) and os.path.exists(l):
            os.remove(b)
    except Exception as e:
        print("nettoyage", name, repr(e), flush=True)
print("finis :", done, "| instables :", unstable, "| échecs :", failed, "| non démarrés :", skipped, "| coupés :", cut,
      f"| durée {(time.time() - T_SCRIPT) * SCALE / 3600:.2f} h", flush=True)
json.dump(dict(done=done, unstable=unstable, failed=failed, skipped=skipped, cut=cut), open(f"{WORK}/resume.json", "w"))
for name in procs:
    print(f"===== {name}", flush=True)
    print("".join(open(f"{WORK}/{name}.out").readlines()[-8:]), flush=True)
'''
src = (TEMPLATE.replace("@@DOC@@", doc).replace("@@QUEUES@@", json.dumps(queues, ensure_ascii=False))
       .replace("@@HOURS@@", str(float(hours))))
compile(src, "runner.py", "exec")
open(f"{d}/runner.py", "w").write(src)
print(f"{d}/runner.py :", {g: [n for n, _ in r] for g, r in queues.items()}, "| plafond", hours, "h")
