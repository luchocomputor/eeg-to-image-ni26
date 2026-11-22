"""THINGS-EEG2 -> format Alljoined du challenge. À exécuter sur Kaggle (CPU, internet).
Pour chaque sujet : brut figshare 18470912, puis pour chaque fichier (session x entraînement/test) le pipeline
officiel neuralbench/neuralset de la tâche image : EEG seuls ; coupe-bande 50 et 60 Hz + harmoniques <= 300 Hz ;
passe-bande 0,1-75 Hz ; rééchantillonnage 120 Hz ; RobustScaler par canal sur l'enregistrement ; nos 32 électrodes
Emotiv par nom (absente = zéros) ; fenêtres de 144 échantillons commençant 0,2 s + 12 éch. avant l'image (ligne de
base et écrêtage +-20 appliqués à l'entraînement, comme pour Alljoined). La marge de +-12 éch. sert au décalage
+-3 et au recalage : les PEV THINGS arrivent ~50 ms plus tôt dans la fenêtre que ceux d'Alljoined (18/11).
Identifiant d'image = rang alphabétique (concept/fichier) des images THINGS-EEG2 = numérotation Alljoined
(entraînement 0-16539, test 16540-16739 ; vérifié image par image le 18/11).
Sortie /kaggle/working/things_sXX.npz : X float16 (n, 32, 144), img, split (0 entraînement / 2 test), ses.
"""
import os, shutil, subprocess, sys, time, traceback, zipfile
import numpy as np

SUBJECTS = [int(s) for s in os.environ.get("THINGS_SUBJECTS", "SUBJ_PLACEHOLDER").split(",")]
PAD = 12                 # marge de chaque côté de la fenêtre officielle de 120 éch.
W = 120 + 2 * PAD
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mne==1.13.2"], check=True)
import mne  # noqa: E402
import requests  # noqa: E402

mne.set_log_level("ERROR")
CH32 = ['Cz', 'FCz', 'AFz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1',
        'Pz', 'POz', 'Oz', 'O2', 'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2', 'F6', 'Fp2']
OUT, TMP = "/kaggle/working", "/tmp/things"
os.makedirs(TMP, exist_ok=True)
NJ = os.cpu_count() or 4
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:6.0f} s]", *a, flush=True)


say("sujets", SUBJECTS, "; disque libre", shutil.disk_usage("/tmp").free // 2**30, "Go ; cpus", NJ)
meta_path = os.path.join(TMP, "image_metadata.npy")
subprocess.run(["curl", "-sL", "-o", meta_path, "https://osf.io/download/qkgtf/"], check=True)
meta = np.load(meta_path, allow_pickle=True).item()
ids = {}
for split, off in (("train", 0), ("test", 16540)):
    names = [f"{c}/{f}" for c, f in zip(meta[f"{split}_img_concepts"], meta[f"{split}_img_files"])]
    pos = {n: i for i, n in enumerate(sorted(names))}
    ids[split] = np.array([off + pos[n] for n in names])       # code c (1..N) -> ids[split][c - 1]
    say(split, len(names), "images ; ordre des métadonnées = ordre alphabétique :", names == sorted(names))
files = requests.get("https://api.figshare.com/v2/articles/18470912/files?page_size=100", timeout=60).json()
url = {f["name"]: f["download_url"] for f in files}


def process(d, split):
    ch = list(d["ch_names"])
    ch[ch.index("stim")] = "STI101"
    info = mne.create_info(ch, sfreq=float(d["sfreq"]), ch_types=list(d["ch_types"]))
    raw = mne.io.RawArray(d["raw_eeg_data"], info, verbose=False)
    ev = mne.find_events(raw, stim_channel="STI101", verbose=False)          # comme neuralfetch
    n_img = len(ids[split])
    ev = ev[(ev[:, 2] >= 1) & (ev[:, 2] <= n_img)]
    onset = (ev[:, 0] - raw.first_samp) / raw.info["sfreq"]
    raw.pick("eeg")
    sf = raw.info["sfreq"]
    notch = []
    for f in (50.0, 60.0):                                                  # comme EegExtractor._notch_filter
        notch.extend(np.arange(f, min(sf / 2, 301), f).tolist())
    raw.notch_filter(notch, picks="all", phase="zero", filter_length="auto", n_jobs=NJ, verbose=False)
    raw.filter(0.1, 75.0, picks="all", skip_by_annotation=("edge", "bad_acq_skip"), n_jobs=NJ, verbose=False)
    raw.resample(120.0, n_jobs=NJ, verbose=False)
    x = raw.get_data()
    med = np.median(x, axis=1, keepdims=True)                               # RobustScaler (sklearn)
    q25, q75 = np.percentile(x, [25, 75], axis=1, keepdims=True)
    sc = q75 - q25
    sc[sc == 0] = 1.0
    x = (x - med) / sc
    upper = [c.upper() for c in raw.ch_names]
    x32 = np.zeros((32, x.shape[1]), np.float32)
    missing = []
    for k, c in enumerate(CH32):
        if c.upper() in upper:
            x32[k] = x[upper.index(c.upper())]
        else:
            missing.append(c)
    i0 = np.round((onset - 0.2) * 120).astype(np.int64) - PAD
    keep = (i0 >= 0) & (i0 + W <= x32.shape[1])
    i0 = i0[keep]
    X = x32[:, i0[:, None] + np.arange(W)].transpose(1, 0, 2).astype(np.float16)   # (n, 32, W)
    return X, ids[split][ev[keep, 2] - 1], missing


for s in SUBJECTS:
    name = f"sub-{s:02d}.zip"
    zp = os.path.join(TMP, name)
    try:
        say("téléchargement", name)
        subprocess.run(["curl", "-sL", "--retry", "5", "-o", zp, url[name]], check=True)
        say("téléchargé", os.path.getsize(zp) // 2**20, "Mo")
        Xs, imgs, splits, sess = [], [], [], []
        with zipfile.ZipFile(zp) as zf:
            members = sorted(m for m in zf.namelist() if "raw_eeg_" in m and m.endswith(".npy"))
            say(len(members), "fichiers :", members[:2], "...")
            for m in members:
                try:
                    split = "train" if "training" in m else "test"
                    ses = int(m.split("ses-")[1][:2]) if "ses-" in m else 0
                    path = zf.extract(m, TMP)
                    d = np.load(path, allow_pickle=True).item()
                    os.remove(path)
                    X, img, missing = process(d, split)
                    del d
                    Xs.append(X)
                    imgs.append(img)
                    splits.append(np.full(len(img), 0 if split == "train" else 2, np.int8))
                    sess.append(np.full(len(img), ses, np.int8))
                    say(m, ":", len(img), "fenêtres ; électrodes absentes", missing,
                        "; écart-type", float(X[:, :, PAD + 24:PAD + 120].astype(np.float32).std()))
                except Exception:
                    traceback.print_exc()
        os.remove(zp)
        X = np.concatenate(Xs)
        img, split, ses = map(np.concatenate, (imgs, splits, sess))
        np.savez(os.path.join(OUT, f"things_s{s:02d}.npz"), X=X, img=img, split=split, ses=ses)
        occ = [CH32.index(c) for c in ("O1", "Oz", "O2")]
        erp = X[:, occ, PAD:PAD + 120].astype(np.float32).mean((0, 1))
        erp -= erp[:24].mean()
        say(f"sujet {s} : {len(img)} fenêtres ({(split == 0).sum()} entraînement, {(split == 2).sum()} test) ; "
            "PEV occipital tous les 100 ms :", np.round(erp[::12], 3).tolist())
    except Exception:
        traceback.print_exc()
        if os.path.exists(zp):
            os.remove(zp)
say("FINI")
