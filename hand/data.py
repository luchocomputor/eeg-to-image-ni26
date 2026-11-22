"""Plomberie : des loaders au contrat du challenge.

Chaque loader rend des triplets (X, y, info) : X tenseur (B, C, T), y tenseur
(B, n_outputs), info un dict. Même contrat que les vrais loaders d'Alljoined.

- simulated_loaders() : tout de suite, sans données (copie du jeu Simulated).
- alljoined_loaders() : ajouté quand les embeddings DINOv2 seront prêts.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "tracks" / "image_decoding"))
from benchmark_utils.data import make_loader  # noqa: E402


def simulated_loaders(n_images=20, n_chans=8, n_times=120, n_outputs=32,
                      n_train=600, n_val=200, n_test=200, seed=0):
    """Une « image » = un gabarit de signal + un embedding aléatoire ;
    une époque = le gabarit de son image + du bruit. Le ridge doit y arriver."""
    rng = np.random.default_rng(seed)
    templates = np.repeat(rng.standard_normal((n_images, n_chans, 1)) * 2.0,
                          n_times, axis=-1)
    embeddings = rng.standard_normal((n_images, n_outputs))

    def windows(n):
        idx = rng.integers(0, n_images, size=n)
        X = templates[idx] + rng.standard_normal((n, n_chans, n_times))
        return X.astype(np.float32), embeddings[idx].astype(np.float32)

    return {
        "train": make_loader(*windows(n_train), batch_size=64, shuffle=True),
        "val": make_loader(*windows(n_val), batch_size=64),
        "test": make_loader(*windows(n_test), batch_size=64),
    }


def alljoined_loaders(batch_size=1024, with_meta=False):
    """Les vrais loaders d'Alljoined-1.6M (train ~1,07 M, val ~270 k, test ~324 k époques),
    depuis le cache préparé le 12/10 : EEG 32 canaux × 120 échantillons (1 s à 120 Hz),
    cible = embedding DINOv2-giant (1536). Construire les loaders prend ~1-2 min."""
    from neuralset.extractors import hf
    # Les poids sont déjà en cache : neuralset retéléchargerait tout le dépôt (.bin compris).
    hf.HuggingFaceMixin._download_huggingface_snapshot = lambda self: None
    from benchmark_utils.nb_task import load_task

    root = Path(__file__).parents[1] / "tracks" / "image_decoding"
    data_dir = root / "data" / "neural_compet"
    loaders, meta = load_task(
        "eeg", "image", data_dir=data_dir, dataset="xu2025alljoined", device="cpu",
        batch_size=batch_size, seed=0, num_workers=0, subset="full",
        overrides={"target.infra.cluster": None,
                   "target.infra.folder": str(data_dir / "cache"),
                   "target.infra.permissions": None},
    )
    return (loaders, meta) if with_meta else loaders
