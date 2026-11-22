"""Exercice 1 — la métrique du challenge, écrite à la main (NumPy seulement).

Codabench : on L2-normalise les prédictions et la galerie, on calcule la
similarité cosinus de chaque prédiction avec toutes les images candidates,
et une requête est juste si la vraie image est parmi les k meilleures.
"""

import numpy as np


def topk_retrieval(pred, gallery, target_idx, k=5):
    """Top-k accuracy de retrieval.

    pred       : (N, D) embeddings prédits à partir de l'EEG
    gallery    : (M, D) embeddings DINOv2 des M images candidates
    target_idx : (N,)   indice, dans gallery, de l'image vraiment vue
    renvoie    : float entre 0 et 1
    """

    pred = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)

    cos_sim = np.dot(pred, gallery.T)

    cos_sim_idx = np.argsort(cos_sim, axis=1)

    cos_sim_idx = cos_sim_idx[:, -k:]

    comp = target_idx[:, None] == cos_sim_idx

    comp = np.any(comp, axis=1)

    return comp.mean()
