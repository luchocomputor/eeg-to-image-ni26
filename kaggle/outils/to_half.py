"""Réécrit des points de contrôle en float16 (place disque ; les originaux float32 restent sur Kaggle)."""
import os, sys
import torch
for p in sys.argv[1:]:
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if all(v.dtype != torch.float32 for v in sd.values()):
        print("déjà float16 :", p); continue
    before = os.path.getsize(p)
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}, p + ".tmp")
    os.replace(p + ".tmp", p)
    print(p, round(before / 1e6), "->", round(os.path.getsize(p) / 1e6), "Mo")
