"""Prépare un cache compact pour l'analyse appariée : prédictions (normalisées) des modèles proxy, galerie, clés."""
import sys
import os; HAND = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
OUT = sys.argv[1]
runs_arg = sys.argv[2:]
sys.argv = [HAND + "eval_scelle_like.py"] + runs_arg
__file__ = HAND + "eval_scelle_like.py"
src = open(HAND + "eval_h1500.py").read()
src = src.split("print(f\"{'modèle'")[0].replace('__file__ = __file__.replace("eval_h1500.py", "eval_scelle_like.py")', '')
exec(src)
import numpy as np, torch
torch.save(dict(preds={r: preds[r].half() for r in runs_arg}, Gh=Gh, true=torch.from_numpy(colh[stim[ph]]),
                subj=torch.from_numpy(subj[ph]), stim=torch.from_numpy(stim[ph])), OUT)
print("ok", len(ph), Mh, np.unique(subj[ph], return_counts=True))
