"""Sorties d'un notebook Kaggle, avec pagination et reprises (python 3.10 du système, paquet kaggle).
  samedi_fetch.py list <kernel-metadata.json>                    -> noms des fichiers, un par ligne
  samedi_fetch.py get  <kernel-metadata.json> <dest> <motif,...> -> télécharge les fichiers correspondants
"""
import fnmatch, json, os, sys, time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

mode, meta = sys.argv[1], sys.argv[2]
owner, slug = json.load(open(meta))["id"].split("/")
api = KaggleApi(); api.authenticate()


def listing():
    files, token = [], None
    with api.build_kaggle_client() as k:
        for _ in range(50):
            r = ApiListKernelSessionOutputRequest()
            r.user_name, r.kernel_slug = owner, slug
            if token:
                r.page_token = token
            resp = k.kernels.kernels_api_client.list_kernel_session_output(r)
            files += list(resp.files)
            token = getattr(resp, "next_page_token", None)
            if not token:
                break
    return files


files = listing()
if mode == "list":
    print("\n".join(f.file_name for f in files))
    sys.exit(0)
dest, pats = sys.argv[3], sys.argv[4].split(",")
bad = 0
for f in files:
    if not any(fnmatch.fnmatch(f.file_name, p) for p in pats):
        continue
    out = os.path.join(dest, f.file_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(4):
        try:
            url = f.url if attempt == 0 else next(g.url for g in listing() if g.file_name == f.file_name)  # lien signé rafraîchi
            with requests.get(url, stream=True, timeout=(20, 90)) as g, open(out + ".part", "wb") as h:      # lecture figée 90 s (coupure, veille) : on réessaie
                g.raise_for_status()
                n = 0
                for chunk in g.iter_content(1 << 22):
                    h.write(chunk); n += len(chunk)
                want = int(g.headers.get("Content-Length") or n)
            assert n == want, f"taille {n} != {want}"
            os.replace(out + ".part", out)
            print(f.file_name, round(n / 1e6, 1), "Mo", flush=True)
            break
        except Exception as e:
            print(f"essai {attempt + 1} raté pour {f.file_name} : {e!r}", flush=True)
            time.sleep(20)
    else:
        bad += 1
        if os.path.exists(out + ".part"):
            os.remove(out + ".part")
sys.exit(1 if bad else 0)
