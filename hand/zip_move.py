"""Zippe un dossier de soumission déjà testé en supprimant chaque fichier dès qu'il est ajouté (disque presque
plein) : make_submission.py --nozip, puis test_submission.py, puis ce script."""
import shutil, sys, zipfile
from pathlib import Path

out = Path(sys.argv[1])
shutil.rmtree(out / "__pycache__", ignore_errors=True)     # laissé par le test (import de submission.py)
z = out.with_suffix(".zip")
assert not z.exists(), z
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    for q in sorted(out.iterdir()):
        assert q.is_file(), q
        f.write(q, q.name)
        q.unlink()
out.rmdir()
print(z, f"{z.stat().st_size / 1e6:.1f} Mo")
