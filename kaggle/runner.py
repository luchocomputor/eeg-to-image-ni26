"""Lance train_v2.py sur Kaggle (GPU) : Alljoined (jeu alljoined-fast) + THINGS-EEG2 (sorties des notebooks
things-prep-*)."""
import glob, os, subprocess, sys

NAME, EXTRA, TIMEOUT = "RUN_NAME", "RUN_ARGS", RUN_TIMEOUT
code = glob.glob("/kaggle/input/**/train_v2.py", recursive=True)[0]
fast = os.path.dirname(glob.glob("/kaggle/input/**/signal_f16.npy", recursive=True)[0])
things = sorted({os.path.dirname(f) for f in glob.glob("/kaggle/input/**/things_s*.npz", recursive=True)})
print("code", code, "| alljoined", fast, "| things", len(things), "dossiers", flush=True)
print(open("/proc/meminfo").read().split("\n")[:2], flush=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
env = dict(os.environ, NI_FAST=fast, NI_OUT="/kaggle/working/outputs")
aug = "--sym --learn_temp --mix 0.3 --shift 3 --chdrop 0.1 --noise 0.1 --scale 0.2 --car --gneg 0.5 --soft 0.02"
cmd = [sys.executable, "-u", code, "--name", NAME, "--arch", "fast", "--device", "cuda", "--things", ",".join(things)]
cmd += aug.split() + EXTRA.split()
print(" ".join(cmd), flush=True)
subprocess.run(cmd, env=env, check=True, timeout=TIMEOUT)   # plafond de quota GPU
