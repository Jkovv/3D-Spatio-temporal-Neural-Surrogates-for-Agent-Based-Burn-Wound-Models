"""Volume-averaged cytokine trajectories for every realisation of every sweep
point: the original run (sweep/outputs/run_XXXX) plus its replicates
(sweep/rep_outputs/rep_XXXX_k). Y has shape (points, realisations, frames,
cytokines); missing or incomplete realisations are NaN."""
import numpy as np, glob, os, re, argparse
from multiprocessing import Pool
CYTS = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
step = lambda f: int(re.search(r"CytoStep_(\d+)", f).group(1))
def vol_avg(run_dir, T):
    fs = sorted(glob.glob(os.path.join(run_dir, "LatticeData", "CytoStep_*.npz")), key=step)
    if len(fs) < T: return None
    out = np.zeros((T, len(CYTS)))
    for j, f in enumerate(fs[:T]):
        d = np.load(f)
        for c, name in enumerate(CYTS): out[j, c] = d[name].mean()
    return out
def one(args):
    sweep_dir, rep_dir, rid, nrep, T = args
    Y = np.full((nrep + 1, T, len(CYTS)), np.nan)
    paths = [os.path.join(sweep_dir, rid)] + [os.path.join(rep_dir, f"rep_{rid[4:]}_{k}") for k in range(1, nrep + 1)]
    for r, p in enumerate(paths):
        v = vol_avg(p, T)
        if v is not None: Y[r] = v
    return Y
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True); ap.add_argument("--replicates", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--nrep", type=int, default=5)
    ap.add_argument("--frames", type=int, default=101); ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    rids = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.sweep, "run_*")))
    with Pool(a.procs) as p:
        Y = np.stack(p.map(one, [(a.sweep, a.replicates, r, a.nrep, a.frames) for r in rids]))
    n_ok = np.isfinite(Y[:, :, 0, 0]).sum(1)
    print(f"{len(rids)} points; realisations per point: min {n_ok.min()}, max {n_ok.max()}, "
          f"points with all {a.nrep + 1}: {(n_ok == a.nrep + 1).sum()}", flush=True)
    np.savez(a.out, Y=Y, runs=rids, cyts=CYTS)
