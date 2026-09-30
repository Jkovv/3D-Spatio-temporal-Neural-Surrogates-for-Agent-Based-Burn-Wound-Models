import numpy as np, glob, os, sys
root = sys.argv[1]
cyts = ["il1", "il6", "il8", "il10", "tgf", "tnf"]
runs = sorted(glob.glob(os.path.join(root, "run_*")))
res = {c: {"gmax": [], "ex_tavg": [], "ex_100": [], "fl_100": []} for c in cyts}
for r in runs:
    steps = sorted(glob.glob(os.path.join(r, "LatticeData", "CytoStep_*.npz")))
    nz = {c: [] for c in cyts}
    for s in steps:
        d = np.load(s)
        for c in cyts:
            nz[c].append((d[c] > 0).mean())
    last = np.load(steps[-1])
    for c in cyts:
        a = last[c].astype(np.float64)
        res[c]["gmax"].append(a.max())
        res[c]["ex_tavg"].append(100 * (1 - np.mean(nz[c])))
        res[c]["ex_100"].append(100 * (a <= 0).mean())
        res[c]["fl_100"].append(100 * (a < 1e-4 * a.max()).mean())
print(f"runs: {len(runs)}, frames per run: {len(steps)}\n")
print(f"{'cyt':5s} {'gmax@100h':>10s} | {'zeros, time-avg (2D def.)':>26s} | {'zeros @100h':>15s} | {'<noise floor @100h':>19s}")
for c in cyts:
    f = lambda k: f"{np.mean(res[c][k]):6.2f} +/- {np.std(res[c][k]):5.2f}"
    print(f"{c:5s} {np.mean(res[c]['gmax']):10.3e} | {f('ex_tavg'):>26s} | {f('ex_100'):>15s} | {f('fl_100'):>19s}")
