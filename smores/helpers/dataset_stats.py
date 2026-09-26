import numpy as np, glob, os, sys
root = sys.argv[1]
cyts = ["il1", "il6", "il8", "il10", "tgf", "tnf"]
runs = sorted(glob.glob(os.path.join(root, "run_*")))
fields = {c: [] for c in cyts}
for r in runs:
    d = np.load(os.path.join(r, "LatticeData", "CytoStep_1000000.npz"))
    for c in cyts:
        fields[c].append(d[c].astype(np.float64))
print(f"runs used: {len(fields['il8'])}\n")
print(f"{'cyt':5s} {'gmax_mean':>10s} | {'exact0':>7s} | {'<1e-4*runmax':>13s} | {'<1e-4*sweepmax':>15s}")
for c in cyts:
    gmax = np.array([a.max() for a in fields[c]])
    sweepmax = gmax.max()
    ex  = [100*(a == 0).mean() for a in fields[c]]
    rel = [100*(a < 1e-4*a.max()).mean() for a in fields[c]]
    glb = [100*(a < 1e-4*sweepmax).mean() for a in fields[c]]
    print(f"{c:5s} {gmax.mean():10.3e} | {np.mean(ex):7.2f} | {np.mean(rel):6.2f}+/-{np.std(rel):5.2f} | {np.mean(glb):7.2f}+/-{np.std(glb):5.2f}")
