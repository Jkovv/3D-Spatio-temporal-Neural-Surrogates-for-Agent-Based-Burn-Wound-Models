import sys, json, numpy as np
sys.path.insert(0, "smore")
from observables import load_sweep, CYTOKINES
from smore_pars import fit_surrogate_one, _saturating
BASE = sys.argv[1]
m = json.load(open(f"{BASE}/smores/manifest.json"))
theta, Y, ids, t = load_sweep(f"{BASE}/sweep/outputs", m["param_names"])
tn = (t - t[0]) / (t[-1] - t[0])
bad = 0
for c, name in enumerate(CYTOKINES):
    stuck, r2 = 0, []
    for i in range(len(ids)):
        y = Y[i, :, c]; p = fit_surrogate_one(t, y); s = max(abs(y).max(), 1e-30)
        stuck += np.allclose(p, [s, 8.0, 0.5], rtol=1e-6)
        ss = ((y - y.mean())**2).sum()
        r2.append(1 - ((y - _saturating(tn, *p))**2).sum() / ss if ss > 0 else np.nan)
    bad += stuck
    print(f"{name:5s} stuck {stuck:3d}/100 | fit R2 median {np.nanmedian(r2):+.3f} mean {np.nanmean(r2):+.3f}", flush=True)
sys.exit(1 if bad > 30 else 0)
