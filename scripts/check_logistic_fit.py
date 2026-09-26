# simplified logistic fit, not the one used in smore pars. use 0.871 from predictive_consistency instead
# Punkt 1: R^2 dopasowania powierzchni saturating-logistic do trajektorii IL-8 z NOWEGO sweepa.
# Zastepuje stara liczbe "~87%" (sekcja External, podpis tab:external, Conclusions).
# Uzycie: $PY scripts/check_logistic_fit.py $BASE/sweep/outputs
import sys, glob, os
import numpy as np
from scipy.optimize import curve_fit

root = sys.argv[1]

def logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))

r2s, fails = [], 0
for d in sorted(glob.glob(os.path.join(root, "run_*"))):
    f = os.path.join(d, "datafiles", "mean_concentration.txt")
    a = np.loadtxt(f, delimiter=",", skiprows=1)
    t = a[:, 0] / 1e4          # MCS -> h
    y = a[:, 1] / a[:, 1].max() # IL-8, znormalizowane (R^2 nie zalezy od skali)
    try:
        p, _ = curve_fit(logistic, t, y, p0=[1.2, 0.05, 50.0], maxfev=50000)
        yh = logistic(t, *p)
        r2s.append(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    except Exception:
        fails += 1

r2s = np.array(r2s)
print(f"runs fitted: {len(r2s)}  failed: {fails}")
print(f"IL-8 logistic fit R2: mean={r2s.mean():.3f} median={np.median(r2s):.3f} "
      f"min={r2s.min():.3f} max={r2s.max():.3f}")
