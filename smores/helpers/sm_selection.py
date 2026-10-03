import numpy as np, glob, os, sys, re
from scipy.optimize import curve_fit
root, out = sys.argv[1], sys.argv[2]
cyts = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
runs = sorted(glob.glob(os.path.join(root, "run_*")))
step = lambda f: int(re.search(r"CytoStep_(\d+)", f).group(1))
if os.path.exists(out):
    Y = np.load(out)["Y"]
else:
    Y = np.zeros((len(runs), 101, len(cyts)))
    for i, r in enumerate(runs):
        fs = sorted(glob.glob(os.path.join(r, "LatticeData", "CytoStep_*.npz")), key=step)
        for j, f in enumerate(fs[:101]):
            d = np.load(f)
            for c, name in enumerate(cyts):
                Y[i, j, c] = d[name].mean()
        if i % 10 == 0: print(f"loaded {i}/{len(runs)}", flush=True)
    np.savez(out, Y=Y, runs=[os.path.basename(r) for r in runs], cyts=cyts)
t = np.linspace(0, 1, Y.shape[1])
def m1(t, c0, p, k): return c0*np.exp(-k*t) + p/k*(1-np.exp(-k*t))
def m2(t, A, r, t0): return A/(1+np.exp(-r*(t-t0)))
def m3(t, c0, p, k, lam):
    d = k - lam; d = np.where(np.abs(d) < 1e-6, 1e-6, d)
    return c0*np.exp(-k*t) + p/d*(np.exp(-lam*t) - np.exp(-k*t))
M = {"M1 prod-decay": (m1, [0.1, 1, 1], ([0, 0, 1e-3], [2, 1e3, 1e3])),
     "M2 logistic":   (m2, [1, 8, 0.5], ([0, 0, -1], [10, 200, 2])),
     "M3 prod-decay-peak": (m3, [0.1, 5, 5, 1], ([0, 0, 1e-3, 0], [2, 1e4, 1e3, 1e3]))}
n = len(t)
print(f"\n{'cytokine':8s} {'model':20s} {'median R2':>9s} {'R2>0.9':>7s} {'sum AIC':>10s} {'AIC wins':>8s}")
for c, name in enumerate(cyts):
    res = {m: {"r2": [], "aic": []} for m in M}
    for i in range(Y.shape[0]):
        y = Y[i, :, c]; s = max(abs(y).max(), 1e-30); z = y/s
        ss = ((z - z.mean())**2).sum()
        for m, (f, p0, b) in M.items():
            try:
                p, _ = curve_fit(f, t, z, p0=p0, bounds=b, maxfev=40000)
                rss = ((z - f(t, *p))**2).sum()
            except Exception:
                rss = ss
            res[m]["r2"].append(1 - rss/ss if ss > 0 else np.nan)
            res[m]["aic"].append(n*np.log(max(rss, 1e-300)/n) + 2*len(p0))
    A = np.array([res[m]["aic"] for m in M]); wins = np.bincount(A.argmin(0), minlength=len(M))
    for k, m in enumerate(M):
        r2 = np.array(res[m]["r2"])
        print(f"{name:8s} {m:20s} {np.nanmedian(r2):9.3f} {np.mean(r2>0.9):7.2f} {A[k].sum():10.0f} {wins[k]:8d}", flush=True)
