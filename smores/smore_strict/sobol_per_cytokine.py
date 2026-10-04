"""Sobol indices of results/sobol_repmean.json broken down per cytokine: mean total-order
(ST) and first-order (S1) index over the four observables of each cytokine, plus the
24-observable means used in the manuscript.  No refitting; reads the stored indices."""
import json, sys, numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else "results/sobol_repmean.json"
d = json.load(open(path))["summaries"]; names = d["param_names"]; pf = d["per_feature"]; feats = list(pf)
cyts = []
for f in feats:
    c = f.split("_")[0]
    if c not in cyts: cyts.append(c)
ST = {c: [np.mean([pf[f]["ST"][i] for f in feats if f.split("_")[0] == c]) for i in range(len(names))] for c in cyts}
S1 = {c: [np.mean([pf[f]["S1"][i] for f in feats if f.split("_")[0] == c]) for i in range(len(names))] for c in cyts}
st_all = [np.mean([max(pf[f]["ST"][i], 0) for f in feats]) for i in range(len(names))]
s1_all = [np.mean([max(pf[f]["S1"][i], 0) for f in feats]) for i in range(len(names))]
print("total-order ST per cytokine (mean over its 4 observables); S1 in brackets")
print(f"{'param':9s}" + "".join(f"{c:>15s}" for c in cyts) + f"{'all 24':>15s}")
for i, p in enumerate(names):
    print(f"{p:9s}" + "".join(f"{ST[c][i]:7.3f} [{S1[c][i]:5.3f}]" for c in cyts) + f"{st_all[i]:7.3f} [{s1_all[i]:5.3f}]")
lab = {"il8": "IL-8", "il1": "IL-1$\\beta$", "il6": "IL-6", "il10": "IL-10", "tnf": "TNF-$\\alpha$", "tgf": "TGF-$\\beta$"}
print("\n% LaTeX rows: parameter & ST per cytokine & mean over 24 observables")
for i in np.argsort(st_all)[::-1]:
    p = names[i].replace("_", "\\_")
    print(f"{p} & " + " & ".join(f"${ST[c][i]:.3f}$" for c in cyts) + f" & ${st_all[i]:.3f}$ \\\\")
print("% columns: " + ", ".join(lab.get(c, c) for c in cyts))
