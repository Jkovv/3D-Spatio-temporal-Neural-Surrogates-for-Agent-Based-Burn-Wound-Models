# Punkt 6: czy surowe pola maja minimum 0 (podpis tabeli datasetu).
# Punkt 7: ile runow jest w loopie i czy run_0062 jest wsrod nich (podpis tab:loop).
# Uzycie: $PY scripts/check_misc.py $BASE
import sys, glob, os, json
import numpy as np

B = sys.argv[1]
cyts = ["il1", "il6", "il8", "il10", "tgf", "tnf"]
mins = {c: [] for c in cyts}; zeros = {c: [] for c in cyts}
for d in sorted(glob.glob(os.path.join(B, "sweep/outputs/run_*"))):
    z = np.load(os.path.join(d, "LatticeData", "CytoStep_1000000.npz"))
    for c in cyts:
        a = z[c]
        mins[c].append(float(a.min())); zeros[c].append(int((a == 0).sum()))
print("=== PUNKT 6: surowe pola, t = 100 h ===")
for c in cyts:
    print(f"{c:5s} min(minimum po runach)={min(mins[c]):.3e}  max(minimum)={max(mins[c]):.3e}  "
          f"runy z min==0: {sum(m == 0 for m in mins[c])}/100  "
          f"srednio dokladnych zer: {np.mean(zeros[c]):.1f} wokseli")

print("\n=== PUNKT 7: loop ===")
R = os.path.join(B, "smores/results")
for f in ["calibration_surrogate_il8.json", "calibration_surrogate_il8_seed1.json",
          "calibration_surrogate_il8_seed100.json"]:
    d = json.load(open(os.path.join(R, f)))
    ids = d.get("run_ids", [])
    gpr = d.get("generalisation_per_run")
    n_gpr = len(gpr) if isinstance(gpr, (list, dict)) else gpr
    print(f"{f}: n_runs={d.get('n_runs')}  len(run_ids)={len(ids)}  "
          f"run_0062 w run_ids={'run_0062' in ids}  generalisation n_runs="
          f"{d['generalisation']['il8'].get('n_runs')}  len(generalisation_per_run)={n_gpr}")
