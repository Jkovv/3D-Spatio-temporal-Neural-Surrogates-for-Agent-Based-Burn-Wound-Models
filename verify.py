#!/usr/bin/env python3
# combi3D/verify.py
#
# Self-check for the whole pipeline WITHOUT needing CompuCell3D.
# Run this first to confirm everything is wired correctly:
#
#     python verify.py
#
# It checks, in order:
#   1. setup_runs.py generates a valid manifest (deterministic, in-bounds).
#   2. param_loader applies a per-run vector correctly (rates + int cell counts).
#   3. param_loader fails LOUD on a bad parameter name / missing file.
#   4. run_sweep stages per-run dirs with a local params.json each.
#   5. a synthetic sweep flows through observables -> Sobol -> SMoRe ParS.
#
# Anything that prints [FAIL] needs attention; all [ok] means the machinery is
# sound and the only remaining unknown is CC3D itself (run one sim to confirm
# it produces datafiles/mean_concentration.txt).

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE  # simulation files live next to this script
SMORE = HERE / "smore"

PARAM_NAMES = ["keil8", "km1il6", "km2il10", "km2tgf", "lnril8", "sigmoidb",
               "init_ec", "init_n", "init_m", "init_f"]

ok = True
def check(label, cond, detail=""):
    global ok
    mark = "[ok]  " if cond else "[FAIL]"
    if not cond:
        ok = False
    print(f"  {mark} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    return cond


def main():
    tmp = Path(tempfile.mkdtemp(prefix="combi3d_verify_"))
    print(f"workspace: {tmp}\n")

    # 1. manifest generation -------------------------------------------------
    print("1. setup_runs.py — manifest generation")
    man_path = tmp / "manifest.json"
    subprocess.run([sys.executable, str(SIM / "setup_runs.py"),
                    "--n-runs", "10", "--seed", "42", "--out", str(man_path)],
                   check=True, capture_output=True)
    man = json.load(open(man_path))
    check("manifest has 10 runs", len(man["runs"]) == 10)
    check("param_names match", man["param_names"] == PARAM_NAMES)
    # determinism
    man_path2 = tmp / "manifest2.json"
    subprocess.run([sys.executable, str(SIM / "setup_runs.py"),
                    "--n-runs", "10", "--seed", "42", "--out", str(man_path2)],
                   check=True, capture_output=True)
    check("same seed -> identical manifest",
          json.load(open(man_path2))["runs"] == man["runs"])
    # in-bounds
    in_bounds = all(
        man["bounds"][n]["low"] - 1e-9 <= r["params"][n] <= man["bounds"][n]["high"] + 1e-9
        for r in man["runs"] for n in PARAM_NAMES)
    check("all sampled values within bounds", in_bounds)
    # cell counts are ints
    ints_ok = all(isinstance(r["params"][n], int)
                  for r in man["runs"]
                  for n in ("init_ec", "init_n", "init_m", "init_f"))
    check("cell counts are integers", ints_ok)

    # 2. param_loader applies a vector ---------------------------------------
    print("\n2. param_loader — applies per-run vector")
    run7 = next(r for r in man["runs"] if r["run_id"] == "run_0007")
    pjson = tmp / "p7.json"
    json.dump(run7, open(pjson, "w"))
    env = dict(os.environ, SMORE_PARAMS=str(pjson))
    code = (
        "import params_biology as b, params_transitions as t;"
        "import json,sys;"
        "print(json.dumps({'init_ec':b.init_ec,'init_n':b.init_n,"
        "'keil8':t.keil8,'lnril8':t.lnril8,'sigmoidb':t.sigmoidb,"
        "'actnr_lnril8':t.actnr[0][1]}))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(SIM), env=env,
                         capture_output=True, text=True)
    vals = json.loads([l for l in out.stdout.splitlines() if l.startswith("{")][0])
    check("init_ec overridden (int)", vals["init_ec"] == run7["params"]["init_ec"]
          and isinstance(vals["init_ec"], int))
    check("keil8 overridden (rate)",
          abs(vals["keil8"] - run7["params"]["keil8"]) < 1e-18)
    check("derived actnr sees overridden lnril8",
          abs(vals["actnr_lnril8"] - run7["params"]["lnril8"]) < 1e-12)

    # 3. fail-loud behaviour -------------------------------------------------
    print("\n3. param_loader — fails loud on bad input")
    bad = tmp / "bad.json"
    json.dump({"params": {"keil8": 1e-8, "not_a_param": 9}}, open(bad, "w"))
    r = subprocess.run([sys.executable, "-c", "import params_transitions"],
                       cwd=str(SIM), env=dict(os.environ, SMORE_PARAMS=str(bad)),
                       capture_output=True, text=True)
    check("unknown param name -> crash", r.returncode != 0 and "unknown" in r.stderr.lower())
    r = subprocess.run([sys.executable, "-c", "import params_transitions"],
                       cwd=str(SIM),
                       env=dict(os.environ, SMORE_PARAMS=str(tmp / "nope.json")),
                       capture_output=True, text=True)
    check("missing file -> crash (no silent baseline)", r.returncode != 0)

    # 4. run_sweep staging ---------------------------------------------------
    print("\n4. run_sweep.py — per-run directory staging")
    (tmp / "combi3D.cc3d").write_text("<Simulation/>\n")
    subprocess.run([sys.executable, str(SIM / "run_sweep.py"),
                    "--manifest", str(man_path), "--mode", "slurm",
                    "--cc3d-python", "/fake/python",
                    "--cc3d-file", str(tmp / "combi3D.cc3d"),
                    "--runs-dir", str(tmp / "runs"),
                    "--out", str(tmp / "outputs"),
                    "--work", str(tmp / "work"), "--account", "x"],
                   check=True, capture_output=True)
    staged_ok = (tmp / "runs" / "run_0007" / "Simulation" / "params.json").exists()
    check("run_0007 staged with local params.json", staged_ok)
    array_ok = (tmp / "work" / "sweep_array.sh").exists()
    check("SLURM array script emitted", array_ok)
    # local params.json loads without env var
    sim7 = tmp / "runs" / "run_0007" / "Simulation"
    out = subprocess.run([sys.executable, "-c",
                          "import params_biology as b;print(b.init_ec)"],
                         cwd=str(sim7),
                         env={k: v for k, v in os.environ.items()
                              if k != "SMORE_PARAMS"},
                         capture_output=True, text=True)
    loaded = out.stdout.strip().splitlines()[-1]
    check("local params.json loads without env var",
          loaded == str(run7["params"]["init_ec"]))

    # 5. calibration pipeline on a synthetic sweep ---------------------------
    print("\n5. smore — observables -> Sobol -> SMoRe ParS (synthetic)")
    syn = tmp / "syn_out"
    _make_synthetic_sweep(man, syn)
    res_path = tmp / "calib.json"
    r = subprocess.run([sys.executable, str(SMORE / "run_calibration.py"),
                        "--sim-root", str(syn), "--manifest", str(man_path),
                        "--top-k", "5", "--n-saltelli", "256",
                        "--out", str(res_path)],
                       capture_output=True, text=True)
    check("calibration runs end-to-end", r.returncode == 0,
          r.stderr[-300:] if r.returncode else "")
    if res_path.exists():
        res = json.load(open(res_path))
        check("Sobol produced a ranking", len(res["sobol"]["ranking"]) == 10)
        check("recovery produced nRMSE", "nrmse_mean" in res["recovery"])
        # the injected drivers should rank near the top
        top5 = set(res["top_k"])
        injected = {"keil8", "km1il6", "km2il10", "km2tgf", "init_ec"}
        overlap = len(top5 & injected)
        check(f"Sobol recovers injected drivers ({overlap}/5 in top-5)",
              overlap >= 3)

    print("\n" + ("ALL CHECKS PASSED — machinery is sound."
                  if ok else "SOME CHECKS FAILED — see [FAIL] above."))
    print("Remaining unknown: CC3D itself. Run one sim and confirm "
          "outputs/<run>/datafiles/mean_concentration.txt appears.")
    sys.exit(0 if ok else 1)


def _make_synthetic_sweep(man, root):
    import numpy as np, csv
    root.mkdir(parents=True, exist_ok=True)
    T = 40
    mcs = np.linspace(0, 1_000_000, T)
    tn = (mcs - mcs[0]) / (mcs[-1] - mcs[0])
    rng = np.random.default_rng(0)
    def sat(A, k, t0): return A / (1 + np.exp(-k * (tn - t0)))
    for run in man["runs"]:
        p = run["params"]; rid = run["run_id"]
        d = root / rid / "datafiles"; d.mkdir(parents=True, exist_ok=True)
        (root / rid / "LatticeData").mkdir(exist_ok=True)
        json.dump({"run_id": rid, "params": p}, open(root / rid / "params.json", "w"))
        il8 = sat(p["keil8"]*1e8*(p["init_ec"]/100.0), 8, 0.4)
        il6 = sat(p["km1il6"]*1e8, 6, 0.5)
        il10 = sat(p["km2il10"]*1e8, 9, 0.6)
        tgf = sat(p["km2tgf"]*1e8, 5, 0.6)
        il1 = sat(0.5, 7, 0.5); tnf = sat(0.4, 7, 0.5)
        arr = np.clip(np.stack([il8, il1, il6, il10, tnf, tgf], 1)
                      + rng.normal(0, 1e-3, (T, 6)), 0, None)
        with open(d / "mean_concentration.txt", "w") as f:
            w = csv.writer(f)
            w.writerow(["meanconcen", "il8mean", "il1mean", "il6mean", "il10mean",
                        "tnfmean", "tgfmean", "il8std", "il1std", "il6std",
                        "il10std", "tnfstd", "tgfstd"])
            for i in range(T):
                w.writerow([int(mcs[i])] + [f"{v:.6e}" for v in arr[i]] + ["0"]*6)


if __name__ == "__main__":
    main()
