#!/usr/bin/env python3
# combi3D/Simulation/smores/helpers/predictive_consistency.py
#
# WHY THIS EXISTS
# Leave-one-out recovery of keil8 from surrogate-derived observables gives
# R2 = 0.191, against 0.375 from ABM observables computed the same way. Taken
# alone that says the recovered parameters are inaccurate. It does NOT say
# whether a model built from them predicts badly.
#
# Those are different failures with different consequences:
#
#   NON-IDENTIFIABLE BUT PREDICTIVE
#     Recovered parameters are wrong, yet the trajectory they imply matches the
#     truth. The parameter space has directions the observables cannot see, so
#     several parameter sets explain the same data. The calibrated model is
#     usable for prediction; it just cannot be read as biology.
#
#   NON-PREDICTIVE
#     Recovered parameters are wrong AND the implied trajectory is wrong. The
#     calibration has genuinely failed.
#
# This script decides which one holds, using SMoRe ParS's own machinery:
#
#   1. Stage-A surfaces A/(1+exp(-k(t-t0))) are fitted to every run's
#      trajectory, giving theta_SM per run.
#   2. For each held-out run i, a GP mapping theta_ABM -> theta_SM is fitted on
#      the other n-1 runs (exactly as leave_one_out_recovery does).
#   3. The recovered theta_ABM[i] is pushed forward through that mapping to
#      predicted surface parameters, and the surface is evaluated on the time
#      grid.
#   4. That predicted trajectory is scored against the run's TRUE ABM
#      trajectory.
#
# Two reference levels make the number interpretable:
#   - surface-fit ceiling: the stage-A surface fitted directly to the true
#     trajectory. No recovery can beat this; it is the cost of the logistic
#     form itself.
#   - forward-map baseline: the surface predicted from the TRUE theta_ABM.
#     This isolates GP mapping error from recovery error.
#
# HONEST SCOPE
# The comparison is between arms that differ only in where the observables came
# from. It does not test the surrogate as a forward model in a calibration loop:
# this surrogate needs ABM frames as input, so it cannot be driven from theta
# alone. That limitation is structural and is reported separately.
#
# Usage:
#   python predictive_consistency.py \
#       --calibration ../../calibration_surrogate_il8.json \
#       --sim-root    ../sweep/outputs \
#       --infer-root  ../../preprocessed_3d_infer \
#       --manifest    ../manifest.json \
#       --smore-dir   ../smore \
#       --cytokine il8 --grid 50 \
#       --out predictive_consistency.json \
#       --tex table_predictive_consistency.tex \
#       --log predictive_consistency.txt

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]


class Tee:
    """Write everything printed to both stdout and a log file, so the terminal
    output is preserved without needing the shell to redirect it."""

    def __init__(self, path):
        self.f = open(path, "w") if path else None

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def saturating(t, A, k, t0):
    return A / (1.0 + np.exp(-k * (t - t0)))


def norm_time(t_grid):
    t = np.asarray(t_grid, float)
    return (t - t[0]) / max(1e-9, (t[-1] - t[0]))


def r2(truth, pred):
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-300 else float("nan")


def grid_mean_traj(infer_dir, cyt_idx):
    """Volume-averaged trajectory in physical units, from the preprocessed
    field. This is the quantity both the ABM arm and the surrogate arm of the
    paired comparison use."""
    fp = Path(infer_dir) / "Y_target.npy"
    mp = Path(infer_dir) / "metadata.json"
    if not (fp.exists() and mp.exists()):
        return None
    cmax = float(json.load(open(mp))["scaling"]["max"][cyt_idx])
    Y = np.load(fp)
    fld = Y[..., cyt_idx] if Y.ndim == 5 else Y
    phys = np.maximum((fld.astype(np.float64) + 1.0) / 2.0 * cmax, 0.0)
    return phys.reshape(phys.shape[0], -1).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", required=True,
                    help="calibration_surrogate_il8.json, holding recovered "
                         "and true theta for both arms")
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--infer-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--smore-dir", required=True)
    ap.add_argument("--cytokine", default="il8", choices=CYTOKINES)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--out", default="predictive_consistency.json")
    ap.add_argument("--tex", default=None)
    ap.add_argument("--log", default="predictive_consistency.txt")
    args = ap.parse_args()

    say = Tee(args.log)
    sys.path.insert(0, str(Path(args.smore_dir).resolve()))
    from observables import load_sweep
    from smore_pars import fit_surrogates, ABMtoSMMapping

    ci = CYTOKINES.index(args.cytokine)
    gtag = f"{args.grid}x{args.grid}x{args.grid}"
    man = json.load(open(args.manifest))
    names = man["param_names"]

    say(f"[1/4] loading calibration results from {args.calibration}")
    cal = json.load(open(args.calibration))
    run_ids_cal = cal.get("run_ids", [])
    arms_in = {}
    for key, label in (("surrogate", "surrogate observables"),
                       ("abm_baseline", "ABM observables")):
        if cal.get(key) and cal[key].get("recovery"):
            rec = cal[key]["recovery"]
            arms_in[key] = {
                "label": label,
                "params": rec["selected_params"],
                "recovered": np.array(rec["recovered"], float),
                "truth": np.array(rec["truth"], float),
            }
            say(f"      {label}: {len(rec['selected_params'])} params, "
                f"{len(rec['recovered'])} runs")
    if not arms_in:
        raise SystemExit("no recovery block found in the calibration file")

    say(f"[2/4] loading trajectories ({args.cytokine}, volume-averaged) ...")
    theta_all, _, run_ids, t_grid = load_sweep(args.sim_root, names)
    traj, keep = [], []
    for rid in run_ids:
        y = grid_mean_traj(Path(args.infer_root) / rid / gtag, ci)
        if y is None:
            continue
        traj.append(y)
        keep.append(rid)
    T = min(len(y) for y in traj)
    Y = np.stack([y[:T] for y in traj])
    idx = [run_ids.index(r) for r in keep]
    theta_all = theta_all[idx]
    tg = np.asarray(t_grid, float)[:T]
    tn = norm_time(tg)
    say(f"      {len(keep)} runs, {T} frames")

    if run_ids_cal and keep != run_ids_cal:
        common = [r for r in keep if r in set(run_ids_cal)]
        say(f"      note: aligning to {len(common)} runs present in both")
        sel = [keep.index(r) for r in common]
        Y = Y[sel]
        theta_all = theta_all[sel]
        keep = common

    say(f"[3/4] fitting stage-A surfaces and reference levels ...")
    Ypad = np.zeros((Y.shape[0], T, 6))
    Ypad[:, :, ci] = Y
    theta_sm, sm_names = fit_surrogates(Ypad, tg)
    cols = slice(ci * 3, (ci + 1) * 3)

    ceiling = np.array([r2(Y[i], saturating(tn, *theta_sm[i, cols]))
                        for i in range(len(Y))])
    say(f"      surface-fit ceiling: R2 = {np.nanmean(ceiling):+.3f} "
        f"+/- {np.nanstd(ceiling):.3f} (median {np.nanmedian(ceiling):+.3f})")
    say("      this is the best any recovery could do; it is the cost of the "
        "logistic form")

    n = len(Y)
    fwd = np.full(n, np.nan)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        m = ABMtoSMMapping().fit(theta_all[tr], theta_sm[tr])
        pred_sm = m.predict_sm(theta_all[i][None, :])[0]
        fwd[i] = r2(Y[i], saturating(tn, *pred_sm[cols]))
    say(f"      forward-map baseline: R2 = {np.nanmean(fwd):+.3f} "
        f"+/- {np.nanstd(fwd):.3f} (median {np.nanmedian(fwd):+.3f})")
    say("      surfaces predicted from the TRUE theta; isolates GP mapping error")

    say(f"\n[4/4] trajectories implied by recovered parameters ...")
    results = {}
    for key, arm in arms_in.items():
        sel_idx = [names.index(p) for p in arm["params"]]
        rec, tru = arm["recovered"], arm["truth"]
        m_ = min(len(rec), n)
        scores = np.full(m_, np.nan)
        for i in range(m_):
            tr = [j for j in range(n) if j != i]
            mp = ABMtoSMMapping().fit(theta_all[tr][:, sel_idx], theta_sm[tr])
            pred_sm = mp.predict_sm(rec[i][None, :])[0]
            scores[i] = r2(Y[i], saturating(tn, *pred_sm[cols]))

        par_r2 = {}
        for c, p in enumerate(arm["params"]):
            if tru[:, c].std() > 1e-15:
                par_r2[p] = r2(tru[:, c], rec[:, c])

        results[key] = {
            "label": arm["label"],
            "trajectory_r2_mean": float(np.nanmean(scores)),
            "trajectory_r2_std": float(np.nanstd(scores)),
            "trajectory_r2_median": float(np.nanmedian(scores)),
            "trajectory_r2_frac_above_0.9": float(np.nanmean(scores > 0.9)),
            "trajectory_r2_per_run": {keep[i]: float(scores[i])
                                      for i in range(m_)},
            "parameter_r2": par_r2,
        }
        say(f"\n  [{arm['label']}]")
        say(f"      trajectory R2 : {np.nanmean(scores):+.3f} "
            f"+/- {np.nanstd(scores):.3f}  (median {np.nanmedian(scores):+.3f}, "
            f"{100*np.nanmean(scores > 0.9):.0f}% above 0.9)")
        say(f"      parameter  R2 : " + ", ".join(
            f"{p}={v:+.3f}" for p, v in par_r2.items()))

    bundle = {
        "mode": "predictive_consistency",
        "cytokine": args.cytokine, "n_runs": len(keep), "run_ids": keep,
        "reference_levels": {
            "surface_fit_ceiling_mean": float(np.nanmean(ceiling)),
            "surface_fit_ceiling_std": float(np.nanstd(ceiling)),
            "forward_map_baseline_mean": float(np.nanmean(fwd)),
            "forward_map_baseline_std": float(np.nanstd(fwd)),
        },
        "arms": results,
    }
    json.dump(bundle, open(args.out, "w"), indent=2)
    say(f"\nDONE -> {args.out}")

    say("\n=== IDENTIFIABILITY VERSUS PREDICTION ===")
    say(f"  surface-fit ceiling   : {np.nanmean(ceiling):+.3f}")
    say(f"  forward-map baseline  : {np.nanmean(fwd):+.3f}")
    for key, r in results.items():
        pr = r["parameter_r2"]
        best = max(pr.values()) if pr else float("nan")
        say(f"  {r['label']:22s} trajectory {r['trajectory_r2_mean']:+.3f}"
            f"   best parameter {best:+.3f}")
    say("")
    say("  If trajectory R2 stays close to the ceiling while parameter R2 is")
    say("  low, the calibration is non-identifiable but predictive: the")
    say("  observables cannot separate parameter sets that imply the same")
    say("  dynamics. If both are low, the calibration itself has failed.")

    if args.tex:
        rows = []
        for key, r in results.items():
            pr = r["parameter_r2"]
            best = max(pr.values()) if pr else float("nan")
            rows.append(f"{r['label']} & ${r['trajectory_r2_mean']:+.3f}$ & "
                        f"${r['trajectory_r2_median']:+.3f}$ & "
                        f"${100*r['trajectory_r2_frac_above_0.9']:.0f}\\%$ & "
                        f"${best:+.3f}$ \\\\")
        tex = f"""\\begin{{table}}[!htbp]
\\centering\\footnotesize
\\caption{{Identifiability contrasted with predictive accuracy
({args.cytokine.upper()}, {len(keep)} runs, leave-one-out). For each held-out
run the recovered ABM parameters are pushed through the fitted
parameter-to-surface mapping and the resulting trajectory is scored against that
run's true trajectory. The surface-fit ceiling
(${np.nanmean(ceiling):+.3f}$) is the accuracy of the logistic surface fitted
directly to the truth, and bounds what any recovery can achieve; the
forward-map baseline (${np.nanmean(fwd):+.3f}$) uses the true parameters and so
isolates mapping error. A high trajectory $R^2$ alongside a low parameter $R^2$
indicates a calibration that is not identifiable yet still predictive.}}
\\label{{tab:predictive_consistency}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Observables & \\multicolumn{{3}}{{c}}{{Trajectory $R^2$}} & Best parameter $R^2$ \\\\
\\cmidrule(lr){{2-4}}
 & mean & median & $>0.9$ & \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        Path(args.tex).parent.mkdir(parents=True, exist_ok=True)
        open(args.tex, "w").write(tex)
        say(f"\nLaTeX table -> {args.tex}")

    say(f"log -> {args.log}")
    say.close()


if __name__ == "__main__":
    main()