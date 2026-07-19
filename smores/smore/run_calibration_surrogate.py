#!/usr/bin/env python3
# combi3D/Simulation/smore/run_calibration_surrogate.py
#
# SMoRe ParS calibration with the SURROGATE IN THE LOOP.
#
# Difference from run_calibration.py: the scalar observables fed to the Sobol
# analysis and to SMoRe ParS are computed from the trained neural surrogate's
# predicted cytokine fields, not read from the raw ABM output. For each sweep
# run the surrogate predicts the volumetric field at every timestep, each frame
# is spatially averaged to give the mean-concentration trajectory il*_mean(t),
# and those trajectories are the observable input -- exactly the quantity
# observables.py otherwise reads from mean_concentration.txt.
#
# CYTOKINE SCOPE (report this in the paper):
# The surrogate is trained at one sweep point (run_0062) and its normalisation
# is frozen at that point's statistics. Measured across the sweep, that frozen
# scale transfers cleanly for IL-8 (~1.6% of active voxels clipped) and IL-10
# (~0.4%), but not for IL-1beta (~64%) or IL-6 (~87%), which are near-absent at
# run_0062 and therefore set a clip threshold far below the sweep-wide range.
# Predictions for those cytokines would be made from systematically flattened
# inputs, so they are excluded by default. The default --cytokines il8 il10 is
# the pair the surrogate benchmark itself is built on (the dense and sparse
# endpoints), keeping the forward model and the calibration on the same fields.
#
# HONEST SCOPE: the surrogate predicts one cytokine per trained model, and its
# trunk input still carries the ABM's other channels and cell masks for the run
# being predicted. This is a surrogate-in-the-loop with ABM-supplied context,
# not a standalone replacement for the simulator.
#
# The script reports, per cytokine and per run, how well the surrogate
# reproduces that run's true mean-concentration trajectory (generalisation R2),
# and re-runs the same Sobol + recovery pipeline on the ABM observables so the
# two can be compared directly. Matching ranking and recovery is evidence the
# surrogate can stand in for the ABM; divergence is a measured limit.
#
# Usage:
#   python run_calibration_surrogate.py \
#       --sim-root    ../sweep/outputs \
#       --infer-root  ../../preprocessed_3d_infer \
#       --weights-dir ../../models/deeponet_3d \
#       --manifest    ../manifest.json \
#       --model deeponet --run-tag run_0062 --grid 50 \
#       --cytokines il8 il10 --top-k 5 \
#       --out calibration_surrogate_results.json

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# --- make the training modules importable regardless of working directory ----
# train_deeponet_3d.py lives in scripts/deeponet/, train_unet_3d.py in
# scripts/unet/. This file sits in smores/smore/, so walk up to the burns root.
_HERE = Path(__file__).resolve()
_BURNS_ROOT = _HERE.parents[2]                 # smores/smore/ -> smores/ -> burns/
for _sub in ("scripts/deeponet", "scripts/unet"):
    _p = _BURNS_ROOT / _sub
    if _p.is_dir():
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(_HERE.parent))

# Existing SMoRe ParS pipeline -- reused UNCHANGED.
from observables import load_sweep, summarize_observable, CYTOKINES
from sensitivity import emulator_sobol, select_top_k
from smore_pars import fit_surrogates, leave_one_out_recovery


def feature_names_for(cyts):
    """FEATURE_NAMES restricted to the cytokines actually used."""
    return [f"{c}_{stat}" for c in cyts
            for stat in ("final", "mean", "max", "auc")]


def load_surrogate(model_name, weights_path, grid, best_params):
    """Rebuild the trained network and load its weights. The architecture is
    imported from the training module, so it cannot drift from the weights."""
    import importlib
    if model_name == "deeponet":
        mod = importlib.import_module("train_deeponet_3d")
        model = mod.DeepONet(hidden=best_params["hidden"], p=best_params["p"])
        dummy_b = np.zeros((1, 8), np.float32)
        dummy_t = np.zeros((1, 16, 25), np.float32)
        _ = model([dummy_b, dummy_t], training=False)
        model.load_weights(weights_path)
        return mod, model
    elif model_name == "unet":
        mod = importlib.import_module("train_unet_3d")
        model = mod.build_unet3d(grid_size=grid, in_channels=22,
                                 base_filters=best_params["base_filters"],
                                 depth=best_params["depth"],
                                 dropout=best_params.get("dropout", 0.0))
        model.load_weights(weights_path)
        return mod, model
    raise SystemExit(f"unknown --model {model_name}")


def _denorm(x_scaled, clip_max):
    return (np.asarray(x_scaled, np.float64) + 1.0) / 2.0 * clip_max


def predict_meanconc(mod, model, model_name, run_dir, cyt_idx, clip_max):
    """Predict this run's fields and spatially average each frame, giving the
    mean-concentration trajectory in physical units. Inputs are the run's own
    preprocessed frames, so this measures single-step generalisation to an
    unseen parameter point without autoregressive drift.

    The branch/trunk inputs are built by the training module's own functions, so
    the encoding cannot drift from what the loaded weights were trained on."""
    if model_name == "deeponet":
        Xb = np.load(run_dir / "X_branch.npy").astype(np.float32)
        Xt = np.load(run_dir / "X_trunk.npy").astype(np.float32)
        N = Xb.shape[0]
        G = int(round(Xt.shape[1] ** (1.0 / 3.0)))
        Xbranch = mod.build_branch_inputs(Xb, Xt, cyt_idx)
        Xtrunk = mod.build_trunk_inputs(Xb, Xt)
        del Xb, Xt
        Yp = mod.predict_full(model, Xbranch, Xtrunk).reshape(N, G, G, G)
        del Xbranch, Xtrunk
    else:
        X = np.load(run_dir / "X_unet.npy").astype(np.float32)
        Yp = model.predict(X, batch_size=1, verbose=0)[..., 0]
        del X
    Yp_phys = np.maximum(_denorm(Yp, clip_max), 0.0)
    return Yp_phys.reshape(Yp_phys.shape[0], -1).mean(axis=1)


def true_meanconc(run_dir, cyt_idx, clip_max):
    """Same quantity from the run's own ABM target field: the reference the
    surrogate is scored against."""
    Y = np.load(run_dir / "Y_target.npy").astype(np.float32)[..., cyt_idx]
    Y_phys = _denorm(Y, clip_max)
    return Y_phys.reshape(Y_phys.shape[0], -1).mean(axis=1)


def r2(truth, pred):
    truth = np.asarray(truth, float); pred = np.asarray(pred, float)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0


def build_surrogate_Y(args, names, cyts):
    """Return (theta, Y_sur, Y_abm, run_ids, t_grid, diagnostics). Y_sur holds
    surrogate-derived mean-concentration trajectories, Y_abm the ABM ones, both
    restricted to `cyts` and on the same time grid, so the two pipelines run on
    identical footing."""
    infer_root = Path(args.infer_root)
    weights_dir = Path(args.weights_dir)
    gtag = f"{args.grid}x{args.grid}x{args.grid}"

    # theta and run ordering come from the existing ABM loader (params.json), so
    # the theta<->observable pairing matches the ABM-based pipeline exactly.
    theta, Y_abm_full, run_ids, t_grid = load_sweep(args.sim_root, names)
    T = Y_abm_full.shape[1]

    cyt_idx = [CYTOKINES.index(c) for c in cyts]
    Y_abm = Y_abm_full[:, :, cyt_idx]                  # (n_runs, T, len(cyts))
    Y_sur = np.zeros_like(Y_abm)
    diagnostics = {c: {} for c in cyts}
    missing_runs = set()

    for j, (c, ci) in enumerate(zip(cyts, cyt_idx)):
        res_path = weights_dir / f"res_{c}_{args.run_tag}_{args.grid}_{args.seed}.json"
        w_path = weights_dir / f"weights_{c}_{args.run_tag}_{args.grid}_{args.seed}.weights.h5"
        if not res_path.exists() or not w_path.exists():
            raise SystemExit(
                f"missing weights for {c}: expected {w_path.name} and "
                f"{res_path.name} in {weights_dir}. Train it, or drop {c} from "
                f"--cytokines.")
        best = json.load(open(res_path))["best_params"]
        mod, model = load_surrogate(args.model, str(w_path), args.grid, best)
        print(f"[{c}] loaded {args.model} ({w_path.name})")

        t0 = time.time()
        for ri, rid in enumerate(run_ids):
            run_dir = infer_root / rid / gtag
            if not run_dir.exists():
                missing_runs.add(rid)
                continue
            meta = json.load(open(run_dir / "metadata.json"))
            clip_max = float(meta["scaling"]["max"][ci])
            mc = predict_meanconc(mod, model, args.model, run_dir, ci, clip_max)
            Tc = min(len(mc), T)
            Y_sur[ri, :Tc, j] = mc[:Tc]
            diagnostics[c][rid] = {
                "gen_r2": r2(true_meanconc(run_dir, ci, clip_max)[:Tc], mc[:Tc]),
                "clip_frac": (meta.get("clipping_vs_frozen_scale", {})
                                  .get(c, {}).get("clipped_frac")),
                "n_pred": int(Tc),
            }
        vals = [d["gen_r2"] for d in diagnostics[c].values()]
        if vals:
            print(f"[{c}] mean-conc R2 over {len(vals)} runs: "
                  f"{np.mean(vals):.3f} +/- {np.std(vals):.3f} "
                  f"(min {np.min(vals):.3f})  [{time.time()-t0:.0f}s]")

    if missing_runs:
        print(f"[warn] {len(missing_runs)} run(s) had no preprocessed inputs "
              f"under {infer_root} and contribute zeros: "
              f"{sorted(missing_runs)[:5]}{'...' if len(missing_runs) > 5 else ''}")
    return theta, Y_sur, Y_abm, run_ids, t_grid, diagnostics


def run_pipeline(theta, Y, t_grid, names, bounds, cyts, args, label):
    """Run the unchanged Sobol + SMoRe ParS pipeline on one observable tensor."""
    feats = summarize_observable(Y)
    fnames = feature_names_for(cyts)
    print(f"  [{label}] observable matrix {feats.shape}")
    sob = emulator_sobol(theta, feats, fnames, names, bounds,
                         n_saltelli=args.n_saltelli)
    if args.params:
        unknown = [p for p in args.params if p not in names]
        if unknown:
            raise SystemExit(f"--params has unknown names: {unknown}")
        top_k, mode = list(args.params), "explicit"
    else:
        top_k = select_top_k(sob["ranking"], args.top_k)
        mode = f"sobol_top_{args.top_k}"
    print(f"  [{label}] ranking (mean ST):")
    for r in sob["ranking"]:
        print(f"      {r['param']:10s} ST={r['ST_mean']:.4f}")
    theta_sm, _ = fit_surrogates(Y, t_grid)
    rec = leave_one_out_recovery(theta, theta_sm, names, bounds,
                                 [names.index(s) for s in top_k])
    print(f"  [{label}] recovery:")
    for pn in rec["selected_params"]:
        print(f"      {pn:10s} nRMSE={rec['nrmse_per_param'][pn]:.3f} "
              f"R2={rec['r2_per_param'][pn]:+.3f}")
    return {"sobol": sob, "selection_mode": mode, "top_k": top_k, "recovery": rec}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-root", required=True,
                    help="ABM sweep root (theta + reference trajectories).")
    ap.add_argument("--infer-root", required=True,
                    help="Preprocessed inputs in the surrogate's training scale.")
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", choices=["deeponet", "unet"], default="deeponet")
    ap.add_argument("--run-tag", default="run_0062")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--cytokines", nargs="+", default=["il8", "il10"],
                    help="Cytokines to build observables from. Default il8 il10: "
                         "the pair whose frozen-scale clipping is negligible and "
                         "on which the surrogate benchmark is built.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--params", nargs="+", default=None)
    ap.add_argument("--n-saltelli", type=int, default=1024)
    ap.add_argument("--skip-abm-baseline", action="store_true",
                    help="Skip the ABM-observable run used for comparison.")
    ap.add_argument("--out", default="calibration_surrogate_results.json")
    args = ap.parse_args()

    bad = [c for c in args.cytokines if c not in CYTOKINES]
    if bad:
        raise SystemExit(f"--cytokines has unknown names: {bad} "
                         f"(known: {CYTOKINES})")

    man = json.load(open(args.manifest))
    names, bounds = man["param_names"], man["bounds"]

    print(f"[1/3] surrogate-derived observables "
          f"({args.model}, {args.run_tag}, seed {args.seed}, "
          f"cytokines {args.cytokines}) ...")
    theta, Y_sur, Y_abm, run_ids, t_grid, diag = build_surrogate_Y(
        args, names, args.cytokines)

    print(f"[2/3] Sobol + SMoRe ParS on SURROGATE observables ...")
    sur = run_pipeline(theta, Y_sur, t_grid, names, bounds,
                       args.cytokines, args, "surrogate")

    abm = None
    if not args.skip_abm_baseline:
        print(f"[3/3] same pipeline on ABM observables (baseline) ...")
        abm = run_pipeline(theta, Y_abm, t_grid, names, bounds,
                           args.cytokines, args, "ABM")

    gen = {}
    for c, runs in diag.items():
        v = [d["gen_r2"] for d in runs.values()]
        cf = [d["clip_frac"] for d in runs.values() if d["clip_frac"] is not None]
        if v:
            gen[c] = {"gen_r2_mean": float(np.mean(v)),
                      "gen_r2_std": float(np.std(v)),
                      "gen_r2_min": float(np.min(v)),
                      "n_runs": len(v),
                      "clip_frac_mean": float(np.mean(cf)) if cf else None}

    bundle = {
        "mode": "surrogate_in_the_loop",
        "model": args.model, "run_tag": args.run_tag, "seed": args.seed,
        "cytokines": args.cytokines,
        "n_runs": len(run_ids), "run_ids": run_ids,
        "generalisation": gen, "generalisation_per_run": diag,
        "surrogate": sur, "abm_baseline": abm,
    }
    json.dump(bundle, open(args.out, "w"), indent=2)
    print(f"\nDONE -> {args.out}")

    if abm is not None:
        print("\n=== SURROGATE vs ABM (the comparison that matters) ===")
        s_rank = [r["param"] for r in sur["sobol"]["ranking"]]
        a_rank = [r["param"] for r in abm["sobol"]["ranking"]]
        print(f"  Sobol top-3  surrogate: {s_rank[:3]}")
        print(f"  Sobol top-3  ABM      : {a_rank[:3]}")
        print(f"  same top-3 (unordered): {set(s_rank[:3]) == set(a_rank[:3])}")
        print("  recovery R2 per parameter (surrogate vs ABM):")
        for p in sur["recovery"]["selected_params"]:
            rs = sur["recovery"]["r2_per_param"].get(p)
            ra = abm["recovery"]["r2_per_param"].get(p)
            if rs is not None and ra is not None:
                print(f"    {p:10s} {rs:+.3f}  vs  {ra:+.3f}   (d={rs-ra:+.3f})")
    print("\nReport in the paper: observables came from surrogate predictions; "
          f"scope {args.cytokines}; per-cytokine generalisation R2 is in "
          f"'generalisation'; excluded cytokines and why in the clipping table.")


if __name__ == "__main__":
    main()