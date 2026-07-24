#!/usr/bin/env python3
# combi3D/Simulation/smore/compare_observables.py
#
# Does the choice of observable determine what SMoRe ParS can recover?
#
# The recovery reported so far uses the volume-averaged concentration trajectory
# reduced to four scalars per cytokine. That average collapses 125,000 voxels
# into one number per frame and discards the spatial structure the 3D model
# exists to produce. This script tests, on ABM fields only, whether keeping the
# spatial structure changes which parameters are identifiable.
#
# ABM ONLY, DELIBERATELY. No surrogate is involved. The question here is whether
# the observable is the limiting factor, independently of surrogate fidelity.
# If spatial observables improve recovery on ABM fields, the observable choice
# matters and the surrogate experiment should be rerun with them. If they do
# not, the limit lies elsewhere and the surrogate run can be interpreted as it
# stands. Both answers are informative.
#
# Everything downstream is unchanged: the same emulator_sobol, fit_surrogates
# and leave_one_out_recovery are called for both observable sets, on identical
# theta and identical runs, so the comparison isolates the observable.
#
# Usage:
#   python compare_observables.py \
#       --sim-root   ../sweep/outputs \
#       --infer-root ../../preprocessed_3d_infer \
#       --manifest   ../manifest.json \
#       --cytokines il8 --grid 50 --top-k 5 \
#       --out observable_comparison.json \
#       --tex ../../paper3d/table_observables.tex

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from observables import load_sweep, summarize_observable, CYTOKINES
from sensitivity import emulator_sobol, select_top_k
from smore_pars import fit_surrogates, leave_one_out_recovery
from spatial_observables import (summarize_spatial, spatial_feature_names,
                                 frame_descriptors, wound_centre)


def mean_feature_names(cyts):
    return [f"{c}_{s}" for c in cyts for s in ("final", "mean", "max", "auc")]


def load_fields_and_reduce(infer_root, run_ids, cyt_idx, cyts, grid,
                           centre_mode, verbose_every=20):
    """Stream over runs, computing both observable sets per run so the full
    field stack (~10 GB) is never held in memory at once.

    Returns (Y_mean, feats_spatial, kept_ids):
      Y_mean        (n, T, n_cyt)  volume-averaged trajectories, physical units
      feats_spatial (n, n_feat)    spatial features
    """
    gtag = f"{grid}x{grid}x{grid}"
    Y_mean_rows, spat_rows, kept = [], [], []
    t0 = time.time()

    for i, rid in enumerate(run_ids):
        d = Path(infer_root) / rid / gtag
        fp, mp = d / "Y_target.npy", d / "metadata.json"
        if not (fp.exists() and mp.exists()):
            continue
        meta = json.load(open(mp))
        Y = np.load(fp)
        if Y.ndim == 4:
            Y = Y[..., None]

        per_cyt_mean, per_cyt_feats = [], []
        for c, ci in zip(cyts, cyt_idx):
            cmax = float(meta["scaling"]["max"][ci])
            fld = Y[..., ci] if Y.shape[-1] > 1 else Y[..., 0]
            phys = np.maximum((fld.astype(np.float64) + 1.0) / 2.0 * cmax, 0.0)
            per_cyt_mean.append(phys.reshape(phys.shape[0], -1).mean(axis=1))

            ctr = wound_centre(phys[0], mode=centre_mode)
            dd = frame_descriptors(phys, ctr)
            row = []
            for k in range(dd["radial"].shape[1]):
                row.append(_reduce_time(dd["radial"][:, k]))
            for k in range(dd["layer"].shape[1]):
                row.append(_reduce_time(dd["layer"][:, k]))
            for key in ("active", "centroid_r", "spread", "q50a", "q90a", "peak"):
                row.append(_reduce_time(dd[key]))
            per_cyt_feats.append(np.concatenate(row))

        Y_mean_rows.append(np.stack(per_cyt_mean, axis=1))   # (T, n_cyt)
        spat_rows.append(np.concatenate(per_cyt_feats))
        kept.append(rid)
        del Y

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  [{i+1}/{len(run_ids)}] {time.time()-t0:.0f}s", flush=True)

    T = min(r.shape[0] for r in Y_mean_rows)
    Y_mean = np.stack([r[:T] for r in Y_mean_rows], axis=0)
    return Y_mean, np.vstack(spat_rows), kept


def _reduce_time(series):
    s = np.asarray(series, np.float64)
    T = len(s)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return np.array([s[-1], s.mean(), s.max(), _trap(s) / max(1, T - 1)])


def run_recovery(theta, feats, fnames, Y_for_surface, t_grid, names, bounds,
                 args, label):
    """Sobol on `feats`, then SMoRe ParS recovery. The stage-one surface fit
    always uses the mean trajectories (that is what fit_surrogates expects);
    only the Sobol screen and the observable matrix differ between arms."""
    print(f"\n  [{label}] observables {feats.shape}")
    sob = emulator_sobol(theta, feats, fnames, names, bounds,
                         n_saltelli=args.n_saltelli)
    if args.params:
        top_k, mode = list(args.params), "explicit"
    else:
        top_k = select_top_k(sob["ranking"], args.top_k)
        mode = f"sobol_top_{args.top_k}"
    print(f"  [{label}] Sobol ranking:")
    for r in sob["ranking"][:5]:
        print(f"      {r['param']:10s} ST={r['ST_mean']:.4f}")

    theta_sm, _ = fit_surrogates(Y_for_surface, t_grid)
    rec = leave_one_out_recovery(theta, theta_sm, names, bounds,
                                 [names.index(s) for s in top_k])
    print(f"  [{label}] recovery:")
    extra = {}
    for pn in rec["selected_params"]:
        p = rec["selected_params"].index(pn)
        recv = np.array([x[p] for x in rec["recovered"]], float)
        true = np.array([x[p] for x in rec["truth"]], float)
        corr = float(np.corrcoef(recv, true)[0, 1]) if true.std() > 0 else float("nan")
        sd_ratio = float(recv.std() / true.std()) if true.std() > 0 else float("nan")
        bias = float((recv.mean() - true.mean()) / true.std()) if true.std() > 0 else float("nan")
        extra[pn] = {"corr": corr, "sd_ratio": sd_ratio, "bias_sd": bias}
        print(f"      {pn:10s} R2={rec['r2_per_param'][pn]:+.3f} "
              f"nRMSE={rec['nrmse_per_param'][pn]:.3f} corr={corr:+.3f}")
    return {"sobol": sob, "selection_mode": mode, "top_k": top_k,
            "recovery": rec, "diagnostics": extra}


def make_tex(res_mean, res_spat, cyts, n_runs, path):
    """LaTeX table contrasting the two observable sets."""
    params = sorted(set(res_mean["recovery"]["selected_params"])
                    | set(res_spat["recovery"]["selected_params"]))
    rows = []
    for p in params:
        rm = res_mean["recovery"]["r2_per_param"].get(p)
        rs = res_spat["recovery"]["r2_per_param"].get(p)
        cm = res_mean["diagnostics"].get(p, {}).get("corr")
        cs = res_spat["diagnostics"].get(p, {}).get("corr")
        f = lambda v: "--" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"${v:+.3f}$"
        pn = p.replace("_", r"\_")
        rows.append(f"\\texttt{{{pn}}} & {f(rm)} & {f(cm)} & {f(rs)} & {f(cs)} \\\\")

    n_m = len(mean_feature_names(cyts))
    n_s = len(spatial_feature_names(cyts))
    cyt_s = ", ".join(c.upper() for c in cyts)

    tex = f"""\\begin{{table}}[!htbp]
\\centering\\footnotesize
\\caption{{Leave-one-out parameter recovery under two observable definitions,
both computed from the same ABM fields over the same {n_runs} sweep runs
({cyt_s}). The volume-averaged set reduces each cytokine to four scalars of its
volume-mean trajectory ({n_m} features); the spatially structured set adds
radial shells about the wound centre, per-depth-layer means, active-region
fraction, spatial moments, and active-voxel quantiles ({n_s} features). $R^2$ is
the coefficient of determination between recovered and true values;
$\\rho$ is their correlation, which separates loss of ordering from loss of
scale. Parameters are those selected by the Sobol screen in each arm.}}
\\label{{tab:observable_comparison}}
\\begin{{tabular}}{{lcccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Volume-averaged}} & \\multicolumn{{2}}{{c}}{{Spatially structured}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
Parameter & $R^2$ & $\\rho$ & $R^2$ & $\\rho$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    open(path, "w").write(tex)
    print(f"\nLaTeX table -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--infer-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cytokines", nargs="+", default=["il8"])
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--centre", choices=["geometric", "field"],
                    default="geometric")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--params", nargs="+", default=None)
    ap.add_argument("--n-saltelli", type=int, default=1024)
    ap.add_argument("--out", default="observable_comparison.json")
    ap.add_argument("--tex", default=None)
    args = ap.parse_args()

    bad = [c for c in args.cytokines if c not in CYTOKINES]
    if bad:
        raise SystemExit(f"unknown cytokines: {bad} (known: {CYTOKINES})")
    cyt_idx = [CYTOKINES.index(c) for c in args.cytokines]

    man = json.load(open(args.manifest))
    names, bounds = man["param_names"], man["bounds"]

    print(f"[1/3] theta and run order from the ABM sweep ...")
    theta, _, run_ids, t_grid = load_sweep(args.sim_root, names)
    print(f"      {len(run_ids)} runs, {len(names)} parameters")

    print(f"[2/3] reading fields, computing both observable sets "
          f"(cytokines {args.cytokines}) ...")
    Y_mean, feats_spat, kept = load_fields_and_reduce(
        args.infer_root, run_ids, cyt_idx, args.cytokines, args.grid,
        args.centre)

    if len(kept) != len(run_ids):
        keep_set = set(kept)
        idx = [i for i, r in enumerate(run_ids) if r in keep_set]
        theta = theta[idx]
        print(f"      {len(kept)}/{len(run_ids)} runs had usable fields")
    T = Y_mean.shape[1]
    t_grid = np.asarray(t_grid, float)[:T]

    feats_mean = summarize_observable(Y_mean)
    print(f"      volume-averaged: {feats_mean.shape[1]} features")
    print(f"      spatial        : {feats_spat.shape[1]} features")

    cv = feats_spat.std(0) / (np.abs(feats_spat.mean(0)) + 1e-30)
    dead = int((cv < 0.01).sum())
    if dead:
        print(f"      note: {dead} spatial feature(s) vary by <1% across runs "
              f"and carry little signal")

    print(f"\n[3/3] Sobol + SMoRe ParS under each observable definition ...")
    res_mean = run_recovery(theta, feats_mean, mean_feature_names(args.cytokines),
                            Y_mean, t_grid, names, bounds, args, "volume-averaged")
    res_spat = run_recovery(theta, feats_spat, spatial_feature_names(args.cytokines),
                            Y_mean, t_grid, names, bounds, args, "spatial")

    bundle = {
        "mode": "observable_comparison_abm_only",
        "cytokines": args.cytokines, "centre_mode": args.centre,
        "n_runs": len(kept), "run_ids": kept,
        "n_features": {"mean": int(feats_mean.shape[1]),
                       "spatial": int(feats_spat.shape[1])},
        "spatial_feature_cv": {n: float(v) for n, v in
                               zip(spatial_feature_names(args.cytokines), cv)},
        "volume_averaged": res_mean, "spatial": res_spat,
    }
    json.dump(bundle, open(args.out, "w"), indent=2)
    print(f"\nDONE -> {args.out}")

    print("\n=== DOES THE OBSERVABLE CHOICE MATTER? ===")
    allp = sorted(set(res_mean["recovery"]["selected_params"])
                  | set(res_spat["recovery"]["selected_params"]))
    print(f"  {'param':10s} {'R2 mean':>9s} {'R2 spatial':>11s} {'delta':>8s}")
    for p in allp:
        rm = res_mean["recovery"]["r2_per_param"].get(p)
        rs = res_spat["recovery"]["r2_per_param"].get(p)
        if rm is None or rs is None:
            print(f"  {p:10s} {'--' if rm is None else f'{rm:+9.3f}'} "
                  f"{'--' if rs is None else f'{rs:+11.3f}'} "
                  f"{'(not selected in both arms)':>8s}")
        else:
            print(f"  {p:10s} {rm:+9.3f} {rs:+11.3f} {rs-rm:+8.3f}")
    print("\n  If the spatial column is clearly higher, the observable was the "
          "limiting factor\n  and the surrogate experiment should be rerun with "
          "spatial observables.")

    if args.tex:
        make_tex(res_mean, res_spat, args.cytokines, len(kept), args.tex)


if __name__ == "__main__":
    main()