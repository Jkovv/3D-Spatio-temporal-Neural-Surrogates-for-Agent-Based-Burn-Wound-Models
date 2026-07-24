#!/usr/bin/env python3
# combi3D/Simulation/helpers/decompose_observable_gap.py
#
# WHY THIS EXISTS
# Three recovery numbers for keil8 have been measured, and they disagree:
#
#   0.838  SMoRe ParS on mean_concentration.txt   (ABM native observable)
#   0.375  SMoRe ParS on grid-averaged fields     (preprocessed, clipped)
#   0.103  SMoRe ParS on spatial observables      (preprocessed, clipped)
#
# The first two differ in TWO ways at once, so the comparison behind them is not
# paired and cannot be interpreted as it stands:
#
#   (a) DEFINITION. mean_concentration.txt is written by the ABM as
#       np.mean(il8_list), where il8_list holds one concentration sampled at
#       each cell's centre of mass (xCOM, yCOM, zCOM). The grid variant averages
#       over all 125,000 voxels. Cells occupy ~0.1-0.2% of the lattice, so these
#       are different quantities, differing by ~170x.
#
#   (b) CLIPPING. The preprocessed fields are percentile-clipped for surrogate
#       training. Cells sit where concentration is highest, so cell-sampled
#       values are exactly the ones the clip saturates: sampling the
#       preprocessed field at cell voxels returns clip_max at t=0.
#
# This script separates (a) from (b) by computing all four combinations from the
# RAW LatticeData fields, which are unclipped and in physical units:
#
#                        raw fields        clipped fields
#   cell-sampled         A                 C
#   grid-averaged        B                 D
#
#   A vs B  -> effect of the observable definition, clipping held out
#   A vs C  -> effect of clipping, definition held fixed
#   D       -> reproduces the 0.375 arm as a consistency check
#
# The point is not to rescue any particular number but to report an honest
# decomposition: how much of the recovery advantage of the ABM's native
# observable comes from sampling at cells, and how much from being unclipped.
# Whatever the split, it is stated rather than assumed.
#
# NOTE ON WHAT THIS DOES NOT DO
# It does not involve the surrogate. The surrogate can only ever produce
# clipped-space predictions, so arms A and B are not reachable from it. That
# asymmetry is itself a result and belongs in the paper's limitations.
#
# Usage:
#   python decompose_observable_gap.py \
#       --sim-root   ../smores/sweep/outputs \
#       --infer-root ../preprocessed_3d_infer \
#       --manifest   ../smores/manifest.json \
#       --smore-dir  ../smores/smore \
#       --cytokine il8 --grid 50 --top-k 5 \
#       --out observable_gap_decomposition.json \
#       --tex table_observable_gap.tex

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]

# CellStep cell_type ids that count as cells for the cell-sampled observable.
# 0 is medium. The ABM samples at the centre of mass of every cell it iterates
# over, so every non-zero type is included unless --cell-types narrows it.
DEFAULT_CELL_TYPES = None   # None -> all non-zero


def _step_of(path):
    m = re.search(r"_(\d+)\.npz$", str(path))
    return int(m.group(1)) if m else -1


def load_raw_run(run_dir, cyt, cell_types=None):
    """Read one run's raw LatticeData.

    Returns (mean_cells, mean_grid) as (T,) arrays in physical units:
      mean_cells : concentration averaged over voxels occupied by cells,
                   the closest reconstruction of the ABM's own observable
      mean_grid  : concentration averaged over the whole lattice
    """
    cyto = sorted(glob.glob(str(Path(run_dir) / "LatticeData" / "CytoStep_*.npz")),
                  key=_step_of)
    cell = sorted(glob.glob(str(Path(run_dir) / "LatticeData" / "CellStep_*.npz")),
                  key=_step_of)
    if not cyto or not cell:
        return None, None
    n = min(len(cyto), len(cell))

    mc = np.zeros(n)
    mg = np.zeros(n)
    for t in range(n):
        f = np.load(cyto[t])[cyt].astype(np.float64)
        ct = np.load(cell[t])["cell_type"]
        mask = (ct > 0) if cell_types is None else np.isin(ct, list(cell_types))
        mg[t] = f.mean()
        mc[t] = f[mask].mean() if mask.any() else 0.0
    return mc, mg


def load_clipped_run(infer_dir, cyt_idx, cell_types=None, cell_mask_from_raw=None):
    """Same two observables, computed from the preprocessed (clipped) field.

    The preprocessed tensor stores cytokines in [-1, 1] plus binary cell masks,
    so the cell mask is taken from its mask channels when available and from the
    raw CellStep otherwise (passed in as cell_mask_from_raw)."""
    fp = Path(infer_dir) / "Y_target.npy"
    mp = Path(infer_dir) / "metadata.json"
    bp = Path(infer_dir) / "X_branch.npy"
    if not (fp.exists() and mp.exists()):
        return None, None
    meta = json.load(open(mp))
    cmax = float(meta["scaling"]["max"][cyt_idx])

    Y = np.load(fp)
    fld = Y[..., cyt_idx] if Y.ndim == 5 else Y
    phys = np.maximum((fld.astype(np.float64) + 1.0) / 2.0 * cmax, 0.0)
    T = phys.shape[0]

    masks = None
    if bp.exists():
        Xb = np.load(bp, mmap_mode="r")
        masks = np.asarray(Xb[:T, 0, ..., 6:]).max(axis=-1) > 0.5
    elif cell_mask_from_raw is not None:
        masks = cell_mask_from_raw[:T]

    mg = phys.reshape(T, -1).mean(axis=1)
    mc = np.zeros(T)
    for t in range(T):
        if masks is not None and masks[t].any():
            mc[t] = phys[t][masks[t]].mean()
    return mc, mg


def summarise(Y):
    """(n_runs, T) -> (n_runs, 4): final, mean, max, auc. Matches
    observables.summarize_observable for a single cytokine, so the four arms are
    reduced identically."""
    n, T = Y.shape
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return np.stack([Y[:, -1], Y.mean(axis=1), Y.max(axis=1),
                     _trap(Y, axis=1) / max(1, T - 1)], axis=1)


def recovery_for(theta, feats, fnames, Y_traj, t_grid, names, bounds,
                 args, label, mods):
    """Sobol screen + leave-one-out SMoRe ParS recovery for one arm."""
    emulator_sobol = mods["emulator_sobol"]
    select_top_k = mods["select_top_k"]
    fit_surrogates = mods["fit_surrogates"]
    leave_one_out_recovery = mods["leave_one_out_recovery"]

    sob = emulator_sobol(theta, feats, fnames, names, bounds,
                         n_saltelli=args.n_saltelli)
    if args.params:
        top_k = list(args.params)
    else:
        top_k = select_top_k(sob["ranking"], args.top_k)

    theta_sm, _ = fit_surrogates(Y_traj[:, :, None], t_grid)
    rec = leave_one_out_recovery(theta, theta_sm, names, bounds,
                                 [names.index(s) for s in top_k])

    diag = {}
    for pn in rec["selected_params"]:
        p = rec["selected_params"].index(pn)
        r = np.array([x[p] for x in rec["recovered"]], float)
        t = np.array([x[p] for x in rec["truth"]], float)
        diag[pn] = {
            "corr": float(np.corrcoef(r, t)[0, 1]) if t.std() > 0 else float("nan"),
            "sd_ratio": float(r.std() / t.std()) if t.std() > 0 else float("nan"),
            "bias_sd": float((r.mean() - t.mean()) / t.std()) if t.std() > 0 else float("nan"),
        }

    print(f"  [{label}] top-{args.top_k}: {top_k}")
    for pn in rec["selected_params"]:
        print(f"      {pn:10s} R2={rec['r2_per_param'][pn]:+.3f} "
              f"nRMSE={rec['nrmse_per_param'][pn]:.3f} "
              f"corr={diag[pn]['corr']:+.3f}")
    return {"sobol_ranking": sob["ranking"], "top_k": top_k,
            "recovery": {k: rec[k] for k in
                         ("selected_params", "r2_per_param", "nrmse_per_param")},
            "diagnostics": diag}


def make_tex(res, cyt, n_runs, path, focus):
    """Decomposition table."""
    arms = [("raw_cell", "Raw fields, cell-sampled",
             "the ABM's own observable, reconstructed"),
            ("raw_grid", "Raw fields, grid-averaged", "definition changed"),
            ("clip_cell", "Clipped fields, cell-sampled", "clipping added"),
            ("clip_grid", "Clipped fields, grid-averaged",
             "both changed; the arm the surrogate can reach")]
    rows = []
    for key, label, note in arms:
        a = res.get(key)
        if a is None:
            rows.append(f"{label} & -- & -- & \\textit{{{note}}} \\\\")
            continue
        r2 = a["recovery"]["r2_per_param"].get(focus)
        co = a["diagnostics"].get(focus, {}).get("corr")
        f = lambda v: "--" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"${v:+.3f}$"
        rows.append(f"{label} & {f(r2)} & {f(co)} & \\textit{{{note}}} \\\\")

    tex = f"""\\begin{{table}}[!htbp]
\\centering\\footnotesize
\\caption{{Decomposition of the recovery gap for \\texttt{{{focus}}}
({cyt.upper()}, {n_runs} runs). The observable the ABM writes during simulation
is the concentration sampled at each cell's centre of mass and averaged over
cells; the surrogate can only produce percentile-clipped fields, from which that
quantity is not recoverable, because cells occupy the voxels the clip saturates.
The four arms separate the two confounded factors: sampling at cells versus
averaging over the lattice, and unclipped versus clipped fields. $R^2$ is
leave-one-out recovery against the known sweep parameters; $\\rho$ is the
correlation between recovered and true values, which distinguishes loss of
ordering from loss of scale.}}
\\label{{tab:observable_gap}}
\\begin{{tabular}}{{lccl}}
\\toprule
Observable & $R^2$ & $\\rho$ & \\\\
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
    ap.add_argument("--sim-root", required=True,
                    help="ABM sweep root holding run_*/LatticeData/")
    ap.add_argument("--infer-root", required=True,
                    help="preprocessed_3d_infer root (clipped fields)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--smore-dir", required=True,
                    help="directory holding observables.py, sensitivity.py, "
                         "smore_pars.py")
    ap.add_argument("--cytokine", default="il8", choices=CYTOKINES)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--cell-types", nargs="+", type=int, default=None,
                    help="cell_type ids to sample at (default: all non-zero)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--params", nargs="+", default=None)
    ap.add_argument("--n-saltelli", type=int, default=1024)
    ap.add_argument("--max-runs", type=int, default=None,
                    help="limit for a quick check")
    ap.add_argument("--focus", default="keil8",
                    help="parameter highlighted in the table")
    ap.add_argument("--out", default="observable_gap_decomposition.json")
    ap.add_argument("--tex", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.smore_dir).resolve()))
    from observables import load_sweep
    from sensitivity import emulator_sobol, select_top_k
    from smore_pars import fit_surrogates, leave_one_out_recovery
    mods = {"emulator_sobol": emulator_sobol, "select_top_k": select_top_k,
            "fit_surrogates": fit_surrogates,
            "leave_one_out_recovery": leave_one_out_recovery}

    man = json.load(open(args.manifest))
    names, bounds = man["param_names"], man["bounds"]
    ci = CYTOKINES.index(args.cytokine)
    gtag = f"{args.grid}x{args.grid}x{args.grid}"

    print("[1/3] theta and run order from the sweep ...")
    theta, Y_txt, run_ids, t_grid = load_sweep(args.sim_root, names)
    if args.max_runs:
        run_ids = run_ids[:args.max_runs]
        theta = theta[:args.max_runs]
        Y_txt = Y_txt[:args.max_runs]
    print(f"      {len(run_ids)} runs")

    print(f"[2/3] reading raw and clipped fields ({args.cytokine}) ...")
    ct = set(args.cell_types) if args.cell_types else None
    rows = {"raw_cell": [], "raw_grid": [], "clip_cell": [], "clip_grid": []}
    keep, t0 = [], time.time()
    for i, rid in enumerate(run_ids):
        mc_r, mg_r = load_raw_run(Path(args.sim_root) / rid, args.cytokine, ct)
        if mc_r is None:
            continue
        mc_c, mg_c = load_clipped_run(Path(args.infer_root) / rid / gtag, ci, ct)
        if mc_c is None:
            continue
        rows["raw_cell"].append(mc_r)
        rows["raw_grid"].append(mg_r)
        rows["clip_cell"].append(mc_c)
        rows["clip_grid"].append(mg_c)
        keep.append(rid)
        if (i + 1) % 20 == 0:
            print(f"      [{i+1}/{len(run_ids)}] {time.time()-t0:.0f}s", flush=True)

    if not keep:
        raise SystemExit("no run had both raw LatticeData and preprocessed fields")
    idx = [run_ids.index(r) for r in keep]
    theta = theta[idx]
    Y_txt_keep = Y_txt[idx][:, :, ci]

    T = min(min(len(a) for a in rows[k]) for k in rows)
    arms = {k: np.stack([a[:T] for a in v]) for k, v in rows.items()}
    tg = np.asarray(t_grid, float)[:T]
    print(f"      {len(keep)} runs usable, {T} frames")

    # sanity: does the reconstructed cell-sampled observable track the ABM's own?
    Tt = min(T, Y_txt_keep.shape[1])
    a = arms["raw_cell"][:, :Tt].ravel()
    b = Y_txt_keep[:, :Tt].ravel()
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    if ok.sum() > 10:
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        ratio = float(np.median(a[ok] / b[ok]))
        print(f"      reconstructed cell-sampled vs mean_concentration.txt: "
              f"corr={r:+.3f}, median ratio={ratio:.3f}")
        print("      (ratio near 1 and high corr means the reconstruction is "
              "faithful; the ABM samples at cell centres of mass whereas this "
              "averages over occupied voxels, so exact equality is not expected)")
        recon = {"corr_vs_abm_txt": r, "median_ratio_vs_abm_txt": ratio}
    else:
        recon = {}

    print(f"\n[3/3] Sobol + SMoRe ParS per arm ...")
    labels = {"raw_cell": "raw, cell-sampled", "raw_grid": "raw, grid-averaged",
              "clip_cell": "clipped, cell-sampled",
              "clip_grid": "clipped, grid-averaged"}
    fnames = [f"{args.cytokine}_{s}" for s in ("final", "mean", "max", "auc")]
    res = {}
    for k, Y in arms.items():
        res[k] = recovery_for(theta, summarise(Y), fnames, Y, tg,
                              names, bounds, args, labels[k], mods)

    bundle = {"mode": "observable_gap_decomposition",
              "cytokine": args.cytokine, "n_runs": len(keep), "run_ids": keep,
              "reconstruction_check": recon, "arms": res}
    json.dump(bundle, open(args.out, "w"), indent=2)
    print(f"\nDONE -> {args.out}")

    f = args.focus
    g = lambda k: res[k]["recovery"]["r2_per_param"].get(f)
    print(f"\n=== DECOMPOSITION FOR {f} ===")
    print(f"  raw,     cell-sampled : {g('raw_cell')}")
    print(f"  raw,     grid-averaged: {g('raw_grid')}")
    print(f"  clipped, cell-sampled : {g('clip_cell')}")
    print(f"  clipped, grid-averaged: {g('clip_grid')}")
    rc, rg, cc = g("raw_cell"), g("raw_grid"), g("clip_cell")
    if None not in (rc, rg):
        print(f"\n  observable definition costs : {rg - rc:+.3f} "
              f"(cell-sampled -> grid-averaged, clipping held out)")
    if None not in (rc, cc):
        print(f"  clipping costs              : {cc - rc:+.3f} "
              f"(raw -> clipped, definition held fixed)")
    print("\n  The surrogate can only reach the clipped, grid-averaged arm.")

    if args.tex:
        make_tex(res, args.cytokine, len(keep), args.tex, f)


if __name__ == "__main__":
    main()