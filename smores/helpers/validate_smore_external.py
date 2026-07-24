#!/usr/bin/env python3
# smores/helpers/validate_smore_external.py
#
# SMoRe ParS on an external, experimental parameter sweep.
#
# WHY THIS DATASET AND NOT TUMOUR GROWTH
# An earlier attempt used tumour volume measurements. Those are observations,
# not a sweep: no ground-truth parameters exist behind each trajectory, so only
# the stage-A surface fit could be checked, never the recovery step that is the
# point of SMoRe ParS. Patients under immunotherapy also have shrinking
# tumours, which a saturating-logistic surface cannot represent at all.
#
# Gong and Ying (Sci Rep 16:2375, 2025; doi:10.1038/s41598-025-32144-1, CC-BY)
# published something structurally different: a designed sweep. 870 growth
# curves of E. coli, 98 time points each, from crossing
#   - 5 strains of known genome size (N0, N7, N14, N20, N28), with
#   - 29 chemically defined media of known composition (8 components),
# in 6 replicates. Every curve has a known input vector, exactly as every ABM
# run has a known theta, so the whole pipeline applies:
#
#   stage A  fit A/(1+exp(-k(t-t0))) per curve             -> theta_SM
#   Sobol    which inputs drive the observables
#   stage B  GP mapping inputs -> theta_SM, leave-one-out  -> recovery
#
# This is the analysis run on the burn ABM, applied to laboratory measurements
# nobody simulated. Bacterial growth is also the regime the surface was made
# for: lag, exponential, stationary. If recovery fails here, it is not because
# the functional form is wrong for the data.
#
# PUBLISHED REFERENCE POINTS
# The source paper reports, from gradient-boosted trees and SHAP, that genome
# size dominates the three growth parameters (K, r, lag), while glucose
# dominates overall curve shape. Sobol indices here can be read against that.

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Genome sizes (Mb) for the strain series: W3110 wild type and progressively
# reduced derivatives. Only ordering and relative spacing matter here.
GENOME_MB = {"N0": 4.63, "N7": 4.35, "N14": 3.98, "N20": 3.79, "N28": 3.62}

# Log-scale columns from the source Table S1; the media were designed on a
# logarithmic grid, so these are the natural coordinates.
CHEM_COLS = ["logThiamine", "logK+", "logPO43-", "logFe2+",
             "logSO42-", "logNH4+", "logMg2+", "logGlucose"]


class Tee:
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


def r2(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-300 else float("nan")


def load_sweep(curves_path, design_path, media_path, say):
    """Assemble (theta, Y, t_grid, curve_ids, param_names).

    theta : (n, 9) genome size in Mb plus 8 log concentrations
    Y     : (n, T) OD600 trajectories on a common time grid
    """
    import pandas as pd

    cur = pd.read_excel(curves_path)
    tcol = cur.columns[0]
    t_grid = cur[tcol].to_numpy(float)
    say(f"      curves file: {cur.shape[1]-1} trajectories, {len(t_grid)} time "
        f"points ({t_grid[0]:.1f}-{t_grid[-1]:.1f} h)")

    des = pd.read_excel(design_path)
    dcols = {c.lower().strip(): c for c in des.columns}
    c_id = dcols.get("growth curve id")
    c_gen = dcols.get("genome id")
    c_med = dcols.get("medium no.")
    if not all([c_id, c_gen, c_med]):
        raise SystemExit(f"design columns not found in {list(des.columns)}")

    med = pd.read_excel(media_path)
    m_id = next(c for c in med.columns if c.strip() == "Medium No.")
    have = {c.strip(): c for c in med.columns}
    missing = [c for c in CHEM_COLS if c.strip() not in have]
    if missing:
        raise SystemExit(f"medium columns missing: {missing}")
    chem = {}
    for _, row in med.iterrows():
        chem[str(row[m_id]).strip()] = np.array(
            [float(row[have[c.strip()]]) for c in CHEM_COLS])

    theta, Y, ids, masks = [], [], [], []
    drop_missing = drop_flat = 0
    n_trimmed = 0
    for _, row in des.iterrows():
        cid = str(row[c_id]).strip()
        gen = str(row[c_gen]).strip()
        mno = str(row[c_med]).strip()
        if cid not in cur.columns or gen not in GENOME_MB or mno not in chem:
            drop_missing += 1
            continue
        y = cur[cid].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 10:
            drop_flat += 1
            continue
        if not ok.all():
            n_trimmed += 1
        theta.append(np.concatenate([[GENOME_MB[gen]], chem[mno]]))
        Y.append(np.where(ok, y, np.nan))
        masks.append(ok)
        ids.append(cid)

    if drop_missing:
        say(f"      {drop_missing} rows dropped: curve, strain or medium not "
            f"found")
    if drop_flat:
        say(f"      {drop_flat} curves dropped: fewer than 10 finite points")
    if n_trimmed:
        say(f"      {n_trimmed} curves have missing points; these are dropped "
            f"point-wise and the curve is kept, following the source "
            f"publication's own handling (x = x[~np.isnan(x)] in their "
            f"DTWImplement.py). Curves are therefore of unequal length, which "
            f"the per-curve surface fit accommodates because time is "
            f"normalised to [0,1] separately for each.")
    say(f"      curves showing no growth are RETAINED: reduced-genome strains "
        f"fail to grow in some media, and the source publication records those "
        f"as observations (K = r = t = 0), not missing data. Excluding them "
        f"would remove exactly the cases where the genome effect is strongest.")

    names = ["genome_Mb"] + [c.replace("log", "log_") for c in CHEM_COLS]
    return (np.vstack(theta), np.vstack(Y), np.vstack(masks), t_grid, ids,
            names)


def summarise(Y, masks=None):
    """(n, T) -> (n, 4): final, mean, max, auc. Same reduction as used on the
    ABM trajectories, so the two analyses are comparable.

    Curves may have missing points, which are dropped rather than imputed, so
    each row is reduced over its own finite entries only."""
    n, T = Y.shape
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    out = np.zeros((n, 4))
    for i in range(n):
        y = Y[i]
        y = y[np.isfinite(y)] if masks is None else y[masks[i]]
        if y.size == 0:
            continue
        out[i] = [y[-1], y.mean(), y.max(),
                  _trap(y) / max(1, len(y) - 1)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curves", required=True, help="Table S2 xlsx")
    ap.add_argument("--design", required=True, help="Table S3 xlsx")
    ap.add_argument("--media", required=True, help="Table S1 xlsx")
    ap.add_argument("--smore-dir", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-saltelli", type=int, default=1024)
    ap.add_argument("--max-curves", type=int, default=None,
                    help="subsample for a quick check")
    ap.add_argument("--out", default="external_validation.json")
    ap.add_argument("--tex", default=None)
    ap.add_argument("--log", default="external_validation.txt")
    args = ap.parse_args()

    say = Tee(args.log)
    sys.path.insert(0, str(Path(args.smore_dir).resolve()))
    from sensitivity import emulator_sobol, select_top_k
    from smore_pars import fit_surrogate_one, leave_one_out_recovery

    say("[1/4] loading the external sweep ...")
    theta, Y, masks, t_grid, ids, names = load_sweep(
        args.curves, args.design, args.media, say)
    if args.max_curves and len(ids) > args.max_curves:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(ids), args.max_curves, replace=False))
        theta, Y, masks = theta[sel], Y[sel], masks[sel]
        ids = [ids[i] for i in sel]
        say(f"      subsampled to {len(ids)} curves")
    say(f"      {len(ids)} usable curves, {theta.shape[1]} inputs")
    say(f"      inputs: {names}")
    say(f"      OD600 range {np.nanmin(Y):.4f}-{np.nanmax(Y):.4f}")

    bounds = {n: {"low": float(theta[:, j].min()),
                  "high": float(theta[:, j].max())}
              for j, n in enumerate(names)}
    const = [n for n in names if bounds[n]["high"] - bounds[n]["low"] < 1e-12]
    if const:
        say(f"      note: {const} do not vary in this design and cannot be "
            f"identified; kept for completeness, expect zero sensitivity")

    say("\n[2/4] stage A: fitting the saturating-logistic surface ...")
    import time
    theta_sm = np.zeros((len(ids), 3))
    fit_r2 = np.full(len(ids), np.nan)
    tn = (t_grid - t_grid[0]) / max(1e-9, t_grid[-1] - t_grid[0])
    t_start = time.time()
    for i in range(len(ids)):
        m = masks[i]
        ti, yi = t_grid[m], Y[i][m]
        p = np.asarray(fit_surrogate_one(ti, yi), float)
        theta_sm[i] = p
        tni = (ti - ti[0]) / max(1e-9, ti[-1] - ti[0])
        fit_r2[i] = r2(yi, saturating(tni, *p))
        if (i + 1) % 200 == 0:
            say(f"      [{i+1}/{len(ids)}] {time.time()-t_start:.0f}s")

    # A flat curve has zero variance, so R2 is undefined for it. Those runs are
    # kept in theta_SM (the fitted amplitude is near zero, which is the correct
    # description) but excluded from the fit-quality summary, which would
    # otherwise be dominated by an undefined quantity.
    grew = np.nanmax(Y, axis=1) > 1e-6
    say(f"      {int(grew.sum())}/{len(ids)} curves show growth, "
        f"{int((~grew).sum())} do not")
    g = fit_r2[grew]
    say(f"      on growing curves: R2 = {np.nanmean(g):+.4f} +/- "
        f"{np.nanstd(g):.4f} (median {np.nanmedian(g):+.4f}, "
        f"min {np.nanmin(g):+.4f})")
    say(f"      above 0.99: {int(np.nansum(g > 0.99))}/{len(g)}")
    say(f"      above 0.90: {int(np.nansum(g > 0.90))}/{len(g)}")
    say(f"      for reference, the same surface reaches ~0.87 on the ABM's "
        f"IL-8 trajectories")

    say("\n[3/4] Sobol screen on the observables ...")
    feats = summarise(Y, masks)
    fnames = ["od_final", "od_mean", "od_max", "od_auc"]
    sob = emulator_sobol(theta, feats, fnames, names, bounds,
                         n_saltelli=args.n_saltelli)
    say("      total-order indices:")
    for r in sob["ranking"]:
        say(f"        {r['param']:14s} ST = {r['ST_mean']:.4f}")
    top_k = select_top_k(sob["ranking"], args.top_k)
    say(f"      top-{args.top_k}: {top_k}")
    say(f"      the source publication reports genome size as dominant for the "
        f"growth parameters, glucose for curve shape")

    say("\n[4/4] stage B: leave-one-out recovery of experimental conditions ...")
    sel_idx = [names.index(p) for p in top_k]
    say(f"      {len(ids)} leave-one-out GP fits; this is the slow step "
        f"(roughly quadratic in the number of curves)")
    t_b = time.time()
    rec = leave_one_out_recovery(theta, theta_sm, names, bounds, sel_idx)
    say(f"      done in {time.time()-t_b:.0f}s")
    diag = {}
    for pn in rec["selected_params"]:
        j = rec["selected_params"].index(pn)
        rv = np.array([x[j] for x in rec["recovered"]], float)
        tv = np.array([x[j] for x in rec["truth"]], float)
        diag[pn] = {
            "corr": float(np.corrcoef(rv, tv)[0, 1]) if tv.std() > 0 else float("nan"),
            "sd_ratio": float(rv.std() / tv.std()) if tv.std() > 0 else float("nan"),
        }
        say(f"      {pn:14s} R2 = {rec['r2_per_param'][pn]:+.3f}  "
            f"nRMSE = {rec['nrmse_per_param'][pn]:.3f}  "
            f"corr = {diag[pn]['corr']:+.3f}")

    bundle = {
        "mode": "smore_pars_on_external_experimental_sweep",
        "dataset": "E. coli growth curves, Gong and Ying, Sci Rep 16:2375 "
                   "(2025), doi:10.1038/s41598-025-32144-1, CC-BY",
        "n_curves": len(ids), "n_timepoints": int(len(t_grid)),
        "inputs": names,
        "surface": "A / (1 + exp(-k (t - t0))), time normalised to [0,1]",
        "n_growing": int(grew.sum()),
        "n_no_growth": int((~grew).sum()),
        "stage_a_fit_growing_curves_only": {
            "n": int(grew.sum()),
            "r2_mean": float(np.nanmean(fit_r2[grew])),
            "r2_std": float(np.nanstd(fit_r2[grew])),
            "r2_median": float(np.nanmedian(fit_r2[grew])),
            "r2_min": float(np.nanmin(fit_r2[grew])),
            "n_above_0.99": int(np.nansum(fit_r2[grew] > 0.99)),
            "n_above_0.90": int(np.nansum(fit_r2[grew] > 0.90)),
        },
        "sobol": sob["ranking"],
        "top_k": top_k,
        "recovery": {k: rec[k] for k in
                     ("selected_params", "r2_per_param", "nrmse_per_param")},
        "diagnostics": diag,
        "curve_ids": ids,
    }
    json.dump(bundle, open(args.out, "w"), indent=2)
    say(f"\nDONE -> {args.out}")

    if args.tex:
        rows = []
        for pn in rec["selected_params"]:
            st = next((r["ST_mean"] for r in sob["ranking"]
                       if r["param"] == pn), float("nan"))
            safe = pn.replace("_", "\\_")
            rows.append(f"\\texttt{{{safe}}} & ${st:.3f}$ & "
                        f"${rec['r2_per_param'][pn]:+.3f}$ & "
                        f"${diag[pn]['corr']:+.3f}$ \\\\")
        f_ = bundle["stage_a_fit_growing_curves_only"]
        tex = (
            "\\begin{table}[!htbp]\n\\centering\\footnotesize\n"
            "\\caption{SMoRe ParS applied to an external experimental "
            f"parameter sweep: {len(ids)} \\textit{{Escherichia coli}} growth "
            f"curves at {len(t_grid)} time points, obtained by crossing five "
            "strains of known genome size with 29 chemically defined media. "
            "Unlike a series of clinical measurements, this design has known "
            "inputs behind every trajectory, so the recovery step can be "
            "evaluated exactly as on the agent-based model. The stage-A "
            f"surface reaches a median $R^2$ of {f_['r2_median']:+.3f} on the "
            f"{f_['n']} curves that show growth "
            f"({f_['n_above_0.90']}/{f_['n']} above $0.90$); the remaining "
            f"{bundle['n_no_growth']} curves are strain-medium combinations in "
            "which no growth occurs, retained as observations. $S_T$ is the total-order Sobol index; $R^2$ and "
            "$\\rho$ describe leave-one-out recovery of the experimental "
            "conditions from the fitted surface parameters.}\n"
            "\\label{tab:external_validation}\n"
            "\\begin{tabular}{lccc}\n\\toprule\n"
            "Input & $S_T$ & Recovery $R^2$ & $\\rho$ \\\\\n\\midrule\n"
            + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        Path(args.tex).parent.mkdir(parents=True, exist_ok=True)
        open(args.tex, "w").write(tex)
        say(f"LaTeX table -> {args.tex}")

    say(f"log -> {args.log}")
    say.close()


if __name__ == "__main__":
    main()