#!/usr/bin/env python3

import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

CYT_LABEL = {
    "il1": r"IL-1$\beta$", "il6": "IL-6", "il8": "IL-8",
    "il10": "IL-10", "tnf": r"TNF-$\alpha$", "tgf": r"TGF-$\beta$",
}
STAT_ORDER = ["final", "mean", "max", "auc"]


# imports

def import_smore(smore_dir):
    smore_dir = os.path.abspath(smore_dir)
    if not os.path.isdir(smore_dir):
        sys.exit(f"--smore-dir not found: {smore_dir}")
    sys.path.insert(0, smore_dir)

    try:
        from observables import (load_sweep, summarize_observable,
                                 FEATURE_NAMES)
    except ImportError as exc:
        sys.exit(f"Could not import observables.py from {smore_dir}: {exc}")

    fit_gp, reused = None, False
    try:
        import sensitivity as sens
        fit_gp = getattr(sens, "_fit_gp", None)
        if callable(fit_gp):
            reused = True
            print("[info] reusing sensitivity._fit_gp() -- this table describes "
                  "the same emulator the Sobol indices are computed on")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[warn] could not import sensitivity.py ({exc})", file=sys.stderr)

    if not reused:
        print("[warn] sensitivity._fit_gp not found; using the replica below. "
              "Verify it still matches before quoting this table alongside "
              "the indices.", file=sys.stderr)
        fit_gp = replica_fit_gp

    return load_sweep, summarize_observable, FEATURE_NAMES, fit_gp, reused


def replica_fit_gp(theta, y):
    """
    Byte-for-byte replica of sensitivity._fit_gp as of the version this script was written against. Only used if the import fails.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                  WhiteKernel)
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(theta)
    ys = StandardScaler().fit(y.reshape(-1, 1))
    Xt = xs.transform(theta)
    yt = ys.transform(y.reshape(-1, 1)).ravel()
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(theta.shape[1]),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-3, (1e-6, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=10, alpha=1e-10)
    gp.fit(Xt, yt)

    def predict(theta_new):
        return ys.inverse_transform(
            gp.predict(xs.transform(theta_new)).reshape(-1, 1)).ravel()

    return predict, gp


# CV
def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def cv_pooled(theta, y, splitter, fit_gp):
    """
    Pooled out-of-fold R^2 -- one score over all held-out predictions, not an
    average of per-fold scores. Scalers and kernel are refitted per fold, so
    no information from the held-out points reaches the model.
    """
    pred = np.empty_like(y, dtype=float)
    for tr, te in splitter.split(theta):
        predict, _ = fit_gp(theta[tr], y[tr])
        pred[te] = predict(theta[te])
    return r2(y, pred)


# LaTeX

def split_name(name):
    for stat in STAT_ORDER:
        if name.endswith("_" + stat):
            return name[: -(len(stat) + 1)], stat
    return name, ""


def to_latex(names, r5, rloo, do_loo, n_runs, folds):
    groups, loo_groups = {}, {}
    for i, name in enumerate(names):
        cyt = split_name(name)[0]
        groups.setdefault(cyt, []).append(r5[i])
        if do_loo:
            loo_groups.setdefault(cyt, []).append(rloo[i])

    ncol = "lcccc" if do_loo else "lccc"
    head = (r"\textbf{Cytokine} & \textbf{Obs.} & \textbf{mean }$\mathbf{R^2}$"
            r" & \textbf{min}--\textbf{max} & \textbf{LOO mean} \\" if do_loo
            else r"\textbf{Cytokine} & \textbf{Obs.} & "
                 r"\textbf{mean }$\mathbf{R^2}$ & \textbf{min}--\textbf{max} \\")

    lines = [
        r"\begin{table}[h]",
        r"\centering\footnotesize",
        r"\caption{Cross-validated predictive accuracy of the per-observable "
        r"Gaussian-process emulators over the " + str(n_runs) + r"-run sweep. "
        r"$R^2$ is pooled out-of-fold (" + str(folds) + r"-fold, shuffled), "
        r"with the emulator refitted on each training fold, so it measures "
        r"how closely the response surface on which the Sobol indices are "
        r"computed tracks the ABM at parameter points the emulator did not "
        r"see. Indices for observables whose emulator is weak should be read "
        r"with corresponding caution.}",
        r"\label{tab:app_emulator}",
        r"\begin{tabular}{" + ncol + "}",
        r"\toprule", head, r"\midrule",
    ]

    for cyt, vals in groups.items():
        vals = np.asarray(vals)
        cells = [CYT_LABEL.get(cyt, cyt.replace("_", r"\_")), str(len(vals)),
                 f"${vals.mean():+.3f}$",
                 f"${vals.min():+.3f}$--${vals.max():+.3f}$"]
        if do_loo:
            cells.append(f"${np.mean(loo_groups[cyt]):+.3f}$")
        lines.append(" & ".join(cells) + r" \\")

    allv = np.asarray(r5)
    cells = [r"\textbf{All}", str(len(allv)),
             f"$\\mathbf{{{allv.mean():+.3f}}}$",
             f"${allv.min():+.3f}$--${allv.max():+.3f}$"]
    if do_loo:
        cells.append(f"${np.mean(rloo):+.3f}$")
    lines += [r"\midrule", " & ".join(cells) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# main

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-root", required=True,
                    help="directory holding the run_* folders")
    ap.add_argument("--manifest", required=True,
                    help="sweep manifest; only param_names is read")
    ap.add_argument("--smore-dir", default=os.path.join(here, "..", "smore"))
    ap.add_argument("--out-tex", default="appendix_emulator.tex")
    ap.add_argument("--out-csv", default="emulator_cv.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--loo", action="store_true",
                    help="also compute leave-one-out R^2 (slow)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    (load_sweep, summarize_observable, FEATURE_NAMES,
     fit_gp, reused) = import_smore(args.smore_dir)

    man = json.load(open(args.manifest))
    param_names = man["param_names"]

    theta, Y, run_ids, t_grid = load_sweep(args.sim_root, param_names)
    theta = np.asarray(theta, dtype=float)
    feats = summarize_observable(Y)
    print(f"[info] {len(run_ids)} runs | theta {theta.shape} | "
          f"features {feats.shape}")

    if feats.shape[1] != len(FEATURE_NAMES):
        sys.exit(f"feature matrix has {feats.shape[1]} columns but "
                 f"FEATURE_NAMES has {len(FEATURE_NAMES)}")
    if len(run_ids) < 10:
        sys.exit(f"only {len(run_ids)} runs -- too few for a meaningful CV")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    loo = LeaveOneOut()

    names, r5, rloo = [], [], []
    print()
    for j, name in enumerate(FEATURE_NAMES):
        y = np.asarray(feats[:, j], dtype=float)
        if not np.all(np.isfinite(y)):
            print(f"  {name:16s} skipped (non-finite)", file=sys.stderr)
            continue
        if np.std(y) == 0:
            print(f"  {name:16s} skipped (constant across sweep)",
                  file=sys.stderr)
            continue
        a = cv_pooled(theta, y, kf, fit_gp)
        names.append(name)
        r5.append(a)
        tail = ""
        if args.loo:
            b = cv_pooled(theta, y, loo, fit_gp)
            rloo.append(b)
            tail = f"   LOO {b:+.3f}"
        print(f"  {name:16s} {args.folds}-fold {a:+.3f}{tail}", flush=True)

    if not names:
        sys.exit("no usable observables")

    for path in (args.out_tex, args.out_csv):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    with open(args.out_csv, "w") as fh:
        fh.write(f"observable,r2_{args.folds}fold"
                 + (",r2_loo" if args.loo else "") + "\n")
        for i, name in enumerate(names):
            fh.write(f"{name},{r5[i]:.6f}"
                     + (f",{rloo[i]:.6f}" if args.loo else "") + "\n")

    with open(args.out_tex, "w") as fh:
        fh.write(to_latex(names, r5, rloo, args.loo,
                          len(run_ids), args.folds) + "\n")

    arr = np.asarray(r5)
    print(f"\n[ok] per-observable scores -> {args.out_csv}")
    print(f"[ok] LaTeX table -> {args.out_tex}")
    print(f"\nmean {args.folds}-fold R^2 = {arr.mean():+.3f} "
          f"(min {arr.min():+.3f}, max {arr.max():+.3f})")
    if not reused:
        print("\n[!] emulator was replicated, not imported - verify it "
              "matches sensitivity.py before quoting this table.")
    weak = [(n, v) for n, v in zip(names, arr) if v < 0.5]
    if weak:
        print(f"\n[!] {len(weak)} observable(s) below R^2 = 0.5.")
        for n, v in weak:
            print(f"    {n:16s} {v:+.3f}")


if __name__ == "__main__":
    main()