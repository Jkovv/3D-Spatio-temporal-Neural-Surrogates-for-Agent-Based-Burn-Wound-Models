#!/usr/bin/env python3

import argparse
import inspect
import json
import os
import sys
import warnings

import numpy as np

try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass


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
        sys.exit(f"Could not import observables.py: {exc}")
    try:
        import sensitivity as sens
    except ImportError as exc:
        sys.exit(f"Could not import sensitivity.py: {exc}")
    if not hasattr(sens, "emulator_sobol"):
        sys.exit("sensitivity.py has no emulator_sobol()")
    return load_sweep, summarize_observable, FEATURE_NAMES, sens


def call_sobol(sens, theta, feats, names, param_names, bounds,
               n_saltelli, verbose=False):
    """
    Bind to emulator_sobol by introspection: the five arguments whose meaning
    is unambiguous go positionally, anything that looks like a sample-size
    knob gets n_saltelli, everything else keeps its default.
    """
    fn = sens.emulator_sobol
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    args = [theta, feats, names, param_names, bounds]
    if len(params) < 5:
        sys.exit(f"emulator_sobol takes {len(params)} args; expected >= 5.\n"
                 f"signature: {sig}")

    kwargs = {}
    for p in params[5:]:
        if "saltelli" in p.name.lower() or p.name.lower() in ("n", "n_samples"):
            kwargs[p.name] = n_saltelli
    if verbose:
        print(f"[info] emulator_sobol{sig}")
        print(f"[info] calling with 5 positional + {kwargs}")
    return fn(*args, **kwargs)


def extract_ranking(res, param_names):
    """Pull {param: ST} out of whatever emulator_sobol returns."""
    if isinstance(res, dict):
        if "ranking" in res:
            out = {}
            for row in res["ranking"]:
                if isinstance(row, dict):
                    key = row.get("param") or row.get("name")
                    val = row.get("ST_mean", row.get("ST"))
                    out[key] = float(val)
                else:                                   # (name, value) pairs
                    out[row[0]] = float(row[1])
            return out
        for key in ("ST_mean", "ST", "total_order"):
            if key in res:
                v = res[key]
                if isinstance(v, dict):
                    return {k: float(x) for k, x in v.items()}
                return dict(zip(param_names, [float(x) for x in v]))
    sys.exit(f"could not find a ranking in emulator_sobol's return "
             f"(keys: {list(res) if isinstance(res, dict) else type(res)})")


# output

def report(full, filt, param_names, kept, dropped, threshold):
    rank_full = {p: i + 1 for i, p in enumerate(
        sorted(full, key=full.get, reverse=True))}
    rank_filt = {p: i + 1 for i, p in enumerate(
        sorted(filt, key=filt.get, reverse=True))}
    order = sorted(param_names, key=lambda p: -full[p])

    print(f"\n{'parameter':12s} {'S_T all':>9s} {'rank':>5s}   "
          f"{'S_T kept':>9s} {'rank':>5s}   {'move':>5s}")
    print("-" * 58)
    for p in order:
        move = rank_full[p] - rank_filt[p]
        arrow = f"{move:+d}" if move else "="
        print(f"{p:12s} {full[p]:9.3f} {rank_full[p]:5d}   "
              f"{filt[p]:9.3f} {rank_filt[p]:5d}   {arrow:>5s}")

    top2_full = [p for p in order[:2]]
    top2_filt = sorted(filt, key=filt.get, reverse=True)[:2]
    print()
    if top2_full == top2_filt:
        print(f"[ok] the two leading parameters are unchanged "
              f"({', '.join(top2_full)}) -- the ranking is robust to dropping "
              f"the poorly emulated observables.")
    else:
        print(f"[!] leaders change: {top2_full} -> {top2_filt}. "
              f"The filtered ranking is the defensible one; report it as the "
              f"main result and the unfiltered one as the sensitivity check.")

    biggest = max(param_names, key=lambda p: abs(full[p] - filt[p]))
    print(f"[info] largest index shift: {biggest} "
          f"{full[biggest]:+.3f} -> {filt[biggest]:+.3f} "
          f"({filt[biggest] - full[biggest]:+.3f})")
    return order, rank_full, rank_filt


def to_latex(order, full, filt, rank_full, rank_filt, kept, dropped,
             threshold):
    lines = [
        r"\begin{table}[h]",
        r"\centering\footnotesize",
        r"\caption{Sobol total-order indices computed on all "
        + str(len(kept) + len(dropped)) + r" observables and on the "
        + str(len(kept)) + r" whose emulator reaches a cross-validated "
        r"$R^2 \geq " + f"{threshold:g}" + r"$ (Table~\ref{tab:app_emulator}). "
        r"Observables the emulator cannot reproduce contribute emulator "
        r"variance rather than ABM response, so the filtered column is the "
        r"better-founded ranking; agreement between the two is evidence that "
        r"the leading parameters do not depend on that choice.}",
        r"\label{tab:app_sobol_filtered}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{All }$\mathbf{S_T}$ & \textbf{Rank}"
        r" & \textbf{Filtered }$\mathbf{S_T}$ & \textbf{Rank} \\",
        r"\midrule",
    ]
    for p in order:
        name = p.replace("_", r"\_")
        lines.append(f"\\texttt{{{name}}} & ${full[p]:.3f}$ & {rank_full[p]} "
                     f"& ${filt[p]:.3f}$ & {rank_filt[p]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
              r"% observables dropped at threshold "
              + f"{threshold:g}: " + ", ".join(dropped)]
    return "\n".join(lines)


# main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cv-csv", required=True,
                    help="emulator_cv.csv produced by emulator_cv.py")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-saltelli", type=int, default=1024)
    ap.add_argument("--smore-dir", default=os.path.join(here, "..", "smore"))
    ap.add_argument("--out-tex", default="sobol_filtered.tex")
    ap.add_argument("--out-csv", default="sobol_filtered.csv")
    ap.add_argument("--show-signature", action="store_true")
    args = ap.parse_args()

    (load_sweep, summarize_observable,
     FEATURE_NAMES, sens) = import_smore(args.smore_dir)

    man = json.load(open(args.manifest))
    param_names = man["param_names"]
    bounds = man["bounds"]

    theta, Y, run_ids, _ = load_sweep(args.sim_root, param_names)
    theta = np.asarray(theta, dtype=float)
    feats = summarize_observable(Y)
    print(f"[info] {len(run_ids)} runs | features {feats.shape}")

    # CV scores
    scores = {}
    with open(args.cv_csv) as fh:
        header = fh.readline().strip().split(",")
        col = 1
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) > col:
                scores[parts[0]] = float(parts[col])
    if not scores:
        sys.exit(f"no scores read from {args.cv_csv}")

    keep_idx, kept, dropped = [], [], []
    for i, name in enumerate(FEATURE_NAMES):
        s = scores.get(name)
        if s is None:
            print(f"[warn] {name} absent from CV file; keeping it",
                  file=sys.stderr)
            keep_idx.append(i); kept.append(name); continue
        (kept if s >= args.threshold else dropped).append(name)
        if s >= args.threshold:
            keep_idx.append(i)

    print(f"[info] threshold R^2 >= {args.threshold:g}: "
          f"keeping {len(kept)}, dropping {len(dropped)}")
    if dropped:
        print(f"[info] dropped: {', '.join(dropped)}")
    if len(kept) < 4:
        sys.exit("too few observables survive the threshold; lower it")

    print("\n[1/2] Sobol on all observables ...", flush=True)
    res_full = call_sobol(sens, theta, feats, FEATURE_NAMES, param_names,
                          bounds, args.n_saltelli, args.show_signature)
    full = extract_ranking(res_full, param_names)

    print("[2/2] Sobol on well-emulated observables ...", flush=True)
    sub_names = [FEATURE_NAMES[i] for i in keep_idx]
    res_filt = call_sobol(sens, theta, feats[:, keep_idx], sub_names,
                          param_names, bounds, args.n_saltelli)
    filt = extract_ranking(res_filt, param_names)

    order, rank_full, rank_filt = report(full, filt, param_names,
                                         kept, dropped, args.threshold)

    for path in (args.out_tex, args.out_csv):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(args.out_csv, "w") as fh:
        fh.write("param,ST_all,rank_all,ST_filtered,rank_filtered\n")
        for p in order:
            fh.write(f"{p},{full[p]:.6f},{rank_full[p]},"
                     f"{filt[p]:.6f},{rank_filt[p]}\n")

    with open(args.out_tex, "w") as fh:
        fh.write(to_latex(order, full, filt, rank_full, rank_filt,
                          kept, dropped, args.threshold) + "\n")

    print(f"\n[ok] {args.out_csv}\n[ok] {args.out_tex}")


if __name__ == "__main__":
    main()