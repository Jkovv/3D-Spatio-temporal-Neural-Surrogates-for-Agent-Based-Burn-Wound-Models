#!/usr/bin/env python3
# combi3D/compute_speedup.py
#
# Computes the speedup and break-even N* for the paper's cost table, from
# times you measured IN THIS PIPELINE — nothing is carried over from the thesis.
#
#   speedup = T_ABM / T_infer
#   N*      = T_train / (T_ABM - T_infer)
#
# where:
#   T_ABM   : one ABM run on rome/CPU  (from time_abm.slurm -> timing_abm_summary.txt)
#   T_train : surrogate training time  (measured when you train on the NEW data, GPU)
#   T_infer : surrogate inference time (same, GPU)
#
# All three come from the new SMoRe ParS pipeline on the new sweep data.
#
# Usage — either pass the numbers directly:
#   python compute_speedup.py --t-abm 1820 --t-train-unet 240 --t-infer-unet 1.8 \
#                             --t-train-deeponet 545 --t-infer-deeponet 35
#
# or point it at the ABM summary and pass only the surrogate numbers:
#   python compute_speedup.py --abm-summary sweep/timing_abm_summary.txt \
#                             --t-train-unet 240 --t-infer-unet 1.8 \
#                             --t-train-deeponet 545 --t-infer-deeponet 35

import argparse
import sys


def read_t_abm(path):
    """Pull T_ABM_mean_seconds (and SD if present) from a timing summary file."""
    mean = sd = None
    with open(path) as f:
        for line in f:
            if line.startswith("T_ABM_mean_seconds"):
                mean = float(line.split("=")[1])
            elif line.startswith("T_ABM_sd_seconds"):
                sd = float(line.split("=")[1])
    if mean is None:
        sys.exit(f"[compute_speedup] could not find T_ABM_mean_seconds in {path}")
    return mean, sd


def main():
    ap = argparse.ArgumentParser(description="Speedup / N* for the cost table.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--t-abm", type=float, help="ABM run time (s), measured on rome.")
    src.add_argument("--abm-summary", help="Path to timing_abm_summary.txt.")

    ap.add_argument("--t-train-unet", type=float, required=True)
    ap.add_argument("--t-infer-unet", type=float, required=True)
    ap.add_argument("--t-train-deeponet", type=float, required=True)
    ap.add_argument("--t-infer-deeponet", type=float, required=True)
    args = ap.parse_args()

    if args.abm_summary:
        t_abm, t_abm_sd = read_t_abm(args.abm_summary)
    else:
        t_abm, t_abm_sd = args.t_abm, None

    models = {
        "U-Net":    (args.t_train_unet, args.t_infer_unet),
        "DeepONet": (args.t_train_deeponet, args.t_infer_deeponet),
    }

    sd_str = f" (+/- {t_abm_sd:.2f})" if t_abm_sd is not None else ""
    print(f"\nT_ABM = {t_abm:.2f} s{sd_str}   [rome / CPU, new sweep data]\n")
    print(f"{'model':10s} {'T_train(s)':>11s} {'T_infer(s)':>11s} "
          f"{'speedup':>10s} {'N*':>8s}")
    print("-" * 54)
    for name, (t_train, t_infer) in models.items():
        if t_abm <= t_infer:
            speedup = float("inf")
            nstar = float("inf")
        else:
            speedup = t_abm / t_infer
            nstar = t_train / (t_abm - t_infer)
        print(f"{name:10s} {t_train:11.1f} {t_infer:11.2f} "
              f"{speedup:9.1f}x {nstar:8.1f}")
    print()
    print("Paste these into tab:cost_3d (Speed-up and N* columns).")
    print("speedup = how many ABM runs' worth of compute one surrogate query saves;")
    print("N*      = surrogate queries needed to amortise its own training cost.")


if __name__ == "__main__":
    main()
