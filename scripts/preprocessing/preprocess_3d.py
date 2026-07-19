#!/usr/bin/env python3
"""
preprocess_3d.py

Converts raw CompuCell3D sweep output into the .npy tensors the surrogate
training scripts consume.

Input  : <sim-root>/run_XXXX/LatticeData/{CytoStep,CellStep}_*.npz
Output : <out-root>/run_XXXX/<G>x<G>x<G>/{X_unet,X_branch,X_trunk,
                                          Y_target,Y_masks_spatial}.npy
                                         + metadata.json

Reads the exact NPZ format the sweep produces:
  CytoStep_*.npz  keys: il8, il1, il6, il10, tnf, tgf   (each (G,G,G))
  CellStep_*.npz  key : cell_type                        ((G,G,G) uint8)

Windowing: (t, t+1) -> t+2, look-back 2 frames, giving T-2 windows from T
frames (101 frames -> 99 windows). Clip scale is kurtosis-adaptive and fitted
on the training split only (first --train-n windows), matching the 2D protocol.

Two scaling modes:
  (default)      fit a fresh kurtosis-adaptive clip scale on the train split of
                 the run(s) being processed. Use this for the benchmark run you
                 will TRAIN on.
  --scale-from   load cmax / cmax_raw / percentiles from an existing
                 metadata.json (e.g. the benchmark run_0062) and REUSE them,
                 fitting nothing. Use this to preprocess the other sweep runs
                 for surrogate inference, so their inputs live in exactly the
                 normalisation the trained surrogate expects. A frozen scale is
                 the correct choice at inference time -- the same reason any ML
                 pipeline fits the scaler on train and applies it unchanged to
                 new data -- and this mode additionally reports, per run, the
                 fraction of voxels that exceed the frozen cmax, i.e. how far
                 each new run falls outside the range the surrogate was trained
                 on.

Usage (single benchmark run -- fits its own scale, then train the surrogate):
    python preprocess_3d.py --sim-root ../../sweep/outputs \\
        --out-root ../../preprocessed_3d --grid 50 --runs run_0062

Usage (all other sweep runs -- reuse run_0062's scale for inference):
    python preprocess_3d.py --sim-root ../../sweep/outputs \\
        --out-root ../../preprocessed_3d_infer --grid 50 \\
        --scale-from ../../preprocessed_3d/run_0062/50x50x50/metadata.json
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
MASK_TYPE_IDS = [
    1,  # endothelial   -> mask_e
    6,  # neutrophilndn -> mask_ndn
    5,  # neutrophila   -> mask_na
    8,  # macrophage1   -> mask_m1
    9,  # macrophage2   -> mask_m2
]
MASK_NAMES = ["mask_e", "mask_ndn", "mask_na", "mask_m1", "mask_m2"]

# 70 train, 10 val, 19 test
DEFAULT_TRAIN_N = 70


def clip_percentile_from_kurtosis(kappa: float) -> Optional[float]:
    if kappa >= 600:
        return 98.0
    elif kappa >= 300:
        return 98.5
    elif kappa >= 100:
        return 99.0
    elif kappa >= 20:
        return 99.5
    else:
        return None


NOISE_FLOOR_FRAC = 1e-4


def floor_noise(cyto: np.ndarray, cmax_raw: np.ndarray) -> np.ndarray:
    """Zero out values below NOISE_FLOOR_FRAC * cmax_raw, per cytokine.
    cmax_raw : (6,) per-cytokine true maxima (used only to set the floor scale).
    """
    floors = NOISE_FLOOR_FRAC * cmax_raw  # (6,)
    return np.where(cyto >= floors, cyto, 0.0).astype(cyto.dtype)


def fit_cmax_per_cytokine(train_cyto: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, List[Optional[float]],
                                     np.ndarray]:
    """
    Fit one clip value cmax per cytokine from TRAIN-SPLIT data only.
    Returns
        cmax     : (6,) per-cytokine clip value (percentile or true max)
        kappas   : (6,) per-cytokine excess kurtosis (post-floor)
        qs       : list of 6 percentiles (or None where no clipping)
        cmax_raw : (6,) per-cytokine TRUE maxima (pre-floor; sets the floor)
    """
    flat = train_cyto.reshape(-1, 6).astype(np.float64)
    cmax_raw = np.maximum(flat.max(axis=0), 1e-300)  # true per-cytokine maxima
    floored = floor_noise(flat, cmax_raw)            # zero the noise tail
    cmax = np.zeros(6, dtype=np.float64)
    kappas = np.zeros(6, dtype=np.float64)
    qs: List[Optional[float]] = []
    for c in range(6):
        col = floored[:, c]
        # excess kurtosis (Fisher), unbiased - on the floored distribution
        kappa = float(scipy_kurtosis(col, fisher=True, bias=False))
        q = clip_percentile_from_kurtosis(kappa)
        if q is None:
            cmax[c] = float(np.max(col))
        else:
            pv = float(np.percentile(col, q))
            # Safeguard: if the chosen percentile still lands at/near zero
            # (extremely sparse signal even after flooring), fall back to the
            # true max so the signal is not collapsed to +1.
            cmax[c] = pv if pv > NOISE_FLOOR_FRAC * cmax_raw[c] else float(np.max(col))
        kappas[c] = kappa
        qs.append(q)
    cmax = np.maximum(cmax, 1e-12)
    return cmax, kappas, qs, cmax_raw


def load_external_scale(scale_from: Path
                        ) -> Tuple[np.ndarray, np.ndarray, List[Optional[float]],
                                   np.ndarray]:
    """
    Load a previously fitted scale from an existing metadata.json, so new runs
    are normalised exactly as the training run was. Returns the same tuple as
    fit_cmax_per_cytokine: (cmax, kappas, qs, cmax_raw).
    """
    if not scale_from.exists():
        sys.exit(f"ERROR: --scale-from file not found: {scale_from}")
    meta = json.load(open(scale_from))
    sc = meta.get("scaling", {})
    if "max" not in sc or "cmax_raw" not in sc:
        sys.exit(f"ERROR: {scale_from} has no scaling.max / scaling.cmax_raw; "
                 f"is it a preprocess_3d metadata.json?")
    cmax = np.asarray(sc["max"], dtype=np.float64)
    cmax_raw = np.asarray(sc["cmax_raw"], dtype=np.float64)
    kappas = np.asarray(sc.get("excess_kurtosis", [0.0] * 6), dtype=np.float64)
    qs_raw = sc.get("clip_percentile", [-1.0] * 6)
    qs: List[Optional[float]] = [None if (q is None or q < 0) else float(q)
                                 for q in qs_raw]
    if cmax.shape != (6,) or cmax_raw.shape != (6,):
        sys.exit(f"ERROR: {scale_from} scaling arrays are not length 6.")
    print(f"[scale] reusing frozen scale from {scale_from}")
    return cmax, kappas, qs, cmax_raw


def clipping_report(cyto: np.ndarray, cmax: np.ndarray,
                    cmax_raw: np.ndarray) -> dict:
    """
    Per-cytokine fraction of ACTIVE voxels (above the noise floor) whose value
    exceeds the frozen cmax and is therefore clipped. This quantifies how far a
    run falls outside the training run's dynamic range: ~0 means the frozen
    scale fits this run well; a large fraction means the surrogate is being
    asked to predict a run whose signal it never saw at training scale.
    """
    floored = floor_noise(cyto, cmax_raw).reshape(-1, 6)
    out = {}
    for c, name in enumerate(CYTOKINE_NAMES):
        col = floored[:, c]
        active = col > 0.0
        n_active = int(active.sum())
        if n_active == 0:
            out[name] = {"clipped_frac": 0.0, "n_active": 0, "max_over_cmax": 0.0}
            continue
        n_clip = int((col[active] > cmax[c]).sum())
        out[name] = {
            "clipped_frac": n_clip / n_active,
            "n_active": n_active,
            "max_over_cmax": float(col.max() / max(cmax[c], 1e-30)),
        }
    return out


def apply_cyto_scaling(cyto: np.ndarray, cmax: np.ndarray,
                       cmax_raw: np.ndarray) -> np.ndarray:
    """
    Floor the numerical-diffusion tail, clip to [0, cmax], then map to [-1, 1]
    """
    floored = floor_noise(cyto, cmax_raw)
    clipped = np.minimum(floored, cmax)
    return (2.0 * clipped / cmax - 1.0).astype(np.float32)


# filesystem discovery
def discover_runs(sim_root: Path, requested: Optional[List[str]]) -> List[Path]:
    pat = re.compile(r"^run_\d+$")
    runs = sorted(p for p in sim_root.iterdir()
                  if p.is_dir() and pat.match(p.name))
    if requested:
        runs = [r for r in runs if r.name in requested]
    if not runs:
        raise FileNotFoundError(f"No run_* directories under {sim_root}")
    return runs


def list_mcs_steps(run_dir: Path) -> List[int]:
    """MCS values for which BOTH CytoStep and CellStep exist."""
    lattice_dir = run_dir / "LatticeData"
    if not lattice_dir.exists():
        return []
    cyto_pat = re.compile(r"CytoStep_(\d{7})\.npz$")
    cyto_steps = {int(m.group(1)) for f in lattice_dir.iterdir()
                  if (m := cyto_pat.match(f.name))}
    cell_pat = re.compile(r"CellStep_(\d{7})\.npz$")
    cell_steps = {int(m.group(1)) for f in lattice_dir.iterdir()
                  if (m := cell_pat.match(f.name))}
    common = sorted(cyto_steps & cell_steps)
    only_cyto = cyto_steps - cell_steps
    only_cell = cell_steps - cyto_steps
    if only_cyto:
        warnings.warn(f"{run_dir.name}: {len(only_cyto)} CytoStep files have "
                      f"no matching CellStep - skipped.")
    if only_cell:
        warnings.warn(f"{run_dir.name}: {len(only_cell)} CellStep files have "
                      f"no matching CytoStep - skipped.")
    return common


# per-frame loaders
def load_cytostep(path: Path, grid: int) -> np.ndarray:
    with np.load(path) as data:
        try:
            stacked = np.stack([data[c] for c in CYTOKINE_NAMES], axis=-1)
        except KeyError as e:
            raise KeyError(f"{path} missing cytokine key {e}") from e
    if stacked.shape[:3] != (grid, grid, grid):
        raise ValueError(f"{path}: shape {stacked.shape[:3]} != ({grid},{grid},{grid})")
    return stacked.astype(np.float32)


def load_cellstep_to_mask(path: Path, grid: int) -> np.ndarray:
    with np.load(path) as data:
        if "cell_type" not in data:
            raise KeyError(f"{path} missing 'cell_type' key (found: {list(data.keys())})")
        ct = data["cell_type"]
    if ct.shape != (grid, grid, grid):
        raise ValueError(f"{path}: cell_type shape {ct.shape} != ({grid},{grid},{grid})")
    mask = np.zeros((grid, grid, grid, 5), dtype=np.float32)
    for ch, tid in enumerate(MASK_TYPE_IDS):
        mask[..., ch] = (ct == tid).astype(np.float32)
    return mask


def load_run_trajectory(run_dir: Path, grid: int
                        ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    steps = list_mcs_steps(run_dir)
    if not steps:
        raise FileNotFoundError(f"{run_dir}/LatticeData has no matching "
                                f"CytoStep_/CellStep_ pairs")
    cyto_list, mask_list = [], []
    for mcs in steps:
        cyto_list.append(load_cytostep(
            run_dir / "LatticeData" / f"CytoStep_{mcs:07d}.npz", grid))
        mask_list.append(load_cellstep_to_mask(
            run_dir / "LatticeData" / f"CellStep_{mcs:07d}.npz", grid))
    return (np.stack(cyto_list, axis=0),
            np.stack(mask_list, axis=0),
            steps)


# windows
def build_windows(cyto_traj: np.ndarray, mask_traj: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = cyto_traj.shape[0]
    if T < 3:
        raise ValueError(f"Need >=3 frames per run, got {T}")
    feats = np.concatenate([cyto_traj, mask_traj], axis=-1)  # (T,G,G,G,11)
    Xb_win = np.stack([feats[:-2], feats[1:-1]], axis=1).astype(np.float32)
    Y_win = cyto_traj[2:].astype(np.float32)
    Ym_win = mask_traj[2:].astype(np.float32)
    t_idx = np.arange(2, T, dtype=np.int64)
    return Xb_win, Y_win, Ym_win, t_idx


# trunk input (xyz in [-1, 1] + t in [-1, 1])
def build_trunk_xyzt(G: int, t_norms: np.ndarray) -> np.ndarray:
    """Per-sample (G^3, 4) array: (x, y, z) in [-1, 1]^3 + t_norm in [-1, 1]."""
    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
    xyz = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (G^3, 3)
    N = t_norms.shape[0]
    Xt = np.empty((N, G ** 3, 4), dtype=np.float32)
    Xt[:, :, :3] = xyz[None, :, :]
    Xt[:, :, 3] = t_norms[:, None]
    return Xt


# process + write a single run (loads, scales, saves, frees) -------------
def process_one_run(run_dir, G, cmax, kappas, qs, cmax_raw, qs_json_src,
                    N_req, train_n, out_root, frozen, scale_from):
    """Load one run, window it, scale with the given (frozen or fitted) scale,
    write the .npy tensors + metadata, and return its clip report (or None).
    Everything for this run is released when the function returns."""
    cyto_traj, mask_traj, steps = load_run_trajectory(run_dir, G)
    T = cyto_traj.shape[0]
    Xb_raw, Y_raw, Ym, t_idx = build_windows(cyto_traj, mask_traj)
    t_norms = 2.0 * (t_idx - 2) / max(1, (T - 1 - 2)) - 1.0
    n_cell = int(mask_traj.sum())
    del cyto_traj, mask_traj  # free the big trajectories immediately

    n_total = Xb_raw.shape[0]
    if n_total < N_req:
        warnings.warn(f"{run_dir.name}: only {n_total} windows, requested {N_req}")
        N = n_total
    else:
        N = N_req
        idx = np.linspace(0, n_total - 1, N).round().astype(int)
        Xb_raw = Xb_raw[idx]; Y_raw = Y_raw[idx]; Ym = Ym[idx]
        t_norms = t_norms[idx]

    clip_rep = None
    if frozen:
        clip_rep = clipping_report(Y_raw, cmax, cmax_raw)
        worst = max(clip_rep.items(), key=lambda kv: kv[1]["clipped_frac"])
        print(f"  [clip] {run_dir.name}: worst {worst[0]} "
              f"{100 * worst[1]['clipped_frac']:.2f}% active voxels clipped "
              f"(max/cmax={worst[1]['max_over_cmax']:.2f})")

    Xb_cyt = apply_cyto_scaling(Xb_raw[..., :6], cmax, cmax_raw)
    Xb_msk = Xb_raw[..., 6:].astype(np.float32)
    Xb = np.concatenate([Xb_cyt, Xb_msk], axis=-1)
    del Xb_raw, Xb_cyt, Xb_msk
    Y = apply_cyto_scaling(Y_raw, cmax, cmax_raw); del Y_raw
    X_unet = (Xb.transpose(0, 2, 3, 4, 1, 5)
                .reshape(N, G, G, G, 22).astype(np.float32))
    X_trunk = build_trunk_xyzt(G, t_norms.astype(np.float32))

    out_dir = out_root / run_dir.name / f"{G}x{G}x{G}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_unet.npy", X_unet)
    np.save(out_dir / "X_branch.npy", Xb.astype(np.float32))
    np.save(out_dir / "X_trunk.npy", X_trunk)
    np.save(out_dir / "Y_target.npy", Y.astype(np.float32))
    np.save(out_dir / "Y_masks_spatial.npy", Ym.astype(np.float32))
    del X_unet, Xb, X_trunk, Y, Ym

    metadata = {
        "grid": G, "n_samples": int(N), "run": run_dir.name,
        "train_n": int(train_n), "cytokine_names": CYTOKINE_NAMES,
        "mask_channels": [{"name": n, "type_id": t}
                          for n, t in zip(MASK_NAMES, MASK_TYPE_IDS)],
        "scaling": {
            "kind": "kurtosis_adaptive_clip_to_minus1_plus1",
            "source": ("frozen:" + str(scale_from)) if frozen
                      else "fit_on_train_split",
            "max": cmax.tolist(),
            "clip_percentile": [(-1.0 if q is None else q) for q in qs],
            "excess_kurtosis": kappas.tolist(),
            "noise_floor_frac": NOISE_FLOOR_FRAC,
            "cmax_raw": cmax_raw.tolist(),
            "scope": ("reused frozen scale (inference on unseen run)" if frozen
                      else "fit on train split (first train_n target windows)"),
            "schedule": "q=98.0(k>=600),98.5(300-600),99.0(100-300),"
                        "99.5(20-100),none(k<20)",
            "formula_scaled": "x_scaled = 2*min(x,cmax)/cmax - 1",
            "formula_denorm": "x_phys = (x_scaled + 1)/2 * cmax",
            "coord_range": "xyz in [-1,1], t in [-1,1]",
        },
        "window": {"input_frames": 2, "target_offset": 1,
                   "description": "(t, t+1) -> t+2"},
    }
    if clip_rep is not None:
        metadata["clipping_vs_frozen_scale"] = clip_rep
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[write] {out_dir}  (N={N}, T={T}, cell-voxels={n_cell})")
    return clip_rep


# main assembly
def assemble(args):
    import gc
    sim_root = args.sim_root.resolve()
    out_root = args.out_root.resolve()
    if not sim_root.exists():
        sys.exit(f"ERROR: --sim-root not found: {sim_root}")
    runs = discover_runs(sim_root, args.runs)
    frozen = args.scale_from is not None
    G = args.grid
    print(f"[info] sim_root = {sim_root}")
    print(f"[info] found {len(runs)} run(s)")
    print(f"[info] grid    = {G}")
    print(f"[info] scaling = {'FROZEN from ' + str(args.scale_from) if frozen else 'fit on train split'}")

    if frozen:
        # Frozen scale: stream run-by-run, never holding more than one run in
        # RAM. This is what avoids the OOM kill on a 100-run sweep.
        cmax, kappas, qs, cmax_raw = load_external_scale(args.scale_from)
        print(f"\n[info] per-cytokine clip (frozen):")
        for name, v, k, q in zip(CYTOKINE_NAMES, cmax, kappas, qs):
            qstr = "no-clip" if q is None else f"p{q}"
            print(f"         {name:5s}: q={qstr:8s}  cmax={v:.3e}")
        print()
        done = 0
        for run_dir in runs:
            try:
                process_one_run(run_dir, G, cmax, kappas, qs, cmax_raw, None,
                                args.n_samples, args.train_n, out_root, True,
                                args.scale_from)
                done += 1
            except Exception as e:
                print(f"  [ERROR] {run_dir.name}: {e} -- skipping")
            gc.collect()
        print(f"\n[done]  {done}/{len(runs)} run(s) written to {out_root}")
        return

    # Non-frozen (fit-on-train): used for the single benchmark run only, so
    # holding it in memory is fine. Load, fit scale, write.
    per_run = []
    for run_dir in runs:
        print(f"[load] {run_dir.name} ...", end=" ", flush=True)
        cyto_traj, mask_traj, steps = load_run_trajectory(run_dir, G)
        T = cyto_traj.shape[0]
        Xb_win, Y_win, Ym_win, t_idx = build_windows(cyto_traj, mask_traj)
        t_norms = 2.0 * (t_idx - 2) / max(1, (T - 1 - 2)) - 1.0
        per_run.append({"name": run_dir.name, "Xb_raw": Xb_win, "Y_raw": Y_win,
                        "Ym": Ym_win, "t_norms": t_norms})
        print(f"T={T}, windows={Xb_win.shape[0]}, cell-voxels={int(mask_traj.sum())}")

    train_targets = []
    for r in per_run:
        n = min(args.train_n, r["Y_raw"].shape[0])
        train_targets.append(r["Y_raw"][:n].reshape(-1, 6))
    cmax, kappas, qs, cmax_raw = fit_cmax_per_cytokine(
        np.concatenate(train_targets, axis=0))
    print(f"\n[info] per-cytokine clip (fit on train split):")
    for name, v, k, q in zip(CYTOKINE_NAMES, cmax, kappas, qs):
        qstr = "no-clip" if q is None else f"p{q}"
        print(f"         {name:5s}: kurtosis={k:10.2f}  q={qstr:8s}  cmax={v:.3e}")
    print()

    N_req = args.n_samples
    for r in per_run:
        Xb_raw = r["Xb_raw"]; Y_raw = r["Y_raw"]; Ym = r["Ym"]; t_norms = r["t_norms"]
        n_total = Xb_raw.shape[0]
        if n_total < N_req:
            warnings.warn(f"{r['name']}: only {n_total} windows, requested {N_req}")
            N = n_total
        else:
            N = N_req
            idx = np.linspace(0, n_total - 1, N).round().astype(int)
            Xb_raw = Xb_raw[idx]; Y_raw = Y_raw[idx]; Ym = Ym[idx]
            t_norms = t_norms[idx]
        Xb_cyt = apply_cyto_scaling(Xb_raw[..., :6], cmax, cmax_raw)
        Xb = np.concatenate([Xb_cyt, Xb_raw[..., 6:].astype(np.float32)], axis=-1)
        Y = apply_cyto_scaling(Y_raw, cmax, cmax_raw)
        X_unet = (Xb.transpose(0, 2, 3, 4, 1, 5)
                    .reshape(N, G, G, G, 22).astype(np.float32))
        X_trunk = build_trunk_xyzt(G, t_norms.astype(np.float32))
        out_dir = out_root / r["name"] / f"{G}x{G}x{G}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X_unet.npy", X_unet)
        np.save(out_dir / "X_branch.npy", Xb.astype(np.float32))
        np.save(out_dir / "X_trunk.npy", X_trunk)
        np.save(out_dir / "Y_target.npy", Y.astype(np.float32))
        np.save(out_dir / "Y_masks_spatial.npy", Ym.astype(np.float32))
        metadata = {
            "grid": G, "n_samples": int(N), "run": r["name"],
            "train_n": int(args.train_n), "cytokine_names": CYTOKINE_NAMES,
            "mask_channels": [{"name": n, "type_id": t}
                              for n, t in zip(MASK_NAMES, MASK_TYPE_IDS)],
            "scaling": {
                "kind": "kurtosis_adaptive_clip_to_minus1_plus1",
                "source": "fit_on_train_split",
                "max": cmax.tolist(),
                "clip_percentile": [(-1.0 if q is None else q) for q in qs],
                "excess_kurtosis": kappas.tolist(),
                "noise_floor_frac": NOISE_FLOOR_FRAC,
                "cmax_raw": cmax_raw.tolist(),
                "scope": "fit on train split (first train_n target windows)",
                "schedule": "q=98.0(k>=600),98.5(300-600),99.0(100-300),"
                            "99.5(20-100),none(k<20)",
                "formula_scaled": "x_scaled = 2*min(x,cmax)/cmax - 1",
                "formula_denorm": "x_phys = (x_scaled + 1)/2 * cmax",
                "coord_range": "xyz in [-1,1], t in [-1,1]",
            },
            "window": {"input_frames": 2, "target_offset": 1,
                       "description": "(t, t+1) -> t+2"},
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[write] {out_dir}  (N={N})")
    print(f"\n[done]  {len(per_run)} run(s) written to {out_root}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-root", type=Path, required=True,
                   help="Folder containing run_*/LatticeData/{Cyto,Cell}Step_*.npz")
    p.add_argument("--out-root", type=Path, default=Path("./preprocessed_3d"),
                   help="Output root; {G}x{G}x{G}/ subdir will be created.")
    p.add_argument("--grid", type=int, required=True,
                   help="Lattice edge length L (must match params_grid.L).")
    p.add_argument("--n-samples", type=int, default=99,
                   help="Target sample count after concatenating runs (default 99).")
    p.add_argument("--train-n", type=int, default=DEFAULT_TRAIN_N,
                   help="Number of leading windows used to fit the clip scale "
                        "(default 70, matching the 70/10/19 chronological split). "
                        "Ignored when --scale-from is given.")
    p.add_argument("--scale-from", type=Path, default=None,
                   help="Path to an existing metadata.json whose scaling.max / "
                        "cmax_raw / clip_percentile will be reused instead of "
                        "fitting a fresh scale. Use this to preprocess runs for "
                        "inference with the trained surrogate's normalisation.")
    p.add_argument("--runs", nargs="+", default=None,
                   help="Process only these run_XXXX names.")
    return p.parse_args()


if __name__ == "__main__":
    assemble(parse_args())