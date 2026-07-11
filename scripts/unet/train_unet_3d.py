#!/usr/bin/env python3
"""
train_unet_3d.py

3D U-Net surrogate for volumetric cytokine fields (thesis architecture).
Trains on ONE run (chronological 70/10/19 split over its 99 time-windows).
seed 42 tunes hyperparameters with Optuna; seeds 1 and 100 reuse the seed-42
configuration to test stability.

Usage:
    python train_unet_3d.py --cytokine il8 --seed 42 --run run_0062 \\
        --data ../../preprocessed_3d --out ../../models/unet_3d
"""

import os
import json
import argparse
import random
import time
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def configure_gpu(use_mixed_precision=True):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"  [warn] couldn't set memory_growth on {gpu}: {e}")
    if gpus:
        print(f"  [gpu] {len(gpus)} GPU(s) available")
        if use_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
                print(f"  [gpu] mixed precision: mixed_bfloat16 (A100 optimised)")
            except Exception:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        print(f"  [gpu] NO GPU DETECTED - running on CPU (slow).")


configure_gpu()

N_TRIALS    = 20
TUNE_EPOCHS = 20
FULL_EPOCHS = 200


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# model (3D)
def _conv_block(x, filters, kernel_size=3):
    x = tf.keras.layers.Conv3D(filters, kernel_size, padding="same",
                               activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv3D(filters, kernel_size, padding="same",
                               activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def _encoder_block(x, filters):
    skip = _conv_block(x, filters)
    pool = tf.keras.layers.MaxPooling3D(pool_size=2, padding="same")(skip)
    return skip, pool


def _decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv3DTranspose(filters, 2, strides=2, padding="same",
                                        activation="relu")(x)
    xs = [int(x.shape[d]) for d in (1, 2, 3)]
    ss = [int(skip.shape[d]) for d in (1, 2, 3)]
    if xs != ss:
        crop = [[max(0, (xs[d] - ss[d]) // 2),
                 max(0, xs[d] - ss[d] - (xs[d] - ss[d]) // 2)] for d in range(3)]
        pad  = [[max(0, (ss[d] - xs[d]) // 2),
                 max(0, ss[d] - xs[d] - (ss[d] - xs[d]) // 2)] for d in range(3)]
        if any(c[0] + c[1] > 0 for c in crop):
            x = tf.keras.layers.Cropping3D(cropping=tuple(map(tuple, crop)))(x)
        if any(p[0] + p[1] > 0 for p in pad):
            x = tf.keras.layers.ZeroPadding3D(padding=tuple(map(tuple, pad)))(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = _conv_block(x, filters)
    return x


def build_unet3d(grid_size: int, in_channels: int = 22,
                 out_channels: int = 1, base_filters: int = 16,
                 depth: int = 3, dropout: float = 0.0):
    inputs = tf.keras.Input(shape=(grid_size, grid_size, grid_size, in_channels))

    skips = []
    x = inputs
    for i in range(depth):
        f = base_filters * (2 ** i)
        skip, x = _encoder_block(x, f)
        skips.append(skip)

    x = _conv_block(x, base_filters * (2 ** depth))
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    for i in reversed(range(depth)):
        f = base_filters * (2 ** i)
        x = _decoder_block(x, skips[i], f)

    outputs = tf.keras.layers.Conv3D(out_channels, 1, padding="same",
                                     activation="linear")(x)
    return tf.keras.Model(inputs, outputs, name="unet3d")


# metrics (3D)
def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))


def _inv_fisher_z(z):
    return float(np.tanh(z))


def compute_2d_slice_metrics(yt, yp, clip_max):
    """Per-axis 2D mid-slice metrics on the three orthogonal mid-planes."""
    T = yt.shape[0]; G = yt.shape[1]
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0
    mid = G // 2

    out = {}
    for axis_name, sl in (("xy_midplane_z",  np.s_[:, :, :, mid, 0]),
                          ("xz_midplane_y",  np.s_[:, :, mid, :, 0]),
                          ("yz_midplane_x",  np.s_[:, mid, :, :, 0])):
        gts = yt[sl]; prs = yp[sl]
        r2s, ssims, n_skip = [], [], 0
        for t in range(T):
            gt = gts[t]; pr = prs[t]
            if np.std(gt) > 1e-12:
                r2s.append(float(r2_score(gt.flatten(), pr.flatten())))
            else:
                n_skip += 1
            dr = float(np.max(gt) - np.min(gt))
            if dr > 1e-12:
                ssims.append(float(ssim(gt, pr, data_range=fixed_dr)))
        out[axis_name] = {
            "R2":   float(np.mean(r2s))   if r2s   else 0.0,
            "SSIM": float(np.mean(ssims)) if ssims else 0.0,
            "Skipped_Frames": n_skip,
        }
    return out


def calculate_metrics(y_true, y_pred, masks, clip_max):
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    y_t = y_true[:min_t]
    y_p = np.maximum(y_pred[:min_t], 0.0)
    m_s = np.max(masks[:min_t], axis=-1, keepdims=True)

    sq_diff = np.square(y_t - y_p)
    rmse = float(np.sqrt(np.sum(sq_diff * m_s) / (np.sum(m_s) + 1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(sq_diff)))
    r2 = float(r2_score(y_t.flatten(), y_p.flatten()))

    per_t_r2 = []
    for t in range(min_t):
        gt_f = y_t[t].flatten(); pr_f = y_p[t].flatten()
        per_t_r2.append(float(r2_score(gt_f, pr_f)) if np.std(gt_f) > 1e-12 else np.nan)

    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices, n_empty = [], 0
    z_corrs = []
    ssims, n_ssim_skip = [], 0
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0

    for t in range(min_t):
        gt = y_t[t, :, :, :, 0]; pr = y_p[t, :, :, :, 0]
        g_b = (gt > dice_thr).astype(float); p_b = (pr > dice_thr).astype(float)
        if np.sum(g_b) + np.sum(p_b) == 0:
            n_empty += 1
        else:
            dices.append((2.0 * np.sum(g_b * p_b)) / (np.sum(g_b) + np.sum(p_b) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val):
                z_corrs.append(_fisher_z(r_val))
        dr = float(np.max(gt) - np.min(gt))
        if dr > 1e-12:
            ssims.append(float(ssim(gt, pr, data_range=fixed_dr)))
        else:
            n_ssim_skip += 1

    spatial_corr = _inv_fisher_z(float(np.mean(z_corrs))) if z_corrs else 0.0

    return {
        "Global_R2":           r2,
        "Per_Timestep_R2":     per_t_r2,
        "Masked_RMSE":         rmse,
        "Unmasked_RMSE":       unmasked_rmse,
        "Avg_Dice":            float(np.mean(dices)) if dices else 0.0,
        "Dice_Empty_Skipped":  n_empty,
        "Spatial_Correlation": spatial_corr,
        "SSIM":                float(np.mean(ssims)) if ssims else 0.0,
        "SSIM_Skipped_Frames": n_ssim_skip,
        "Slice_2D":            compute_2d_slice_metrics(y_t, y_p, clip_max),
    }


def denormalize(scaled, clip_max):
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max


# optuna
def make_objective(X_train, Y_train, X_val, Y_val, grid_size, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()

        base_filters = trial.suggest_categorical("base_filters", [8, 16, 32])
        depth        = trial.suggest_categorical("depth",        [2, 3])
        dropout      = trial.suggest_float("dropout", 0.0, 0.3)
        lr           = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [1, 2])

        model = build_unet3d(grid_size=grid_size, base_filters=base_filters,
                             depth=depth, dropout=dropout)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=TUNE_EPOCHS, batch_size=batch_size, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True)],
        )
        return float(min(history.history["val_loss"]))
    return objective


# pipeline
def run_pipeline(grid, seed, cytokine, run_name,
                 data_root="./preprocessed_3d", out_root="./models/unet_3d"):
    set_seed(seed)
    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx = cyt_names.index(cytokine.lower())

    data_path = Path(f"{data_root}/{run_name}/{grid}x{grid}x{grid}")
    out_dir = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"  [skip] {data_path} does not exist")
        return

    X = np.load(data_path / "X_unet.npy").astype(np.float32)
    Y = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)

    with open(data_path / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    in_channels = X.shape[-1]  # 22

    X_train, Y_train = X[:70],   Y[:70]
    X_val,   Y_val   = X[70:80], Y[70:80]

    suffix = f"{cytokine}_{run_name}_{grid}_{seed}"

    if seed == 42:
        print(f"\nOptuna [{cytokine.upper()}] {run_name} {grid}x{grid}x{grid} - "
              f"{N_TRIALS} trials x {TUNE_EPOCHS} epochs...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            make_objective(X_train, Y_train, X_val, Y_val, grid, 42),
            n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,),
        )
        best = study.best_params
        optuna_val = float(study.best_value)
        print(f"  Best: {best}  |  val_loss = {optuna_val:.6f}")
    else:
        ref_path = out_dir / f"res_{cytokine}_{run_name}_{grid}_42.json"
        print(f"  Loading HP from {ref_path.name}")
        with open(ref_path) as f:
            ref = json.load(f)
        best = ref["best_params"]
        optuna_val = ref["optuna_best_val_loss"]

    tf.keras.backend.clear_session()
    set_seed(seed)

    model = build_unet3d(grid_size=grid, in_channels=in_channels,
                         base_filters=best["base_filters"], depth=best["depth"],
                         dropout=best["dropout"])
    model.compile(optimizer=tf.keras.optimizers.Adam(best["learning_rate"]),
                  loss="mse")

    print(f"Final training [{cytokine.upper()}] {run_name} {grid}x{grid}x{grid}...")

    t_train_start = time.time()
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=FULL_EPOCHS, batch_size=best["batch_size"], verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ],
    )
    train_elapsed = time.time() - t_train_start
    print(f"  Training time: {train_elapsed:.1f}s")

    t_pred_start = time.time()
    Y_p_scaled = model.predict(X, batch_size=1, verbose=0)
    pred_elapsed = time.time() - t_pred_start
    print(f"  Prediction time (all {X.shape[0]} samples): {pred_elapsed:.1f}s")

    Y_p_phys = denormalize(Y_p_scaled, clip_max)
    Y_a_phys = denormalize(Y,          clip_max)

    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine, "run": run_name,
        "best_params": best,
        "optuna_best_val_loss": optuna_val,
        "train_time_seconds": round(train_elapsed, 2),
        "pred_time_seconds": round(pred_elapsed,  2),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Y_a_phys[80:90], Y_p_phys[80:90], M[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Y_a_phys[90:99], Y_p_phys[90:99], M[90:99], clip_max),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"DONE: {out_dir}/res_{suffix}.json")


def discover_runs(data_root: Path, grid: int):
    pat = re.compile(r"^run_\d+$")
    runs = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and pat.match(p.name):
            if (p / f"{grid}x{grid}x{grid}").exists():
                runs.append(p.name)
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid",     type=int, default=50)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--run",      type=str, default="run_0062",
                        help="Specific run name. Default: run_0062 (benchmark run).")
    parser.add_argument("--data",     type=str, default="./preprocessed_3d",
                        help="Root folder containing <run>/<G>x<G>x<G>/*.npy.")
    parser.add_argument("--out",      type=str, default="./models/unet_3d",
                        help="Output folder for weights + JSON.")
    args = parser.parse_args()

    data_root = Path(args.data)

    if args.grid is not None:
        grids = [args.grid]
    else:
        first_run = next((p for p in sorted(data_root.iterdir())
                          if p.is_dir() and p.name.startswith("run_")), None)
        if first_run is None:
            raise FileNotFoundError(f"No run_* dirs in {data_root}")
        grids = sorted(int(d.name.split("x")[0])
                       for d in first_run.iterdir() if d.is_dir())

    for grid_size in grids:
        if args.run is not None:
            runs = [args.run]
        else:
            runs = discover_runs(data_root, grid_size)
            if not runs:
                print(f"  [skip] no runs with {grid_size}x{grid_size}x{grid_size} found")
                continue
            print(f"\n[info] grid {grid_size}x{grid_size}x{grid_size}: "
                  f"iterating {len(runs)} runs")

        for run_name in runs:
            run_pipeline(grid_size, args.seed, args.cytokine, run_name,
                         data_root=args.data, out_root=args.out)
