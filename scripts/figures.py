#!/usr/bin/env python3
"""
Figure generation for the 3D burn-wound surrogate + calibration paper.

Benchmark run: run_0062 (central point of the LHS sweep, chosen by
helpers/select_benchmark_run.py), 50^3, seeds 1/42/100.

Two model families:
    DeepONet (gated trunk)  -> models/deeponet_3d/res_<cyt>_run_0062_50_<seed>.json
    U-Net (3D conv)         -> models/unet_3d/res_<cyt>_run_0062_50_<seed>.json

Calibration (Sobol + SMoRe ParS) reads the calibration bundles under
<sweep-root>/ (default: smores/):
    calibration_results_topk10.json          (full ranking + recovery)
    calibration_results_topk5_sobol.json
    calibration_results_topk5_identifiable.json

FIGURE GROUPS (select with --figs):
  Surrogate (chart, read JSON only):
    B1 -> F_accuracy      grouped Global R2 (Near/Far), error bars over seeds
    B4 -> F_metrics       SSIM / Dice / Corr (near) per model x cytokine
    B3 -> F_speedup       inference speed-up vs ABM (needs ABM_RUNTIME_S[50])
    E4 -> F_midplane_r2   R2 on xy/xz/yz mid-planes (near)
  Surrogate (field, need preprocessed + weights):
    S  -> F_concentrations  six-cytokine mean trajectories, split shaded
    A  -> F_eda_slices      xy/xz/yz mid-plane slices, IL-8 vs IL-10 (GT)
    E  -> F_recon_slices    GT / Pred / |diff| slices, DeepONet, both cytokines
          F_diff_models     cross-model |diff| slices, DeepONet vs U-Net
  Calibration (chart, read calibration JSON only):
    K1 -> F_sobol         Sobol total-order ranking (bar)
    K2 -> F_recovery      recovered-vs-true scatter + per-param R2 +
                          sensitivity-vs-identifiability scatter (init_ec)

Default (no --figs): builds every chart + field figure + tables.

Chart groups read only JSON and run anywhere. Field groups load the
preprocessed data (S/A) or trained weights (E).

Output: figures_3d/
"""

import os, json, argparse, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# config 
RUN   = "run_0062"
GRID  = 50
SEEDS = [1, 42, 100]
CYTS  = [("il8", "IL-8"), ("il10", "IL-10")]

MODELS = ["DeepONet", "U-Net"]
MODEL_DIR = {"DeepONet": "deeponet_3d", "U-Net": "unet_3d"}

# Full palette from the 2D script. In 3D only DeepONet/U-Net are used, so the
# colours originally assigned to PI-DeepONet / STA-LSTM / PINN are free and get
# reused for the calibration panels (kinetic vs initial-population).
P = {"DeepONet": "#ffd700", "PI-DeepONet": "#ffb14e", "U-Net": "#fa8775",
     "STA-LSTM": "#ea5f94", "PINN": "#cd34b5"}
CC = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4", "#42d4f4"]

COL = {"DeepONet": P["DeepONet"], "U-Net": P["U-Net"]}
MK  = {"DeepONet": "o",       "U-Net": "s"}
LS  = {"DeepONet": "-",       "U-Net": "-"}

# Calibration panels: kinetic vs initial-population, using colours from CC
# (blue = CC[3], red = CC[0]).
TYPE_COL = {"kinetic": CC[3], "initial": CC[0]}

# Consistent colormap for all field slices (the dark purple-to-yellow one).
FIELD_CMAP = "magma"
ERR_CMAP   = "inferno"

EBAR_CAPSIZE = 6
EBAR_KW = dict(ecolor="black", elinewidth=1.6, capthick=1.6)

# ABM reference wall-clock for one 100h trajectory at 50^3 (run_0062 canary).
ABM_RUNTIME_S = {50: 4984.06}

# Paths. Models + preprocessed live under burns/ ; sweep + calibration under
# burns/smores/ . Defaults assume the script is run from burns/.
MODELS_ROOT = Path("./models")
PREP_ROOT   = Path("./preprocessed_3d")
SWEEP_ROOT  = Path("./smores")            # holds calibration_results_*.json + sweep/
FIGDIR      = Path("./figures_3d")

CYT_INDEX = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}
CYT_ALL = [("il8", "IL-8"), ("il1", r"IL-1$\beta$"), ("il6", "IL-6"),
           ("il10", "IL-10"), ("tnf", r"TNF-$\alpha$"), ("tgf", r"TGF-$\beta$")]
SPLIT_SPANS = {"Train": (0, 72), "Val": (72, 82), "Near": (82, 91), "Far": (91, 101)}
SPLIT_COL   = {"Train": "green", "Val": "orange", "Near": "dodgerblue", "Far": "red"}


def setup():
    import shutil
    use_tex = shutil.which("latex") is not None
    if use_tex:
        try:
            plt.rcParams["text.usetex"] = True
            f = plt.figure(); f.text(0.5, 0.5, r"$x_1$"); f.canvas.draw(); plt.close(f)
        except Exception:
            use_tex = False
            plt.rcParams["text.usetex"] = False
    plt.rcParams.update({
        "text.usetex": use_tex, "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    if use_tex:
        print("  LaTeX rendering on")
    else:
        print("  LaTeX not found, using mathtext")


def savef(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p = FIGDIR / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"    -> {p}")


def _bar_labels(ax, xs, means, stds, fmt="{:.3f}", fs=6):
    """Place value+/-spread labels. For positive bars, above bar top (incl std).
    For negative bars, just ABOVE the x-axis (y slightly >0) so the text never
    overlaps the downward bar."""
    for x, mu, sd in zip(xs, means, stds):
        if mu is None:
            continue
        sd = sd or 0
        lbl = fmt.format(mu) + (f"\n$\\pm${fmt.format(sd)}" if sd is not None else "")
        if mu >= 0:
            y = mu + sd + 0.02
            va = "bottom"
        else:
            # negative bar: label sits just above the axis line
            y = 0.02
            va = "bottom"
        ax.text(x, y, lbl, ha="center", va=va, fontsize=fs, fontweight="bold")


# JSON metric helpers 
def load_res(model, cyt, seed):
    p = MODELS_ROOT / MODEL_DIR[model] / f"res_{cyt}_{RUN}_{GRID}_{seed}.json"
    return json.load(open(p)) if p.exists() else None


def horizon(res, which):
    if not res:
        return {}
    for k in res.get("results", {}):
        if which in k.lower():
            return res["results"][k]
    return {}


def metric_seeds(model, cyt, key, which="near"):
    vals = []
    for s in SEEDS:
        h = horizon(load_res(model, cyt, s), which)
        v = h.get(key)
        if v is None or (isinstance(v, float) and v != v):
            continue
        vals.append(float(v))
    return vals


def mean_std(model, cyt, key, which="near"):
    v = metric_seeds(model, cyt, key, which)
    if not v:
        return None, None, 0
    if len(v) == 1:
        return v[0], 0.0, 1
    return float(np.mean(v)), float(np.std(v, ddof=0)), len(v)


def time_seeds(model, cyt, field):
    vals = []
    for s in SEEDS:
        r = load_res(model, cyt, s)
        if not r:
            continue
        v = r.get(field)
        if v and v > 0:
            vals.append(float(v))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


#  SURROGATE CHART FIGURES
def figB1():
    """Global R2 (Near & Far) grouped bars, error bars over seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    all_lo, all_hi = [], []
    for ax, (cyt, cl) in zip(axes, CYTS):
        x = np.arange(len(MODELS)); w = 0.35
        for j, which in enumerate(["near", "far"]):
            mus, sds = [], []
            for m in MODELS:
                mu, sd, _ = mean_std(m, cyt, "Global_R2", which)
                mus.append(mu if mu is not None else 0)
                sds.append(sd if sd is not None else 0)
            off = (j - 0.5) * w
            hatch = "" if which == "near" else "//"
            ax.bar(x + off, mus, w, yerr=sds, capsize=EBAR_CAPSIZE,
                   color=[COL[m] for m in MODELS], edgecolor="black", lw=0.4,
                   hatch=hatch, alpha=0.9 if which == "near" else 0.65,
                   error_kw=EBAR_KW)
            _bar_labels(ax, x + off, mus, sds, fs=5.5)
            for mu, sd in zip(mus, sds):
                all_lo.append(mu - sd); all_hi.append(mu + sd)
        ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=9)
        ax.set_title(cl, fontsize=10)
        ax.axhline(0, color="black", lw=0.5); ax.axhline(1, color="gray", ls=":", alpha=0.4)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
    # shared y-limits with headroom for labels + error caps
    lo = min(all_lo); hi = max(all_hi)
    ylo = min(-0.12, lo - 0.12) if lo < 0 else 0
    for ax in axes:
        ax.set_ylim(ylo, max(1.15, hi + 0.12))
    axes[0].set_ylabel(r"Global $R^2$")
    handles = [Patch(fc="lightgray", ec="black", hatch="", label="Near (t82-91)"),
               Patch(fc="lightgray", ec="black", hatch="//", label="Far (t92-100)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    fig.text(0.5, -0.14, f"Error bars: $\\pm$1 s.d. over seeds {SEEDS}.",
             ha="center", fontsize=7, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savef(fig, "F2_accuracy")


def figB4():
    """SSIM / Dice / Corr (near) grouped bars."""
    mets = [("SSIM", "SSIM"), ("Avg_Dice", "Dice"), ("Spatial_Correlation", "Corr")]
    hatches = ["", "//", ".."]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (cyt, cl) in zip(axes, CYTS):
        x = np.arange(len(MODELS)); w = 0.25
        for j, (key, lab) in enumerate(mets):
            mus, sds = [], []
            for m in MODELS:
                mu, sd, _ = mean_std(m, cyt, key, "near")
                mus.append(mu if mu is not None else 0)
                sds.append(sd if sd is not None else 0)
            off = (j - 1) * w
            ax.bar(x + off, mus, w, yerr=sds, capsize=EBAR_CAPSIZE,
                   color=[COL[m] for m in MODELS], edgecolor="black", lw=0.3,
                   hatch=hatches[j], alpha=0.6 + j * 0.12, error_kw=EBAR_KW)
            _bar_labels(ax, x + off, mus, sds, fs=5)
        ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=9)
        ax.set_title(cl, fontsize=10); ax.axhline(0, color="black", lw=0.5)
        ax.set_ylim(0, 1.18); ax.set_xlim(-0.6, len(MODELS) - 0.4)
    axes[0].set_ylabel("Score (near)")
    handles = [Patch(fc="lightgray", ec="black", hatch=hatches[j], label=mets[j][1])
               for j in range(len(mets))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savef(fig, "F2_metrics")


def figB3():
    """Inference speed-up vs ABM (log y)."""
    abm = ABM_RUNTIME_S.get(GRID)
    if abm is None:
        print("    [B3] no ABM runtime set; skipping."); return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    labels, vals, errs, cols = [], [], [], []
    for m in MODELS:
        for cyt, cl in CYTS:
            pt_mu, pt_sd = time_seeds(m, cyt, "pred_time_seconds")
            if pt_mu is None or pt_mu <= 0:
                continue
            sp = abm / pt_mu
            sp_err = abm * (pt_sd or 0) / (pt_mu ** 2)
            labels.append(f"{m}\n{cl}"); vals.append(sp); errs.append(sp_err)
            cols.append(COL[m])
    x = np.arange(len(labels))
    ax.bar(x, vals, color=cols, edgecolor="black", lw=0.4, yerr=errs,
           capsize=EBAR_CAPSIZE, error_kw=EBAR_KW)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(top=max(v + e for v, e in zip(vals, errs)) * 4)
    ax.set_ylabel(r"Inference speed-up vs ABM ($\times$)")
    for xi, v, e in zip(x, vals, errs):
        elbl = f"\\,$\\pm$\\,{e:.0f}"      # always show, even if 0
        ax.text(xi, (v + e) * 1.25, f"{v:.0f}$\\times${elbl}", ha="center",
                fontsize=6.5, fontweight="bold")
    fig.tight_layout()
    savef(fig, "F4_speedup")


def figE4():
    """R2 on xy/xz/yz mid-planes (near), grouped bars."""
    planes = [("xy_midplane_z", "xy"), ("xz_midplane_y", "xz"), ("yz_midplane_x", "yz")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    all_lo, all_hi = [], []
    for ax, (cyt, cl) in zip(axes, CYTS):
        x = np.arange(len(planes)); w = 0.35
        for j, m in enumerate(MODELS):
            r2s, sds = [], []
            for pkey, _ in planes:
                vals = []
                for s in SEEDS:
                    h = horizon(load_res(m, cyt, s), "near")
                    sl = h.get("Slice_2D", {}).get(pkey, {})
                    v = sl.get("R2")
                    if v is not None and not (isinstance(v, float) and v != v):
                        vals.append(float(v))
                r2s.append(np.mean(vals) if vals else 0)
                sds.append(np.std(vals, ddof=0) if len(vals) > 1 else 0)
            xs = x + (j - 0.5) * w
            ax.bar(xs, r2s, w, yerr=sds, capsize=EBAR_CAPSIZE, color=COL[m],
                   edgecolor="black", lw=0.4, label=m, error_kw=EBAR_KW)
            _bar_labels(ax, xs, r2s, sds, fs=5.5)
            for mu, sd in zip(r2s, sds):
                all_lo.append(mu - sd); all_hi.append(mu + sd)
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in planes])
        ax.set_title(cl, fontsize=10); ax.axhline(0, color="black", lw=0.5)
        ax.set_xlim(-0.6, len(planes) - 0.4)
    lo = min(all_lo); hi = max(all_hi)
    ylo = min(-0.15, lo - 0.12) if lo < 0 else 0
    for ax in axes:
        ax.set_ylim(ylo, max(1.15, hi + 0.12))
    axes[0].set_ylabel(r"Slice $R^2$ (near, mean over seeds)")
    handles = [Patch(fc=COL[m], ec="black", label=m) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savef(fig, "F3_midplane_r2")


#  CALIBRATION CHART FIGURES  (new: Sobol + recovery)
def _load_calib(name="calibration_results_topk10.json"):
    for base in (SWEEP_ROOT, Path(".")):
        p = base / name
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError(f"{name} not found under {SWEEP_ROOT} or ./")


# type of each parameter, for colouring (kinetic vs initial-population)
PARAM_TYPE = {
    "keil8": "kinetic", "km1il6": "kinetic", "km2il10": "kinetic",
    "km2tgf": "kinetic", "lnril8": "kinetic", "sigmoidb": "kinetic",
    "init_ec": "initial", "init_n": "initial", "init_m": "initial",
    "init_f": "initial",
}


def figK1():
    """Sobol total-order ranking (horizontal bar), coloured by param type."""
    d = _load_calib()
    ranking = d["sobol"]["ranking"]
    params = [r["param"] for r in ranking]
    st = [r["ST_mean"] for r in ranking]
    cols = [TYPE_COL[PARAM_TYPE.get(p, "kinetic")] for p in params]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y = np.arange(len(params))[::-1]         # highest at top
    ax.barh(y, st, color=cols, edgecolor="black", lw=0.4)
    for yi, v in zip(y, st):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(params, fontsize=8)
    ax.set_xlabel(r"Sobol total-order index $S_T$ (mean over observables)")
    ax.set_xlim(0, max(st) * 1.18)
    handles = [Patch(fc=TYPE_COL["kinetic"], ec="black", label="kinetic"),
               Patch(fc=TYPE_COL["initial"], ec="black", label="initial population")]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    savef(fig, "F4_sobol")


def figK2():
    """Recovery panel: (a) recovered-vs-true for the two best params,
    (b) recovery R2 per param, (c) sensitivity vs identifiability scatter."""
    d = _load_calib()
    ranking = d["sobol"]["ranking"]
    st_by = {r["param"]: r["ST_mean"] for r in ranking}
    rec = d["recovery"]
    params = rec["selected_params"]
    r2 = rec["r2_per_param"]
    recovered = np.array(rec["recovered"])   # (100, 10)
    truth = np.array(rec["truth"])           # (100, 10)

    fig = plt.figure(figsize=(11.0, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # (a) recovered vs true for the two best-recovered parameters.
    # Colours: green (CC[2]) + purple (CC[4]) from the cytokine palette --
    # chosen because the recovery panels already use CC[3] (blue) and CC[0]
    # (red) for kinetic/initial, so these two are free and don't clash.
    ax = fig.add_subplot(gs[0, 0])
    best2 = sorted(params, key=lambda p: r2[p], reverse=True)[:2]
    marks = ["o", "s"]
    scatter_cols = [CC[2], CC[4]]        # green, purple
    for pi, p in enumerate(best2):
        j = params.index(p)
        t = truth[:, j]; rv = recovered[:, j]
        # normalise both to [0,1] by the true range, so two params share axes
        lo, hi = t.min(), t.max()
        tn = (t - lo) / (hi - lo + 1e-12)
        rn = (rv - lo) / (hi - lo + 1e-12)
        ax.scatter(tn, rn, s=18, marker=marks[pi], alpha=0.6,
                   color=scatter_cols[pi],
                   edgecolors="black", linewidths=0.3,
                   label=f"{p} ($R^2$={r2[p]:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    ax.set_xlabel("True (normalised)"); ax.set_ylabel("Recovered (normalised)")
    ax.set_title("Recovered vs.\\ true", fontsize=10)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.4)

    # (b) recovery R2 per parameter, ordered by Sobol rank
    ax = fig.add_subplot(gs[0, 1])
    order = [r["param"] for r in ranking]
    vals = [r2[p] for p in order]
    cols = [TYPE_COL[PARAM_TYPE.get(p, "kinetic")] for p in order]
    x = np.arange(len(order))
    ax.bar(x, vals, color=cols, edgecolor="black", lw=0.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"Recovery $R^2$")
    ax.set_title("Recovery by parameter", fontsize=10)
    ax.set_ylim(min(vals) - 0.2, 1.05)

    # (c) sensitivity vs identifiability
    ax = fig.add_subplot(gs[0, 2])
    for p in order:
        c = TYPE_COL[PARAM_TYPE.get(p, "kinetic")]
        ax.scatter(st_by[p], r2[p], s=45, color=c, edgecolors="black", linewidths=0.4)
        # annotate the standout cases
        if p in ("init_ec", "keil8", "sigmoidb"):
            ax.annotate(p, (st_by[p], r2[p]), textcoords="offset points",
                        xytext=(6, 4), fontsize=7)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(r"Sobol $S_T$ (sensitivity)")
    ax.set_ylabel(r"Recovery $R^2$ (identifiability)")
    ax.set_title("Sensitivity $\\neq$ identifiability", fontsize=10)
    # shade the "sensitive but not identifiable" region
    xlim = ax.get_xlim()
    ax.axhspan(ax.get_ylim()[0], 0, xmin=0, xmax=1, color="#e6194b", alpha=0.05)

    handles = [Patch(fc=TYPE_COL["kinetic"], ec="black", label="kinetic"),
               Patch(fc=TYPE_COL["initial"], ec="black", label="initial population")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8)
    savef(fig, "F5_recovery")


#  FIELD FIGURES (preprocessed + weights)
def _dp():
    return PREP_ROOT / RUN / f"{GRID}x{GRID}x{GRID}"

def _clip_max(cyt):
    meta = json.load(open(_dp() / "metadata.json"))
    return float(meta["scaling"]["max"][CYT_INDEX[cyt]])

def _denorm(x, cmax):
    return (np.asarray(x, np.float64) + 1.0) / 2.0 * cmax

def _raw_field(cyt, frame):
    idx = CYT_INDEX[cyt]
    Y = np.load(_dp() / "Y_target.npy").astype(np.float32)[..., idx]
    return _denorm(Y[frame], _clip_max(cyt))

def _raw_trajectory():
    Y = np.load(_dp() / "Y_target.npy").astype(np.float32)
    meta = json.load(open(_dp() / "metadata.json"))
    out = np.empty(Y.shape, np.float64)
    for cyt, idx in CYT_INDEX.items():
        out[..., idx] = _denorm(Y[..., idx], float(meta["scaling"]["max"][idx]))
    return out

def _ortho_slices(fig, gs_row, vol, cmap, label, vmax=None, show_titles=True, cbar=True):
    G = vol.shape[0]; h = G // 2
    if vmax is None:
        pos = vol[vol > 0]
        vmax = np.percentile(pos, 99.5) if pos.size else 1.0
    planes = [(vol[:, :, h], "xy"), (vol[:, h, :], "xz"), (vol[h, :, :], "yz")]
    for j, (sl, name) in enumerate(planes):
        ax = fig.add_subplot(gs_row[j])
        im = ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(label, fontsize=8, labelpad=2)
        if show_titles:
            ax.set_title(name, fontsize=8)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format="%.0e")
            cb.ax.tick_params(labelsize=4.5, pad=1)
            cb.ax.yaxis.get_offset_text().set_fontsize(4)

def figS():
    """Six-cytokine mean trajectories, chronological split shaded."""
    raw = _raw_trajectory()
    T = raw.shape[0]; t = np.arange(T)
    CC = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4", "#42d4f4"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for i, (cyt, cl) in enumerate(CYT_ALL):
        ax = axes[i // 3, i % 3]; idx = CYT_INDEX[cyt]
        mc = raw[..., idx].reshape(T, -1).mean(axis=1)
        ax.plot(t, mc, color=CC[i], lw=1.4)
        for lb, (t0, t1) in SPLIT_SPANS.items():
            ax.axvspan(t0, t1, alpha=0.06 if lb == "Train" else 0.10, color=SPLIT_COL[lb])
        ax.set_title(cl, fontsize=10)
        if i >= 3:
            ax.set_xlabel("Time (h)")
        if i % 3 == 0:
            ax.set_ylabel("Mean conc.")
        ax.set_xlim(0, T - 1)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))
    handles = [Patch(fc=SPLIT_COL[s], alpha=0.3, label=s) for s in SPLIT_SPANS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9, edgecolor="gray")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    savef(fig, "F1_concentrations")

def figA(frame=88):
    """xy/xz/yz mid-plane slices, IL-8 vs IL-10, ground truth."""
    fig = plt.figure(figsize=(7.6, 4.6))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.45)
    for row, (cyt, cl) in enumerate(CYTS):
        vol = _raw_field(cyt, frame)
        _ortho_slices(fig, [gs[row, j] for j in range(3)], vol,
                      FIELD_CMAP, cl)
    savef(fig, "F1_eda_slices")

# weight-based prediction (for GT vs Pred slice figures) 
def _load_field_deps():
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return tf

def _build_deeponet(tf, hidden, p):
    class Branch(tf.keras.layers.Layer):
        def __init__(s, h, p, **k): super().__init__(**k); s.f1 = tf.keras.layers.Dense(h, activation="relu"); s.f2 = tf.keras.layers.Dense(p)
        def call(s, x, training=False): return s.f2(s.f1(x))
    class Trunk(tf.keras.layers.Layer):
        def __init__(s, h, p, **k):
            super().__init__(**k)
            s.U = tf.keras.layers.Dense(h, activation="tanh"); s.V = tf.keras.layers.Dense(h, activation="tanh")
            s.W1a = tf.keras.layers.Dense(h, activation="relu"); s.W1b = tf.keras.layers.Dense(h)
            s.W2a = tf.keras.layers.Dense(h, activation="relu"); s.W2b = tf.keras.layers.Dense(h); s.out = tf.keras.layers.Dense(p)
        def call(s, x):
            u = s.U(x); v = s.V(x); h = s.W1b(s.W1a(x)); h = h * u + (1 - h) * v
            h = s.W2b(s.W2a(h)); h = h * u + (1 - h) * v; return s.out(h)
    class DON(tf.keras.Model):
        def __init__(s, h, p): super().__init__(); s.branch = Branch(h, p); s.trunk = Trunk(h, p); s.bias = s.add_weight(shape=(1,), initializer="zeros", trainable=True)
        def call(s, inp, training=False):
            b = s.branch(inp[0], training=training); t = s.trunk(inp[1])
            return tf.expand_dims(tf.einsum("bp,bnp->bn", b, t) + s.bias, -1)
    return DON(hidden, p)

def _build_unet(tf, grid, in_ch, base_filters, depth, dropout):
    def cb(x, f):
        x = tf.keras.layers.Conv3D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv3D(f, 3, padding="same", activation="relu")(x)
        return tf.keras.layers.BatchNormalization()(x)
    inp = tf.keras.Input(shape=(grid, grid, grid, in_ch)); sk = []; x = inp
    for i in range(depth):
        s = cb(x, base_filters * (2 ** i)); sk.append(s)
        x = tf.keras.layers.MaxPooling3D(2, padding="same")(s)
    x = cb(x, base_filters * (2 ** depth))
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    for i in reversed(range(depth)):
        x = tf.keras.layers.Conv3DTranspose(base_filters * (2 ** i), 2, strides=2, padding="same", activation="relu")(x)
        s = sk[i]
        xs = [int(x.shape[d]) for d in (1, 2, 3)]; ss = [int(s.shape[d]) for d in (1, 2, 3)]
        if xs != ss:
            crop = [[max(0, (xs[d] - ss[d]) // 2), max(0, xs[d] - ss[d] - (xs[d] - ss[d]) // 2)] for d in range(3)]
            pad = [[max(0, (ss[d] - xs[d]) // 2), max(0, ss[d] - xs[d] - (ss[d] - xs[d]) // 2)] for d in range(3)]
            if any(c[0] + c[1] > 0 for c in crop): x = tf.keras.layers.Cropping3D(tuple(map(tuple, crop)))(x)
            if any(pp[0] + pp[1] > 0 for pp in pad): x = tf.keras.layers.ZeroPadding3D(tuple(map(tuple, pad)))(x)
        x = tf.keras.layers.Concatenate()([x, s]); x = cb(x, base_filters * (2 ** i))
    return tf.keras.Model(inp, tf.keras.layers.Conv3D(1, 1, padding="same")(x))

def _load_weights_robust(model, w_path):
    import h5py
    try:
        model.load_weights(str(w_path)); return model
    except Exception:
        pass
    h5map, all_h5 = {}, []
    with h5py.File(str(w_path), "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                a = np.array(obj)
                h5map.setdefault("/".join(name.replace("\\", "/").split("/")[-2:]), []).append(a)
                all_h5.append(a)
        f.visititems(visit)
    used = [False] * len(all_h5); assigned = []
    for w in model.weights:
        wn = (w.path if hasattr(w, "path") else w.name).replace(":0", "")
        key = "/".join(wn.split("/")[-2:]); chosen = None
        for a in h5map.get(key, []):
            if a.shape == tuple(w.shape): chosen = a; break
        if chosen is None:
            for i, a in enumerate(all_h5):
                if not used[i] and a.shape == tuple(w.shape): chosen = a; used[i] = True; break
        if chosen is None: raise ValueError(f"no match {wn}")
        assigned.append(chosen)
    model.set_weights(assigned); return model


def _predict_field(tf, model_name, cyt, frame, seed=42):
    idx = CYT_INDEX[cyt]; G = GRID; dp = _dp(); cmax = _clip_max(cyt)
    Y = np.load(dp / "Y_target.npy").astype(np.float32)[..., idx:idx + 1]
    wdir = MODELS_ROOT / MODEL_DIR[model_name]
    suffix = f"{cyt}_{RUN}_{GRID}_{seed}"
    res = json.load(open(wdir / f"res_{suffix}.json")); bp = res["best_params"]
    wpath = wdir / f"weights_{suffix}.weights.h5"

    if model_name == "DeepONet":
        Xb = np.load(dp / "X_branch.npy").astype(np.float32)
        Xt = np.load(dp / "X_trunk.npy").astype(np.float32)
        N = Xb.shape[0]
        f0 = Xb[:, 0, :, :, :, idx]
        mask = (Xb[:, 0, :, :, :, 6:].max(-1) > 0.5).astype(np.float32)
        xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
        xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
        br = np.zeros((N, 8), np.float32)
        for i in range(N):
            f = f0[i]; m = mask[i]; na = float(m.sum()) + 1e-6
            br[i] = [(float(f.max()) + 1) / 2, (float(f.mean()) + 1) / 2, float(f.std()),
                     float((xx * m).sum() / na), float((yy * m).sum() / na), float((zz * m).sum() / na),
                     na / G ** 3, float(Xt[i, 0, 3])]
        vals = Xb.transpose(0, 2, 3, 4, 1, 5).reshape(N, G ** 3, 22).astype(np.float32)
        tr = np.concatenate([Xt[:, :, :3].astype(np.float32), vals], -1)
        model = _build_deeponet(tf, bp["hidden"], bp["p"])
        ch = min(int(bp.get("chunk_size", 4096)), tr.shape[1])
        _ = model([tf.constant(br[:1]), tf.constant(tr[:1, :ch])], training=False)
        _load_weights_robust(model, wpath)
        out = np.zeros((G ** 3, 1), np.float32); xb = tf.constant(br[frame:frame + 1])
        for s in range(0, G ** 3, ch):
            e = min(s + ch, G ** 3)
            out[s:e] = model([xb, tf.constant(tr[frame:frame + 1, s:e])], training=False).numpy()[0]
        pred = out.reshape(G, G, G)
    else:
        X = np.load(dp / "X_unet.npy").astype(np.float32)
        model = _build_unet(tf, G, X.shape[-1], bp["base_filters"], bp["depth"], bp.get("dropout", 0))
        _load_weights_robust(model, wpath)
        pred = model.predict(X[frame:frame + 1], verbose=0)[0, ..., 0]

    gt = _denorm(Y[frame, :, :, :, 0], cmax)
    pred = np.maximum(_denorm(pred, cmax), 0.0)
    return gt, pred


def figE(frame=88):
    """GT / Pred / |diff| slices (DeepONet) + cross-model |diff| slices."""
    tf = _load_field_deps()
    data = {}
    for model in MODELS:
        for cyt, cl in CYTS:
            gt, pred = _predict_field(tf, model, cyt, frame)
            data[(model, cyt)] = (gt, pred, np.abs(pred - gt))

    # F_recon_slices: rows = cytokine x {GT,Pred,err}, cols = xy/xz/yz (DeepONet)
    fig = plt.figure(figsize=(7.0, 8.4))
    gs = fig.add_gridspec(len(CYTS) * 3, 3, hspace=0.28, wspace=0.18)
    for ci, (cyt, cl) in enumerate(CYTS):
        gt, pred, err = data[("DeepONet", cyt)]
        pos = gt[gt > 0]; vmax = np.percentile(pos, 99.5) if pos.size else 1.0
        b = ci * 3; top = (ci == 0)
        _ortho_slices(fig, [gs[b, j] for j in range(3)], gt, "magma", f"{cl} GT", vmax, show_titles=top)
        _ortho_slices(fig, [gs[b + 1, j] for j in range(3)], pred, "magma", f"{cl} Pred", vmax, show_titles=False)
        _ortho_slices(fig, [gs[b + 2, j] for j in range(3)], err, "inferno", f"{cl} abs. error", show_titles=False)
    savef(fig, "F3_recon_slices")

    # F_diff_models: cross-model |diff|, rows=models, cols=cytokine x plane
    planes = [("xy", lambda v: v[:, :, v.shape[2] // 2]),
              ("xz", lambda v: v[:, v.shape[1] // 2, :]),
              ("yz", lambda v: v[v.shape[0] // 2, :, :])]
    np_ = len(planes)
    width_ratios = []
    for _ in CYTS:
        width_ratios += [1] * np_ + [0.45]
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(len(MODELS), len(width_ratios), width_ratios=width_ratios,
                          hspace=0.06, wspace=0.20, left=0.05, right=0.97, top=0.84, bottom=0.04)
    emax = {cyt: (max(np.percentile(data[(m, cyt)][2], 99) for m in MODELS) or 1e-15)
            for cyt, _ in CYTS}
    block = np_ + 1
    yz_axes, first_axes = {}, {}
    for mj, m in enumerate(MODELS):
        for ci, (cyt, cl) in enumerate(CYTS):
            for pj, (pname, slf) in enumerate(planes):
                col = ci * block + pj
                ax = fig.add_subplot(gs[mj, col])
                ax.imshow(slf(data[(m, cyt)][2]).T, origin="lower", cmap="inferno",
                          vmin=0, vmax=emax[cyt], aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if mj == 0:
                    ax.set_title(f"{cl}\n{pname}", fontsize=7)
                if col == 0:
                    ax.set_ylabel(m, fontsize=8)
                if pj == np_ - 1:
                    yz_axes[cyt] = ax
                if pj == 0:
                    first_axes.setdefault((ci, cyt), ax)
    import matplotlib.cm as _cm
    fig.canvas.draw()
    cyts = [c for c, _ in CYTS]
    for ci, (cyt, cl) in enumerate(CYTS):
        yz_pos = yz_axes[cyt].get_position(fig)
        right_limit = first_axes[(ci + 1, cyts[ci + 1])].get_position(fig).x0 if ci + 1 < len(cyts) else 0.99
        cb_x = yz_pos.x1 + 0.012; cb_w = 0.010
        if cb_x + cb_w > right_limit - 0.03:
            cb_x = max(yz_pos.x1 + 0.004, right_limit - 0.03 - cb_w)
        cax = fig.add_axes([cb_x, 0.04, cb_w, 0.80])
        sm = _cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=0, vmax=emax[cyt]))
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=5, pad=2)
        cb.set_label(f"{cl} abs. error", fontsize=6, labelpad=3)
    savef(fig, "F3_diff_models")


#  DISPATCH
CHART_FIGS = {"B1": figB1, "B4": figB4, "B3": figB3, "E4": figE4,
              "K1": figK1, "K2": figK2}
FIELD_FIGS = {"A": figA, "S": figS, "E": figE}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figs", nargs="*", default=None,
                    help="Which: B1 B4 B3 E4 K1 K2 (chart), A S E (field). Default: all.")
    ap.add_argument("--models-root", default="./models")
    ap.add_argument("--prep-root", default="./preprocessed_3d")
    ap.add_argument("--sweep-root", default="./smores",
                    help="Dir holding calibration_results_*.json (and sweep/).")
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--out", default="./figures_3d")
    args = ap.parse_args()

    global MODELS_ROOT, PREP_ROOT, SWEEP_ROOT, FIGDIR, GRID
    MODELS_ROOT = Path(args.models_root)
    PREP_ROOT = Path(args.prep_root)
    SWEEP_ROOT = Path(args.sweep_root)
    FIGDIR = Path(args.out)
    GRID = args.grid

    setup()
    figs = args.figs if args.figs else list(CHART_FIGS.keys()) + list(FIELD_FIGS.keys())
    for f in figs:
        target = CHART_FIGS.get(f) or FIELD_FIGS.get(f)
        if target is None:
            print(f"  unknown fig: {f}"); continue
        tag = "field" if f in FIELD_FIGS else "chart"
        print(f"  {f} ({tag})")
        try:
            target()
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
    print(f"Done -> {FIGDIR}/")

if __name__ == "__main__":
    main()