import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import figures as F
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt = F.plt
if hasattr(F, "setup"):
    F.setup()
MK = {1: "o", 42: "s", 100: "^"}


def seed_vals(m, cyt, which, get):
    out = []
    for s in F.SEEDS:
        r = F.load_res(m, cyt, s)
        if r is None:
            continue
        v = get(F.horizon(r, which))
        if v is not None and v == v:
            out.append((s, float(v)))
    return out


def draw(ax, xc, pairs, color, hatch, alpha, w, fs):
    vs = [v for _, v in pairs]
    mu = float(np.mean(vs))
    ax.bar(xc, mu, w, color=color, edgecolor="black", lw=0.4, hatch=hatch, alpha=alpha)
    for (s, v), o in zip(pairs, np.linspace(-w * 0.25, w * 0.25, len(pairs))):
        ax.scatter(xc + o, v, marker=MK.get(s, "o"), s=14, facecolor="white",
                   edgecolor="black", lw=0.8, zorder=3)
    top = max(vs + [mu])
    ax.text(xc, max(top, 0) + 0.03, f"{mu:.3f}", ha="center", va="bottom", fontsize=fs)
    return min(vs + [mu]), top


def seedh():
    return [Line2D([], [], marker=MK[s], ls="", mfc="white", mec="black", label=f"seed {s}")
            for s in F.SEEDS]


def ylim(axes, lo, hi):
    ylo = min(lo) - 0.1 if min(lo) < 0 else 0
    for ax in axes:
        ax.set_ylim(ylo, max(1.08, max(hi) + 0.1))


def finish(fig, handles, name):
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.12, "Bars: mean over seeds; markers: individual seeds.",
             ha="center", fontsize=7, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    F.savef(fig, name)


def B1():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    lo, hi = [], []
    for ax, (cyt, cl) in zip(axes, F.CYTS):
        x = np.arange(len(F.MODELS)); w = 0.35
        for j, which in enumerate(["near", "far"]):
            for i, m in enumerate(F.MODELS):
                p = seed_vals(m, cyt, which, lambda h: h.get("Global_R2"))
                if not p:
                    continue
                a, b = draw(ax, x[i] + (j - 0.5) * w, p, F.COL[m],
                            "" if which == "near" else "//",
                            0.9 if which == "near" else 0.65, w, 5.5)
                lo.append(a); hi.append(b)
        ax.set_xticks(x); ax.set_xticklabels(F.MODELS, fontsize=9); ax.set_title(cl, fontsize=10)
        ax.axhline(0, color="black", lw=0.5); ax.axhline(1, color="gray", ls=":", alpha=0.4)
        ax.set_xlim(-0.6, len(F.MODELS) - 0.4)
    ylim(axes, lo, hi); axes[0].set_ylabel(r"Global $R^2$")
    finish(fig, [Patch(fc="lightgray", ec="black", label="Near (t82-91)"),
                 Patch(fc="lightgray", ec="black", hatch="//", label="Far (t92-100)")] + seedh(),
           "F2_accuracy")


def B4():
    mets = [("SSIM", "SSIM", ""), ("Avg_Dice", "Dice", "//"), ("Spatial_Correlation", "Corr", "..")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    lo, hi = [], []
    for ax, (cyt, cl) in zip(axes, F.CYTS):
        x = np.arange(len(F.MODELS)); w = 0.25
        for j, (key, lab, hat) in enumerate(mets):
            for i, m in enumerate(F.MODELS):
                p = seed_vals(m, cyt, "near", lambda h, k=key: h.get(k))
                if not p:
                    continue
                a, b = draw(ax, x[i] + (j - 1) * w, p, F.COL[m], hat, 0.6 + j * 0.12, w, 5)
                lo.append(a); hi.append(b)
        ax.set_xticks(x); ax.set_xticklabels(F.MODELS, fontsize=9); ax.set_title(cl, fontsize=10)
        ax.axhline(0, color="black", lw=0.5); ax.set_xlim(-0.6, len(F.MODELS) - 0.4)
    ylim(axes, lo, hi); axes[0].set_ylabel("Score (near)")
    finish(fig, [Patch(fc="lightgray", ec="black", hatch=h, label=l) for _, l, h in mets] + seedh(),
           "F2_metrics")


def E4():
    planes = [("xy_midplane_z", "xy"), ("xz_midplane_y", "xz"), ("yz_midplane_x", "yz")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    lo, hi = [], []
    for ax, (cyt, cl) in zip(axes, F.CYTS):
        x = np.arange(len(planes)); w = 0.35
        for j, m in enumerate(F.MODELS):
            for i, (pk, _) in enumerate(planes):
                p = seed_vals(m, cyt, "near",
                              lambda h, k=pk: h.get("Slice_2D", {}).get(k, {}).get("R2"))
                if not p:
                    continue
                a, b = draw(ax, x[i] + (j - 0.5) * w, p, F.COL[m], "", 0.9, w, 5.5)
                lo.append(a); hi.append(b)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in planes]); ax.set_title(cl, fontsize=10)
        ax.axhline(0, color="black", lw=0.5); ax.set_xlim(-0.6, len(planes) - 0.4)
    ylim(axes, lo, hi); axes[0].set_ylabel(r"Slice $R^2$ (near)")
    finish(fig, [Patch(fc=F.COL[m], ec="black", label=m) for m in F.MODELS] + seedh(),
           "F3_midplane_r2")


if __name__ == "__main__":
    B1(); B4(); E4()
    print("seed-dot figures written")
