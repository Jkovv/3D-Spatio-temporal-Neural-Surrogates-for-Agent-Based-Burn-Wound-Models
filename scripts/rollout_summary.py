#!/usr/bin/env python3
"""Table and figure of the autoregressive rollout (rollout_3d.py): R2 in the early and late
test windows per model and seed, and R2 per step ahead with persistence as reference.

usage:  python scripts/rollout_summary.py --rollout models/rollout_3d --figdir figures_3d
"""
import json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"deeponet": "#ffd700", "unet": "#fa8775", "persistence": "#222222"}
LBL = {"deeponet": "DeepONet", "unet": "U-Net", "persistence": "Persistence"}
MK = {1: "^", 42: "o", 100: "s"}
CYTS = [("il8", "IL-8"), ("il10", "IL-10")]
Y_FLOOR = -0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default="models/rollout_3d"); ap.add_argument("--figdir", default="figures_3d")
    ap.add_argument("--run", default="run_0062"); ap.add_argument("--grid", type=int, default=50)
    a = ap.parse_args()
    rd = Path(a.rollout)
    res = {}                                                   # (kind, seed) -> {cyt: results}
    for cyt, _ in CYTS:
        p = rd / f"rollout_persistence_{cyt}_{a.run}_{a.grid}.json"
        if p.exists(): res.setdefault(("persistence", None), {})[cyt] = json.load(open(p))["results"]
        for f in sorted(rd.glob(f"rollout_*_{cyt}_{a.run}_{a.grid}_*.json")):
            d = json.load(open(f)); res.setdefault((d["model"], d["seed"]), {})[cyt] = d["results"]
    start = json.load(open(rd / f"rollout_persistence_il8_{a.run}_{a.grid}.json"))["start_window"]
    order = sorted(res, key=lambda k: ({"persistence": 0, "deeponet": 1, "unet": 2}[k[0]], k[1] or 0))

    def g(k, cyt, win):
        r = res[k].get(cyt); return r[win]["Global_R2"] if r else np.nan
    def f(v): return "--" if not np.isfinite(v) else (f"${v:.3f}$" if v > -10 else "$<-10$")
    rows = []
    for k in order:
        name = LBL[k[0]] + ("" if k[1] is None else f", seed {k[1]}")
        rows.append(f"{name} & " + " & ".join(f(g(k, c, w)) for c, _ in CYTS
                    for w in ("Near_Horizon_t82_t91", "Far_Horizon_t92_t100")) + " \\\\")
    (rd / "tab_rollout.tex").write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Autoregressive rollout from window {start} ($t={start + 2}$\\,h): the predicted cytokine channel is fed back into the inputs of the following windows, the other channels and the cell masks still come from the ABM. Global $R^2$ in the early ($t=82$-91\\,h, 1-10 steps ahead) and late ($t=92$-100\\,h, 11-19 steps) windows; persistence holds the last true frame. Teacher-forced one-step values are in Table~\\ref{{tab:results_3d}}.}}
\\label{{tab:rollout}}
\\centering\\footnotesize
\\begin{{tabular}}{{lcccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textbf{{IL-8}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{IL-10}}}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
\\textbf{{Predictor}} & \\textbf{{early}} & \\textbf{{late}} & \\textbf{{early}} & \\textbf{{late}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    print(f"wrote {rd / 'tab_rollout.tex'}")

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (cyt, lbl) in zip(axes, CYTS):
        for k in order:
            r = res[k].get(cyt)
            if not r: continue
            y = np.array(r["r2_per_step"], float); x = np.arange(1, len(y) + 1)
            kind, seed = k
            below = y < Y_FLOOR; yc = np.where(below, Y_FLOOR, y)
            style = dict(color=COL[kind], lw=1.6 if kind != "persistence" else 1.4, ls="--" if kind == "persistence" else "-",
                         marker=None if kind == "persistence" else MK.get(seed, "o"), ms=3.5, mec="white", mew=0.5,
                         label=LBL[kind] + ("" if seed is None else f" seed {seed}"), zorder=3 if kind == "persistence" else 2)
            ax.plot(x, yc, **style)
            if below.any():
                ax.plot(x[below], yc[below], ls="none", marker="v", color=COL[kind], ms=4, zorder=4)
        ax.axhline(0, color="#999999", lw=0.6, zorder=1)
        ax.set_xlabel("steps ahead (h)"); ax.set_title(lbl, fontsize=9, loc="left")
        ax.set_xticks([1, 5, 10, 15, 19]); ax.set_ylim(Y_FLOOR - 0.05, 1.02)
        ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    axes[0].set_ylabel("global $R^2$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, ncol=4, bbox_to_anchor=(0.5, 0.0))
    axes[0].text(1, Y_FLOOR + 0.04, f"triangles: $R^2 < {Y_FLOOR}$", fontsize=6, color="#555555", va="bottom")
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    Path(a.figdir).mkdir(parents=True, exist_ok=True)
    out = Path(a.figdir) / "F6_rollout.png"; fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); print(f"wrote {out} and .pdf")


if __name__ == "__main__":
    main()
