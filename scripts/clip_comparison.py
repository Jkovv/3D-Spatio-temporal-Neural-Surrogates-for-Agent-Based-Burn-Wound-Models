#!/usr/bin/env python3
"""Clipped vs unclipped normalisation on a second realisation of the benchmark point (run_0062).

Every model is trained on its own normalisation (clipcmp_train.slurm) and evaluated against the
same unclipped ground truth: one step ahead (teacher forcing) and in an autoregressive rollout
from window --start. Persistence and linear extrapolation are scored against the same truth.
Metrics: calculate_metrics() of the training scripts; the Dice threshold and SSIM range use the
clip value of the clipped preprocessing for every row, so that only the predictions differ.

usage:  python scripts/clip_comparison.py --work /path/to/clipcmp --models models/clip_comparison
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

R = DON = None                                                       # imported in main(), so that --table-only needs no TensorFlow

WINDOWS = {"Near_Horizon_t82_t91": (80, 90), "Far_Horizon_t92_t100": (90, 99)}
LBL = {("deeponet", "clip"): "DeepONet, clipped", ("deeponet", "noclip"): "DeepONet, unclipped",
       ("unet", "clip"): "U-Net, clipped", ("unet", "noclip"): "U-Net, unclipped"}


def load(prep, run, grid):
    d = Path(prep) / run / f"{grid}x{grid}x{grid}"
    return {"Xb": np.load(d / "X_branch.npy").astype(np.float32), "Xt": np.load(d / "X_trunk.npy").astype(np.float32),
            "Y": np.load(d / "Y_target.npy").astype(np.float32), "M": np.load(d / "Y_masks_spatial.npy").astype(np.float32),
            "meta": json.load(open(d / "metadata.json"))}


def one_step(kind, model, D, idx):
    G = D["Xb"].shape[2]
    if kind == "deeponet":
        p = DON.predict_full(model, DON.build_branch_inputs(D["Xb"], D["Xt"], idx), DON.build_trunk_inputs(D["Xb"], D["Xt"]))
        return p[:, :, 0].reshape(-1, G, G, G)
    X = D["Xb"].transpose(0, 2, 3, 4, 1, 5).reshape(-1, G, G, G, 22)
    return model.predict(X, batch_size=1, verbose=0)[..., 0]


def scores(truth, pred, M, cref, start):
    """Window metrics for a prediction covering windows start..N-1."""
    return {k: DON.calculate_metrics(truth[lo:hi], pred[lo - start:hi - start], M[lo:hi], cref) for k, (lo, hi) in WINDOWS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--models", default="models/clip_comparison")
    ap.add_argument("--run", default="run_0062"); ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--start", type=int, default=80); ap.add_argument("--cytokines", nargs="+", default=["il8"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 100])
    ap.add_argument("--table-only", action="store_true")                 # rebuild the table from clip_comparison.json
    a = ap.parse_args()
    mdir = Path(a.models)
    if a.table_only:
        write_table(json.load(open(mdir / "clip_comparison.json")), mdir / "tab_clip_comparison.tex", mdir); return
    global R, DON
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rollout_3d as R
    DON = R.DON
    data = {s: load(Path(a.work) / f"prep_{s}", a.run, a.grid) for s in ("clip", "noclip")}
    s0 = a.start
    out = {"run": a.run, "start_window": s0, "cytokines": {}}

    for cyt in a.cytokines:
        idx = R.CYTS.index(cyt)
        cmax = {s: float(data[s]["meta"]["scaling"]["max"][idx]) for s in data}
        truth = DON.denormalize(data["noclip"]["Y"][..., idx:idx + 1], cmax["noclip"])          # unclipped truth
        truth_c = DON.denormalize(data["clip"]["Y"][..., idx:idx + 1], cmax["clip"])            # clipped truth (as in Table 1)
        M = data["noclip"]["M"]; cref = cmax["clip"]
        test = slice(80, 99)
        res = {"clip_value_clipped": cmax["clip"], "scale_unclipped": cmax["noclip"],
               "clip_percentile": data["clip"]["meta"]["scaling"]["clip_percentile"][idx],
               "test_voxels_above_clip": float(np.mean(truth[test] > cref)),
               "test_active_voxels_above_clip": float(np.mean(truth[test][truth[test] > 0] > cref)),
               "unclipped_scale_saturated_voxels": int(np.sum(data["noclip"]["Y"][..., idx] >= 1.0 - 1e-6)),
               "baselines": {}, "models": {}}
        print(f"[{cyt}] clip value {cref:.3e} (q={res['clip_percentile']}); test voxels above it: {100*res['test_voxels_above_clip']:.3f}% "
              f"of all, {100*res['test_active_voxels_above_clip']:.2f}% of active; saturated voxels on the unclipped scale: "
              f"{res['unclipped_scale_saturated_voxels']}", flush=True)

        for name, T in (("unclipped_truth", truth), ("clipped_truth", truth_c)):
            pers = np.empty_like(T); lin = np.empty_like(T)
            pers[1:] = T[:-1]; pers[0] = T[0]
            lin[2:] = np.maximum(2 * T[1:-1] - T[:-2], 0.0); lin[:2] = pers[:2]
            res["baselines"][name] = {"persistence": scores(T, pers, M, cref, 0), "linear_extrapolation": scores(T, lin, M, cref, 0)}
        held = np.repeat(truth[s0 - 1:s0], truth.shape[0] - s0, axis=0)                         # last true frame before the rollout
        res["baselines"]["unclipped_truth"]["persistence_rollout"] = scores(truth, held, M, cref, s0)
        res["baselines"]["unclipped_truth"]["persistence_rollout"]["r2_per_step"] = R.r2_per_step(truth[s0:], held)
        b = res["baselines"]["unclipped_truth"]
        print(f"  persistence one-step R2 {b['persistence']['Near_Horizon_t82_t91']['Global_R2']:.4f} / "
              f"{b['persistence']['Far_Horizon_t92_t100']['Global_R2']:.4f}; rollout "
              f"{b['persistence_rollout']['Near_Horizon_t82_t91']['Global_R2']:.4f} / {b['persistence_rollout']['Far_Horizon_t92_t100']['Global_R2']:.4f}", flush=True)

        for (kind, scale), lbl in LBL.items():
            wdir = mdir / f"{kind}_{scale}"
            D = data[scale]
            for seed in a.seeds:
                if not (wdir / f"weights_{cyt}_{a.run}_{a.grid}_{seed}.weights.h5").exists():
                    print(f"  [skip] {lbl} seed {seed}: no weights"); continue
                model = R.load_model(kind, wdir, cyt, a.run, a.grid, seed)
                p1 = DON.denormalize(one_step(kind, model, D, idx)[..., None], cmax[scale])
                pr = DON.denormalize(R.rollout(kind, model, D["Xb"], D["Xt"], idx, s0)[..., None], cmax[scale])
                r = {"one_step_vs_unclipped": scores(truth, p1, M, cref, 0),
                     "one_step_vs_own_target": scores(DON.denormalize(D["Y"][..., idx:idx + 1], cmax[scale]), p1, M, cref, 0),
                     "rollout_vs_unclipped": scores(truth, pr, M, cref, s0)}
                r["rollout_vs_unclipped"]["r2_per_step"] = R.r2_per_step(truth[s0:], pr)
                res["models"].setdefault(f"{kind}_{scale}", {})[str(seed)] = r
                e, l = "Near_Horizon_t82_t91", "Far_Horizon_t92_t100"
                print(f"  {lbl:20s} seed {seed:<3d} one-step R2 {r['one_step_vs_unclipped'][e]['Global_R2']:.4f} / "
                      f"{r['one_step_vs_unclipped'][l]['Global_R2']:.4f} (vs own target {r['one_step_vs_own_target'][e]['Global_R2']:.4f}); "
                      f"rollout {r['rollout_vs_unclipped'][e]['Global_R2']:.3f} / {r['rollout_vs_unclipped'][l]['Global_R2']:.3f}", flush=True)
        out["cytokines"][cyt] = res

    mdir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(mdir / "clip_comparison.json", "w"), indent=1, default=float)
    print(f"saved {mdir / 'clip_comparison.json'}")
    write_table(out, mdir / "tab_clip_comparison.tex", mdir)


def write_table(out, path, mdir=None):
    e, l = "Near_Horizon_t82_t91", "Far_Horizon_t92_t100"
    def cell(vals):
        v = np.array([x for x in vals if np.isfinite(x)])
        if not v.size: return "--"
        f = lambda x: f"{x:.3f}" if x > -10 else "<-10"
        return f"${f(v.mean())}$" if v.size == 1 else f"${f(v.mean())}$ ({f(v.min())}-{f(v.max())})"
    ceil = {}
    if mdir is not None and (Path(mdir) / "clip_ceiling.json").exists():
        ceil = json.load(open(Path(mdir) / "clip_ceiling.json"))
    blocks = []
    for cyt, res in out["cytokines"].items():
        b = res["baselines"]["unclipped_truth"]
        rows = []
        if cyt in ceil:
            rows.append(f"Clipped field itself (upper bound) & ${ceil[cyt][e]['r2_clipped_field_vs_full']:.3f}$ & "
                        f"${ceil[cyt][l]['r2_clipped_field_vs_full']:.3f}$ & -- & -- \\\\")
        rows += [f"Persistence & ${b['persistence'][e]['Global_R2']:.3f}$ & ${b['persistence'][l]['Global_R2']:.3f}$ & "
                f"${b['persistence_rollout'][e]['Global_R2']:.3f}$ & ${b['persistence_rollout'][l]['Global_R2']:.3f}$ \\\\",
                f"Linear extrapolation & ${b['linear_extrapolation'][e]['Global_R2']:.3f}$ & ${b['linear_extrapolation'][l]['Global_R2']:.3f}$ & -- & -- \\\\"]
        for key, lbl in (("deeponet_clip", "DeepONet, clipped"), ("deeponet_noclip", "DeepONet, unclipped"),
                         ("unet_clip", "U-Net, clipped"), ("unet_noclip", "U-Net, unclipped")):
            m = res["models"].get(key, {})
            g = lambda part, w: [s[part][w]["Global_R2"] for s in m.values()]
            rows.append(f"{lbl} & {cell(g('one_step_vs_unclipped', e))} & {cell(g('one_step_vs_unclipped', l))} & "
                        f"{cell(g('rollout_vs_unclipped', e))} & {cell(g('rollout_vs_unclipped', l))} \\\\")
        name = {"il8": "IL-8", "il10": "IL-10"}.get(cyt, cyt)
        blocks.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{name}}} (clip at q = {res['clip_percentile']}; "
                      f"{100*res['test_active_voxels_above_clip']:.1f}\\% of active test voxels above it)}} \\\\\n" + "\n".join(rows))
    body = "\n\\midrule\n".join(blocks)
    path.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Clipped vs unclipped normalisation on a second realisation of the benchmark point (run~0062, same parameters). Each network is trained and tuned (same Optuna search space) on its own normalisation, and every row is scored against the same unclipped ground truth: global $R^2$ one step ahead (teacher forcing) and in an autoregressive rollout from $t={out['start_window'] + 2}$\\,h, in the early ($t=82$-91\\,h) and late ($t=92$-100\\,h) windows. Networks: mean over seeds 1, 42 and 100, range in brackets. Clipped field itself: the clipped target scored against the full field, the highest $R^2$ any predictor of the clipped field can reach.}}
\\label{{tab:clip_comparison}}
\\centering\\footnotesize
\\begin{{tabular}}{{lcccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textbf{{One step}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Rollout}}}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
\\textbf{{Predictor}} & \\textbf{{early}} & \\textbf{{late}} & \\textbf{{early}} & \\textbf{{late}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
