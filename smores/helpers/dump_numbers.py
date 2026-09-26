import json, glob, os, sys
import numpy as np
B = sys.argv[1]
R = f"{B}/smores/results"
out = []
def p(s=""): out.append(str(s))

p("=== SURROGATES: per model x cytokine x horizon, mean +/- SD over seeds 1/42/100 ===")
for model in ["deeponet_3d", "unet_3d"]:
    for cyt in ["il8", "il10"]:
        runs = [json.load(open(f"{B}/models/{model}/res_{cyt}_run_0062_50_{s}.json")) for s in (1, 42, 100)]
        for hor in runs[0]["results"]:
            keys = [k for k, v in runs[0]["results"][hor].items() if isinstance(v, (int, float))]
            for k in keys:
                vals = [r["results"][hor][k] for r in runs]
                p(f"{model} {cyt} {hor} {k}: mean={np.mean(vals):.4g} sd={np.std(vals):.4g} per-seed={[round(v,4) for v in vals]}")
            nested = {k: v for k, v in runs[0]["results"][hor].items() if isinstance(v, dict)}
            for k in nested:
                for kk in nested[k]:
                    try:
                        vals = [r["results"][hor][k][kk] for r in runs]
                        if all(isinstance(v, (int, float)) for v in vals):
                            p(f"{model} {cyt} {hor} {k}.{kk}: mean={np.mean(vals):.4g} sd={np.std(vals):.4g}")
                    except Exception:
                        pass
        tr = [r.get("train_time_seconds") for r in runs]; pr = [r.get("pred_time_seconds") for r in runs]
        p(f"{model} {cyt} train_s={tr} pred_s={pr}")
    p()

p("=== SOBOL + RECOVERY (topk10) ===")
d = json.load(open(f"{R}/calibration_results_topk10.json"))
for row in d["sobol"]["ranking"]:
    p(f"ST {row['param']}: {row['ST_mean']:.4f}")
for k in d["recovery"]["r2_per_param"]:
    p(f"recovery {k}: R2={d['recovery']['r2_per_param'][k]:.4f} nRMSE={d['recovery']['nrmse_per_param'][k]:.4f}")
p(f"nrmse_mean: {d['recovery']['nrmse_mean']:.4f}")
for f in ["calibration_results_topk5_sobol.json", "calibration_results_topk5_identifiable.json"]:
    dd = json.load(open(f"{R}/{f}"))
    p(f"{f}: params={dd['recovery']['selected_params']} nrmse_mean={dd['recovery']['nrmse_mean']:.4f}")
p()

p("=== LOOP ===")
for s, f in {"1": "calibration_surrogate_il8_seed1.json", "42": "calibration_surrogate_il8.json", "100": "calibration_surrogate_il8_seed100.json"}.items():
    d = json.load(open(f"{R}/{f}"))
    g = d["generalisation"]["il8"]
    sr = d["surrogate"].get("recovery", d["surrogate"])["r2_per_param"]
    ar = d["abm_baseline"].get("recovery", d["abm_baseline"])["r2_per_param"]
    p(f"seed {s}: field={g['gen_r2_mean']:.4f}+/-{g['gen_r2_std']:.4f} clip={g['clip_frac_mean']:.4f} | surrogate {sr} | abm {ar}")
p()

p("=== RIDGE ===")
p(json.dumps(json.load(open(f"{R}/ridge_init_ec_keil8.json")), indent=1))
p()
for f in [f"{B}/smores/helpers/identifiability_report.txt", f"{B}/smores/helpers/speedup_report.txt",
          f"{B}/smores/helpers/sobol_convergence_report.txt", f"{R}/emulator_cv.csv"]:
    p(f"=== {os.path.basename(f)} ===")
    p(open(f).read() if os.path.exists(f) else "MISSING")

open(f"{B}/paper_numbers.txt", "w").write("\n".join(out))
print(f"wrote {B}/paper_numbers.txt ({len(out)} lines)")
