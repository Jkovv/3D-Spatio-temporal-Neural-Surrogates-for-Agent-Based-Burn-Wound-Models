"""Builds the LaTeX tables of the SMoRe ParS part of the paper directly from the result files,
so that no number in the manuscript is typed by hand.  Also writes numbers.txt: every number
quoted in the text, with the file and key it comes from.

usage:  python smore_strict/make_tables.py --results results_v3 --sobol results/sobol_repmean.json --ecoli-npz results_v3/ecoli_arrays.npz
output: <results>/tables/*.tex and <results>/tables/numbers.txt
Missing result files are skipped with a message.
"""
import json, os, sys, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results_v3")
ap.add_argument("--sobol", default="results/sobol_repmean.json")
ap.add_argument("--ecoli-npz", default=None)   # optional: measurement window per strain, from the cached curves
a = ap.parse_args()
OUT = os.path.join(a.results, "tables"); os.makedirs(OUT, exist_ok=True)
NUM = []                                                   # (text, value, source)

def load(name):
    p = os.path.join(a.results, name)
    if not os.path.exists(p): print(f"  missing {p}: skipped"); return None
    return json.load(open(p))
def note(text, value, src): NUM.append((text, value, src))
def r2(x): return "--" if x is None or not np.isfinite(x) else f"${x:+.3f}$"
def f2(x): return "--" if x is None or not np.isfinite(x) else f"${x:.2f}$"
def pct(x): return "--" if x is None or not np.isfinite(x) else f"${100*x:.0f}\\%$"
def tex(p): return p.replace("_", "\\_")
def write(name, s): open(os.path.join(OUT, name), "w").write(s); print(f"  wrote {os.path.join(OUT, name)}")

# ---------------------------------------------------------------- Sobol (per cytokine)
ST_all, ST_max, ST_arg = {}, {}, {}
if os.path.exists(a.sobol):
    d = json.load(open(a.sobol))["summaries"]; names = d["param_names"]; pf = d["per_feature"]; feats = list(pf)
    cyts = []
    for f in feats:
        c = f.split("_")[0]
        if c not in cyts: cyts.append(c)
    lab = {"il8": "IL-8", "il1": "IL-1$\\beta$", "il6": "IL-6", "il10": "IL-10", "tnf": "TNF-$\\alpha$", "tgf": "TGF-$\\beta$"}
    per = {c: [float(np.mean([pf[f]["ST"][i] for f in feats if f.split("_")[0] == c])) for i in range(len(names))] for c in cyts}
    for i, p in enumerate(names):
        ST_all[p] = float(np.mean([max(pf[f]["ST"][i], 0) for f in feats]))
        c_best = max(cyts, key=lambda c: per[c][i]); ST_max[p] = per[c_best][i]; ST_arg[p] = lab.get(c_best, c_best)
    order = sorted(names, key=lambda p: -ST_all[p])
    hdr_sobol = " & ".join("\\textbf{" + lab.get(c, c) + "}" for c in cyts)
    rows = "\n".join(f"{tex(p)} & " + " & ".join(f"${max(per[c][names.index(p)], 0):.3f}$" for c in cyts) + f" & ${ST_all[p]:.3f}$ \\\\" for p in order)
    write("tab_sobol_per_cytokine.tex", f"""\\begin{{table}}[!htbp]
\\caption{{Sobol total-order index $S_T$ per cytokine (mean over the four observables of each cytokine) and over all 24 observables. Each secretion rate acts on its own cytokine; \\texttt{{sigmoidb}} acts on several, which is why it leads the 24-observable mean.}}
\\label{{tab:sobol_per_cytokine}}
\\centering\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l{'c'*len(cyts)}c}}
\\toprule
\\textbf{{Parameter}} & {hdr_sobol} & \\textbf{{All 24}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    for p in ("keil8", "init_ec", "km2tgf", "sigmoidb"):
        if p in names: note(f"S_T of {p}: all 24 observables / max over cytokines ({ST_arg[p]})", f"{ST_all[p]:.3f} / {ST_max[p]:.3f}", a.sobol)
else:
    print(f"  missing {a.sobol}: Sobol columns left empty")

# ---------------------------------------------------------------- main ABM inference
main = load("smore_main.json")
if main:
    J = main["jain"]; P = J["params"]
    order = sorted(P, key=lambda p: -ST_all.get(p, 0))
    rows = "\n".join(f"{tex(p)} & {('$%.3f$' % ST_all[p]) if p in ST_all else '--'} & "
                     f"{('$%.3f$' % ST_max[p]) if p in ST_max else '--'} & {r2(J['per_param'][p]['r2_median_accepted'])} & "
                     f"{r2(J['per_param'][p].get('r2_inverse_regression'))} & {f2(J['per_param'][p]['coverage95'])} & {f2(J['per_param'][p]['sd_ratio'])} \\\\"
                     for p in order)
    rg = J.get("ridge", {}); n_chain = sum(1 for i in J["sampler_info"] if (i.get("mcmc_steps") or 0) > 0)
    exc = {k[:-3]: v["library_unidentified"] for k, v in J["surfaces"].items() if k.endswith("_lo") and v["library_unidentified"] > 0}
    write("tab_recovery.tex", f"""\\begin{{table}}[!htbp]
\\caption{{SMoRe ParS inference, leave-one-out over the {J['n_folds']} sweep points, all ten parameters inferred jointly. $S_T$: Sobol total-order index, mean over the 24 observables and maximum over cytokines (Table~\\ref{{tab:sobol_per_cytokine}}). $R^2$: median of the admissible region against the true value. Inverse $R^2$: model-free reference, a Gaussian-process regression from the fitted SM parameters to each ABM parameter. Containment: share of folds in which the central 95\\% of the region contains the true value. Width: standard deviation of the region divided by that of a uniform distribution over the sweep range (1 = whole range). The region is non-empty in {J['n_nonempty']} folds; the true parameter vector is admissible in {100*J['truth_admissible_all_folds']:.0f}\\% of all folds. {n_chain} regions were too small for the quasi-random screen and were completed by hit-and-run chains, all with split $\\hat R \\le {J.get('rhat_target', 1.1)}$ (maximum ${J['rhat_max_over_folds']:.3f}$).}}
\\label{{tab:recovery}}
\\centering\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Parameter}} & $\\mathbf{{S_T}}$ & $\\mathbf{{\\max_c S_T}}$ & $\\mathbf{{R^2}}$ & \\textbf{{Inverse }}$\\mathbf{{R^2}}$ & \\textbf{{Containment}} & \\textbf{{Width}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    note("main: non-empty folds", f"{J['n_nonempty']}/{J['n_folds']}", "smore_main.json jain.n_nonempty")
    note("main: true vector admissible (all folds)", f"{J['truth_admissible_all_folds']:.3f}", "smore_main.json jain.truth_admissible_all_folds")
    note("main: region volume fraction, median", f"{J['volume_fraction']['median']:.2e}", "smore_main.json jain.volume_fraction.median")
    note("main: ridge width along init_ec+keil8 / init_ec-keil8", f"{rg.get('sd_ratio_sum', np.nan):.3f} / {rg.get('sd_ratio_difference', np.nan):.3f}", "smore_main.json jain.ridge")
    note("main: library points left out of a surface (unidentified SM parameter)", str(exc), "smore_main.json jain.surfaces[*].library_unidentified")
    note("main: surfaces dropped", str(J.get("surfaces_dropped", [])), "smore_main.json jain.surfaces_dropped")
    s = J["surfaces"]; r = np.array([v["loo_r2"] for v in s.values()], float); w = np.array([v["within_nsd"] for v in s.values()], float)
    note("main: bound surfaces LOO R2 median (min)", f"{np.nanmedian(r):.2f} ({np.nanmin(r):.2f})", "smore_main.json jain.surfaces[*].loo_r2")
    note("main: held-out bounds within the 2 SD band, median (min)", f"{np.nanmedian(w):.2f} ({np.nanmin(w):.2f})", "smore_main.json jain.surfaces[*].within_nsd")
    note("main: folds that needed the hit-and-run fallback", str(sum(1 for i in J["sampler_info"] if i.get("mcmc_steps", 0) > 0)), "smore_main.json jain.sampler_info")
    note("main: max R-hat over folds", f"{J['rhat_max_over_folds']}", "smore_main.json jain.rhat_max_over_folds")
    note("main: folds excluded from sample statistics (chain not converged)", str(J.get("n_nonconverged", "n/a")), "smore_main.json jain.n_nonconverged")
    note("main: Bergman pointwise acceptance, share of folds with no accepted sweep point", f"{main['frac_empty']:.2f}", "smore_main.json frac_empty")
    for p in order:
        q = J["per_param"][p]
        note(f"main {p}: R2 region / R2 inverse / containment / width", f"{q['r2_median_accepted']:+.3f} / {q.get('r2_inverse_regression', np.nan):+.3f} / {q['coverage95']:.2f} / {q['sd_ratio']:.2f}", "smore_main.json jain.per_param")

# ---------------------------------------------------------------- appendix: variants
var = [(n, load(f)) for n, f in (("Main", "smore_main.json"), ("Every 5th frame", "smore_main_thin5.json"), ("Quadrature", "smore_quadrature.json"),
                                 ("Top five", "smore_top5.json"), ("One SD", "smore_nsd1.json"))]
var = [(n, d["jain"]) for n, d in var if d]
if var:
    P = var[0][1]["params"]
    def cell(n, J, p):
        if p not in J["params"] or n == "One SD": return "--"          # one SD: only containment of the true vector is read
        q = J["per_param"][p]; return f"{r2(q['r2_median_accepted'])} (${q['coverage95']:.2f}$)"
    rows = "\n".join(f"{tex(p)} & " + " & ".join(cell(n, J, p) for n, J in var) + " \\\\" for p in P)
    hdr_var = " & ".join("\\textbf{" + n + "}" for n, _ in var)
    tail = ("Non-empty folds & " + " & ".join(f"${J['n_nonempty']}$" for _, J in var) + " \\\\\n"
            "True vector admissible & " + " & ".join(pct(J["truth_admissible_all_folds"]) for _, J in var) + " \\\\\n"
            "Region volume ($10^{-4}$ of the box) & " + " & ".join(f"${J['volume_fraction']['median']*1e4:.1f}$" for _, J in var) + " \\\\")
    write("tab_app_ident.tex", f"""\\begin{{table}}[h]
\\centering\\footnotesize
\\caption{{SMoRe ParS under five settings, leave-one-out over the sweep points: $R^2$ of the region median, with marginal containment in parentheses. Main: ten parameters, all 101 frames, bounds widened by two Gaussian-process standard deviations, replicate standard error as the SM uncertainty (IL-8: fit residual). Every 5th frame: as Main, with the SMs fitted to 21 of the 101 frames. Quadrature: SM misfit added in quadrature to the replicate standard error for every cytokine. Top five: only the five parameters with the highest Sobol index inferred. One SD: bounds widened by one standard deviation; the regions are then so small that the sampler cannot describe them reliably, and only non-emptiness and the admissibility of the true vector are reported.}}
\\label{{tab:app_ident}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l{'c'*len(var)}}}
\\toprule
\\textbf{{Parameter}} & {hdr_var} \\\\
\\midrule
{rows}
\\midrule
{tail}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    for n, J in var:
        note(f"variant {n}: non-empty / true vector admissible / volume fraction median / max R-hat / folds excluded (chain not converged)",
             f"{J['n_nonempty']}/{J['n_folds']} / {J['truth_admissible_all_folds']:.2f} / {J['volume_fraction']['median']:.2e} / {J['rhat_max_over_folds']} / {J.get('n_nonconverged', 'n/a')}", f"{n} jain")
        rg = J.get("ridge", {})
        if rg: note(f"variant {n}: ridge width sum / difference", f"{rg.get('sd_ratio_sum', np.nan):.3f} / {rg.get('sd_ratio_difference', np.nan):.3f}", f"{n} jain.ridge")
        if n != "One SD":
            for p in J["params"]:
                q = J["per_param"][p]
                note(f"variant {n} {p}: R2 region / containment / width", f"{q['r2_median_accepted']:+.3f} / {q['coverage95']:.2f} / {q['sd_ratio']:.2f}", f"{n} jain.per_param")

# ---------------------------------------------------------------- surrogate in the loop
loops = [(s, load(f"loop_seed{s}.json")) for s in (1, 42, 100)]
loops = [(s, d) for s, d in loops if d]
if loops:
    ARMS = [("abm", "ABM output"), ("sur", "Surrogate, no allowance"), ("sur_val", "Surrogate, validation tolerance"),
            ("sur_cv", "Surrogate, per-frame tolerance"), ("sur_conf", "Surrogate, error propagated in SM space"),
            ("abm_err_conf", "Control: ABM + transplanted error")]
    def trip(arm, f):
        vals = []
        for s, d in loops:
            J = d["arms"].get(arm); vals.append(f(J) if J else np.nan)
        return vals
    def fmt(vals, kind):
        if kind == "pct": return " / ".join("--" if not np.isfinite(v) else f"{100*v:.0f}" for v in vals)
        if kind == "vf": return " / ".join("--" if not np.isfinite(v) else f"{100*v:.0f}" for v in vals)
        return " / ".join("--" if not np.isfinite(v) else f"{v:+.2f}" for v in vals)
    rows = []
    for arm, lbl in ARMS:
        if not any(arm in d["arms"] for _, d in loops): continue
        ne = trip(arm, lambda J: J["n_nonempty"]/J["n_folds"]); ta = trip(arm, lambda J: J["truth_admissible_all_folds"])
        calibrated = np.nanmin(ta) >= 0.9
        vf = trip(arm, lambda J: J["volume_fraction"]["median"]) if calibrated else [np.nan]*len(loops)
        k8 = trip(arm, lambda J: J["per_param"]["keil8"]["r2_median_accepted"]) if calibrated else [np.nan]*len(loops)
        ie = trip(arm, lambda J: J["per_param"]["init_ec"]["r2_median_accepted"]) if calibrated else [np.nan]*len(loops)
        rows.append(f"{lbl} & {fmt(ne, 'pct')} & {fmt(ta, 'pct')} & {fmt(vf, 'vf')} & {fmt(k8, 'r2')} & {fmt(ie, 'r2')} \\\\")
        for s, d in loops:
            J = d["arms"].get(arm)
            if J: note(f"loop seed {s} {arm}: non-empty / true vector admissible / volume fraction median / R2 keil8 / R2 init_ec",
                       f"{J['n_nonempty']}/{J['n_folds']} / {J['truth_admissible_all_folds']:.2f} / {J['volume_fraction']['median']:.3f} / "
                       f"{J['per_param']['keil8']['r2_median_accepted']:+.3f} / {J['per_param']['init_ec']['r2_median_accepted']:+.3f}", f"loop_seed{s}.json arms.{arm}")
    err = " / ".join(f"{100*np.mean(d['surrogate_rel_rms']):.1f}" for _, d in loops)
    val = " / ".join(f"{100*d['tolerance']['val_rel_rms'][0]:.1f}" for _, d in loops)
    tr2 = " / ".join(f"{np.mean(d['surrogate_traj_r2']):.3f}" for _, d in loops)
    for s, d in loops:
        note(f"loop seed {s}: surrogate relative RMS error mean / validation-frame error on run_0062 / trajectory R2 mean",
             f"{np.mean(d['surrogate_rel_rms']):.4f} / {d['tolerance']['val_rel_rms'][0]:.4f} / {np.mean(d['surrogate_traj_r2']):.3f}", f"loop_seed{s}.json")
    write("tab_loop.tex", f"""\\begin{{table}}[!htbp]
\\caption{{Surrogate in the loop, IL-8 only, leave-one-out over the {loops[0][1]['n_points']} sweep points other than the surrogate's training run; values for the training seeds {' / '.join(str(s) for s, _ in loops)}. The library of bound surfaces always comes from the ABM; only the data interval of the held-out point changes. Relative RMS error of the surrogate's volume-averaged trajectory over the sweep: {err}\\%; on the validation frames of its own training run: {val}\\%; trajectory $R^2$: {tr2}. Non-empty: share of points with a non-empty admissible region. Admissible: share of points whose true parameter vector lies in the region. Volume, $R^2$: median share of the parameter box covered by the region, and $R^2$ of the region median, given only for arms in which the true vector lies in the region in at least 90\\% of points.}}
\\label{{tab:loop}}
\\centering\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Data from}} & \\textbf{{Non-empty (\\%)}} & \\textbf{{Admissible (\\%)}} & \\textbf{{Volume (\\%)}} & $\\mathbf{{R^2}}$ \\texttt{{keil8}} & $\\mathbf{{R^2}}$ \\texttt{{init\\_ec}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

# ---------------------------------------------------------------- E. coli
real, syn, syn4, cont, quad = (load("ecoli_real.json"), load("ecoli_synthetic.json"), load("ecoli_synthetic_4strains.json"),
                               load("ecoli_real_continuous.json"), load("ecoli_real_quadrature.json"))
if real:
    names = real["names"]
    def col(d, p, key):
        return d["per_param"][p].get(key, np.nan) if d and p in d["per_param"] else np.nan
    rows = "\n".join(f"\\texttt{{{tex(p)}}} & {r2(col(real, p, 'r2_median_accepted'))} & {r2(col(real, p, 'r2_inverse_regression'))} & {f2(col(real, p, 'sd_ratio'))} & "
                     f"{r2(col(syn, p, 'r2_median_accepted'))} & {f2(col(syn, p, 'sd_ratio'))} & "
                     f"{r2(col(syn4, p, 'r2_median_accepted'))} & {f2(col(syn4, p, 'sd_ratio'))} \\\\" for p in names)
    def summ(d):
        if not d: return dict(ne="--", ta="--", nc="--", ng="--", gin="--", gmode="--", chance="--")
        ds = d.get("design_summary", {})
        return dict(ne=f"{d['n_nonempty']}/{d['n_folds']}", ta=pct(d["truth_admissible_all_folds"]),
                    nc=f"{np.median(ds['admissible_conditions']):.0f}/{ds['n_conditions']}" if ds else "--",
                    ng=f"{np.median(ds['distinct_genomes']):.0f}/{ds['n_genomes']}" if ds else "--",
                    gin=pct(ds.get("genome_in_set", np.nan)) if ds else "--", gmode=pct(ds.get("genome_mode_correct", np.nan)) if ds else "--",
                    chance=pct(ds.get("genome_chance", np.nan)) if ds else "--")
    S = {k: summ(d) for k, d in (("real", real), ("syn", syn), ("syn4", syn4))}
    act = (f"plateau $\\leftarrow$ \\texttt{{{tex(syn['synthetic']['plateau'])}}}, rate $\\leftarrow$ \\texttt{{{tex(syn['synthetic']['rate'])}}}, "
           f"midpoint $\\leftarrow$ \\texttt{{{tex(syn['synthetic']['midpoint'])}}}") if syn else "--"
    pairs = ", ".join(f"\\texttt{{{tex(x)}}}/\\texttt{{{tex(y)}}} ($r={c:.2f}$)" for x, y, c in real["design"]["correlated_pairs"])
    line = lambda key, lbl: f"{lbl}: " + ", ".join(f"{S[k][key]} ({n})" for k, n in (("real", "real"), ("syn", "control, five strains"), ("syn4", "control, four strains")))
    write("tab_external.tex", f"""\\begin{{table}}[!htbp]
\\caption{{SMoRe ParS on the external \\textit{{Escherichia coli}} sweep~\\cite{{gongying2025}}, leave-one-out over the growing conditions, admissible region evaluated on the conditions of the experimental design. Real: the measured curves. Control: the same design, replicate noise and missing readings, with curves generated from a known dependence ({act}), each spanning the 5--95\\% range of the real values; in the four-strain control the 4.63\\,Mb strain, whose readings end at 24\\,h, is left out. $R^2$, inverse $R^2$ and width as in Table~\\ref{{tab:recovery}} (width relative to the spread of the input over the design). In the design {pairs} vary together, so they can only be recovered jointly. {line('ne', 'Non-empty regions')}. {line('ta', 'True condition admissible')}. {line('nc', 'Median number of admissible conditions')}. {line('gin', 'True genome among the admissible conditions')}. {line('gmode', 'Genome with the most admissible conditions is the true one')}; chance level {S['real']['chance']} with five strains, {S['syn4']['chance']} with four.}}
\\label{{tab:external}}
\\centering\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{\\textbf{{Real curves}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Control, five strains}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Control, four strains}}}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-6}}\\cmidrule(lr){{7-8}}
\\textbf{{Input}} & $\\mathbf{{R^2}}$ & \\textbf{{Inv. }}$\\mathbf{{R^2}}$ & \\textbf{{Width}} & $\\mathbf{{R^2}}$ & \\textbf{{Width}} & $\\mathbf{{R^2}}$ & \\textbf{{Width}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    for lbl, d in (("real", real), ("control, five strains", syn), ("control, four strains", syn4), ("real, continuous box", cont), ("real, quadrature", quad)):
        if not d: continue
        ds = d.get("design_summary")
        note(f"E. coli {lbl}: non-empty / true condition admissible / volume fraction median",
             f"{d['n_nonempty']}/{d['n_folds']} / {d['truth_admissible_all_folds']:.2f} / {d['volume_fraction']['median']:.3f}", f"ecoli ({lbl}) top level")
        if ds:
            note(f"E. coli {lbl}: median admissible conditions / genomes / media; genome identified share",
                 f"{np.median(ds['admissible_conditions']):.0f}/{ds['n_conditions']} / {np.median(ds['distinct_genomes']):.0f}/{ds['n_genomes']} / "
                 f"{np.median(ds['distinct_media']):.0f}/{ds['n_media']}; {ds['genome_identified_frac']:.2f}", f"ecoli ({lbl}) design_summary")
            if "genome_in_set" in ds:
                note(f"E. coli {lbl}: genome in admissible set / most-admissible genome correct / nearest-to-median genome correct / chance",
                     f"{ds['genome_in_set']:.2f} / {ds['genome_mode_correct']:.2f} / {ds['genome_nearest_correct']:.2f} / {ds['genome_chance']:.2f}", f"ecoli ({lbl}) design_summary")
        s = d["surfaces"]; w = np.array([v["within_nsd"] for v in s.values()], float); r = np.array([v["loo_r2"] for v in s.values()], float)
        note(f"E. coli {lbl}: held-out bounds within the 2 SD band, median / surfaces LOO R2 median", f"{np.nanmedian(w):.2f} / {np.nanmedian(r):.2f}", f"ecoli ({lbl}) surfaces")
        for p in names:
            q = d["per_param"][p]
            note(f"E. coli {lbl} {p}: R2 region / R2 inverse / containment / width",
                 f"{q['r2_median_accepted']:+.3f} / {q.get('r2_inverse_regression', np.nan):+.3f} / {q['coverage95']:.2f} / {q['sd_ratio']:.2f}", f"ecoli ({lbl}) per_param")
        if d.get("synthetic", {}).get("unidentified_by_genome"):
            note(f"E. coli {lbl}: unidentified SM parameters per genome (conditions)", str(d["synthetic"]["unidentified_by_genome"]), f"ecoli ({lbl}) synthetic.unidentified_by_genome")
    note("E. coli: logistic SM median R2 on replicate means / below 0.9", f"{np.median(real['sm_r2']):.3f} / {sum(v < 0.9 for v in real['sm_r2'])}", "ecoli_real.json sm_r2")
    if a.ecoli_npz and os.path.exists(a.ecoli_npz):
        Z = np.load(a.ecoli_npz, allow_pickle=True); th, Y, tg = Z["theta"], Z["Y"], Z["t_grid"]; gc = [str(n) for n in Z["names"]].index("genome_Mb")
        for g in np.unique(th[:, gc]):
            last = [int(np.max(np.where(np.isfinite(y))[0])) for y in Y[th[:, gc] == g]]
            note(f"E. coli data: genome {g:.2f} Mb, last reading (median over curves), hours", f"{tg[int(np.median(last))]:.1f} of {tg[-1]:.1f}", a.ecoli_npz)

with open(os.path.join(OUT, "numbers.txt"), "w") as fh:
    for t, v, src in NUM: fh.write(f"{t}: {v}    [{src}]\n")
print(f"  wrote {os.path.join(OUT, 'numbers.txt')} ({len(NUM)} numbers)")
