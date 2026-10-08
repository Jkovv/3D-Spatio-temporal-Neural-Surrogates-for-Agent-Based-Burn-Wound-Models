"""SMoRe ParS on the E. coli growth-curve sweep of Gong & Ying (Sci Rep 2025): 5 genomes x 29 media
= 145 conditions, 6 replicate curves each, with the pipeline of smore_pars_strict.py (logistic SM (M2),
profile-likelihood bounds, GP bound surfaces, leave-one-out admissible regions).

--support design (default) evaluates the region on the 145 conditions of the experiment; grid and
continuous use the product grid of levels or the continuous box (v1).
--mode synthetic is the positive control: real design, replicate noise and missing values, with curves
generated from a known dependence of plateau, rate and midpoint on the three --active inputs.

Input: the three source spreadsheets (--curves --design --media; --save-npz caches them) or such a
cache (--from-npz).
"""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helpers"))
import smore_pars_strict as S
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument("--curves"); ap.add_argument("--design"); ap.add_argument("--media")
ap.add_argument("--from-npz"); ap.add_argument("--save-npz")
ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=8)
ap.add_argument("--n-cand", type=int, default=131072); ap.add_argument("--n-cand-max", type=int, default=4194304)
ap.add_argument("--n-min", type=int, default=300); ap.add_argument("--n-sd", type=float, default=2.0)
ap.add_argument("--min-od", type=float, default=0.05)
ap.add_argument("--sigma-mode", choices=["paper", "quadrature", "se"], default="paper")
ap.add_argument("--mode", choices=["real", "synthetic"], default="real")
ap.add_argument("--active", nargs=3, default=["genome_Mb", "log_K+", "log_NH4+"])   # acting on plateau, rate, midpoint
ap.add_argument("--effect", type=float, default=1.0)     # synthetic: 1 = each active input spans the 5-95% range of the real SM parameter
ap.add_argument("--synth-seed", type=int, default=0); ap.add_argument("--folds", type=int, default=None)
ap.add_argument("--synth-q", type=float, nargs=2, default=[0.05, 0.95])   # synthetic: quantile range of the real SM parameters spanned by each active input
ap.add_argument("--fits-only", action="store_true")                        # stop after the SM fits (diagnostics of the synthetic control)
ap.add_argument("--drop-genome", type=float, default=None)                 # leave out every curve of this genome size (Mb), e.g. a strain with a shorter window
ap.add_argument("--support", choices=["design", "grid", "continuous"], default="design")
ap.add_argument("--continuous", action="store_true")    # same as --support continuous (v1 search space; for comparison)
a = ap.parse_args()
say = lambda msg: print(msg, flush=True)
S.SIGMA_MODE = a.sigma_mode

if a.from_npz:
    D = np.load(a.from_npz, allow_pickle=True)
    theta, Y, t_grid, names = np.asarray(D["theta"], float), np.asarray(D["Y"], float), np.asarray(D["t_grid"], float), [str(x) for x in D["names"]]
else:
    from validate_smore_external import load_sweep
    theta, Y, _masks, t_grid, ids, names = load_sweep(a.curves, a.design, a.media, say)
    theta, Y, t_grid = np.asarray(theta, float), np.asarray(Y, float), np.asarray(t_grid, float)
    if a.save_npz: np.savez(a.save_npz, theta=theta, Y=Y, t_grid=t_grid, names=np.array(names)); say(f"cached arrays in {a.save_npz}")

if a.drop_genome is not None:
    gcol0 = names.index("genome_Mb"); keep_c = np.abs(theta[:, gcol0] - a.drop_genome) > 1e-6
    # window diagnostic: last finite reading per strain (the measurement window differs between strains)
    for g in np.unique(theta[:, gcol0]):
        last = [int(np.max(np.where(np.isfinite(y))[0])) for y in Y[theta[:, gcol0] == g]]
        say(f"  genome {g:.2f} Mb: last finite reading at index {int(np.median(last))} of {Y.shape[1]-1} (median over curves)")
    theta, Y = theta[keep_c], Y[keep_c]; say(f"dropped genome {a.drop_genome} Mb: {int((~keep_c).sum())} curves removed, {len(Y)} kept")
# group replicate curves by identical input vector (condition)
keys = [tuple(np.round(r, 9)) for r in theta]; cond = {}
for i, k in enumerate(keys): cond.setdefault(k, []).append(i)
conds = list(cond)
th = np.array([theta[cond[k][0]] for k in conds]); R = max(len(v) for v in cond.values()); T = Y.shape[1]
Yrep = np.full((len(conds), R, T), np.nan)
for ci, k in enumerate(conds):
    for r, i in enumerate(cond[k]): Yrep[ci, r] = Y[i]
say(f"{len(Y)} curves -> {len(conds)} conditions, replicates per condition: min {min(len(v) for v in cond.values())} max {R}")

def mean_se(Yr):
    nfin = np.sum(np.isfinite(Yr), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(Yr, axis=1); se = np.nanstd(Yr, axis=1, ddof=1)/np.sqrt(np.maximum(nfin, 1))
    return mu, se
mu, se = mean_se(Yrep)
grow = np.nanmax(mu, axis=1) >= a.min_od
say(f"conditions with growth (max mean OD >= {a.min_od}): {grow.sum()} of {len(conds)}; the rest are excluded")
th_design = th.copy()                                     # all 145 realisable conditions (5 genomes x 29 media)
keep = np.where(grow)[0]; th, Yrep, mu, se = th[keep], Yrep[keep], mu[keep], se[keep]
runs = [f"cond_{i:03d}" for i in keep]
S.CYT_SM = {"od": "M2"}                                   # logistic growth as the surrogate model
tn = np.linspace(0, 1, T)

def fit_all(mu, se):
    def jobs():
        for i in range(len(runs)):
            ok = np.isfinite(mu[i]) & np.isfinite(se[i])
            yield (mu[i][ok][:, None], se[i][ok][:, None], tn[ok], ["od"], a.sigma_mode)
    if a.procs > 1:
        with Pool(a.procs) as pool: return pool.map(S.fit_point, list(jobs()))
    return [S.fit_point(j) for j in jobs()]

fits = fit_all(mu, se)
res = {"mode": a.mode, "names": names, "n_conditions": int(len(runs)), "sigma_mode": a.sigma_mode}
if a.mode == "synthetic":
    # positive control: real design, real residual noise and missing values, known signal
    B = np.array([f["od"]["best"] for f in fits])                          # (n, 3): A, r, t0 of the real curves
    logA, logr, t0 = np.log(B[:, 0]), np.log(np.maximum(B[:, 1], 1e-6)), B[:, 2]
    Xn = (th - th.min(0))/np.maximum(th.max(0) - th.min(0), 1e-300)
    U = (Xn - Xn.mean(0))/np.maximum(Xn.std(0), 1e-12)
    # one active input per SM parameter, linear on the normalised design scale over the 5-95% range of the
    # real values (log scale for plateau and rate)
    ix = {p: names.index(p) for p in a.active}
    pA, pr, pt = a.active
    def span(v, x):
        q5, q95 = np.quantile(v, a.synth_q); return 0.5*(q5 + q95) + a.effect*(x - 0.5)*(q95 - q5)
    A_s = np.exp(span(logA, Xn[:, ix[pA]])); r_s = np.exp(span(logr, Xn[:, ix[pr]])); t_s = span(t0, Xn[:, ix[pt]])
    signal = S.m2(tn[None, :], A_s[:, None], r_s[:, None], t_s[:, None])         # (n, T)
    resid = (Yrep - mu[:, None, :])*(A_s/np.maximum(np.nanmax(mu, axis=1), 1e-300))[:, None, None]
    Yrep = signal[:, None, :] + resid                                           # NaNs (missing readings) carried over
    mu, se = mean_se(Yrep)
    fits = fit_all(mu, se)
    res["synthetic"] = {"plateau": pA, "rate": pr, "midpoint": pt, "effect": a.effect, "quantiles": list(a.synth_q),
                        "inactive": [p for p in names if p not in a.active]}
    say(f"positive control: plateau <- {pA}; rate <- {pr}; midpoint <- {pt}; each spanning {a.effect} x the "
        f"{100*a.synth_q[0]:.0f}-{100*a.synth_q[1]:.0f}% range of the real values")
    # unidentified SM parameters per genome
    gcol = names.index("genome_Mb"); unid = {}
    for g in np.unique(np.round(th[:, gcol], 6)):
        sel = [f for f, gg in zip(fits, th[:, gcol]) if abs(gg - g) < 1e-6]
        unid[str(g)] = {nm: int(sum(bool(f["od"]["unidentified"][j]) for f in sel)) for j, nm in enumerate(fits[0]["od"]["names"])}; unid[str(g)]["n"] = len(sel)
    res["synthetic"]["unidentified_by_genome"] = unid
    say(f"  unidentified SM parameters per genome (count of conditions): {unid}")
    if a.fits_only:
        res["fits"] = {r: f for r, f in zip(runs, fits)}; res["synthetic"]["A_s"] = A_s.tolist(); res["synthetic"]["r_s"] = r_s.tolist(); res["synthetic"]["t_s"] = t_s.tolist()
        json.dump(res, open(a.out, "w"), indent=1, default=float); say(f"saved {a.out} (fits only)"); raise SystemExit
    say(f"  generated ranges: plateau {A_s.min():.3f}-{A_s.max():.3f}, rate {r_s.min():.1f}-{r_s.max():.1f}, midpoint {t_s.min():.3f}-{t_s.max():.3f} (time scaled to 0-1)")
r2 = [f["od"]["r2"] for f in fits if f.get("od")]
say(f"logistic SM fit on replicate means: median R2 {np.median(r2):.3f}, min {np.min(r2):.3f}, below 0.9: {sum(v < 0.9 for v in r2)} of {len(r2)}")
# design diagnostics: discrete levels and correlation between inputs over the conditions
lv = {p: int(len(np.unique(np.round(th[:, j], 9)))) for j, p in enumerate(names)}
cc = np.corrcoef(th.T); np.fill_diagonal(cc, 0)
i1, i2 = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
pairs = [(names[i], names[j], round(float(cc[i, j]), 2)) for i in range(len(names)) for j in range(i+1, len(names)) if abs(cc[i, j]) > 0.5]
say(f"design: levels per input {lv}; input pairs with |correlation| > 0.5 over the conditions: {pairs}")
res["design"] = {"levels": lv, "correlated_pairs": pairs, "n_media": int(len(np.unique(np.round(th_design[:, 1:], 9), axis=0)))}
bounds = {p: {"low": float(th_design[:, j].min()), "high": float(th_design[:, j].max())} for j, p in enumerate(names)}
folds = list(range(a.folds)) if a.folds else None
# region support: the 145 realisable conditions (default), the product grid of levels, or the continuous box
support = "continuous" if a.continuous else a.support
levels = [np.unique(np.round(th_design[:, j], 9)) for j in range(len(names))] if support == "grid" else None
cands = th_design if support == "design" else None
X, acc, info = S.jain_region(th, fits, runs, names, bounds, n_cand=a.n_cand, procs=a.procs, n_sd=a.n_sd,
                             n_min=a.n_min, n_cand_max=a.n_cand_max, folds=folds, levels=levels, candidates=cands)
lo_b = np.array([bounds[p]["low"] for p in names]); hi_b = np.array([bounds[p]["high"] for p in names])
if support == "design": ref_sd = ((th_design - lo_b)/(hi_b - lo_b)).std(0)
elif support == "grid": ref_sd = np.array([np.std((l - lo_b[i])/(hi_b[i] - lo_b[i])) for i, l in enumerate(levels)])
else: ref_sd = None
J = S.summarise_region(X, acc, names, info, n_sd=a.n_sd, ref_sd=ref_sd)
res["support"] = {"design": f"design ({len(th_design)} conditions)", "grid": "product grid of levels", "continuous": "continuous box"}[support]
S.print_region(J, names, f"E. coli, {a.mode}, {res['support']}")
if support == "design":
    # number of admissible conditions, and of distinct genomes and media among them
    rnd = lambda Z: np.round(Z, 6)
    n_adm = np.array([len(A) for A in acc]); n_gen = np.array([len(np.unique(rnd(A[:, 0]))) for A in acc])
    n_med = np.array([len(np.unique(rnd(A[:, 1:]), axis=0)) if len(A) else 0 for A in acc])
    n_med_all = len(np.unique(rnd(th_design[:, 1:]), axis=0)); n_gen_all = len(np.unique(rnd(th_design[:, 0])))
    # map every admissible point back to its design condition, and classify the genome of the held-out curve
    Dn = (th_design - lo_b)/(hi_b - lo_b); key = {tuple(np.round(r, 6)): i for i, r in enumerate(Dn)}
    adm_idx = [[key.get(tuple(np.round(r, 6)), -1) for r in A] for A in acc]
    gl = np.unique(rnd(th_design[:, 0])); fold_ids = folds if folds is not None else list(range(len(th)))
    g_true = rnd(th[fold_ids, 0]); cls = []
    for k, A in enumerate(acc):
        if len(A) == 0: cls.append({"in_set": False, "mode": None, "nearest": None}); continue
        ga = rnd(lo_b[0] + A[:, 0]*(hi_b[0] - lo_b[0])); cnt = {float(g): int(np.sum(ga == g)) for g in np.unique(ga)}   # back to Mb
        mode = max(cnt, key=lambda g: (cnt[g], -abs(g - np.median(ga))))        # most admissible conditions; ties -> nearest the median
        nearest = float(gl[np.argmin(np.abs(gl - np.median(ga)))])              # genome level nearest the region median
        cls.append({"in_set": bool(g_true[k] in cnt), "mode": mode, "nearest": nearest, "counts": cnt})
    g_in = float(np.mean([c["in_set"] for c in cls])); g_mode = float(np.mean([c["mode"] == g_true[k] for k, c in enumerate(cls)]))
    g_near = float(np.mean([c["nearest"] == g_true[k] for k, c in enumerate(cls)])); chance = 1.0/len(gl)
    res["design_summary"] = {"n_conditions": int(len(th_design)), "n_genomes": int(n_gen_all), "n_media": int(n_med_all),
                             "admissible_conditions": n_adm.tolist(), "distinct_genomes": n_gen.tolist(), "distinct_media": n_med.tolist(),
                             "genome_identified_frac": float(np.mean(n_gen == 1)), "medium_identified_frac": float(np.mean(n_med == 1)),
                             "admissible_idx": adm_idx, "genome_true": g_true.tolist(), "genome_classification": cls,
                             "genome_in_set": g_in, "genome_mode_correct": g_mode, "genome_nearest_correct": g_near, "genome_chance": chance}
    say(f"  genome of the held-out curve: in the admissible set in {g_in:.2f} of folds; the genome with most admissible conditions "
        f"is the true one in {g_mode:.2f}; the level nearest the region median is the true one in {g_near:.2f} (chance {chance:.2f})")
    say(f"  admissible conditions per held-out curve: median {np.median(n_adm):.0f} of {len(th_design)} (range {n_adm.min()}-{n_adm.max()}); "
        f"distinct genomes: median {np.median(n_gen):.0f} of {n_gen_all}; distinct media: median {np.median(n_med):.0f} of {n_med_all}")
    say(f"  genome uniquely identified in {np.mean(n_gen == 1):.2f} of folds, medium uniquely identified in {np.mean(n_med == 1):.2f}")
if support == "continuous":
    say("  note: continuous box with grid inputs: extreme levels lie on the box boundary, where the marginal central-95%\n"
        "  coverage cannot contain them; use the joint check (true vector admissible) for calibration.")
res.update(J); res["replicates_per_condition"] = np.sum(np.isfinite(Yrep).any(axis=2), axis=1).tolist(); res["sm_r2"] = r2
res["fits"] = {r: f for r, f in zip(runs, fits)}
json.dump(res, open(a.out, "w"), indent=1, default=float)
say(f"saved {a.out}")
