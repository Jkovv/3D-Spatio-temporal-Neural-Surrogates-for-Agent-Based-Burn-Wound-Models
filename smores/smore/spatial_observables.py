#!/usr/bin/env python3
# combi3D/Simulation/smore/spatial_observables.py
#
# Spatially structured observables for SMoRe ParS.
#
# WHY THIS EXISTS
# observables.py reduces each run to the volume-averaged concentration
# trajectory il*_mean(t), then to four scalars per cytokine. That average
# collapses 125,000 voxels into one number per frame and discards every bit of
# spatial structure -- which is the one thing a 3D model provides over a 2D one.
#
# The measured consequence: with surrogate-derived observables, keil8 recovery
# drops from R2 = 0.838 (ABM observables) to 0.191, with corr(recovered, true)
# falling 0.915 -> 0.536 while variance and bias stay intact. Coarse-effect
# parameters (init_ec, the number of constitutive sources) are over-weighted;
# subtle-effect ones (keil8, a secretion rate) are lost. A plausible reading is
# that keil8 shapes the *profile* of the diffusion front rather than its overall
# level, so volume-averaging removes its signature before SMoRe ParS sees it.
#
# This module tests that reading. It computes observables that keep spatial
# structure, and is otherwise a drop-in replacement: summarize_spatial(F) takes
# volumetric fields and returns a (n_runs, n_features) matrix in the same form
# emulator_sobol() and fit_surrogates() already consume.
#
# WHAT IT COMPUTES (per cytokine, per frame, then reduced over time)
#   radial   : concentration in R shells about the wound centre -- captures how
#              far and how steeply the field falls off, which a rate parameter
#              should influence independently of the overall level
#   layer    : mean per depth band along z -- physiologically the axis that
#              defines burn severity, and the axis where the paper's mid-plane
#              analysis already found anisotropy
#   active   : fraction of voxels above a threshold -- how far the signal
#              spreads, rather than how strong it is on average
#   moments  : centroid distance from wound centre and spatial SD of the
#              concentration distribution -- shape without a binning choice
#   active-voxel quantiles and peak : median and 90th percentile computed over
#              non-empty voxels only, plus the field maximum. Whole-field
#              quantiles are useless on a sparse field -- if the signal occupies
#              0.1% of voxels, even the 99th percentile sits in the background --
#              so these condition on the active set instead. This is the regime
#              IL-10 lives in, where the mean-based observable failed
#              (gen R2 = -1124) because averaging over a ~99.99% empty field
#              amplified background error to ~500x the signal. Whether they
#              rescue IL-10 is an open question this experiment tests.
#
# HONEST NOTE ON INTERPRETATION
# If recovery improves with these observables, the earlier limit was about
# observable design, not surrogate fidelity, and the finding becomes a design
# recommendation. If recovery does not improve, the fidelity claim is
# strengthened, because the loss then survives observables that preserve the
# spatial information. Both outcomes are informative; neither is assumed here.
#
# Usage as a library:
#     from spatial_observables import summarize_spatial, spatial_feature_names
#     feats = summarize_spatial(F, cyts, centre="mask")   # F: (n_runs,T,G,G,G)
#     names = spatial_feature_names(cyts)
#
# Usage as a check on one run's fields:
#     python spatial_observables.py --fields path/to/Y_target.npy --cyt-idx 0

import argparse

import numpy as np

# Feature blocks, in the order summarize_spatial emits them.
N_RADIAL = 4      # radial shells
N_LAYER = 4       # depth bands along z
ACTIVE_THRESH = 0.05   # fraction of that run's own field max


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------

_GRID_CACHE = {}


def _coords(G):
    """Voxel coordinate grids, cached per grid size."""
    if G not in _GRID_CACHE:
        ax = np.arange(G, dtype=np.float32)
        xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
        _GRID_CACHE[G] = (xx, yy, zz)
    return _GRID_CACHE[G]


def wound_centre(field_t0, mask=None, mode="geometric",
                 scaled=False, clip_max=None):
    """Reference point for the radial profiles.

    mode='geometric' : centre of the lattice. Safe default, no assumptions.
    mode='field'     : intensity-weighted centroid of the first frame.
    mode='mask'      : centroid of the supplied cell mask.

    Which is right depends on how the wound is initialised in the 3D
    configuration. If the injury is centred in the domain, 'geometric' and
    'field' agree and the choice does not matter; if it is not, 'field' or
    'mask' follows the wound. Report which was used.
    """
    G = field_t0.shape[0]
    if mode == "geometric":
        return np.array([(G - 1) / 2.0] * 3, dtype=np.float32)
    xx, yy, zz = _coords(G)
    w = mask if (mode == "mask" and mask is not None) else field_t0
    w = np.asarray(w, np.float64)
    if scaled and not (mode == "mask" and mask is not None):
        if clip_max is None:
            raise ValueError("scaled=True requires clip_max")
        w = (w + 1.0) / 2.0 * float(clip_max)
    w = np.maximum(w, 0.0)
    tot = w.sum()
    if tot <= 0:
        return np.array([(G - 1) / 2.0] * 3, dtype=np.float32)
    return np.array([(xx * w).sum() / tot,
                     (yy * w).sum() / tot,
                     (zz * w).sum() / tot], dtype=np.float32)


def _radius_bins(G, centre, n_shells):
    """Assign every voxel to a radial shell about `centre`. Shell edges are
    equal-width in radius out to the largest radius fully inside the lattice."""
    xx, yy, zz = _coords(G)
    r = np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (zz - centre[2]) ** 2)
    r_max = float(min(centre.min(), (G - 1) - centre.max()))
    if r_max <= 0:
        r_max = float(r.max())
    edges = np.linspace(0.0, r_max, n_shells + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_shells - 1)
    return idx, r


# ----------------------------------------------------------------------------
# per-frame spatial descriptors
# ----------------------------------------------------------------------------

def frame_descriptors(F, centre, n_radial=N_RADIAL, n_layer=N_LAYER,
                      thresh_frac=ACTIVE_THRESH, scaled=False, clip_max=None):
    """Spatial descriptors for one run's field sequence.

    F : (T, G, G, G). Either physical units (non-negative) or the preprocessed
        [-1, 1] representation, in which case pass scaled=True and clip_max.

    Denormalisation is not cosmetic here. In [-1, 1] the background sits at -1,
    so the intensity-weighted moments sum signed weights, the total passes
    through zero, and the centroid diverges. The quantiles break too, because
    "active" cannot mean "> 0" when 0 is mid-range. Everything below therefore
    assumes a non-negative field with background at 0.
    """
    F = np.asarray(F, np.float64)
    if scaled:
        if clip_max is None:
            raise ValueError("scaled=True requires clip_max (metadata "
                             "scaling.max for this cytokine)")
        F = np.maximum((F + 1.0) / 2.0 * float(clip_max), 0.0)
    T, G = F.shape[0], F.shape[1]
    flat = F.reshape(T, -1)

    rid, _ = _radius_bins(G, centre, n_radial)
    rid_flat = rid.ravel()
    counts_r = np.bincount(rid_flat, minlength=n_radial).astype(np.float64)
    radial = np.stack(
        [np.bincount(rid_flat, weights=flat[t], minlength=n_radial) /
         np.maximum(counts_r, 1.0) for t in range(T)], axis=0)      # (T, n_radial)

    edges = np.linspace(0, G, n_layer + 1).astype(int)
    layer = np.stack(
        [F[:, :, :, edges[k]:edges[k + 1]].reshape(T, -1).mean(axis=1)
         for k in range(n_layer)], axis=1)                          # (T, n_layer)

    fmax = flat.max()
    thr = thresh_frac * fmax if fmax > 0 else 0.0
    active = (flat > thr).mean(axis=1)                              # (T,)

    xx, yy, zz = _coords(G)
    cx, cy, cz = xx.ravel(), yy.ravel(), zz.ravel()
    tot = flat.sum(axis=1)
    ok = tot > 0
    safe = np.where(ok, tot, 1.0)
    mx = flat @ cx / safe
    my = flat @ cy / safe
    mz = flat @ cz / safe
    centroid_r = np.zeros(T)
    spread = np.zeros(T)
    if ok.any():
        dx = mx[ok] - centre[0]; dy = my[ok] - centre[1]; dz = mz[ok] - centre[2]
        centroid_r[ok] = np.sqrt(dx * dx + dy * dy + dz * dz)
        r2 = flat[ok] @ (cx ** 2 + cy ** 2 + cz ** 2) / safe[ok]
        var = r2 - (mx[ok] ** 2 + my[ok] ** 2 + mz[ok] ** 2)
        spread[ok] = np.sqrt(np.maximum(var, 0.0))

    # Quantiles over the WHOLE field are useless on a sparse one: if the signal
    # occupies 0.1% of voxels, even the 99th percentile sits in the background.
    # These are therefore computed over active voxels only (> 0), which is the
    # regime IL-10 lives in. Frames with no active voxel yield 0.
    q50a = np.zeros(T)
    q90a = np.zeros(T)
    peak = flat.max(axis=1)
    for t in range(T):
        v = flat[t][flat[t] > 0]
        if v.size:
            q50a[t] = np.quantile(v, 0.50)
            q90a[t] = np.quantile(v, 0.90)

    return {"radial": radial, "layer": layer, "active": active,
            "centroid_r": centroid_r, "spread": spread,
            "q50a": q50a, "q90a": q90a, "peak": peak}


# ----------------------------------------------------------------------------
# reduction over time -> one feature vector per run
# ----------------------------------------------------------------------------

def _reduce_time(series):
    """(T,) -> [final, mean, max, auc], matching observables.summarize_observable
    so the two observable sets are compared on equal footing."""
    s = np.asarray(series, np.float64)
    T = len(s)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return np.array([s[-1], s.mean(), s.max(), _trap(s) / max(1, T - 1)])


def summarize_spatial(F, cyts, centre_mode="geometric", masks=None,
                      n_radial=N_RADIAL, n_layer=N_LAYER,
                      scaled=False, clip_max=None):
    """Volumetric fields -> per-run spatial feature matrix.

    F     : (n_runs, T, G, G, G, n_cyt) or (n_runs, T, G, G, G) for one cytokine
    cyts  : cytokine names, len == n_cyt
    masks : optional (n_runs, G, G, G) cell masks, used when centre_mode='mask'
    scaled: True if F is the preprocessed [-1,1] representation rather than
            physical units; then clip_max must give scaling.max per cytokine
            (scalar for one channel, or a sequence of length n_cyt)

    Returns (n_runs, n_features); pair with spatial_feature_names(cyts).
    """
    F = np.asarray(F)
    if F.ndim == 5:
        F = F[..., None]
    n_runs, T = F.shape[0], F.shape[1]
    n_cyt = F.shape[-1]
    if len(cyts) != n_cyt:
        raise ValueError(f"{len(cyts)} cytokine names for {n_cyt} channels")

    out = []
    for i in range(n_runs):
        row = []
        for c in range(n_cyt):
            fld = F[i, ..., c]
            cm = None
            if scaled:
                cm = clip_max[c] if np.ndim(clip_max) else clip_max
            ctr = wound_centre(fld[0],
                               mask=None if masks is None else masks[i],
                               mode=centre_mode, scaled=scaled, clip_max=cm)
            d = frame_descriptors(fld, ctr, n_radial, n_layer,
                                  scaled=scaled, clip_max=cm)
            for k in range(n_radial):
                row.append(_reduce_time(d["radial"][:, k]))
            for k in range(n_layer):
                row.append(_reduce_time(d["layer"][:, k]))
            for key in ("active", "centroid_r", "spread", "q50a", "q90a", "peak"):
                row.append(_reduce_time(d[key]))
        out.append(np.concatenate(row))
    return np.vstack(out)


def spatial_feature_names(cyts, n_radial=N_RADIAL, n_layer=N_LAYER):
    stats = ("final", "mean", "max", "auc")
    names = []
    for c in cyts:
        for k in range(n_radial):
            names += [f"{c}_radial{k}_{s}" for s in stats]
        for k in range(n_layer):
            names += [f"{c}_layer{k}_{s}" for s in stats]
        for key in ("active", "centroidr", "spread", "q50a", "q90a", "peak"):
            names += [f"{c}_{key}_{s}" for s in stats]
    return names


def n_features_per_cytokine(n_radial=N_RADIAL, n_layer=N_LAYER):
    return 4 * (n_radial + n_layer + 6)


# ----------------------------------------------------------------------------
# standalone check on a single run
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Inspect spatial observables for one run's field file.")
    ap.add_argument("--fields", required=True,
                    help="Y_target.npy, shape (T,G,G,G,n_cyt) or (T,G,G,G)")
    ap.add_argument("--cyt-idx", type=int, default=0)
    ap.add_argument("--cyt-name", default="il8")
    ap.add_argument("--centre", choices=["geometric", "field"],
                    default="geometric")
    ap.add_argument("--metadata", default=None,
                    help="metadata.json beside the field file. If given, the "
                         "field is treated as the preprocessed [-1,1] form and "
                         "denormalised with scaling.max for this cytokine.")
    args = ap.parse_args()

    Y = np.load(args.fields)
    if Y.ndim == 5:
        Y = Y[..., args.cyt_idx]
    print(f"fields {Y.shape}  min={Y.min():.3e}  max={Y.max():.3e}")

    scaled, cmax = False, None
    if args.metadata:
        import json
        cmax = float(json.load(open(args.metadata))["scaling"]["max"][args.cyt_idx])
        scaled = True
        print(f"denormalising with clip_max = {cmax:.4e}")
    elif Y.min() < -0.5:
        print("WARNING: field looks like the [-1,1] preprocessed form but no "
              "--metadata given; moments and quantiles will be wrong.")

    ctr = wound_centre(Y[0], mode=args.centre, scaled=scaled, clip_max=cmax)
    print(f"wound centre ({args.centre}): {ctr}")

    d = frame_descriptors(Y, ctr, scaled=scaled, clip_max=cmax)
    print(f"\nradial shells (frame 0 -> last):")
    for k in range(d['radial'].shape[1]):
        print(f"  shell {k}: {d['radial'][0, k]:+.3e} -> {d['radial'][-1, k]:+.3e}")
    print(f"depth bands (frame 0 -> last):")
    for k in range(d['layer'].shape[1]):
        print(f"  band  {k}: {d['layer'][0, k]:+.3e} -> {d['layer'][-1, k]:+.3e}")
    print(f"active fraction : {d['active'][0]:.4f} -> {d['active'][-1]:.4f}")
    print(f"centroid radius : {d['centroid_r'][0]:.2f} -> {d['centroid_r'][-1]:.2f}")
    print(f"spatial spread  : {d['spread'][0]:.2f} -> {d['spread'][-1]:.2f}")
    print(f"median (active) : {d['q50a'][0]:+.3e} -> {d['q50a'][-1]:+.3e}")
    print(f"q90    (active) : {d['q90a'][0]:+.3e} -> {d['q90a'][-1]:+.3e}")
    print(f"peak            : {d['peak'][0]:+.3e} -> {d['peak'][-1]:+.3e}")

    feats = summarize_spatial(Y[None, ..., None], [args.cyt_name],
                              centre_mode=args.centre,
                              scaled=scaled, clip_max=cmax)
    names = spatial_feature_names([args.cyt_name])
    print(f"\nfeature vector: {feats.shape[1]} features "
          f"({n_features_per_cytokine()} per cytokine)")
    nz = np.abs(feats[0]) > 0
    print(f"non-zero: {nz.sum()}/{len(nz)}")
    if not nz.all():
        print("  all-zero features (they carry no signal and will not help "
              "the recovery):")
        for nm in np.array(names)[~nz][:10]:
            print(f"    {nm}")


if __name__ == "__main__":
    main()