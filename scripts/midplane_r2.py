import sys, json, functools, numpy as np
sys.path.insert(0, "scripts")
_orig = np.load
@functools.lru_cache(maxsize=6)
def _cached(p): return _orig(p)
np.load = lambda p, *a, **k: _cached(str(p))
import figures as F
tf = F._load_field_deps(); G = F.GRID; mid = G // 2
PL = {"xy_midplane_z": np.s_[:, :, mid], "xz_midplane_y": np.s_[:, mid, :], "yz_midplane_x": np.s_[mid, :, :]}
for model in F.MODELS:
    for cyt, _ in F.CYTS:
        for seed in (1, 42, 100):
            g = {k: [] for k in PL}; p = {k: [] for k in PL}
            for fr in range(80, 90):
                gt, pr = F._predict_field(tf, model, cyt, fr, seed=seed)
                gt, pr = np.squeeze(gt), np.squeeze(pr)
                for k, sl in PL.items():
                    g[k].append(gt[sl].ravel()); p[k].append(pr[sl].ravel())
            f = F.MODELS_ROOT / F.MODEL_DIR[model] / f"res_{cyt}_{F.RUN}_{G}_{seed}.json"
            d = json.load(open(f)); s2 = d["results"]["Near_Horizon_t82_t91"].setdefault("Slice_2D", {})
            for k in PL:
                a, b = np.concatenate(g[k]), np.concatenate(p[k])
                s2.setdefault(k, {})["R2_perframe_mean"] = s2[k].get("R2")
                s2[k]["R2"] = float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
            json.dump(d, open(f, "w"), indent=4)
            print(model, cyt, seed, {k: round(s2[k]["R2"], 3) for k in PL}, flush=True)
