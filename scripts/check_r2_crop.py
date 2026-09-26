import sys, json, functools
import numpy as np
sys.path.insert(0, "scripts")
_orig = np.load
@functools.lru_cache(maxsize=6)
def _cached(p): return _orig(p)
np.load = lambda p, *a, **k: _cached(str(p))
import figures as F
tf = F._load_field_deps()
G = F.GRID
for model in F.MODELS:
    for cyt, _ in F.CYTS:
        for seed in (1, 42, 100):
            g_all, p_all, per_frame = [], [], []
            for fr in range(80, 90):
                gt, pred = F._predict_field(tf, model, cyt, fr, seed=seed)
                g, p = gt[3:-3,3:-3,3:-3].ravel(), pred[3:-3,3:-3,3:-3].ravel()
                g_all.append(g); p_all.append(p)
                per_frame.append(1 - ((g-p)**2).sum() / ((g-g.mean())**2).sum())
            g = np.concatenate(g_all); p = np.concatenate(p_all)
            pooled = 1 - ((g-p)**2).sum() / ((g-g.mean())**2).sum()
            f = F.MODELS_ROOT / F.MODEL_DIR[model] / f"res_{cyt}_{F.RUN}_{G}_{seed}.json"
            glob_r2 = json.load(open(f))["results"]["Near_Horizon_t82_t91"]["Global_R2"]
            print(f"{model:9s} {cyt:5s} seed{seed:<3d} volume_pooled={pooled:.3f} "
                  f"volume_perframe_mean={np.mean(per_frame):.3f} | training Global_R2={glob_r2:.3f}", flush=True)
