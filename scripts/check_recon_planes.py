# Punkt 2: czy opis rekonstrukcji w appendixie zgadza sie z danymi (DeepONet, seed 42, klatka 88).
# Punkt 4: SSIM na przekrojach (paper twierdzi, ze jest liczony).
# Uruchamiac z $BASE (import figures z scripts/). Wymaga modulu TensorFlow (GPU).
import sys, functools
import numpy as np
sys.path.insert(0, "scripts")
_orig = np.load
@functools.lru_cache(maxsize=6)
def _c(p): return _orig(p)
np.load = lambda p, *a, **k: _c(str(p))
import figures as F
from skimage.metrics import structural_similarity as ssim

tf = F._load_field_deps()
G = F.GRID; h = G // 2
planes = {"xy": lambda v: v[:, :, h], "xz": lambda v: v[:, h, :], "yz": lambda v: v[h, :, :]}

print("=== CZESC A: figura rekonstrukcji (DeepONet, seed 42, indeks klatki 88) ===")
for cyt in ["il8", "il10"]:
    gt, pred = F._predict_field(tf, "DeepONet", cyt, 88, seed=42)
    for k, fn in planes.items():
        g, p = fn(gt), fn(pred); e = np.abs(p - g)
        i = np.unravel_index(np.argmax(g), g.shape)
        gm = g.max()
        core = g > 0.5 * gm
        ring = (g > 0.05 * gm) & (g <= 0.5 * gm)
        bg = g <= 0.05 * gm
        m = lambda msk: float(e[msk].mean()) if msk.any() else float("nan")
        print(f"{cyt:5s} {k}: GT max={gm:.2e} | pred w tym wokselu={p[i]:.2e} | "
              f"max |err|={e.max():.2e} ({100*e.max()/gm:.1f}% GT max) | "
              f"sredni |err|: rdzen zrodla={m(core):.2e} pierscien={m(ring):.2e} tlo={m(bg):.2e}")

print("\n=== CZESC B: SSIM na przekrojach (near, okna 80-89, L = c_max) ===")
for model in F.MODELS:
    for cyt, _ in F.CYTS:
        L = float(F._clip_max(cyt))
        for seed in (1, 42, 100):
            acc = {k: [] for k in planes}
            for fr in range(80, 90):
                gt, pred = F._predict_field(tf, model, cyt, fr, seed=seed)
                for k, fn in planes.items():
                    acc[k].append(ssim(fn(gt), fn(pred), data_range=L))
            print(f"{model:9s} {cyt:5s} seed{seed:<3d} " +
                  " ".join(f"{k}={np.mean(v):.3f}" for k, v in acc.items()), flush=True)
