#!/usr/bin/env python3
"""Upper bound for any predictor of the clipped field, scored against the full field: global R2 of
min(y, clip) against y in the early and late test windows, with the share of field variance above
the clip value. Reads the clipped and unclipped preprocessing of clip_comparison.

usage:  python scripts/clip_ceiling.py --work /path/to/clipcmp --out models/clip_comparison/clip_ceiling.json
"""
import json, argparse
from pathlib import Path
import numpy as np

CYTS = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
WINDOWS = {"Near_Horizon_t82_t91": (80, 90), "Far_Horizon_t92_t100": (90, 99)}

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--run", default="run_0062"); ap.add_argument("--grid", type=int, default=50)
ap.add_argument("--cytokines", nargs="+", default=["il8"])
a = ap.parse_args()
sub = f"{a.run}/{a.grid}x{a.grid}x{a.grid}"
d_full, d_clip = Path(a.work) / "prep_noclip" / sub, Path(a.work) / "prep_clip" / sub
out = {}
for cyt in a.cytokines:
    i = CYTS.index(cyt)
    scale = json.load(open(d_full / "metadata.json"))["scaling"]["max"][i]
    clip = json.load(open(d_clip / "metadata.json"))["scaling"]["max"][i]
    Y = (np.load(d_full / "Y_target.npy")[..., i].astype(np.float64) + 1) / 2 * scale
    out[cyt] = {"clip_value": clip}
    for w, (lo, hi) in WINDOWS.items():
        y = Y[lo:hi]; yc = np.minimum(y, clip); sst = ((y - y.mean()) ** 2).sum()
        out[cyt][w] = {"r2_clipped_field_vs_full": float(1 - ((y - yc) ** 2).sum() / sst),
                       "voxels_above_clip": float(np.mean(y > clip)), "max_over_clip": float(y.max() / clip)}
        print(f"{cyt} {w}: R2 of the clipped field against the full field {out[cyt][w]['r2_clipped_field_vs_full']:.4f}; "
              f"voxels above clip {100 * out[cyt][w]['voxels_above_clip']:.2f}%; max/clip {out[cyt][w]['max_over_clip']:.1f}")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"saved {a.out}")
