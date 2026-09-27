# 3D Spatio-temporal Neural Surrogates for Agent-Based Burn-Wound Models

Volumetric neural surrogates for a three-dimensional agent-based model of the post-burn immune response, together with the global sensitivity analysis and SMoRe ParS parameter recovery those surrogates were built to make affordable. 

All results come from a single 100-run Latin-hypercube sweep. The ABM has no fixed RNG seed, so the sweep, surrogates, sensitivity analysis, recovery and figures were all recomputed together on this sweep. Surrogate hyperparameters were tuned with Optuna on seed 42 and reused unchanged for seeds 1 and 100.

---
## Headline results

| | |
|---|---|
| Dense cytokine (IL-8) | DeepONet R² = 0.999, 3D U-Net R² = 0.987 - effectively a tie |
| Sparse cytokine (IL-10) | DeepONet R² = 0.929 (0.89-0.96 across seeds; 0.956 at the far horizon); U-Net 0.742 (0.62-0.82; 0.703 far) |
| Mid-plane accuracy | On IL-10 xy is the weakest plane (DeepONet 0.80, U-Net 0.73) and yz the strongest (0.97, 0.95), for both models |
| Inference speed-up | U-Net ≈ 3100-3400×, DeepONet ≈ 120× per volumetric read-out |
| Sensitivity | `sigmoidb` clearly first (S_T = 0.445); `keil8`, `init_ec` and `km2il10` (0.13-0.17) have no stable order |
| Recovery | `keil8` R² = +0.871, `sigmoidb` +0.512; all four initial-population parameters negative |
| Sensitivity ≠ identifiability | `init_ec` ranks third by Sobol yet recovers at -0.122; final IL-8 tracks the product `init_ec` × `keil8` (\|r\| = 0.984), a frozen-endothelium ridge |
| Recovery target choice | Swapping `init_ec` for `km2tgf` in the top 5 lowers mean nRMSE by 11% |
| Surrogate in the loop | Field R² 0.93-0.98 across seeds, `keil8` recovery -0.07 to +0.12 against +0.81 from the ABM's own IL-8 observables |
| External stress test | Surface fits *E. coli* growth curves better than the ABM (median R² = 0.986); none of the top inputs recovered |

---

## Layout

```
.
├── models/      # DeepONet and U-Net: result files and trained weights
├── scripts/     # preprocessing, training, evaluation and figure scripts
├── figures/     # the seven manuscript figures
├── figures_3d/  # individual panels the manuscript figures are assembled from
├── smores/      # sweep, sensitivity, and calibration - see smores/README.md
├── .gitattributes
└── .gitignore
```

**`models/`** holds the two architectures carried forward from the 2D benchmark (https://zenodo.org/records/20465819).

**`scripts/`** holds the preprocessing from the 2D benchmark - the kurtosis-adaptive percentile-clipping normalisation, the two-frame look-back, and the chronological 70/10/19 split - adapted to 3D by removing a numerical-diffusion noise floor and computing the kurtosis and clip percentile over the voxels above it. It also holds the evaluation metrics (global R², masked RMSE, Dice, volumetric SSIM, Fisher-z-pooled spatial correlation) and the orthogonal mid-plane metrics added for the 3D setting (`midplane_r2.py`). 
`figures.py`, `fig_seeddots.py` and `assemble_panels.py` rebuild all figures (`rebuild_figures.slurm`).

**`figures/`** maps one-to-one onto the manuscript:

| File | Content |
|---|---|
| `fig1_data.png` | Cytokine trajectories and the dense/sparse contrast (slices at t = 88 h) |
| `fig2_architectures.png` | The two benchmarked architectures |
| `fig3_accuracy.png` | Volumetric accuracy and metrics, per seed |
| `fig4_midplane.png` | Mid-plane reconstruction at t = 88 h and per-plane R² |
| `fig5_cost.png` | Inference speed-up over one ABM trajectory |
| `fig6_sobol.png` | Sobol total-order indices, ranked |
| `fig7_recovery.png` | Recovery, and sensitivity against identifiability |

**`smores/`** is the calibration pipeline: the 100-run Latin-hypercube sweep,
the emulator-based Sobol screen, the SMoRe ParS recovery, the surrogate-in-the-loop
experiment, and the `init_ec` × `keil8` ridge analysis (`smores/helpers/ridge_raw.py`).

---

## Quickstart

```bash
git clone <repo> && cd <repo>

# calibration: sensitivity first, then SMoRe ParS on the identifiable subset
cd smores
python smore/run_calibration.py --sim-root sweep/outputs \
    --manifest manifest.json --top-k 5 --out calibration_results.json
```

See `smores/README.md` for the sweep, the emulator audit, and the filtered
Sobol ranking, and `models/` for surrogate training.

Surrogate results in the paper were produced on a single NVIDIA A100 on Snellius.

---
