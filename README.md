# 3D Spatio-temporal Neural Surrogates for Agent-Based Burn-Wound Models

Volumetric (50³) neural surrogates for a three-dimensional agent-based model of
the post-burn immune response, together with the global sensitivity analysis and
SMoRe ParS parameter recovery those surrogates were built to make affordable.

---
## Headline results

| | |
|---|---|
| Dense cytokine (IL-8) | DeepONet R² = 0.999, 3D U-Net R² = 0.993 - effectively a tie |
| Sparse cytokine (IL-10) | DeepONet R² = 0.959; U-Net collapses to 0.308, and to -0.399 on the yz mid-plane |
| Repeatability on IL-10 | U-Net seed spread ±0.18 against DeepONet's ±0.01 |
| Inference speed-up | U-Net ≈ 2900-3100×, DeepONet ≈ 120× per volumetric read-out |
| Sensitivity | Two parameters carry the variance: `sigmoidb` (0.457), `keil8` (0.216) |
| Recovery | `keil8` R² = +0.833; all four initial-population parameters negative |
| Sensitivity ≠ identifiability | `init_ec` ranks third by Sobol yet recovers at -0.454 - a frozen-endothelium collinearity |
| Surrogate in the loop | Field R² 0.987-0.998 across seeds, `keil8` recovery +0.191 to -0.077 |
| External validation | Surface fits *E. coli* growth curves better than the ABM (median R² = 0.986); none of the nine inputs recovered |

---

## Layout

```
.
├── models/     # DeepONet and 3D U-Net: architectures, training, tuned configs
├── scripts/    # preprocessing, evaluation metrics, and the experiment drivers
├── figures/    # the eight manuscript figures
├── smores/     # sweep, sensitivity, and calibration - see smores/README.md
├── .gitattributes
└── .gitignore
```

**`models/`** holds the two architectures carried forward from the 2D
benchmark (https://zenodo.org/records/20465819). 

**`scripts/`** holds the preprocessing carried unchanged from the 2D benchmark
- the kurtosis-adaptive percentile-clipping normalisation, the two-frame
look-back, and the chronological 70/10/19 split - plus the evaluation metrics
(global R², masked RMSE, Dice, volumetric SSIM, Fisher-z-pooled spatial
correlation) and the orthogonal mid-plane metrics added for the 3D setting.

**`figures/`** maps one-to-one onto the manuscript:

| File | Content |
|---|---|
| `fig1_data.png` | Cytokine trajectories and the dense/sparse contrast |
| `fig2_architectures.png` | The two benchmarked architectures |
| `fig3_accuracy.png` | Volumetric accuracy and seed spread |
| `fig4_recon_slices.png` | Qualitative reconstruction on IL-10 |
| `fig5_midplane_r2.png` | Per-plane R², showing the anisotropy |
| `fig6_speedup.png` | Inference speed-up over one ABM trajectory |
| `fig7_sobol.png` | Sobol total-order indices, ranked |
| `fig8_recovery.png` | Recovery, and sensitivity against identifiability |

**`smores/`** is the calibration pipeline: the 100-run Latin-hypercube sweep,
the emulator-based Sobol screen, and the SMoRe ParS recovery. 
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

Surrogate results in the paper were produced on a single NVIDIA A100 on
Snellius, matching the 2D benchmark so the cross-dimensional comparison is not
confounded by hardware.

---

## Data

The parameter sweep, trained surrogate weights, analysis scripts, and every
result file needed to reproduce the figures and tables are deposited on Zenodo:

**[10.5281/zenodo.20700318](https://doi.org/10.5281/zenodo.20700318)**

The external growth-curve dataset used for the independent validation is the
published supplementary material of Gong and Ying (2025), reused under CC BY
4.0. It is not redistributed here.

---
