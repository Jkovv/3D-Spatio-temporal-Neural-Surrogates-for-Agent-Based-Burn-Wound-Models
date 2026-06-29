from params_grid import volumeconv, s_mcs, h_mcs, scale_factor

# Diffusion coefficients (cm² s⁻¹ → lattice units MCS⁻¹) 
Dil8  = 2.09e-6 * s_mcs / volumeconv
Dil1  = 3.00e-7 * s_mcs / volumeconv
Dil6  = 8.49e-8 * s_mcs / volumeconv
Dil10 = 1.45e-8 * s_mcs / volumeconv
Dtnf  = 4.07e-9 * s_mcs / volumeconv
Dtgf  = 2.60e-7 * s_mcs / volumeconv

# Decay rates (h⁻¹ → MCS⁻¹) 
muil8  = 0.2              * h_mcs
muil1  = 0.6              * h_mcs
muil6  = 0.5              * h_mcs
muil10 = 0.5              * h_mcs
mutnf  = 0.5 * 0.225      * h_mcs
mutgf  = 0.5 * (1 / 25)   * h_mcs

# Saturation concentrations (mol cm⁻³) 
cil8  = 2e-9
cil1  = 5e-9
cil6  = 5e-9
cil10 = 5e-9
ctnf  = 5e-9
ctgf  = 5e-9

# Lifespans (MCS) 
lifespane   = 1_000_000 # endothelial - permanent
lifespannr  = 20 # resting neutrophil
lifespanm   = 24 # monocyte
lifespanmr  = 1_000_000 # resting macrophage - permanent
lifespanf   = 1_000_000 # fibroblast - permanent
timeforgrowth = 0.5 # fraction of lifespan before division eligible

# dDivision probabilities 
divpre =  1
divprnr= -1
divprm = -1
divprf = int((lifespanf - lifespanf * timeforgrowth) * 2)

# scaled initial cell counts 
def _scaled(n500, name):
    val = max(1, round(n500 * scale_factor))
    if round(n500 * scale_factor) < 1:
        print(f"WARNING: {name} count rounded to 0 - flooring to 1")
    return val

init_ec = _scaled(1000,"init_ec") # endothelial cells (tissue scaffold)
init_n = _scaled(1100,"init_n") # neutrophils (blood + tissue combined)
init_m = _scaled(1000,"init_m") # monocytes
init_my = _scaled(25, "init_my") # myofibroblasts

# Replenishment rates (cells per replenishment interval) 
replen_n  = _scaled(10,  "replen_n")
replen_f  = _scaled(10,  "replen_f")
replen_my = _scaled(3,   "replen_my")
replen_m  = _scaled(100, "replen_m")

# Per-run parameter overrides (SMoRe ParS / sweep) 
# Applied last so the sampled vector can vary cell counts. 
from transcriptomics_overrides import apply_biology as _apply_biology
_apply_biology(globals())
