from params_grid import volumeconv, h_mcs
from params_biology import cil8, cil1, cil6, cil10, ctnf, ctgf

# cytokine production / secretion rates (mol cm⁻³ h⁻¹ -> lattice units MCS⁻¹)
keil8      = 234   * 1e-5 * volumeconv * h_mcs # endothelial -> IL-8
kndnil8    = 1.46  * 1e-5 * volumeconv * h_mcs # NDN neutrophil -> IL-8
thetanail8 = 3.024 * 1e-5 * volumeconv * h_mcs # activated neutrophil consumes IL-8
knail1     = 225   * 1e-5 * volumeconv * h_mcs # activated neutrophil -> IL-1
km1il6     = 250   * 1e-5 * volumeconv * h_mcs # M1 macrophage -> IL-6
km2il10    = 45    * 1e-5 * volumeconv * h_mcs # M2 macrophage -> IL-10
knatnf     = 250   * 1e-5 * volumeconv * h_mcs # activated neutrophil -> TNF
km1tnf     = 70    * 1e-5 * volumeconv * h_mcs # M1 macrophage -> TNF
km2tgf     = 280   * 1e-5 * volumeconv * h_mcs # M2 macrophage -> TGF-β1

# Sigmoid activation weights (dimensionless) 
# Positive = activating signal; negative = inhibiting signal.
lnril8  =  0.25 # neutrophil recruitment <- IL-8
lnril6  =  0.25 # neutrophil recruitment <- IL-6
lnril1  =  0.25 # neutrophil recruitment <- IL-1
lnrtnf  =  0.25 # neutrophil recruitment <- TNF
tnril10 = -0.5  # neutrophil recruitment <- IL-10 (inhibitory)
lmril6  =  0.5  # monocyte -> resting macrophage <- IL-6
lmrtnf  =  0.5  # monocyte -> resting macrophage <- TNF
tmril10 = -0.5  # monocyte -> resting macrophage <- IL-10 (inhibitory)
lm1il10 =  1.0  # M1 -> M2 transition <- IL-10
lftgf   =  1.0  # fibroblast activation <- TGF-β1

# Transition thresholds (dimensionless) 
tranril6 = 1.0    # monocyte -> resting macrophage threshold for IL-6

# Sigmoid shape parameters 
sigmoida = 1
sigmoidb = 4

# Michaelis-Menten kinetics (M2 lifespan modulation) 
vmax = -17.79826
km   = -963211.7

# Per-run parameter overrides (SMoRe ParS / sweep) 
# Applies any per-run vector from $SMORE_PARAMS to the scalar names above.
# MUST come BEFORE the activation arrays so that arrays derived from the sigmoid
# weights (lnril8, lmril6, ...) pick up the overridden values, not the originals.
from transcriptomics_overrides import apply_transitions as _apply_transitions
_apply_transitions(globals())

# Activation parameter arrays (built from possibly-overridden weights) 
# Each entry: [saturation_concentration, sigmoid_weight]
actnr  = [[cil8, lnril8], [cil6, lnril6], [cil1, lnril1], [ctnf, lnrtnf], [cil10, tnril10]]
actmr  = [[cil6, lmril6], [ctnf, lmrtnf], [cil10, tmril10]]
actm1  = [[cil10, lm1il10]]
actf   = [[ctgf, lftgf]]
tranmr = [[cil6, tranril6]]