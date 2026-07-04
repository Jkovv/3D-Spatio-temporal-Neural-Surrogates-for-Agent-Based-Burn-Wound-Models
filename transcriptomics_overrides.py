# combi3D/Simulation/transcriptomics_overrides.py
#
# *** PER-RUN PARAMETER INJECTION POINT (SMoRe ParS / sweep) ***
#
# Kept as a thin, explicit module. The real logic lives in param_loader.py.
# params_biology.py and params_transitions.py each call the matching
# apply_* function below with their own globals(), so there is NO stack
# introspection and NO ambiguity about which module is being written.
#
# Mechanism:
#   driver writes run_XXXX_params.json   (one "runs[i].params" dict)
#   driver sets  SMORE_PARAMS=/abs/path/run_XXXX_params.json
#   driver copies that JSON next to the run's LatticeData/ output
#   -> CompuCell3D launches; the vector is applied deterministically.
#
# With SMORE_PARAMS unset, this is a no-op baseline run (logged by param_loader).

import sys as _sys
from param_loader import apply_to as _apply_to

BIOLOGY_NAMES = {
    "init_ec", "init_n", "init_m", "init_f", "init_my",
    "replen_n", "replen_f", "replen_my", "replen_m",
}
TRANSITION_NAMES = {
    "keil8", "kndnil8", "thetanail8", "knail1",
    "km1il6", "km2il10", "knatnf", "km1tnf", "km2tgf",
    "lnril8", "lnril6", "lnril1", "lnrtnf", "tnril10",
    "lmril6", "lmrtnf", "tmril10", "lm1il10", "lftgf",
    "tranril6", "sigmoida", "sigmoidb",
}


def apply_biology(ns):
    """Override cell-count names in the params_biology namespace."""
    applied = _apply_to(ns, BIOLOGY_NAMES)
    if applied:
        print(f"[transcriptomics_overrides] params_biology overrode "
              f"{sorted(applied)}", file=_sys.stderr)
    return applied


def apply_transitions(ns):
    """Override rate / weight names in the params_transitions namespace."""
    applied = _apply_to(ns, TRANSITION_NAMES)
    if applied:
        print(f"[transcriptomics_overrides] params_transitions overrode "
              f"{sorted(applied)}", file=_sys.stderr)
    return applied
