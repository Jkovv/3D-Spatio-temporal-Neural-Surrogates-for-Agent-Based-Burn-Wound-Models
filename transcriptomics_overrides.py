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
    applied = _apply_to(ns, BIOLOGY_NAMES)
    if applied:
        print(f"[transcriptomics_overrides] params_biology overrode "
              f"{sorted(applied)}", file=_sys.stderr)
    return applied


def apply_transitions(ns):
    applied = _apply_to(ns, TRANSITION_NAMES)
    if applied:
        print(f"[transcriptomics_overrides] params_transitions overrode "
              f"{sorted(applied)}", file=_sys.stderr)
    return applied