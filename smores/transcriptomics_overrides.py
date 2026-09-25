from param_loader import apply_to as _apply_to

_BIOLOGY_NAMES = [
    "init_ec", "init_n", "init_m", "init_f", "init_my",
    "replen_n", "replen_f", "replen_my", "replen_m",
]

_TRANSITION_NAMES = [
    "keil8", "kndnil8", "thetanail8", "knail1", "km1il6", "km2il10",
    "knatnf", "km1tnf", "km2tgf",
    "lnril8", "lnril6", "lnril1", "lnrtnf", "tnril10",
    "lmril6", "lmrtnf", "tmril10", "lm1il10", "lftgf", "tranril6",
    "sigmoida", "sigmoidb",
]


def apply_biology(namespace):
    """Override cell-count names in the params_biology namespace."""
    applied = _apply_to(namespace, _BIOLOGY_NAMES)
    if applied:
        print(f"[transcriptomics_overrides] params_biology overrode {applied}")
    return applied


def apply_transitions(namespace):
    """Override rate / weight names in the params_transitions namespace."""
    applied = _apply_to(namespace, _TRANSITION_NAMES)
    if applied:
        print(f"[transcriptomics_overrides] params_transitions overrode {applied}")
    return applied
