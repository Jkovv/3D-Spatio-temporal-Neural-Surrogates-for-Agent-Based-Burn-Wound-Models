# combi3D/Simulation/param_loader.py
#
# Central loader for per-run SMoRe ParS / sweep parameters.
#
# Mechanism (decided to match Jain et al. 2022 SMoRe ParS workflow, where each
# ABM trajectory must be uniquely paired with its sampled theta_ABM vector):
#
#   1. The sweep driver writes ONE JSON file per run containing that run's
#      parameter vector (a single "runs[i].params" dict from the manifest).
#   2. Before launching CompuCell3D, the driver sets the environment variable
#         SMORE_PARAMS=/abs/path/to/run_XXXX_params.json
#   3. params_biology.py and params_transitions.py both call get_overrides(),
#      which reads that JSON exactly once (cached) and returns the vector.
#   4. The same JSON is copied next to the run's LatticeData/ output so that
#      preprocessing and SMoRe ParS can recover theta_ABM directly from disk.
#
# Design guarantees (these are the things the old single-file star-import
# mechanism did NOT give, and which caused runs to silently fall back to
# baseline):
#   - FAIL LOUD on unknown parameter names (typo / manifest mismatch -> crash,
#     never a silent no-op).
#   - FAIL LOUD on non-finite / out-of-type values.
#   - LOG CLEARLY when no SMORE_PARAMS is set (baseline run) so an unintended
#     baseline cannot masquerade as a sampled run.
#   - Read the JSON ONCE; both override modules see an identical vector.

import json
import os
import sys

# Whitelist of parameters that are allowed to be overridden.
# A name not in this set is rejected (catches manifest/code drift early).
# "kind" controls post-processing:
#     "int_count" -> rounded and cast to int (cell counts feed range()).
#     "float"     -> used as-is.
_ALLOWED = {
    # cytokine production / secretion rates (params_transitions)
    "keil8":   "float",
    "kndnil8": "float",
    "thetanail8": "float",
    "knail1":  "float",
    "km1il6":  "float",
    "km2il10": "float",
    "knatnf":  "float",
    "km1tnf":  "float",
    "km2tgf":  "float",
    # sigmoid weights / thresholds (params_transitions)
    "lnril8":  "float",
    "lnril6":  "float",
    "lnril1":  "float",
    "lnrtnf":  "float",
    "tnril10": "float",
    "lmril6":  "float",
    "lmrtnf":  "float",
    "tmril10": "float",
    "lm1il10": "float",
    "lftgf":   "float",
    "tranril6": "float",
    "sigmoida": "float",
    "sigmoidb": "float",
    # initial cell counts (params_biology) - MUST be int, they feed range()
    "init_ec": "int_count",
    "init_n":  "int_count",
    "init_m":  "int_count",
    "init_f":  "int_count",
    "init_my": "int_count",
    # replenishment counts (params_biology) - also feed integer loops
    "replen_n":  "int_count",
    "replen_f":  "int_count",
    "replen_my": "int_count",
    "replen_m":  "int_count",
}

_ENV_VAR = "SMORE_PARAMS"
_cache = None  # parsed {name: value} after type coercion
_loaded = False # whether we've attempted a load this process


def _coerce(name, raw):
    kind = _ALLOWED[name]
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"[param_loader] parameter '{name}' = {raw!r} is not numeric")
    if val != val or val in (float("inf"), float("-inf")):
        raise ValueError(
            f"[param_loader] parameter '{name}' = {raw!r} is not finite")
    if kind == "int_count":
        iv = int(round(val))
        if iv < 1:
            print(f"[param_loader] WARNING: '{name}' rounded to {iv}; "
                  f"flooring to 1", file=sys.stderr)
            iv = 1
        return iv
    return val


def _resolve_param_path():
    env_path = os.environ.get(_ENV_VAR)
    if env_path:
        return env_path, f"${_ENV_VAR}"
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
    if os.path.isfile(local):
        return local, "local params.json"
    return None, "none"


def _load():
    global _cache, _loaded
    if _loaded:
        return _cache
    _loaded = True

    path, source = _resolve_param_path()
    if not path:
        print(f"[param_loader] no {_ENV_VAR} and no local params.json -> "
              f"BASELINE run (no parameter overrides applied).", file=sys.stderr)
        _cache = {}
        return _cache

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[param_loader] parameter file {path!r} (from {source}) does not exist"
            f"Refusing to run on baseline for what was meant to be a sweep run.")

    with open(path) as f:
        doc = json.load(f)

    if "params" in doc and isinstance(doc["params"], dict):
        raw_params = doc["params"]
        run_id = doc.get("run_id", "?")
    else:
        raw_params = doc
        run_id = "?"

    unknown = sorted(set(raw_params) - set(_ALLOWED))
    if unknown:
        raise KeyError(
            f"[param_loader] unknown parameter name(s) in {path}: {unknown}. "
            f"Allowed: {sorted(_ALLOWED)}")

    parsed = {name: _coerce(name, raw_params[name]) for name in raw_params}
    _cache = parsed

    print(f"[param_loader] loaded {len(parsed)} override(s) for run "
          f"'{run_id}' from {path}", file=sys.stderr)
    for k in sorted(parsed):
        print(f"[param_loader]   {k} = {parsed[k]}", file=sys.stderr)
    return _cache


def get_overrides():
    return _load()


def apply_to(namespace, allowed_subset):
    """
    Inject overrides into a module namespace (the dict returned by globals()).
    """
    ov = get_overrides()
    subset = set(allowed_subset)
    applied = []
    for name, value in ov.items():
        if name in subset:
            namespace[name] = value
            applied.append(name)
    return applied