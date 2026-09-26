# combi3D/Simulation/variablevals3D.py
#
# Aggregator module loaded by combi3D.cc3d as a Python <Resource>. It simply
# pulls every parameter name into one namespace so the .cc3d project (and any
# legacy code expecting a single import) has them all available. The actual
# values — including any per-run overrides from params.json — are resolved
# inside params_biology / params_transitions before this import runs.
#
# (Reconstructed from the supervisor's compiled .pyc, whose only contents were
# imports of these three modules.)

from params_grid import *          # noqa: F401, F403
from params_biology import *       # noqa: F401, F403
from params_transitions import *   # noqa: F401, F403
