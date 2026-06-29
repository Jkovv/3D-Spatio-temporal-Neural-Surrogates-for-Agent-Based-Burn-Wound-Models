from fipy import CellVariable, Grid3D, TransientTerm, DiffusionTerm, LinearGMRESSolver, ImplicitSourceTerm
from params_grid import fipy_duration
from params_biology import (
    Dil8, Dil1, Dil6, Dil10, Dtnf, Dtgf,
    muil8, muil1, muil6, muil10, mutnf, mutgf,
)
from params_transitions import (
    keil8, kndnil8, thetanail8, knail1,
    km1il6, km2il10, knatnf, km1tnf, km2tgf,
)
import numpy as np


def tester3D(mcs, cellpresente, cellpresentndn, cellpresentna, cellpresentm1, cellpresentm2, cytokines, mesh):
    """
    Solves 6 coupled diffusion PDEs on a 3D FiPy mesh.
    No blood-border Dirichlet constraints.
    """
    il8  = CellVariable(mesh=mesh, value=cytokines[0])
    il1  = CellVariable(mesh=mesh, value=cytokines[1])
    il6  = CellVariable(mesh=mesh, value=cytokines[2])
    il10 = CellVariable(mesh=mesh, value=cytokines[3])
    tnf  = CellVariable(mesh=mesh, value=cytokines[4])
    tgf  = CellVariable(mesh=mesh, value=cytokines[5])

    mysolver = LinearGMRESSolver()

    eqil8  = TransientTerm() == DiffusionTerm(coeff=Dil8)  - ImplicitSourceTerm(muil8)       + keil8 * cellpresente   + kndnil8 * cellpresentndn - ImplicitSourceTerm(thetanail8 * cellpresentna)
    eqil1  = TransientTerm() == DiffusionTerm(coeff=Dil1)  - ImplicitSourceTerm(muil1)        + knail1 * cellpresentna
    eqil6  = TransientTerm() == DiffusionTerm(coeff=Dil6)  - ImplicitSourceTerm(muil6)        + km1il6 * cellpresentm1
    eqil10 = TransientTerm() == DiffusionTerm(coeff=Dil10) - ImplicitSourceTerm(muil10)       + km2il10 * cellpresentm1
    eqtnf  = TransientTerm() == DiffusionTerm(coeff=Dtnf)  - ImplicitSourceTerm(mutnf)        + knatnf * cellpresentna + km1tnf * cellpresentm1
    eqtgf  = TransientTerm() == DiffusionTerm(coeff=Dtgf)  - ImplicitSourceTerm(mutgf)        + km2tgf * cellpresentm2

    for i in range(fipy_duration):
        eqil8.solve(var=il8,   dt=1.0, solver=mysolver)
        eqil1.solve(var=il1,   dt=1.0, solver=mysolver)
        eqil6.solve(var=il6,   dt=1.0, solver=mysolver)
        eqil10.solve(var=il10, dt=1.0, solver=mysolver)
        eqtnf.solve(var=tnf,   dt=1.0, solver=mysolver)
        eqtgf.solve(var=tgf,   dt=1.0, solver=mysolver)

    # Clip negatives and warn
    for var, name in zip([il8, il1, il6, il10, tnf, tgf],
                         ["il8", "il1", "il6", "il10", "tnf", "tgf"]):
        arr = np.array(var)
        if np.any(arr < 0):
            print(f"WARNING MCS {mcs}: negative values in {name} "
                  f"({np.sum(arr < 0)} voxels). Clipping.")
            var.value = np.clip(var.value, 0, None)

    return (il8, il1, il6, il10, tnf, tgf)
