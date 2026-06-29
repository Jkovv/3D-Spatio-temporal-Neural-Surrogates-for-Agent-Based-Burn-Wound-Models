from cc3d.core.PySteppables import *
from fipy import CellVariable, Grid3D
from solver3D import tester3D
from params_grid import (
    nx, ny, nz, dx, dy, dz,
    total_cytokines, total_celltypes, relaxationmcs,
)
from params_biology import (
    init_ec, init_n, init_m, init_f, init_my,
    replen_n, replen_f, replen_my, replen_m,
    lifespane, lifespannr, lifespanm, lifespanmr, lifespanf,
    divpre, divprnr, divprm, divprf, timeforgrowth,
)
from params_transitions import (
    sigmoida, sigmoidb,
    lnril8, lnril6, lnril1, lnrtnf, tnril10,
    lmril6, lmrtnf, tmril10,
    lm1il10, lftgf, tranril6,
    km, vmax,
)
import gc
import os
import random
import csv
import numpy as np
from numpy import *
from builtins import range

global mesh
global cytokines
global fullFileName

# 3D FiPy mesh 
mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)

# cell-presence indicator arrays
cellpresente    = CellVariable(name="cellpresente",    mesh=mesh, value=0.)
cellpresentndn  = CellVariable(name="cellpresentndn",  mesh=mesh, value=0.)
cellpresentna   = CellVariable(name="cellpresentna",   mesh=mesh, value=0.)
cellpresentm1   = CellVariable(name="cellpresentm1",   mesh=mesh, value=0.)
cellpresentm2   = CellVariable(name="cellpresentm2",   mesh=mesh, value=0.)

# cytokine fields (6): il8, il1, il6, il10, tnf, tgf
cytokines = [CellVariable(name="cytokine", mesh=mesh, value=0.0)
             for _ in range(total_cytokines)]

setlambda = 2000
setSaturationCoef = 10 ** -11

_TYPE_NAMES = {
    0:'Medium',
    1:'endothelial',
    2:'neutrophil',
    3:'monocyte',
    4:'fibroblast',
    5:'neutrophila',
    6:'neutrophilndn',
    7:'monocyter',
    8:'macrophage1',
    9:'macrophage2',
    10:'myofibroblast',
}

def safe_randopos(cell_field, nx, ny, nz):
    while True:
        x = random.randint(0, nx - 1)
        y = random.randint(0, ny - 1)
        z = random.randint(0, nz - 1)
        if cell_field[x, y, z] is None:
            return x, y, z

def sigmoid(x, a, b):
    z = np.exp(-a * (x - b))
    return 1.0 / (1.0 + z)

def modmm(x, km, vmax):
    return abs((vmax * x) / (km + x))

def age_hours(mcs, cell):
    return (mcs - cell.dict["born_mcs"]) / relaxationmcs

class endothelialSteppable(SteppableBasePy):
    def __init__(self, frequency=int(relaxationmcs)):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        global fullFileName

        self.scalarFieldil8 = self.create_scalar_field_py("il8")
        self.scalarFieldil1 = self.create_scalar_field_py("il1")
        self.scalarFieldil6 = self.create_scalar_field_py("il6")
        self.scalarFieldil10 = self.create_scalar_field_py("il10")
        self.scalarFieldtnf = self.create_scalar_field_py("tnf")
        self.scalarFieldtgf = self.create_scalar_field_py("tgf")

        _out_root = self.output_dir or os.path.dirname(os.path.abspath(__file__))
        datafiles_dir = os.path.join(_out_root, "datafiles")
        os.makedirs(datafiles_dir, exist_ok=True)
        fullFileName = os.path.join(datafiles_dir, "creatdoc.txt")
        fileHandle = open(fullFileName, "w")

        self.plot_win = self.add_new_plot_window(
            title='Cell counts',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='Cell types',
            x_scale_type='linear', y_scale_type='linear', grid=False,
            config_options={'legend': True})
        labels = ['Endothelial', 'Neutrophils', 'Monocytes', 'Fibroblast',
                  'Neutrophil A', 'ND Neutrophil', 'Monocyte R',
                  'Macrophage I', 'Macrophage II', 'Myofibroblast']
        colors = ["blue", "brown", "cyan", "violet", "red",
                  "pink", "yellow", "orange", "darkblue", "green"]
        [self.plot_win.add_plot(labels[i], style='Lines', color=colors[i])
         for i in range(len(colors))]

        self.cytokine_plot_win = self.add_new_plot_window(
            title='Mean cytokine concentrations',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='Mean concentration',
            x_scale_type='linear', y_scale_type='linear', grid=False,
            config_options={'legend': True})
        cyto_labels = ['IL-8', 'IL-1', 'IL-6', 'IL-10', 'TNF', 'TGF']
        cyto_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        [self.cytokine_plot_win.add_plot(cyto_labels[i], style='Lines', color=cyto_colors[i])
         for i in range(len(cyto_labels))]

        # Place initial cells
        for i in range(init_ec):
            x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
            self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.ENDOTHELIAL)

        for i in range(init_n):
            x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
            self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.NEUTROPHIL)

        for i in range(init_f):
            x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
            self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.FIBROBLAST)

        for i in range(init_my):
            x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
            self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.MYOFIBROBLAST)

        for i in range(init_m):
            x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
            self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.MONOCYTE)

        # targetVolume=3 requires three sequential shrinks (ΔE=40, 120, 200) making
        # spontaneous CPM death negligible over 10,000-MCS relaxation intervals.
        for cell in self.cell_list:
            cell.targetVolume = 3
            cell.lambdaVolume = 40.0

            if cell.type == 1: # ENDOTHELIAL
                cell.dict["born_mcs"] = 0
                cell.dict["span"] = lifespane
                cell.dict["divide"] = divpre
                cell.dict["dividepr"] = 1

            if cell.type == 2: # NEUTROPHIL
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "cytokine")
                cd.setLambda(setlambda)
                cd.setChemotactTowards("MEDIUM")
                cd.setSaturationCoef(setSaturationCoef)
                cell.dict["span"] = lifespannr
                cell.dict["dividepr"] = divprnr
                # Pre-age by 0..(span-1) hours: prevents any cell dying at the
                # first or second step() call (mcs=0 and mcs=relaxationmcs).
                cell.dict["born_mcs"] = -random.randint(0, lifespannr - 1) * relaxationmcs

            if cell.type == 3: # MONOCYTE
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "cytokine")
                cd.setLambda(setlambda)
                cd.setChemotactTowards("MEDIUM")
                cd.setSaturationCoef(setSaturationCoef)
                cell.dict["span"] = lifespanm
                cell.dict["dividepr"] = divprm
                cell.dict["born_mcs"] = -random.randint(0, lifespanm - 1) * relaxationmcs

            if cell.type == 4: # FIBROBLAST
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "cytokine")
                cd.setLambda(setlambda)
                cd.setChemotactTowards("MEDIUM")
                cd.setSaturationCoef(setSaturationCoef)
                cell.dict["span"] = lifespanf
                cell.dict["divide"] = random.randint(0, divprf)
                cell.dict["dividepr"] = divprf
                cell.dict["born_mcs"] = -random.randint(0, lifespanf - 1) * relaxationmcs

            if cell.type == 10: # MYOFIBROBLAST
                cell.dict["span"] = lifespanf
                cell.dict["divide"] = random.randint(0, divprf)
                cell.dict["dividepr"] = divprf
                cell.dict["born_mcs"] = -random.randint(0, lifespanf - 1) * relaxationmcs

        # PIFF dump: initial cell layout (MCS 0 only) 
        piff_path = os.path.join(
            os.path.dirname(os.path.dirname(fullFileName)), 'initial_cells.piff')
        with open(piff_path, 'w') as piff_f:
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        cell = self.cell_field[x, y, z]
                        if cell is not None:
                            tname = _TYPE_NAMES.get(cell.type, f'Type{cell.type}')
                            piff_f.write(
                                f'{tname} {cell.id} {x} {x} {y} {y} {z} {z}\n')

    def step(self, mcs):
        global cytokines
        global fullFileName

        ccount = np.zeros(total_celltypes + 1)

        # Replenishment - every 10 biological hours (matches 2D reference interval).
        # Re-initialises ALL existing cells as well as new arrivals, matching the
        # 2D reference behaviour where the replenishment loop re-randomises every
        # cell's age regardless of whether it is new or pre-existing.
        if mcs % (relaxationmcs * 10) == 0:
            for i in range(replen_n * 10):
                x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
                self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.NEUTROPHIL)

            for i in range(replen_f * 10):
                x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
                self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.FIBROBLAST)

            for i in range(replen_my * 10):
                x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
                self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.MYOFIBROBLAST)

            for i in range(replen_m * 10):
                x, y, z = safe_randopos(self.cell_field, nx, ny, nz)
                self.cell_field[x:x + 1, y:y + 1, z:z + 1] = self.new_cell(self.MONOCYTE)

            # Re-initialise ALL cells (new and existing) - matches 2D reference which
            # re-randomises every cell's age in this block unconditionally.
            for cell in self.cell_list:
                cell.targetVolume = 3
                cell.lambdaVolume = 40.0

                if cell.type == 1:
                    cell.dict["born_mcs"] = mcs
                    cell.dict["span"] = lifespane
                    cell.dict["divide"] = divpre
                    cell.dict["dividepr"] = 1

                if cell.type == 2:
                    cd = self.chemotaxisPlugin.addChemotaxisData(cell, "cytokine")
                    cd.setLambda(setlambda)
                    cd.setChemotactTowards("MEDIUM")
                    cd.setSaturationCoef(setSaturationCoef)
                    cell.dict["span"] = lifespannr
                    cell.dict["dividepr"] = divprnr
                    cell.dict["born_mcs"] = mcs - random.randint(0, lifespannr) * relaxationmcs

                if cell.type == 3:
                    cd = self.chemotaxisPlugin.addChemotaxisData(cell, "cytokine")
                    cd.setLambda(setlambda)
                    cd.setChemotactTowards("MEDIUM")
                    cd.setSaturationCoef(setSaturationCoef)
                    cell.dict["span"] = lifespanm
                    cell.dict["dividepr"] = divprm
                    cell.dict["born_mcs"] = mcs - random.randint(0, lifespanm) * relaxationmcs

                if cell.type == 4:
                    cd = self.chemotaxisPlugin.addChemotaxisData(cell, "cytokine")
                    cd.setLambda(setlambda)
                    cd.setChemotactTowards("MEDIUM")
                    cd.setSaturationCoef(setSaturationCoef)
                    cell.dict["span"] = lifespanf
                    cell.dict["divide"] = random.randint(0, divprf)
                    cell.dict["dividepr"] = divprf
                    cell.dict["born_mcs"] = mcs - random.randint(0, lifespanf) * relaxationmcs

                if cell.type == 10:
                    cell.dict["span"] = lifespanf
                    cell.dict["divide"] = random.randint(0, divprf)
                    cell.dict["dividepr"] = divprf
                    cell.dict["born_mcs"] = mcs - random.randint(0, lifespanf) * relaxationmcs

        # Build cell-presence arrays
        for cell in self.cell_list:
            xCOM = cell.xCOM
            yCOM = cell.yCOM
            zCOM = cell.zCOM

            if xCOM >= nx:
                xCOM = nx - 1
            if yCOM >= ny:
                yCOM = ny - 1
            if zCOM >= nz:
                zCOM = nz - 1

            pos = int(xCOM) + int(yCOM) * nx + int(zCOM) * nx * ny
            if pos >= nx * ny * nz:
                continue

            if cell.type == 1:
                cellpresente[pos] = 1.
            if cell.type == 6:
                cellpresentndn[pos] = 1.
            if cell.type == 5:
                cellpresentna[pos] = 1.
            if cell.type == 8:
                cellpresentm1[pos] = 1.
            if cell.type == 9:
                cellpresentm2[pos] = 1.

        # Solve PDEs
        cytokines = tester3D(mcs, cellpresente, cellpresentndn, cellpresentna,
                             cellpresentm1, cellpresentm2, cytokines, mesh)

        # Push cytokine arrays into CC3D scalar fields
        self.scalarFieldil8[:]  = np.reshape(cytokines[0], (nx, ny, nz), 'F')
        self.scalarFieldil1[:]  = np.reshape(cytokines[1], (nx, ny, nz), 'F')
        self.scalarFieldil6[:]  = np.reshape(cytokines[2], (nx, ny, nz), 'F')
        self.scalarFieldil10[:] = np.reshape(cytokines[3], (nx, ny, nz), 'F')
        self.scalarFieldtnf[:]  = np.reshape(cytokines[4], (nx, ny, nz), 'F')
        self.scalarFieldtgf[:]  = np.reshape(cytokines[5], (nx, ny, nz), 'F')

        fileDir = os.path.dirname(os.path.abspath(fullFileName))
        _lattice_dir = os.path.join(os.path.dirname(fileDir), "LatticeData")
        os.makedirs(_lattice_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(_lattice_dir, f"CytoStep_{mcs:07d}.npz"),
            il8  = np.array(cytokines[0]).reshape((nz, ny, nx)),
            il1  = np.array(cytokines[1]).reshape((nz, ny, nx)),
            il6  = np.array(cytokines[2]).reshape((nz, ny, nx)),
            il10 = np.array(cytokines[3]).reshape((nz, ny, nx)),
            tnf  = np.array(cytokines[4]).reshape((nz, ny, nx)),
            tgf  = np.array(cytokines[5]).reshape((nz, ny, nx)),
        )

        # Write cell-type lattice snapshot: uint8 array [z, y, x] where each
        # voxel holds the CC3D cell type ID (0 = Medium/empty).
        cell_field = self.cell_field
        cell_type_lattice = np.zeros((nz, ny, nx), dtype=np.uint8)
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    cell = cell_field[x, y, z]
                    if cell is not None:
                        cell_type_lattice[z, y, x] = cell.type
        np.savez_compressed(
            os.path.join(_lattice_dir, f"CellStep_{mcs:07d}.npz"),
            cell_type=cell_type_lattice,
        )

        # Reset presence arrays
        cellpresente[:]   = 0.
        cellpresentndn[:] = 0.
        cellpresentna[:]  = 0.
        cellpresentm1[:]  = 0.
        cellpresentm2[:]  = 0.

        # Per-cell cytokine sampling, state transitions, lifespan checks
        il8_list  = []
        il1_list  = []
        il6_list  = []
        il10_list = []
        tnf_list  = []
        tgf_list  = []

        for cell in self.cell_list:
            xCOM = int(cell.xCOM)
            yCOM = int(cell.yCOM)
            zCOM = int(cell.zCOM)

            if xCOM >= nx:
                xCOM = nx - 1
            if yCOM >= ny:
                yCOM = ny - 1
            if zCOM >= nz:
                zCOM = nz - 1

            ccil8  = self.scalarFieldil8[xCOM,  yCOM, zCOM]
            ccil1  = self.scalarFieldil1[xCOM,  yCOM, zCOM]
            ccil6  = self.scalarFieldil6[xCOM,  yCOM, zCOM]
            ccil10 = self.scalarFieldil10[xCOM, yCOM, zCOM]
            cctnf  = self.scalarFieldtnf[xCOM,  yCOM, zCOM]
            cctgf  = self.scalarFieldtgf[xCOM,  yCOM, zCOM]

            # Per-MCS concentration file
            fileDir = os.path.dirname(os.path.abspath(fullFileName))
            cytoname = fileDir + "/datafiles" + str(mcs) + "concentration.txt"
            if not os.path.exists(cytoname):
                with open(cytoname, 'w') as cytofile:
                    writer = csv.writer(cytofile)
                    writer.writerow(["mcsteps", "xCOM", "yCOM", "zCOM",
                                     "il8", "il1", "il6", "il10", "tnf", "tgf"])
            with open(cytoname, 'a') as cytofile:
                cytowriter = csv.writer(cytofile)
                cytowriter.writerow([str(mcs), str(xCOM), str(yCOM), str(zCOM),
                                     str(ccil8), str(ccil1), str(ccil6),
                                     str(ccil10), str(cctnf), str(cctgf)])

            il8_list.append(ccil8)
            il1_list.append(ccil1)
            il6_list.append(ccil6)
            il10_list.append(ccil10)
            tnf_list.append(cctnf)
            tgf_list.append(cctgf)

            # Age in biological hours - independent of step() call frequency
            cell_age = age_hours(mcs, cell)

            if cell_age > cell.dict["span"]:
                self.delete_cell(cell)
            else:
                if cell_age > timeforgrowth * cell.dict["span"]:
                    cell.targetVolume = 6  # double resting size (3 → 6)

                if cell.type == 2:  # neutrophil -> neutrophila
                    proba = (sigmoid(ccil8 * 10 ** 9, sigmoida, sigmoidb) * lnril8
                             + sigmoid(ccil6 * 10 ** 11, sigmoida, sigmoidb) * lnril6
                             + sigmoid(ccil1 * 10 ** 9, sigmoida, sigmoidb) * lnril1
                             + sigmoid(cctnf * 10 ** 9, sigmoida, sigmoidb) * lnrtnf
                             - sigmoid(ccil10 * 10 ** 12, sigmoida, sigmoidb) * tnril10)
                    if proba > random.random():
                        cell.type = 5

                if cell.type == 7:  # monocyter -> macrophage1
                    proba = (0.1
                             + 0.9 * sigmoid(ccil6 * 10 ** 11, sigmoida, sigmoidb) * lmril6
                             + sigmoid(cctnf * 10 ** 9, sigmoida, sigmoidb) * lmrtnf
                             - sigmoid(ccil10 * 10 ** 12, sigmoida, sigmoidb) * tmril10)
                    if proba > random.random():
                        cell.type = 8
                        cell.dict["born_mcs"] = mcs # reset age at transition
                        cell.dict["span"] = lifespanmr

                if cell.type == 8:  # macrophage1 -> macrophage2
                    proba = (0.1
                             + 0.9 * sigmoid(ccil10 * 10 ** 12, sigmoida, sigmoidb) * lm1il10)
                    if proba > random.random():
                        cell.type = 9
                        cell.dict["born_mcs"] = mcs   # reset age at transition
                        cell.dict["span"] = modmm(mcs, km, vmax)

                if cell.type == 4:  # fibroblast -> myofibroblast
                    proba = 0.1 * sigmoid(cctgf * 10 ** 10, sigmoida, sigmoidb) * lftgf
                    if proba > random.random():
                        cell.type = 10

                if cell.type == 5:  # neutrophila -> neutrophilndn
                    if random.randint(0, 1000) == 50:
                        cell.type = 6

                if cell.type == 3:  # monocyte -> monocyter
                    proba = (0.1
                             + 0.9 * sigmoid(ccil6 * 10 ** 11, sigmoida, sigmoidb) * tranril6)
                    if proba > random.random():
                        cell.type = 7
                        cell.dict["born_mcs"] = mcs # reset age at transition
                        cell.dict["span"] = lifespanmr

                ccount[cell.type] += 1

        # Plot
        labels = ['Endothelial', 'Neutrophils', 'Monocytes', 'Fibroblast',
                  'Neutrophil A', 'ND Neutrophil', 'Monocyte R',
                  'Macrophage I', 'Macrophage II', 'Myofibroblast']
        [self.plot_win.add_data_point(labels[i], mcs, ccount[i + 1])
         for i in range(len(labels))]

        fileDir = os.path.dirname(os.path.abspath(fullFileName))
        namer = fileDir + "/datafiles" + str(mcs) + ".png"
        if not os.path.exists(os.path.dirname(namer)):
            os.makedirs(os.path.dirname(namer))
        self.plot_win.save_plot_as_png(namer, 1200, 1200)

        countname = fileDir + "/cellcount.txt"
        if not os.path.exists(countname):
            with open(countname, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["mcsteps", "1", "2", "3", "4", "5",
                                 "6", "7", "8", "9", "10"])
        with open(countname, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([str(mcs)] + [str(ccount[i]) for i in range(1, 11)])

        il8_mean  = np.mean(il8_list)  if il8_list  else 0.0
        il1_mean  = np.mean(il1_list)  if il1_list  else 0.0
        il6_mean  = np.mean(il6_list)  if il6_list  else 0.0
        il10_mean = np.mean(il10_list) if il10_list else 0.0
        tnf_mean  = np.mean(tnf_list)  if tnf_list  else 0.0
        tgf_mean  = np.mean(tgf_list)  if tgf_list  else 0.0

        il8_std   = np.std(il8_list)   if il8_list  else 0.0
        il1_std   = np.std(il1_list)   if il1_list  else 0.0
        il6_std   = np.std(il6_list)   if il6_list  else 0.0
        il10_std  = np.std(il10_list)  if il10_list else 0.0
        tnf_std   = np.std(tnf_list)   if tnf_list  else 0.0
        tgf_std   = np.std(tgf_list)   if tgf_list  else 0.0

        cytodata = fileDir + "/mean_concentration.txt"
        if not os.path.exists(cytodata):
            with open(cytodata, 'w') as cytofile:
                cytowriter = csv.writer(cytofile)
                cytowriter.writerow(["meanconcen", "il8mean", "il1mean", "il6mean",
                                     "il10mean", "tnfmean", "tgfmean",
                                     "il8std", "il1std", "il6std",
                                     "il10std", "tnfstd", "tgfstd"])
        with open(cytodata, 'a') as cytofile:
            cytowriter = csv.writer(cytofile)
            cytowriter.writerow([str(mcs),
                                 str(il8_mean), str(il1_mean), str(il6_mean),
                                 str(il10_mean), str(tnf_mean), str(tgf_mean),
                                 str(il8_std), str(il1_std), str(il6_std),
                                 str(il10_std), str(tnf_std), str(tgf_std)])

        # Cytokine plot
        cyto_labels = ['IL-8', 'IL-1', 'IL-6', 'IL-10', 'TNF', 'TGF']
        cyto_means  = [il8_mean, il1_mean, il6_mean, il10_mean, tnf_mean, tgf_mean]
        [self.cytokine_plot_win.add_data_point(cyto_labels[i], mcs, cyto_means[i])
         for i in range(len(cyto_labels))]

        cyto_png = fileDir + "/cytokines" + str(mcs) + ".png"
        self.cytokine_plot_win.save_plot_as_png(cyto_png, 1200, 1200)

    def finish(self):
        pass

    def on_stop(self):
        return


class celldivisionSteppable(MitosisSteppableBase):

    def __init__(self, frequency=int(relaxationmcs)):
        MitosisSteppableBase.__init__(self, frequency)

    def step(self, mcs):
        cells_to_divide = []
        for cell in self.cell_list:
            cell_age = age_hours(mcs, cell)
            if (cell.volume >= 6  # doubled from resting targetVolume=3
                    and cell_age > timeforgrowth * cell.dict["span"]
                    and random.randint(0, 1000) == 5000):
                cells_to_divide.append(cell)

        for cell in cells_to_divide:
            self.divide_cell_along_major_axis(cell)

    def update_attributes(self):
        self.parent_cell.targetVolume = 3.0  # return to resting size after division
        # Reset parent age so it starts a fresh lifespan after division
        self.parent_cell.dict["born_mcs"] = self.parent_cell.dict.get("born_mcs", 0)
        self.clone_parent_2_child()
