from cc3d import CompuCellSetup
from params_grid import nx, ny, nz, relaxationmcs
from combi3DSteppables import endothelialSteppable, celldivisionSteppable

def _dump_resolved_params():
    import os, json
    outdir = os.environ.get("SMORE_OUTDIR")
    if not outdir:
        return
    try:
        from param_loader import get_overrides
        import params_biology as _b
        import params_transitions as _t
        resolved = {
            # cell counts (post-scaling, post-override)
            "init_ec": _b.init_ec, "init_n": _b.init_n, "init_m": _b.init_m,
            "init_f": _b.init_f, "init_my": _b.init_my,
            # a representative set of overridable rates/weights
            "keil8": _t.keil8, "km1il6": _t.km1il6, "km2il10": _t.km2il10,
            "km2tgf": _t.km2tgf, "lnril8": _t.lnril8, "sigmoidb": _t.sigmoidb,
        }
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "resolved_params.json"), "w") as f:
            json.dump({"overrides_from_json": get_overrides(),
                       "resolved_values": resolved}, f, indent=2)
    except Exception as e:
        print(f"[combi3D] could not dump resolved params: {e}")

_dump_resolved_params()

def configure_simulation(nx, ny, nz, relaxationmcs):
    from cc3d.core.XMLUtils import ElementCC3D

    xml3d = ElementCC3D("CompuCell3D", {"Revision": "20210612", "Version": "4.2.5"})

    # Metadata
    meta = xml3d.ElementCC3D("Metadata")
    meta.ElementCC3D("NumberOfProcessors", {}, 4)
    meta.ElementCC3D("DebugOutputFrequency", {}, relaxationmcs)

    # Potts
    potts = xml3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions", {"x": nx, "y": ny, "z": nz})
    potts.ElementCC3D("Steps", {}, 1000001)
    potts.ElementCC3D("Temperature", {}, 100.0)
    potts.ElementCC3D("NeighborOrder", {}, 1)
    potts.ElementCC3D("Boundary_x", {}, "Periodic")
    potts.ElementCC3D("Boundary_y", {}, "Periodic")
    potts.ElementCC3D("Boundary_z", {}, "Periodic")

    # Cell types - same 10 as 2D (endothelial frozen, plus Medium)
    cell_type_plugin = xml3d.ElementCC3D("Plugin", {"Name": "CellType"})
    cell_types = [
        (0,  "Medium"),
        (1,  "endothelial"),    # frozen
        (2,  "neutrophil"),
        (3,  "monocyte"),
        (4,  "fibroblast"),
        (5,  "neutrophila"),
        (6,  "neutrophilndn"),
        (7,  "monocyter"),
        (8,  "macrophage1"),
        (9,  "macrophage2"),
        (10, "myofibroblast"),
    ]
    frozen = {"endothelial"}
    for tid, tname in cell_types:
        if tname in frozen:
            cell_type_plugin.ElementCC3D("CellType",
                                         {"TypeId": tid, "TypeName": tname, "Freeze": ""})
        else:
            cell_type_plugin.ElementCC3D("CellType",
                                         {"TypeId": tid, "TypeName": tname})

    # Required geometry / tracking plugins
    xml3d.ElementCC3D("Plugin", {"Name": "CenterOfMass"})
    xml3d.ElementCC3D("Plugin", {"Name": "PixelTracker"})
    xml3d.ElementCC3D("Plugin", {"Name": "Volume"})

    # Contact energies - Medium↔* = 10, cell↔cell = 100
    contact = xml3d.ElementCC3D("Plugin", {"Name": "Contact"})
    mobile = ["neutrophil", "monocyte", "fibroblast", "neutrophila",
              "neutrophilndn", "monocyter", "macrophage1", "macrophage2",
              "myofibroblast"]
    all_types = ["endothelial"] + mobile

    contact.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "Medium"}, 10.0)
    for t in all_types:
        contact.ElementCC3D("Energy", {"Type1": "Medium", "Type2": t}, 10.0)
    for i, t1 in enumerate(all_types):
        for t2 in all_types[i:]:
            contact.ElementCC3D("Energy", {"Type1": t1, "Type2": t2}, 100.0)
    contact.ElementCC3D("NeighborOrder", {}, 1)

    xml3d.ElementCC3D("Plugin", {"Name": "NeighborTracker"})

    # ConnectivityGlobal - prevents cell fragmentation in 3D
    conn = xml3d.ElementCC3D("Plugin", {"Name": "ConnectivityGlobal"})
    for tname in mobile:
        conn.ElementCC3D("Penalty", {"Type": tname}, 10000000)

    # DiffusionSolverFE - zero diffusion/decay; FiPy handles PDE solving
    diff_solver = xml3d.ElementCC3D("Steppable", {"Type": "DiffusionSolverFE"})
    diff_field = diff_solver.ElementCC3D("DiffusionField", {"Name": "cytokine"})
    diff_data = diff_field.ElementCC3D("DiffusionData")
    diff_data.ElementCC3D("FieldName", {}, "cytokine")
    diff_data.ElementCC3D("GlobalDiffusionConstant", {}, 0.0)
    diff_data.ElementCC3D("GlobalDecayConstant", {}, 0)

    # Chemotaxis plugin
    chemo = xml3d.ElementCC3D("Plugin", {"Name": "Chemotaxis"})
    chemo.ElementCC3D("ChemicalField", {"Name": "cytokine"})

    CompuCellSetup.set_simulation_xml_description(xml3d)


configure_simulation(nx, ny, nz, relaxationmcs)

CompuCellSetup.register_steppable(
    steppable=endothelialSteppable(frequency=int(relaxationmcs)))
CompuCellSetup.register_steppable(
    steppable=celldivisionSteppable(frequency=int(relaxationmcs)))

CompuCellSetup.run()
