#!/usr/bin/env python3
# combi3D/Simulation/run_sweep.py
#
# Drives a SMoRe ParS / sweep over the ABM from a manifest.json, using the
# per-run-directory layout (matching the original sweep, but with a validated
# JSON parameter file instead of a generated .py override):
#
#   sweep/runs/<run_id>/Simulation/   <- full copy of the Simulation code
#       + params.json                 <- THIS run's theta vector (validated)
#   sweep/outputs/<run_id>/           <- CC3D output (OUTSIDE the run dir;
#                                        CC3D rejects output under the .cc3d
#                                        parent directory)
#
# Each run reads its own params.json via param_loader (no reliance on the
# environment being passed through CC3D's launcher). The CC3D launch uses the
# real command:
#     <cc3d_python> -m cc3d.run_script --input=<run>/combi3D.cc3d \
#                   --output-dir=<out>
#
# Two modes:
#   --mode local : run sequentially in this process (PoC).
#   --mode slurm : emit a SLURM array script; submit by hand.

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SIM_FILES = [
    "combi3D.py",
    "combi3DSteppables.py",
    "solver3D.py",
    "variablevals3D.py",
    "params_grid.py",
    "params_biology.py",
    "params_transitions.py",
    "transcriptomics_overrides.py",
    "param_loader.py",
]


def load_manifest(path):
    with open(path) as f:
        man = json.load(f)
    if "runs" not in man or not isinstance(man["runs"], list):
        sys.exit(f"[run_sweep] manifest {path} has no 'runs' list")
    return man


def stage_run(run, sim_src, runs_dir, cc3d_src):
    rid = run["run_id"]
    run_dir = runs_dir / rid
    sim_dir = run_dir / "Simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in SIM_FILES if not (sim_src / f).exists()]
    if missing:
        sys.exit(f"[run_sweep] missing simulation files in {sim_src}: {missing}")
    for f in SIM_FILES:
        shutil.copy2(sim_src / f, sim_dir / f)

    with open(sim_dir / "params.json", "w") as f:
        json.dump({"run_id": rid, "params": run["params"]}, f, indent=2)

    if cc3d_src and cc3d_src.exists():
        shutil.copy2(cc3d_src, run_dir / "combi3D.cc3d")

    return rid, run_dir, sim_dir


def cc3d_command(cc3d_python, run_dir, out_dir):
    return [cc3d_python, "-m", "cc3d.run_script",
            f"--input={run_dir}/combi3D.cc3d",
            f"--output-dir={out_dir}"]


def run_local(staged, out_root, cc3d_python):
    out_root.mkdir(parents=True, exist_ok=True)
    for rid, run_dir, sim_dir in staged:
        out_dir = out_root / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = cc3d_command(cc3d_python, run_dir, out_dir)
        print(f"\n[run_sweep] === {rid} ===")
        print(f"[run_sweep] params: {sim_dir/'params.json'}")
        print(f"[run_sweep] cmd: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[run_sweep] {rid} FAILED (exit {e.returncode}). Stopping so "
                  f"the failure is visible.", file=sys.stderr)
            sys.exit(e.returncode)
        shutil.copy2(sim_dir / "params.json", out_dir / "params.json")
        print(f"[run_sweep] {rid} done -> {out_dir}")


def emit_slurm(staged, out_root, cc3d_python, account, work, cpus, hours):
    out_root.mkdir(parents=True, exist_ok=True)
    idx_file = work / "sweep_index.tsv"
    with open(idx_file, "w") as f:
        for rid, run_dir, sim_dir in staged:
            f.write(f"{rid}\t{run_dir}\t{sim_dir}\n")
    n = len(staged)
    account_line = f"#SBATCH --account={account}\n" if account else ""
    script = work / "sweep_array.sh"
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name=combi3d_sweep
#SBATCH --array=1-{n}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={hours:02d}:00:00
#SBATCH --partition=rome
{account_line}#SBATCH --output={work}/slurm_%A_%a.out

set -euo pipefail

# Sweep mode: float32 output + no PNG/plots (headless HPC). Set here so it
# always applies inside the job, regardless of the submitting shell's env.
# Override with --export on sbatch if you ever want the interactive behaviour.
export COMBI3D_SWEEP=1

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{idx_file}")
RID=$(echo "$LINE"  | cut -f1)
RUNDIR=$(echo "$LINE" | cut -f2)
SIMDIR=$(echo "$LINE" | cut -f3)
OUTDIR="{out_root}/$RID"
mkdir -p "$OUTDIR"

echo "[$RID] launching (COMBI3D_SWEEP=$COMBI3D_SWEEP)"
"{cc3d_python}" -m cc3d.run_script --input="$RUNDIR/combi3D.cc3d" --output-dir="$OUTDIR"

cp "$SIMDIR/params.json" "$OUTDIR/params.json"
echo "[$RID] done -> $OUTDIR"
""")
    script.chmod(0o755)
    print(f"[run_sweep] wrote {script}")
    print(f"[run_sweep] wrote {idx_file}  ({n} runs)")
    print(f"[run_sweep] submit with:  sbatch {script}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--mode", choices=["local", "slurm"], default="local")
    ap.add_argument("--cc3d-python", required=True,
                    help="Python interpreter of the CC3D env.")
    ap.add_argument("--sim-src", type=Path, default=HERE)
    ap.add_argument("--cc3d-file", type=Path, default=HERE.parent / "combi3D.cc3d")
    ap.add_argument("--runs-dir", type=Path, default=HERE.parent / "sweep" / "runs")
    ap.add_argument("--out", type=Path, default=HERE.parent / "sweep" / "outputs")
    ap.add_argument("--work", type=Path, default=HERE.parent / "sweep")
    ap.add_argument("--account", default=None)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--hours", type=int, default=12)
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    staged = [stage_run(run, args.sim_src, args.runs_dir, args.cc3d_file)
              for run in man["runs"]]
    print(f"[run_sweep] staged {len(staged)} run dir(s) under {args.runs_dir}")

    if args.mode == "local":
        run_local(staged, args.out, args.cc3d_python)
    else:
        emit_slurm(staged, args.out, args.cc3d_python, args.account,
                   args.work, args.cpus, args.hours)


if __name__ == "__main__":
    main()
