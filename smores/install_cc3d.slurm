#!/bin/bash
#SBATCH --job-name=cc3d_install
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=install_%j.out

# Installer: miniconda + CompuCell3D (headless) + FiPy + calibration deps on
# scratch. The sweep job reuses the env it builds.
#
#   sbatch install_cc3d.slurm
#   tail -f install_<jobid>.out

set -euo pipefail

# ---- paths ----------------------------------------------------------------
SCRATCH=/gpfs/scratch1/shared/jkowalczuk
CONDA_ROOT=$SCRATCH/miniconda3
ENV_NAME=cc3d

echo "[install] host=$(hostname)  date=$(date)"
echo "[install] scratch=$SCRATCH"

# 1. miniconda (skip if already present) ------------------------------------
if [ ! -d "$CONDA_ROOT" ]; then
    echo "[install] downloading miniconda..."
    cd "$SCRATCH"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA_ROOT"
    rm -f miniconda.sh
else
    echo "[install] miniconda already at $CONDA_ROOT, reusing"
fi

source "$CONDA_ROOT/bin/activate"

# 2. CC3D env (headless: cc3d only, no cc3d-player5) -------------------------
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[install] env '$ENV_NAME' exists; reusing. Delete it to rebuild:"
    echo "          conda env remove -n $ENV_NAME"
else
    echo "[install] creating env '$ENV_NAME' with CC3D (this is the slow part)..."
    conda create -n "$ENV_NAME" -c conda-forge -c compucell3d \
        cc3d python=3.10 -y
fi

conda activate "$ENV_NAME"

# 3. FiPy + calibration deps (pip) ------------------------------------------
echo "[install] installing FiPy + calibration packages..."
pip install --no-input fipy scipy scikit-learn SALib

# 4. smoke test --------------------------------------------------------------
echo "[install] smoke test..."
python - <<'PYEOF'
import importlib, sys
mods = ["cc3d", "fipy", "numpy", "scipy", "sklearn", "SALib"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  [ok]  import {m}")
    except Exception as e:
        bad.append((m, str(e)))
        print(f"  [FAIL] import {m}: {e}")
# the sweep calls cc3d.run_script
try:
    import cc3d.run_script  # noqa
    print("  [ok]  cc3d.run_script importable")
except Exception as e:
    bad.append(("cc3d.run_script", str(e)))
    print(f"  [FAIL] cc3d.run_script: {e}")
sys.exit(1 if bad else 0)
PYEOF

echo ""
echo "[install] DONE. CC3D python interpreter is:"
echo "          $CONDA_ROOT/envs/$ENV_NAME/bin/python"
echo "[install] use that as --cc3d-python in run_sweep.py"
