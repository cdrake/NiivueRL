#!/bin/bash
#SBATCH --job-name=niivuerl-goalvec
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/goalvec-%j.out
#SBATCH --error=logs/goalvec-%j.err

# Train the goal-vector network on Theia HPC.
#
# Usage (from repo root):
#     mkdir -p logs
#     sbatch scripts/goal_vector/slurm/train_goal_net.sbatch
#
# Override hyperparameters via environment variables:
#     EPOCHS=40 STRIDES="1 4" SAVE_TAG=multiscale_v1 \
#       sbatch scripts/goal_vector/slurm/train_goal_net.sbatch
#
# Override the SLURM account/partition via sbatch flags:
#     sbatch -A my_account -p gpu-a100 scripts/goal_vector/slurm/train_goal_net.sbatch
#
# What this script does:
#   1. Loads the cluster's Python + CUDA modules.
#   2. Creates (or reuses) a venv at .venv-theia/ with the same packages as
#      the local .venv-tfjs (tensorflow 2.19, tf_keras 2.19, nibabel, tensorflowjs).
#   3. Downloads N_SUBJECTS AOMIC ID1000 subjects to data/aomic_id1000/
#      if they're not already there (idempotent; skips existing files).
#   4. Runs scripts/goal_vector/train_goal_net.py with HPC-scaled hyperparameters.
#   5. Saves model + metadata.json under models/goal_vector_${SAVE_TAG}/.
#
# After the job finishes, run scripts/goal_vector/convert_to_tfjs.py on the
# saved directory to produce the browser-deployable TFJS model.

set -euo pipefail

# ---- Tunables (override via environment) -----------------------------------
N_SUBJECTS="${N_SUBJECTS:-200}"
EPOCHS="${EPOCHS:-40}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PATCH_RADIUS="${PATCH_RADIUS:-3}"
SAMPLE_RADIUS="${SAMPLE_RADIUS:-50}"
STRIDES="${STRIDES:-1}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SEED="${SEED:-0}"
HIERARCHICAL="${HIERARCHICAL:-0}"
SAVE_TAG="${SAVE_TAG:-theia_$(date +%Y%m%d_%H%M%S)}"

# ---- Python + venv setup ---------------------------------------------------
# Theia HPC (USC RC). The anaconda module loads on the login node only; the
# compute nodes can't resolve "python3/anaconda/3.12". The install itself
# lives on shared filesystem, so we call it by absolute path. The pip-
# installed `tensorflow[and-cuda]` bundles its own CUDA + cuDNN compatible
# with the node's NVIDIA driver, so no CUDA module is needed.
ANACONDA_PY=/home/apps/anaconda/3.12/bin/python3
if [[ ! -x "${ANACONDA_PY}" ]]; then
    echo "ERROR: ${ANACONDA_PY} not visible from this node ($(hostname))." >&2
    echo "  Compute nodes may not share /home/apps; ask RC for an alternate path." >&2
    exit 1
fi

# Slurm copies the batch script to a spool dir, so $0 doesn't resolve to the
# repo. Use $SLURM_SUBMIT_DIR (the directory where `sbatch` was called from)
# and fall back to the path-based resolution for direct shell testing.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${REPO_ROOT}"

VENV="${REPO_ROOT}/.venv-theia"
if [[ ! -d "${VENV}" ]]; then
    echo "[setup] creating venv at ${VENV} from ${ANACONDA_PY}"
    "${ANACONDA_PY}" -m venv "${VENV}"
    "${VENV}/bin/pip" install --upgrade pip
    "${VENV}/bin/pip" install \
        "tensorflow[and-cuda]==2.19.*" \
        "tf_keras==2.19.*" \
        "nibabel==5.4.*" \
        "numpy==2.1.*" \
        "awscli"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

echo "[env] python: $(which python)  ($(python --version))"
echo "[env] tf: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
python -c "import tensorflow as tf; print('[env] GPUs visible:', tf.config.list_physical_devices('GPU'))"

# ---- Verify data is pre-staged --------------------------------------------
# Theia compute nodes have unreliable outbound network, so we don't try to
# download AOMIC from S3 here. Run scripts/goal_vector/download.py on the
# login node first; this job just asserts the staged subjects are present.
DATA_DIR="${REPO_ROOT}/data/aomic_id1000"
mkdir -p "${DATA_DIR}"
N_PRESENT=$(find "${DATA_DIR}" -mindepth 2 -name aseg.mgz 2>/dev/null | wc -l)
N_PRESENT=$((N_PRESENT))  # normalize whitespace + force integer
if [ "${N_PRESENT}" -lt "${N_SUBJECTS}" ]; then
    echo "ERROR: only ${N_PRESENT}/${N_SUBJECTS} subjects staged at ${DATA_DIR}." >&2
    echo "  Pre-stage on the login node before sbatch:" >&2
    echo "    cd ${REPO_ROOT} && .venv-theia/bin/python scripts/goal_vector/download.py --n ${N_SUBJECTS} --workers 8" >&2
    exit 1
fi
echo "[data] ${N_PRESENT} subjects staged at ${DATA_DIR}"

# ---- Train -----------------------------------------------------------------
SAVE_DIR="${REPO_ROOT}/models/goal_vector_${SAVE_TAG}"
mkdir -p "${SAVE_DIR}"

EXTRA_ARGS=()
if [[ "${HIERARCHICAL}" == "1" ]]; then
    EXTRA_ARGS+=(--hierarchical)
fi

echo "[train] save_dir=${SAVE_DIR}"
echo "[train] epochs=${EPOCHS} steps/epoch=${STEPS_PER_EPOCH} batch=${BATCH_SIZE} strides=[${STRIDES}] hierarchical=${HIERARCHICAL}"

python scripts/goal_vector/train_goal_net.py \
    --patch-radius "${PATCH_RADIUS}" \
    --sample-radius "${SAMPLE_RADIUS}" \
    --batch-size "${BATCH_SIZE}" \
    --steps-per-epoch "${STEPS_PER_EPOCH}" \
    --epochs "${EPOCHS}" \
    --val-frac "${VAL_FRAC}" \
    --seed "${SEED}" \
    --strides ${STRIDES} \
    --save-dir "${SAVE_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "[done] model at ${SAVE_DIR}"
echo "[done] convert with: python scripts/goal_vector/convert_to_tfjs.py --src ${SAVE_DIR} --dst public/goal_vector_model"
