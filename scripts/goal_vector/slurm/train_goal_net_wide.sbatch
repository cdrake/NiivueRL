#!/bin/bash
#SBATCH --job-name=niivuerl-goalvec-wide
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=logs/goalvec-wide-%j.out
#SBATCH --error=logs/goalvec-wide-%j.err

# Wide-FOV variant of the goal-vector training job.
#
# Differences vs. train_goal_net.sbatch:
#   * patch_radius 3 -> 7 (15^3 patches; 10x more voxels per scale)
#   * strides=[1 2 4] (multi-scale: 15, 30, 60 voxel windows)
#   * hierarchical (one conv branch per scale, no shared weights)
#   * --wide-trunk (Conv -> MaxPool x2 -> Conv -> AvgPool per branch)
#   * 80 epochs x 600 steps to give the wider model time to fit
#   * 18h walltime, 48G mem
#   * sample_radius 80 so the larger patch sees genuinely off-target context
#     during training (otherwise the patch is dominated by target-adjacent
#     voxels and the network can shortcut to the target one-hot).
#
# Usage (from repo root, after data is staged):
#     mkdir -p logs
#     sbatch scripts/goal_vector/slurm/train_goal_net_wide.sbatch
#
# Override hyperparameters via environment variables:
#     EPOCHS=120 SAVE_TAG=wide_v2 \
#       sbatch scripts/goal_vector/slurm/train_goal_net_wide.sbatch
#
# After the job finishes, run scripts/goal_vector/convert_to_tfjs.py against
# the saved directory to produce the browser-deployable TFJS model.

set -euo pipefail

# ---- Tunables (override via environment) -----------------------------------
N_SUBJECTS="${N_SUBJECTS:-200}"
EPOCHS="${EPOCHS:-80}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-600}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PATCH_RADIUS="${PATCH_RADIUS:-7}"
SAMPLE_RADIUS="${SAMPLE_RADIUS:-80}"
STRIDES="${STRIDES:-1 2 4}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SEED="${SEED:-0}"
SAVE_TAG="${SAVE_TAG:-wide_$(date +%Y%m%d_%H%M%S)}"

# ---- Python + venv setup ---------------------------------------------------
ANACONDA_PY=/home/apps/anaconda/3.12/bin/python3
if [[ ! -x "${ANACONDA_PY}" ]]; then
    echo "ERROR: ${ANACONDA_PY} not visible from this node ($(hostname))." >&2
    echo "  Compute nodes may not share /home/apps; ask RC for an alternate path." >&2
    exit 1
fi

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
DATA_DIR="${REPO_ROOT}/data/aomic_id1000"
mkdir -p "${DATA_DIR}"
N_PRESENT=$(find "${DATA_DIR}" -mindepth 2 -name aseg.mgz 2>/dev/null | wc -l)
N_PRESENT=$((N_PRESENT))
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

echo "[train] save_dir=${SAVE_DIR}"
echo "[train] epochs=${EPOCHS} steps/epoch=${STEPS_PER_EPOCH} batch=${BATCH_SIZE}"
echo "[train] patch_radius=${PATCH_RADIUS} sample_radius=${SAMPLE_RADIUS} strides=[${STRIDES}]"
echo "[train] hierarchical=1 wide_trunk=1"

python scripts/goal_vector/train_goal_net.py \
    --patch-radius "${PATCH_RADIUS}" \
    --sample-radius "${SAMPLE_RADIUS}" \
    --batch-size "${BATCH_SIZE}" \
    --steps-per-epoch "${STEPS_PER_EPOCH}" \
    --epochs "${EPOCHS}" \
    --val-frac "${VAL_FRAC}" \
    --seed "${SEED}" \
    --strides ${STRIDES} \
    --hierarchical \
    --wide-trunk \
    --save-dir "${SAVE_DIR}"

echo "[done] model at ${SAVE_DIR}"
echo "[done] convert with: python scripts/goal_vector/convert_to_tfjs.py --src ${SAVE_DIR} --dst public/goal_vector_model_wide"
