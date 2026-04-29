#!/bin/bash
#SBATCH --job-name=niivuerl-goalvec-mni
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/goalvec-mni-%j.out
#SBATCH --error=logs/goalvec-mni-%j.err

# MNI152-bridge variant of the goal-vector training job.
#
# Differences vs. train_goal_net.sbatch:
#   * Mixes a fraction of synthetic MNI152 samples (drawn from public/
#     t1_crop.nii.gz + data/mni152_landmarks.json) into each training batch
#     so the network sees the deployment volume's intensity distribution.
#   * Histogram-matches AOMIC patches into the MNI152 brain-interior range so
#     the AOMIC training signal is presented in MNI's intensity scale.
#   * Baseline architecture (no --hierarchical, no --wide-trunk) since the
#     report's smaller 7^3 model deployed better than the wide variant.
#   * 40 epochs at the baseline cadence.
#
# Usage (from repo root, after AOMIC data is staged):
#     mkdir -p logs
#     sbatch scripts/goal_vector/slurm/train_goal_net_mni.sbatch
#
# Override hyperparameters via environment variables:
#     EPOCHS=80 MNI_FRAC=0.5 SAVE_TAG=mni_v2 \
#       sbatch scripts/goal_vector/slurm/train_goal_net_mni.sbatch
#
# After the job finishes, run scripts/goal_vector/convert_to_tfjs.py against
# the saved directory to produce the browser-deployable TFJS model.

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
MNI_FRAC="${MNI_FRAC:-0.25}"
SAVE_TAG="${SAVE_TAG:-mni_$(date +%Y%m%d_%H%M%S)}"

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

# ---- Verify AOMIC data is pre-staged --------------------------------------
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
echo "[data] ${N_PRESENT} AOMIC subjects staged at ${DATA_DIR}"

# ---- Stage MNI152 volume + dump landmark centroids -----------------------
# The browser bundles public/ into the static site, so re-using it as the
# training source directly would couple training to the deployed copy. Mirror
# it under data/ so the trainer never reaches into public/ at run time.
MNI_VOLUME="${REPO_ROOT}/data/mni152.nii.gz"
MNI_LANDMARKS="${REPO_ROOT}/data/mni152_landmarks.json"
PUBLIC_VOLUME="${REPO_ROOT}/public/t1_crop.nii.gz"
if [[ ! -f "${PUBLIC_VOLUME}" ]]; then
    echo "ERROR: ${PUBLIC_VOLUME} missing; cannot stage MNI volume." >&2
    exit 1
fi
mkdir -p "${REPO_ROOT}/data"
cp -f "${PUBLIC_VOLUME}" "${MNI_VOLUME}"
echo "[mni] staged ${MNI_VOLUME} (from ${PUBLIC_VOLUME})"

# (Re)generate landmark JSON from src/lib/landmarks.ts so the trainer's coords
# always track the TS source of truth.
python scripts/goal_vector/dump_mni_landmarks.py \
    --ts-source "${REPO_ROOT}/src/lib/landmarks.ts" \
    --out "${MNI_LANDMARKS}"
echo "[mni] landmarks at ${MNI_LANDMARKS}"

# ---- Train -----------------------------------------------------------------
SAVE_DIR="${REPO_ROOT}/models/goal_vector_${SAVE_TAG}"
mkdir -p "${SAVE_DIR}"

echo "[train] save_dir=${SAVE_DIR}"
echo "[train] epochs=${EPOCHS} steps/epoch=${STEPS_PER_EPOCH} batch=${BATCH_SIZE}"
echo "[train] patch_radius=${PATCH_RADIUS} sample_radius=${SAMPLE_RADIUS} strides=[${STRIDES}]"
echo "[train] mni_frac=${MNI_FRAC} mni_volume=${MNI_VOLUME}"
echo "[train] architecture=baseline (no --hierarchical, no --wide-trunk)"

python scripts/goal_vector/train_goal_net.py \
    --patch-radius "${PATCH_RADIUS}" \
    --sample-radius "${SAMPLE_RADIUS}" \
    --batch-size "${BATCH_SIZE}" \
    --steps-per-epoch "${STEPS_PER_EPOCH}" \
    --epochs "${EPOCHS}" \
    --val-frac "${VAL_FRAC}" \
    --seed "${SEED}" \
    --strides ${STRIDES} \
    --mni-frac "${MNI_FRAC}" \
    --mni-volume "${MNI_VOLUME}" \
    --mni-landmarks-json "${MNI_LANDMARKS}" \
    --save-dir "${SAVE_DIR}"

echo "[done] model at ${SAVE_DIR}"
echo "[done] convert with: python scripts/goal_vector/convert_to_tfjs.py --src ${SAVE_DIR} --dst public/goal_vector_model_mni"
