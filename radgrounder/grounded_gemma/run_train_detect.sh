#!/bin/bash
# Train the detection model (PaliGemma-2 + frozen FT SigLIP, report + VQA + detection).
#
# For SLURM clusters, submit with e.g.:
#   sbatch --gres=gpu:1 --mem=80G -p gpu run_train_detect.sh
# or run directly on a machine with a GPU:
#   bash run_train_detect.sh
#
# Requirements (see README + docs/DATASET_FORMAT.md):
#   - uv environment created at <repo>/.venv
#   - SIGLIP_CKPT_PATH set to your SigLIP checkpoint (or set siglip_model_path in the config)
#   - RefRad2D data env vars (REFRAD2D_DICOM_DIR, REFRAD2D_VQA_PARQUET, REFRAD2D_SPLIT_DIR,
#     REFRAD2D_SEGMENT_DIR) pointing at data laid out as in docs/DATASET_FORMAT.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

source "${REPO_ROOT}/.venv/bin/activate"
python -c "import torch; print('Torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_PATH="configs/radgrounder_detection.json"

python train_detectgemma.py --config "${CONFIG_PATH}"
echo "job finished."
