#!/bin/bash
# Train on the public external VQA datasets (VQA-RAD + SLAKE combined) — no RefRad2D
# data, no SigLIP checkpoint. A self-contained example so anyone can run training
# end-to-end with just a GPU and the two public downloads. Uses PaliGemma-2's stock
# vision tower (load_siglip_weights=false). This is the MICCAI external-dataset setup.
#
# Setup:
#   1. Download SLAKE 1.0 and VQA-RAD images and set:
#        export SLAKE_ROOT=/path/to/Slake1.0
#        export VQA_RAD_ROOT=/path/to/VQA-RAD
#   2. bash run_train_public_external.sh       (or: sbatch --gres=gpu:1 --mem=50G ...)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

source "${REPO_ROOT}/.venv/bin/activate"
python -c "import torch; print('Torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

: "${SLAKE_ROOT:?Set SLAKE_ROOT to your downloaded SLAKE 1.0 directory (see README)}"
: "${VQA_RAD_ROOT:?Set VQA_RAD_ROOT to your downloaded VQA-RAD directory (see README)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_detectgemma.py --config "configs/train_public_vqa.json"
echo "job finished."
