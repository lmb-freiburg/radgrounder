#!/bin/bash
# Evaluate the segmentation model (PaliGemma-2 + frozen FT SigLIP, all datasets + segmentation).
#
# For SLURM clusters, submit with e.g.:
#   sbatch --gres=gpu:1 --mem=50G -p gpu run_eval_segment.sh
# or just run directly on a machine with a GPU:
#   bash run_eval_segment.sh
#
# Requirements:
#   - uv environment created at <repo>/.venv  (see README: "Environment setup")
#   - SLAKE_ROOT / VQA_RAD_ROOT pointing at the downloaded benchmarks (see README)
#   - LLM_JUDGE_MODEL set (HF id or local path) for --eval_llm_score
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

# Activate the uv environment.
source "${REPO_ROOT}/.venv/bin/activate"
python -c "import torch; print('Torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

# Model under test (override with: MODEL_PATH=/path/to/run bash run_eval_segment.sh)
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/segmentation}"

SEQ_LEN=200
DATASET_SIZE=2000
NORMALIZATION="medgemma"
BATCH=64

echo "Evaluating: ${MODEL_PATH}"

# ---- Open benchmarks (downloadable): SLAKE + VQA-RAD --------------------------
TESTED_DATASET="slake_vqa"
for QT in "vqa_open" "vqa_closed"; do
    python eval_groundedgemma.py --dataset_name "${TESTED_DATASET}" --model_path "${MODEL_PATH}" \
        --seq_len ${SEQ_LEN} --dataset_size ${DATASET_SIZE} --batch_size ${BATCH} \
        --body_part "ALL" --modality "all" --language "en" --question_types "${QT}" \
        --eval_llm_score --normalization "${NORMALIZATION}" \
        -n "tested_on_${TESTED_DATASET}_${QT}"
done

TESTED_DATASET="vqa_rad"
for QT in "vqa" "vqa_open" "vqa_closed"; do
    python eval_groundedgemma.py --dataset_name "${TESTED_DATASET}" --model_path "${MODEL_PATH}" \
        --seq_len ${SEQ_LEN} --dataset_size ${DATASET_SIZE} --batch_size ${BATCH} \
        --body_part "ALL" --modality "all" --language "en" --question_types "${QT}" \
        --eval_llm_score --normalization "${NORMALIZATION}" \
        -n "tested_on_${TESTED_DATASET}_${QT}"
done

# ---- RefRad2D (private; requires your own data — see docs/DATASET_FORMAT.md) ----
# for LANG in "en" "de"; do
#     python eval_groundedgemma.py --dataset_name "refrad2d_v18" --model_path "${MODEL_PATH}" \
#         --seq_len ${SEQ_LEN} --dataset_size ${DATASET_SIZE} --batch_size ${BATCH} \
#         --body_part "ALL" --modality "all" --language "${LANG}" --question_types "vqa" \
#         --eval_llm_score --normalization "${NORMALIZATION}" -n "refrad2d_vqa_${LANG}"
# done
# # Segmentation grounding (G-IoU) on the merged segmentation set:
# for LANG in "en" "de"; do
#     python eval_groundedgemma.py --dataset_name "refrad2d_segment_merged" --model_path "${MODEL_PATH}" \
#         --seq_len ${SEQ_LEN} --dataset_size ${DATASET_SIZE} --batch_size ${BATCH} \
#         --body_part "ALL" --modality "all" --language "${LANG}" --question_types "report" \
#         --only_segmented --selected_dataset "refrad2d_segment" \
#         --normalization "${NORMALIZATION}" -n "refrad2d_segment_report_${LANG}"
# done

echo "job finished."
