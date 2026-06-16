#!/bin/bash
# Start the vLLM server for the LLM-as-judge used by --eval_llm_score.
#
# Runs in the SEPARATE judge environment (.venv-judge) — vLLM is incompatible with the
# main env's transformers version, so it lives on its own (see requirements-judge.txt).
# The eval process talks to this server over HTTP.
#
# Set LLM_JUDGE_MODEL to a HuggingFace id or local path (default google/gemma-3-27b-it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate the judge env if present (created via requirements-judge.txt).
if [ -f "${REPO_ROOT}/.venv-judge/bin/activate" ]; then
    source "${REPO_ROOT}/.venv-judge/bin/activate"
fi

MODEL="${LLM_JUDGE_MODEL:-google/gemma-3-27b-it}"
PORT="${LLM_JUDGE_PORT:-8050}"

vllm serve "${MODEL}" --async-scheduling --port "${PORT}"
