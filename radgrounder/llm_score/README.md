# LLMScore — LLM-as-judge metric

Standard n-gram metrics (CIDEr) penalize semantically-equivalent medical text (e.g.
"1.2 cm" vs "10 mm"). **LLMScore** instead uses an LLM judge (Gemma-3-27B) to compare a
generated report/answer against the ground truth, scoring clinical factuality and semantic
correctness on a 5-point scale normalized to [0, 1] with a short textual justification. (In
the paper, LLMScore reached Pearson r = 0.977 with the mean of three radiologists.)

## Files

| File | Purpose |
|---|---|
| `llm_score_server.py` | `LLMScoreServer` — client that sends (prediction, reference) pairs to the judge over HTTP and parses the score + reason |
| `start_gemma3_server.sh` | launches the judge as an OpenAI-compatible vLLM server |
| `score_system_prompt.md` | the judge's scoring rubric (1–5 scale) |
| `context_medical_report.json` | few-shot context for grading **report** generation |
| `context_medical_vqa.json` | few-shot context for grading **VQA** answers |

## Why a separate environment

vLLM conflicts with `transformers ≥ 4.54` (which the segmentation model needs) — both
register an `aimv2` config. So the judge runs in its **own** env (`.venv-judge`,
`requirements-judge.txt`) and the eval reaches it over HTTP:

```bash
# one-time
uv venv --python 3.10 .venv-judge
uv pip install --python .venv-judge -r requirements-judge.txt

# start the judge (separate shell / GPU)
export LLM_JUDGE_MODEL=google/gemma-3-27b-it      # HF id or local path
bash radgrounder/llm_score/start_gemma3_server.sh
```

Then run any eval with `--eval_llm_score`; it auto-detects the running server. Omit the flag
and no judge is needed.
