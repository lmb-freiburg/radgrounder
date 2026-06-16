"""Stage 3 — extract groundable anatomical keywords from report sentences with an LLM.

For every report sentence, an instruction-tuned LLM (we used ``gpt-oss-120b``) returns
the anatomical structures mentioned and the TotalSegmentator category each maps to.
The result is a JSON dict ``{sentence: {keyword: category}}`` that Stage 4 intersects
with the structures actually segmented in each slice.

The LLM is reached over an OpenAI-compatible HTTP endpoint, so any server works.
We used vLLM in a separate environment (see the repo README's LLM-judge note)::

    vllm serve gpt-oss-120b --async-scheduling      # serves http://localhost:8000

Then, once per language::

    python -m radgrounder.dataset.segmentation.grounding_pipeline.step3_extract_keywords \\
        --reports sentences_en.json --language english \\
        --model gpt-oss-120b --output-dir keywords/english

``--reports`` is either a JSON list of sentences or a parquet (give ``--text-column``).
Requests run concurrently; partial results are checkpointed and resumed, and a merged
``ExtractedKeywords_<language>_<n>_<timestamp>.json`` is written at the end.
"""

import argparse
import asyncio
import glob
import json
import os
from datetime import datetime

import aiohttp
from tqdm.asyncio import tqdm

PROMPT_FILE = {
    "english": "keywords_system_prompt_eng.md",
    "german": "keywords_system_prompt_de.md",
}
SAVE_PREFIX = "keywords_part"


def load_sentences(reports_path, text_column):
    """Load the list of report sentences from a JSON list or a parquet column."""
    if reports_path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(reports_path)
        if text_column is None:
            raise ValueError("--text-column is required when --reports is a parquet file")
        return df[text_column].dropna().astype(str).tolist()
    with open(reports_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Accept a {sentence: ...} or {...: sentence} dict; keys are the sentences.
        return list(data.keys())
    return list(data)


def extract_json(generated_text):
    """Parse the model's JSON array of {keyword, category} into {keyword: category}."""
    try:
        keywords_list = json.loads(generated_text)
        return {item["keyword"]: item["category"] for item in keywords_list}
    except (json.JSONDecodeError, KeyError, TypeError):
        print(f"Could not parse model output: {generated_text[:200]!r}")
        return {}


def already_processed(output_dir):
    """Sentences already present in checkpoint files, so a re-run resumes."""
    done = set()
    for path in glob.glob(os.path.join(output_dir, f"{SAVE_PREFIX}_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                done.update(json.load(f).keys())
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: could not read checkpoint {path}; skipping.")
    return done


def merge_checkpoints(output_dir, language):
    """Combine all checkpoint parts into one final keyword file."""
    merged = {}
    for path in glob.glob(os.path.join(output_dir, f"{SAVE_PREFIX}_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            merged.update(json.load(f))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(output_dir, f"ExtractedKeywords_{language}_{len(merged)}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(merged)} sentences -> {out_path}")
    return out_path


async def fetch(session, url, payload, system_prompt, sentence, semaphore):
    async with semaphore:
        payload = {**payload, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sentence},
        ]}
        try:
            async with session.post(url, headers={"Content-Type": "application/json"}, json=payload) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    content = response["choices"][0]["message"]["content"]
                    return sentence, extract_json(content)
                print(f"HTTP {resp.status} for: {sentence[:80]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"Request failed for {sentence[:80]!r}: {e}")
        return sentence, {}


async def run(sentences, args, system_prompt):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    payload = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0.0}
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    existing = glob.glob(os.path.join(output_dir, f"{SAVE_PREFIX}_*.json"))
    part = (max((int(p.rsplit("_", 1)[-1].split(".")[0]) for p in existing if p.rsplit("_", 1)[-1].split(".")[0].isdigit()), default=-1) + 1)

    buffer = {}
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [fetch(session, args.server_url, payload, system_prompt, s, semaphore) for s in sentences]
        for i, future in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting")):
            sentence, keywords = await future
            buffer[sentence] = keywords
            if (i + 1) % args.save_every == 0:
                _flush(buffer, output_dir, part)
                buffer, part = {}, part + 1
    if buffer:
        _flush(buffer, output_dir, part)

    merge_checkpoints(output_dir, args.language)


def _flush(buffer, output_dir, part):
    path = os.path.join(output_dir, f"{SAVE_PREFIX}_{part}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(buffer, f, ensure_ascii=False, indent=2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", required=True, help="JSON list of sentences, or a parquet (+ --text-column).")
    p.add_argument("--text-column", default=None, help="Column with the report sentence when --reports is a parquet.")
    p.add_argument("--language", required=True, choices=["english", "german"], help="Selects the system prompt.")
    p.add_argument("--output-dir", required=True, help="Where checkpoint parts + the merged keyword file are written.")
    p.add_argument("--server-url", default=os.environ.get("LLM_KEYWORD_SERVER_URL", "http://localhost:8000/v1/chat/completions"))
    p.add_argument("--model", default=os.environ.get("LLM_KEYWORD_MODEL", "gpt-oss-120b"), help="Model name the server serves.")
    p.add_argument("--concurrency", type=int, default=100, help="Max concurrent requests.")
    p.add_argument("--save-every", type=int, default=50000, help="Checkpoint every N completed sentences.")
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--timeout", type=int, default=600, help="Per-request timeout (seconds).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PROMPT_FILE[args.language])
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    sentences = load_sentences(args.reports, args.text_column)
    done = already_processed(args.output_dir)
    if done:
        print(f"Resuming: {len(done)} already processed.")
        sentences = [s for s in sentences if s not in done]
    print(f"Extracting keywords for {len(sentences)} {args.language} sentences.")
    if not sentences:
        merge_checkpoints(args.output_dir, args.language)
        return
    asyncio.run(run(sentences, args, system_prompt))


if __name__ == "__main__":
    main()
