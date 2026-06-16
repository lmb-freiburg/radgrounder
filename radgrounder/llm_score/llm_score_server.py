import datetime
import requests
import os
import asyncio
import aiohttp
import json
from tqdm.asyncio import tqdm
import glob
import time
import numpy as np
import hashlib
from collections import defaultdict

#run the server
#source start_gemma3_server.sh

class LLMScoreServer:
    def __init__(self, model_path=None, port=8050, concurrency_limit=100):
        if model_path is None:
            from radgrounder.paths import LLM_JUDGE_MODEL
            model_path = LLM_JUDGE_MODEL
        self.model_path = model_path
        self.port = port
        self.url = f"http://localhost:{self.port}/v1/chat/completions"
        self.health_url = f"http://localhost:{self.port}/health"
        self.concurrency_limit = concurrency_limit
        
        self.system_prompt_path = os.path.join(os.path.dirname(__file__), "score_system_prompt.md")
        context_vqa_path = os.path.join(os.path.dirname(__file__), "context_medical_vqa.json")
        context_report_path = os.path.join(os.path.dirname(__file__), "context_medical_report.json")

        with open(context_vqa_path, "r", encoding="utf-8") as f:
            self.context_vqa = json.load(f)
        self.context_vqa_by_score = defaultdict(list)
        for example in self.context_vqa:
            score = example.get("score", "unknown")
            self.context_vqa_by_score[score].append(example)
            
        with open(context_report_path, "r", encoding="utf-8") as f:
            self.context_report = json.load(f)
        self.context_report_by_score = defaultdict(list)
        for example in self.context_report:
            score = example.get("score", "unknown")
            self.context_report_by_score[score].append(example)
        
        with open(self.system_prompt_path, "r", encoding="utf-8") as f:
            self.base_system_prompt = f.read().strip()

    def check_server_status(self):
        try:
            response = requests.get(self.health_url)
            if response.status_code == 200:
                print(f"Server at {self.health_url} is running.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(f"Server at {self.health_url} is not running.")
        print("Please start the vLLM server with a command like:")
        print(f'vllm serve "{self.model_path}" --async-scheduling --port {self.port}')
        return False

    def _preprocess_fn(self, row_text, context_type="report"):
        hash_object = hashlib.sha256(row_text.encode('utf-8'))
        seed = int.from_bytes(hash_object.digest()[:4], 'big')
        rng = np.random.default_rng(seed=seed)
        context_examples = []
        context_source = self.context_vqa_by_score if context_type == "vqa" else self.context_report_by_score
        for score in sorted(context_source.keys()):
            examples = context_source[score]
            if not examples:
                continue
            idx = rng.integers(low=0, high=len(examples))
            context_examples.append(examples[idx])
        examples_to_add = "\n".join([json.dumps(example, ensure_ascii=False) for example in context_examples])
        # print(f"Examples to add: {examples_to_add}")
        system_prompt = self.base_system_prompt.replace("{context_examples}", examples_to_add)

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row_text}
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "seed": 42,
        }

    async def _fetch(self, session, headers, triplet, semaphore, context_type):
        async with semaphore:
            user_content = f"Question: {triplet['question']}\nCandidate: {triplet['prediction']}\nReference: {triplet['ground_truth']}"
            data = self._preprocess_fn(user_content, context_type=context_type)
            data["model"] = self.model_path

            try:
                async with session.post(self.url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        response = await resp.json()
                        output = response['choices'][0]['message']['content']
                        return {**triplet, "llm_score": output.strip()}
                    else:
                        error_message = f"Error: status code {resp.status}"
                        print(f"{error_message} for input: {triplet['question']}")
                        return {**triplet, "llm_score": error_message}
            except Exception as e:
                error_message = f"Error: {e}"
                print(f"An exception occurred for input '{triplet['question']}': {e}")
                return {**triplet, "llm_score": error_message}

    async def evaluate_async(self, questions: list[str], predictions: list[str], ground_truth: list[str], context_type: str = "report"):
        if not self.check_server_status():
            return None

        input_triplets = [
            {"question": q, "prediction": p, "ground_truth": gt}
            for q, p, gt in zip(questions, predictions, ground_truth)
        ]

        headers = {"Content-Type": "application/json"}
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout

        # Create tasks and keep their order
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self._fetch(session, headers, triplet, semaphore, context_type)
                for triplet in input_triplets
            ]
            # Gather preserves order
            results = await tqdm.gather(*tasks, desc="Processing LLM Scores")
        
        scores = []
        reasons = []
        for row in results:
            score_text = row.get('llm_score', '')
            reasons.append(score_text)
            try:
                # Extracts the first number from the score string
                score_val = float(score_text.split()[0])
                scores.append(score_val)
            except (ValueError, IndexError):
                scores.append(0)

        valid_scores = [s for s in scores if s != 0]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return mean_score, scores, reasons

    def evaluate(self, questions: list[str], predictions: list[str], ground_truth: list[str], context_type: str = "report"):
        return asyncio.run(self.evaluate_async(questions, predictions, ground_truth, context_type=context_type))

if __name__ == "__main__":
    llm_scorer = LLMScoreServer()

    input_data_path = os.path.join(os.path.dirname(__file__), "sample_llm_score_input.json")
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Input data file not found: {input_data_path}")

    with open(input_data_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    questions = [item["question"] for item in input_data]
    predictions = [item["prediction"] for item in input_data]
    ground_truth = [item["ground_truth"] for item in input_data]

    evaluation_results = llm_scorer.evaluate(questions, predictions, ground_truth, context_type="report")

    if evaluation_results:
        mean_score, scores, reasons = evaluation_results
        print(f"\n--- Evaluation Summary ---")
        print(f"Mean Score: {mean_score:.2f}")
        ref_scores = [item.get("score") for item in input_data if item.get("score") is not None]
        if ref_scores:
            ref_mean = sum(ref_scores) / len(ref_scores)
            print(f"Reference Mean Score: {ref_mean:.2f}")
        
        print("\n--- Detailed Results ---")
        for i, item in enumerate(input_data):
            print(f"Question: {item['question']}")
            print(f"Prediction: {item['prediction']}")
            print(f"Ground Truth: {item['ground_truth']}")
            ref_score = item.get("score")
            if ref_score is not None:
                diff = scores[i] - ref_score
                print(f"Score (AI | Reference | Δ): {scores[i]:.2f} | {ref_score:.2f} | {diff:+.2f}")
            else:
                print(f"Score: {scores[i]:.2f}")
            print(f"Reason: {reasons[i]}\n")
        
        # Save the detailed results to a file
        # os.makedirs(output_dir, exist_ok=True)
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_path = os.path.join(output_dir, f"llm_scores_results_{timestamp}.json")
        # with open(save_path, "w", encoding="utf-8") as f:
        #     json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        # print(f"Detailed results saved to {save_path}")
