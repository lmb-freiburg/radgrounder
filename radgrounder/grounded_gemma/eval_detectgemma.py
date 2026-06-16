import hashlib
from platform import processor
import random
import torch
from torch.utils.data import DataLoader # Add DataLoader import
from PIL import Image
from tqdm import tqdm # For progress bar
import pandas as pd
import argparse
from datasets import Dataset
import os
import functools # Import functools
import multiprocessing as mp # Import multiprocessing
from ovqa.metrics.ngram import NgramMetric
from ovqa.metrics.simple import compare_f1, get_metric_is_equal_default_prep, get_metric_recall_default_prep
from radgrounder.llm_score.llm_score_server import LLMScoreServer
from radgrounder.grounded_gemma.recalculate_refrad2d_vqa_metrics import calculate_refrad2dvqa_open_closed
import time # Add time import
from datetime import datetime # Add datetime import
import json
import re
import numpy as np

# Handle both direct execution and module import

from radgrounder.dataset.dataset_manager import DatasetManager
from radgrounder.dataset.image_preprocessing import NormalizationConfig, NormalizationType
from radgrounder.grounded_gemma.detect_vis_utils import visualize_and_save_detection
from radgrounder.grounded_gemma.g_iou.detect_grounding_evaluator import DetectGroundingEvaluator


# from dataset.vqa_dataset.refrad2d_vqa_dataset import RefRad2DVQA, get_collate_fn
import gc

ENABLE_VISUALIZATIONS = True
max_visualizations = 20


def save_detection_visualizations(batch_inputs, batch_prefixes, batch_predictions, batch_suffixes, batch_infos, 
                                 vis_output_dir, vis_count, max_visualizations, language="en"):
    """
    Save detection visualizations for a batch of samples.
    
    Args:
        batch_inputs: Batch inputs containing pixel_values
        batch_prefixes: List of prefix texts
        batch_predictions: List of prediction texts
        batch_suffixes: List of ground truth texts
        batch_infos: List of sample information dictionaries
        vis_output_dir: Directory to save visualizations
        vis_count: Current visualization count
        max_visualizations: Maximum number of visualizations to save
        
    Returns:
        int: Updated visualization count
    """
    if vis_count >= max_visualizations:
        return vis_count
        
    for i in range(min(len(batch_predictions), max_visualizations - vis_count)):
        try:
            # Extract image from batch (convert from tensor to numpy)
            image_tensor = batch_inputs["pixel_values"][i]  # Shape: [C, H, W]
            # Convert BFloat16 to Float32 to avoid numpy conversion issues
            image_np = image_tensor.float().permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C]
            
            # Handle grayscale images
            if image_np.shape[2] == 1:
                image_np = image_np.squeeze(2)  # Remove channel dimension for grayscale
            
            # Get prediction and ground truth text
            prefix = batch_prefixes[i].strip()
            pred_text = batch_predictions[i].strip()
            gt_text = batch_suffixes[i].strip()
            
            # Create visualization filename
            rsopid = batch_infos[i].get('rsopid', f'sample_{vis_count}')
            vis_filename = f"detection_vis_{language}_{vis_count:03d}_{rsopid}.png"
            vis_path = os.path.join(vis_output_dir, vis_filename)
            
            # Create visualization
            visualize_and_save_detection(
                image=image_np,
                pred_text=pred_text,
                gt_text=gt_text,
                save_path=vis_path,
                prefix=prefix,
                show_original=False,
            )
            
            vis_count += 1
            if vis_count >= max_visualizations:
                break
                
        except Exception as e:
            print(f"Error creating visualization {vis_count}: {e}")
            vis_count += 1  # Still increment to avoid infinite loop
            continue
    
    return vis_count

def load_config(config_path: str) -> dict:
    """Load training configuration from JSON"""
    with open(config_path, "r") as f:
        return json.load(f)

def load_and_evaluate_model(metrics, args):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

    # Construct path to the best checkpoint if the provided path is the parent directory
    if args.base_model:
        print("Evaluating base model without fine-tuning")
        run_name = args.model_path.split("/")[-3]
        model_path = args.model_path
    else:
        # Check if model_path is a checkpoint directory or a run directory
        path_obj = os.path.normpath(args.model_path)
        if os.path.basename(path_obj).startswith("checkpoint-"):
            # It's a specific checkpoint
            model_path = args.model_path
            run_dir = os.path.dirname(os.path.dirname(model_path))
            run_name = os.path.basename(run_dir)
        elif os.path.exists(os.path.join(args.model_path, "config.json")) and not os.path.isdir(
            os.path.join(args.model_path, "final_model")
        ):
            # model_path is itself a HF model directory (e.g. .../final_model)
            model_path = args.model_path
            run_dir = os.path.dirname(path_obj)
            run_name = os.path.basename(run_dir)
        else:
            # It's a run directory, look for final_model or best checkpoint
            run_name = os.path.basename(path_obj)
            final_model_path = os.path.join(args.model_path, "final_model")
            if os.path.exists(final_model_path):
                model_path = final_model_path
            else:
                # Look for checkpoints
                checkpoints_dir = os.path.join(args.model_path, "checkpoints")
                legacy_best_checkpoint_path = os.path.join(args.model_path, "checkpoint-best")
                if os.path.exists(checkpoints_dir):
                    checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")]
                    if checkpoints:
                        # Sort by step number
                        checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
                    model_path = os.path.join(checkpoints_dir, checkpoints[0])
                elif os.path.exists(legacy_best_checkpoint_path):
                    model_path = legacy_best_checkpoint_path
                else:
                    raise FileNotFoundError(f"No final_model or checkpoints found in {args.model_path}")
    
    print(f"Loading model from {model_path}")
    
    # Try to load config from run directory
    if not args.base_model:
        if os.path.basename(model_path) == "final_model":
            run_dir = os.path.dirname(model_path)
        elif "checkpoint-" in os.path.basename(model_path):
            run_dir = os.path.dirname(os.path.dirname(model_path))
        else:
            run_dir = args.model_path
            
        config_file = os.path.join(run_dir, "config.json")
        if os.path.exists(config_file):
            print(f"Loading config from {config_file}")
            config = load_config(config_file)
            
            # Update args with config values if not explicitly provided
            # Note: Command line args should take precedence if they were explicitly set
            # But since we don't know which were explicit, we'll use config values 
            # for key parameters if they exist in config
            
            if "image_size" in config:
                args.img_size = config["image_size"]
            if "max_length" in config:
                args.seq_len = config["max_length"]
            if "normalization" in config:
                args.normalization = config["normalization"]
            if "use_grounded_prompt" in config:
                args.use_grounded_prompt = config["use_grounded_prompt"]
                
            print("Updated evaluation arguments from config file.")
            print(f"Image size: {args.img_size}, Seq length: {args.seq_len}, Normalization: {args.normalization}")


    import torch._dynamo
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 128

    processor = PaliGemmaProcessor.from_pretrained(model_path, use_fast=True)

    token_ids = processor.tokenizer.convert_tokens_to_ids(["<p bbox=", "</p>", "id="])
    for token_id in token_ids:
        if token_id != processor.tokenizer.unk_token_id:
            print(f"Token ID {token_id} exists!")
            decoded_token = processor.tokenizer.decode(token_id)
            print(f"Decoded token: {decoded_token}")
        else:
            print(f"Token ID {token_id} not found (mapped to UNK).")

    # processor.tokenizer.add_special_tokens({'additional_special_tokens': ["<p bbox=", "</p>", "id="]})

    if args.language == "all":
        languages = ["english", "german"]
    elif args.language == "en":
        languages = ["english"]
    else:
        languages = ["german"]
        

    print(f"Evaluating model with languages: {languages}")
    print(f"Prefix options: {args.prefix_style}")
    for language in languages:
        print(f"Evaluating for language: {language}")
            # Load model with appropriate dtype and device mapping
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=args.torch_dtype, 
            device_map="auto" # Use device_map for potentially large models
        )
        # Explicitly disable cache for evaluation
        model.config.use_cache = False 
        print(f"Tokenizer vocab size: {len(processor.tokenizer)}")
        print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")
        print(f"Running evaluation on body part: {args.body_part}")
        model.resize_token_embeddings(len(processor.tokenizer))
        args.language = language
        dataset_size = args.dataset_size if hasattr(args, 'dataset_size') else None
        normalization_config = NormalizationConfig(strategy=NormalizationType(args.normalization))
        dataset_manager = DatasetManager(
            dataset_name=args.dataset_name,
            split="val",
            img_size=args.img_size,
            eval_mode=True,
            language=args.language,
            augment=False,
            dataset_size=dataset_size,
            max_length=args.seq_len,
            body_part=args.body_part,
            modality=args.modality,
            prefix_style=args.prefix_style,
            tokenizer=processor.tokenizer,
            selected_dataset=args.selected_dataset,
            question_types=args.question_types,
            only_segmented=args.only_segmented,
            use_grounded_prompt=args.use_grounded_prompt,
            normalization=normalization_config,
        )
        val_dataset = dataset_manager.dataset
        get_collate_fn = dataset_manager.get_collate_fn
        evaluate_model(model, val_dataset, get_collate_fn, processor, model_path, args.device, args.torch_dtype, args.seq_len, args.batch_size, run_name, metrics, args)



def evaluate_model(model, val_dataset, get_collate_fn, processor, model_path, DEVICE="cuda", TORCH_DTYPE=torch.bfloat16, SEQLEN=100, batch_size=4, run_name="", metrics={}, args=None):
    # results = []
    print("Model device", model.device)
    model.eval() # Set model to evaluation mode

    total_samples = len(val_dataset)
    print(f"Total samples in validation dataset: {total_samples}")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=get_collate_fn(processor, SEQLEN, TORCH_DTYPE, return_infos=True), num_workers=9)  # Use the sampled dataset for DataLoader

    generate_config = {
        "max_new_tokens": SEQLEN,
        "do_sample": False,
        # "use_cache": False,
        "num_beams": 1,
        # "repetition_penalty": 1.5,
    }

    start_time = time.time() # Start timer
    predictions = []
    prefixes = []
    suffixes = []
    infos = []
    remove_token_ids_list = [processor.tokenizer.pad_token_id, processor.tokenizer.eos_token_id]
    remove_token_list = processor.tokenizer.convert_ids_to_tokens(remove_token_ids_list)
    pattern = "|".join(map(re.escape, remove_token_list))
    
    # Create visualization directory
    from radgrounder.paths import OUTPUT_DIR
    vis_output_dir = os.path.join(str(OUTPUT_DIR), "detection_outputs", run_name, args.dataset_name, args.question_types)
    os.makedirs(vis_output_dir, exist_ok=True)
    vis_count = 0
    
    with torch.no_grad(): # Disable gradient calculations for inference
        # Iterate over batches from DataLoader
        for batch_inputs, batch_prefixes, batch_suffixes, batch_infos in tqdm(val_dataloader, desc="Validation Inference"):
            if batch_inputs is None: # Skip empty batches
                continue

            # Generate output for the batch
            # The inputs are already on the correct device and dtype from collate_fn
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].to(DEVICE)
            outputs = model.generate(**batch_inputs, **generate_config)

            # Decode predictions for each item in the batch
            input_len = batch_inputs["input_ids"].shape[1]
            # Ensure generated_ids are on CPU for decoding
            generated_ids = outputs[:, input_len:].cpu()
            batch_predictions = processor.batch_decode(generated_ids, skip_special_tokens=False)
            for i in range(len(batch_predictions)):
                pred_text = batch_predictions[i]
                pred_text = re.sub(pattern, "", pred_text)
                # Keep only the text after "Final Answer:" if present
                marker = "Final Answer:"
                idx = pred_text.find(marker)
                if idx != -1:
                    pred_text = pred_text[idx + len(marker):].strip()
                else:
                    pred_text = pred_text.strip()
                batch_predictions[i] = pred_text

            # Save visualizations for first 20 samples
            if vis_count < max_visualizations and ENABLE_VISUALIZATIONS:
                vis_count = save_detection_visualizations(
                    batch_inputs, batch_prefixes, batch_predictions, batch_suffixes, batch_infos,
                    vis_output_dir, vis_count, max_visualizations, language=args.language
                )
            # else:
            #     exit()
            
            # print(f"Batch predictions: {batch_predictions}")
            # print("batch_suffixes", batch_suffixes)
            prefixes.extend(batch_prefixes)
            suffixes.extend(batch_suffixes)
            predictions.extend(batch_predictions)
            infos.extend(batch_infos)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print visualization summary
    if vis_count > 0:
        print(f"Saved {vis_count} detection visualizations to {vis_output_dir}")
    else:
        print("No visualizations were saved")

    # print(f"Length prefixes: {len(prefixes)}, suffixes: {len(suffixes)}, predicitons {len(predictions)}, infos {len(infos)}")
    regex = r"<p bbox=.*?<seg\d+>>|</p>"
    cleaned_predictions = [re.sub(regex, "", pred) for pred in predictions]
    cleaned_suffixes = [re.sub(regex, "", gt) for gt in suffixes]
    # Calculate metrics
    metrics_dict = {}
    avg_metrics = {}
    # print(metrics)
    for metric_name, metric_func in metrics.items():
        if metric_name == "f1": 
            metrics_dict[metric_name] = [metric_func(cleaned_predictions[i], cleaned_suffixes[i]) for i in range(len(cleaned_predictions))]
            avg_metrics[metric_name] = sum(metrics_dict[metric_name]) / len(metrics_dict[metric_name])
        elif metric_name == "llm_score":
            context_type = args.question_types
            # pattern_to_remove_special_tokens = r"<\s*\/?\s*p\s*bbox\s*=\s*[^>]*>|<\s*\/?\s*p\s*id\s*=\s*[^>]*>"
            score, scores, reasons = metric_func.evaluate(prefixes, cleaned_predictions, cleaned_suffixes, context_type=context_type)
            avg_metrics[metric_name] = (score - 1) / 4
            metrics_dict[metric_name] = scores
            metrics_dict["llm_score_reasons"] = reasons
        elif metric_name == "grounding":
            results = metric_func.evaluate_dataset(predictions, suffixes)
            overall_metrics = results["overall_metrics"]
            detailed_results = results["detailed_results"]
            #overall metrics include "f1_score", "f1_at_05", "precision", "recall", "precision_at_05", "recall_at_05", "average_iou", "average_semantic_score", "average_grounding_iou"
            avg_metrics[metric_name] = overall_metrics
            metrics_dict[metric_name] = detailed_results
            
            keyword_results = metric_func.eval_keyword_metrics(predictions, suffixes)

            avg_metrics[metric_name].update(keyword_results)

        elif hasattr(metric_func, 'update') and hasattr(metric_func, 'compute'): # Check if it's an object like NgramMetric
                metric_func.reset() 
                metric_func.update(cleaned_predictions, cleaned_suffixes) 
                if metric_name == "accuracy" or metric_name == "recall":
                    scores = metric_func.compute_per_datapoint()
                    score = torch.mean(scores).item()
                    avg_metrics[metric_name] = score
                    metrics_dict[metric_name] = scores
                else:
                    avg_metrics[metric_name], metrics_dict[metric_name] = metric_func.compute_aggregated_and_per_datapoint() # Assuming compute gives the score
        else:
            # Fallback or error for unknown metric types
            print(f"Warning: Unknown metric type for {metric_name}")
            metrics_dict[metric_name] = None

    # Convert results to DataFrame and save
    results_df = pd.DataFrame({
        'prefixes': prefixes,
        'predictions': predictions,
        'suffixes': suffixes,
        **metrics_dict,
        "infos": infos
    })
    # Create directory if it doesn't exist
    from radgrounder.paths import VALIDATION_RESULTS_DIR
    output_dir = str(VALIDATION_RESULTS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    hashed_runname_code = hashlib.md5(run_name.encode()).hexdigest()
    hashed_file_name = f"{hashed_runname_code}_eval_{args.language}_{args.notes}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    output_csv_path = os.path.join(output_dir, hashed_file_name) # Add folder path
    results_df.to_csv(output_csv_path, index=False)

    print(f"Inference complete. Results saved to {output_csv_path}")

    # Save experiment results
    from radgrounder.paths import EXPERIMENT_RESULTS_JSON
    experiment_results_path = str(EXPERIMENT_RESULTS_JSON)
    os.makedirs(os.path.dirname(experiment_results_path), exist_ok=True)
    if os.path.exists(experiment_results_path):
        try:
            with open(experiment_results_path, 'r') as f:
                experiment_results = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {experiment_results_path}. Initializing with empty list.")
            experiment_results = [] # Initialize if file is corrupt or empty
    else:
        experiment_results = []


    end_time = time.time() # End timer
    validation_time = end_time - start_time
    hours, remainder = divmod(validation_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    results_log = { "run_name": run_name }
    
    if args.dataset_name == "refrad2d_vqa" and args.question_types == "vqa":
        open_closed_scores = calculate_refrad2dvqa_open_closed(metrics_dict, infos)
        results_log.update(open_closed_scores)
        
    for metric_name in avg_metrics:
        if metric_name == "grounding":
            for sub_metric, value in avg_metrics[metric_name].items():
                metric_key = f"{metric_name}_{sub_metric}"
                results_log[metric_key] = round(value, 4)
                print(f"Average {metric_key}: {value:.4f}")
        else:
            value = round(avg_metrics[metric_name], 4)
            results_log[metric_name] = value
            print(f"Average {metric_name}: {value:.4f}")

    # results_log["language_averaged"] = language_averaged
    results_log["run_hash"] = hashed_runname_code
    results_log["model_path"] = model_path # Add model path
    results_log["csv_path"] = os.path.join(os.path.basename(output_dir), hashed_file_name) # Save relative path
    results_log["dataset_name"] = args.dataset_name
    results_log["modality"] = args.modality
    results_log["question_types"] = args.question_types
    results_log['only_segmented'] = args.only_segmented
    results_log["val_dataset_size"] = total_samples
    results_log["body_part"] = args.body_part if hasattr(args, 'body_part') else None
    results_log["notes"] = args.notes
    results_log["prefix_style"] = args.prefix_style
    results_log["use_grounded_prompt"] = args.use_grounded_prompt
    results_log["language"] = args.language if hasattr(args, 'language') else "all"
    results_log["generate_config"] = generate_config
    results_log["normalization"] = args.normalization
    results_log["validation_time"] = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    results_log["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    experiment_results.append(results_log)

    with open(experiment_results_path, 'w') as f:
        json.dump(experiment_results, f, indent=4)
    print(f"Experiment results saved to {experiment_results_path}")

    if getattr(args, "use_wandb", False):
        import wandb
        print("Logging results to Weights & Biases")
        run = wandb.init(project="refrad2d-eval", name=run_name, notes=args.notes, config=results_log)
        for metric_name in avg_metrics:
            if metric_name == "grounding":
                for sub_metric, value in avg_metrics[metric_name].items():
                    metric_key = f"{metric_name}_{sub_metric}"
                    run.summary[metric_name] = results_log[metric_key]
            else:
                value = round(avg_metrics[metric_name], 4)
                run.summary[metric_name] = results_log[metric_name]

    return results_df


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="Path to the model checkpoint (run dir or final_model/)")
    parser.add_argument("--use_wandb", action="store_true", help="Log eval results to Weights & Biases (off by default; no login needed)")
    parser.add_argument("--eval_list_path", type=str, default="", help="Comma-separated list of model paths to evaluate")
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size for evaluation") # Added batch_size argument
    parser.add_argument("-n", "--notes", type=str, default="")
    parser.add_argument("-i", "--img_size", type=int, default=224, help="Image size for evaluation")
    parser.add_argument("-l", "--language", type=str, default="all", help="Language for evaluation (en, de, all)")
    parser.add_argument("-d", "--dataset_size", type=int, default=2000, help="Size of the dataset for evaluation")
    parser.add_argument("-m", "--model_id", type=str, default="google/paligemma2-3b-pt-224", help="Model ID for processor")
    parser.add_argument("-t", "--torch_dtype", type=str, default="bfloat16", help="Torch dtype for the model")
    parser.add_argument("-s", "--seq_len", type=int, default=100, help="Sequence length for evaluation")
    parser.add_argument("-po", "--prefix_style", type=str, default="both", help="Prefix style", choices=["none", "klinische_angaben", "fragestellung", "both", "random"])
    parser.add_argument("-bp", "--body_part", type=str, default="ALL", help="Body part to filter the dataset by", choices=["ALL", "ABDOMEN", "CHEST", "PELVIS"])
    parser.add_argument("-mod", "--modality", type=str, default="all", help="Modality to filter the dataset by", choices=["ct", "mr", "all"])
    parser.add_argument("--dataset_name", type=str, default="refrad2d_detect_merged", help="Name of the dataset to use for evaluation", choices=["refrad2d_detect_merged","refrad2d_v18", "slake_vqa", "vqa_rad", "refrad2d_detect", "refrad2d_detect_vqa"])
    parser.add_argument("--selected_dataset", type=str, default=None, help="Select a specific dataset to use", choices=[None, "refrad2d_detect_vqa", "refrad2d_detect", "external_dataset"])
    parser.add_argument("--question_types", type=str, default="report", help="Question types- just for logging its always report for detection")
    parser.add_argument("--only_segmented", action="store_true", help="Only include segmented images")
    parser.add_argument("--eval_llm_score", action="store_true", help="Whether to evaluate LLM Score")
    parser.add_argument("--base_model", action="store_true", help="Whether the model is the untrained base model")
    parser.add_argument(
        "--use_grounded_prompt",
        action="store_true",
        help="Use the grounded report generation prompt even if the dataset does not contain grounding labels.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=NormalizationType.DATASET_STATS.value,
        choices=[t.value for t in NormalizationType],
        help="Normalization strategy passed to the dataset manager.",
    )
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Check cuda availability
    args = parser.parse_args()
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics = {
        # "meteor": NgramMetric("METEOR"),
        "cider": NgramMetric("CIDEr"),
        "rouge": NgramMetric("ROUGE"),
        # "spice": NgramMetric("SPICE"),
        "bleu1": NgramMetric("Bleu_1"),
        "bleu4": NgramMetric("Bleu_4"),
        "f1": compare_f1,
        "accuracy": get_metric_is_equal_default_prep(),
        "recall": get_metric_recall_default_prep(),
        "grounding": DetectGroundingEvaluator()
    }
    
    if args.eval_llm_score:
        llm_scorer = LLMScoreServer()
        if llm_scorer.check_server_status():
            print("LLM Score server is running.")
            metrics["llm_score"] = llm_scorer
        else:
            raise RuntimeError(
                "LLM-as-judge server is not running. Start it in the judge env "
                "(see README 'LLM-as-judge'):\n"
                "    source .venv-judge/bin/activate\n"
                "    bash radgrounder/llm_score/start_gemma3_server.sh\n"
                "Or omit --eval_llm_score to skip the LLM metric."
            )

        
    args.device = DEVICE
    args.torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32

    if args.eval_list_path:
        with open(args.eval_list_path, 'r') as f:
            eval_list = f.read().splitlines()
    else:
        eval_list = [args.model_path]

    for model_path in eval_list:
        # try:
            args.model_path = model_path
            print(f"Evaluating model: {model_path}")
            load_and_evaluate_model(metrics, args)
        # except Exception as e:
        #     print(f"Error evaluating model {model_path}: {e}")
        #     continue





