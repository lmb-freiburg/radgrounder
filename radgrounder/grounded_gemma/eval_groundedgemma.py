from platform import processor
import torch
from torch.utils.data import DataLoader # Add DataLoader import
from PIL import Image
from tqdm import tqdm # For progress bar
import pandas as pd
import argparse
import os
import functools # Import functools
import multiprocessing as mp # Import multiprocessing
from radgrounder.grounded_gemma.recalculate_refrad2d_vqa_metrics import calculate_refrad2dvqa_open_closed
from ovqa.metrics.ngram import NgramMetric
from ovqa.metrics.simple import compare_f1, get_metric_is_equal_default_prep, get_metric_recall_default_prep
from radgrounder.llm_score.llm_score_server import LLMScoreServer
import time # Add time import
from datetime import datetime # Add datetime import
import json
import re
from segment_vis_utils import visualize_and_save_segmentation
import numpy as np
import random

from radgrounder.grounded_gemma.g_iou.segment_grounding_evaluator import SegmentGroundingEvaluator
from radgrounder.dataset.dataset_manager import DatasetManager
from radgrounder.dataset.image_preprocessing import NormalizationConfig, NormalizationType
from radgrounder.dataset.segmentation.refrad2d_segment import SEG_START, SEG_END

# from dataset.vqa_dataset.refrad2d_vqa_dataset import RefRad2DVQA, get_collate_fn
import gc

def load_config(config_path: str) -> dict:
    """Load training configuration from JSON"""
    with open(config_path, "r") as f:
        return json.load(f)

def load_and_evaluate_model(metrics, args):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from radgrounder.grounded_gemma.modeling_groundedgemma import GroundedGemmaForConditionalGeneration

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
            if os.path.exists(checkpoints_dir):
                checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    # Sort by step number
                    checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
                    model_path = os.path.join(checkpoints_dir, checkpoints[0])
                else:
                    raise ValueError(f"No checkpoints found in {checkpoints_dir}")
            else:
                raise ValueError(f"No checkpoints directory found in {args.model_path}")
    
    print(f"Loading model from {model_path}")
    
    # Try to load config from run directory
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
        
        # Update args with config values
        if "image_size" in config:
            args.img_size = config["image_size"]
        if "max_length" in config:
            args.seq_len = config["max_length"]
        if "normalization" in config:
            args.normalization = config["normalization"]

            
        print("Updated evaluation arguments from config file.")
        print(f"Image size: {args.img_size}, Seq length: {args.seq_len}, Normalization: {args.normalization}")
        

    import torch._dynamo
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 128

    
    processor = PaliGemmaProcessor.from_pretrained(model_path, use_fast=True)

    token_ids = processor.tokenizer.convert_tokens_to_ids([SEG_START, SEG_END])
    for token_id in token_ids:
        if token_id != processor.tokenizer.unk_token_id:
            print(f"Token ID {token_id} exists!")
            decoded_token = processor.tokenizer.decode(token_id)
            print(f"Decoded token: {decoded_token}")
        else:
            print(f"Token ID {token_id} not found (mapped to UNK).")

    seg_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_END)
    if seg_token_id == processor.tokenizer.unk_token_id:
        raise ValueError(f"Error: {SEG_END} token not found in tokenizer vocabulary The model cannot segment without it!")


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
        model = GroundedGemmaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=args.torch_dtype, 
            device_map="auto" # Use device_map for potentially large models
        )
        # Explicitly disable cache for evaluation
        model.config.use_cache = False 
        print(f"Tokenizer vocab size: {len(processor.tokenizer)}")
        print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")
        print(f"Running evaluation on body part: {args.body_part}")
        # model.resize_token_embeddings(len(processor.tokenizer))
        model.set_seg_token_id(seg_token_id)
        
        
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
            only_segmented=args.only_segmented,
            selected_dataset=args.selected_dataset,
            question_types=args.question_types,
            tokenizer=processor.tokenizer,
            add_other_vqa_datasets=args.add_other_vqa_datasets,
            normalization=normalization_config,
        )
        val_dataset = dataset_manager.dataset
        get_collate_fn = dataset_manager.get_collate_fn
        
        
        evaluate_model(model, val_dataset, get_collate_fn, processor, model_path, args.device, args.torch_dtype, args.seq_len, args.batch_size, run_name, metrics, args)


def organize_segmentation_masks(outputs, segmentation_logits_and_pos, segment_gt, seg_token_pos_gt):
    pred_logits = [[] for _ in range(len(outputs))]
    for seg_logits, seg_token_pos in segmentation_logits_and_pos:
        for logit, (batch_idx, token_pos) in zip(seg_logits, seg_token_pos):
            pred_logits[batch_idx].append(logit)
    
    segment_gt_for_samples = [[] for _ in range(len(outputs))]
    for seg_gt, seg_token_pos in zip(segment_gt, seg_token_pos_gt):
        batch_idx, token_pos = seg_token_pos
        segment_gt_for_samples[batch_idx].append(seg_gt.unsqueeze(0))
        
    return pred_logits, segment_gt_for_samples

def _tensor_list_to_numpy(mask_list, threshold=0.5):
    """Convert a list of mask tensors to a stacked numpy array."""
    if not mask_list:
        return None

    masks = []
    for mask in mask_list:
        mask_tensor = mask.detach().cpu().float()
        if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.squeeze(0)
        elif mask_tensor.ndim > 3:
            mask_tensor = mask_tensor.squeeze()
        mask_np = mask_tensor.numpy()
        if threshold is not None:
            mask_np = (mask_np > threshold).astype(np.float32)
        masks.append(mask_np)

    if not masks:
        return None
    try:
        return np.stack(masks, axis=0)
    except ValueError:
        # Fallback in case masks have inconsistent shapes
        return None


def collect_segmentation_samples(sample_store, max_samples, global_start_idx, pred_seg_logits,
                                 segment_gt_for_samples, batch_inputs, batch_prefixes,
                                 batch_predictions, batch_suffixes):
    """Collect segmentation samples for visualization after evaluation."""
    if max_samples <= 0 or len(sample_store) >= max_samples:
        return

    available_slots = max_samples - len(sample_store)
    num_to_collect = min(len(batch_predictions), available_slots)

    for local_idx in range(num_to_collect):
        image_tensor = batch_inputs["pixel_values"][local_idx].detach().cpu().permute(1, 2, 0).float()
        image_np = image_tensor.numpy()

        pred_masks_np = _tensor_list_to_numpy(pred_seg_logits[local_idx], threshold=0.5)
        gt_masks_np = _tensor_list_to_numpy(segment_gt_for_samples[local_idx], threshold=None)

        sample_store.append({
            "index": global_start_idx + local_idx,
            "image": image_np,
            "prefix": str(batch_prefixes[local_idx]) if batch_prefixes is not None else "",
            "prediction": batch_predictions[local_idx],
            "suffix": batch_suffixes[local_idx],
            "pred_masks": pred_masks_np,
            "gt_masks": gt_masks_np,
        })


def save_visualization_outputs(samples, llm_scores, grounding_details, run_name, args):
    """Save segmentation visualizations annotated with key metrics."""
    if not samples:
        return

    llm_scores = llm_scores or []
    grounding_details = grounding_details or []

    from radgrounder.paths import OUTPUT_DIR
    seg_out_base = os.path.join(str(OUTPUT_DIR), "segmentation_outputs")
    if args.selected_dataset != "None":
        save_dir = f"{seg_out_base}/{run_name}/{args.selected_dataset}/{args.question_types}/{args.language}"
    else:
        save_dir = f"{seg_out_base}/{run_name}/{args.dataset_name}/{args.question_types}/{args.language}"
    os.makedirs(save_dir, exist_ok=True)

    for sample in samples:
        sample_idx = sample["index"]
        llm_score = None
        if sample_idx < len(llm_scores):
            llm_score = llm_scores[sample_idx]

        miou = None
        grounding_iou = None
        if sample_idx < len(grounding_details):
            miou = grounding_details[sample_idx].get("sample_iou")
            grounding_iou = grounding_details[sample_idx].get("sample_grounding_iou")

        metrics_payload = {}
        if llm_score is not None:
            metrics_payload["LLMScore"] = (float(llm_score) - 1) / 4 * 100
        if miou is not None:
            metrics_payload["mIoU"] = float(miou) * 100
        if grounding_iou is not None:
            metrics_payload["G-IoU"] = float(grounding_iou) * 100

        filename = f"sample_{sample_idx:04d}.png"
        save_path = os.path.join(save_dir, filename)

        visualize_and_save_segmentation(
            sample["image"],
            sample["pred_masks"],
            sample["gt_masks"],
            pred_text=sample["prediction"],
            gt_text=sample["suffix"],
            prefix=sample.get("prefix", ""),
            metrics=metrics_payload,
            save_path=save_path,
        )


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

    remove_token_ids_list = [processor.tokenizer.pad_token_id, processor.tokenizer.eos_token_id]
    remove_token_list = processor.tokenizer.convert_ids_to_tokens(remove_token_ids_list)
    remove_pattern = "|".join(map(re.escape, remove_token_list))
    with torch.no_grad(): # Disable gradient calculations for inference
        # Iterate over batches from DataLoader
        if "segment" in args.dataset_name:
            prefixes, suffixes, predictions, infos, grounding_results, visualization_samples = seg_dataset_eval_loop(model, val_dataloader, processor, generate_config, DEVICE, run_name, remove_pattern, metrics, args)
        else:
            prefixes, suffixes, predictions, infos = other_datasets_eval_loop(model, val_dataloader, processor, generate_config, DEVICE, run_name, remove_pattern, args)
            visualization_samples = []

    print(f"Completed inference on {len(predictions)} samples.")
    del model
    gc.collect()
    torch.cuda.empty_cache()


    regex = r"<.*?>|</.*?>"

    cleaned_predictions = [re.sub(regex, "", pred) for pred in predictions]
    cleaned_suffixes = [re.sub(regex, "", gt) for gt in suffixes]
    # print(f"Length prefixes: {len(prefixes)}, suffixes: {len(suffixes)}, predicitons {len(predictions)}, infos {len(infos)}")
    # Calculate metrics
    metrics_dict = {}
    avg_metrics = {}
    # print(metrics)
    for metric_name, metric_func in metrics.items():
        if metric_name == "f1": 
            metrics_dict[metric_name] = [metric_func(cleaned_predictions[i], cleaned_suffixes[i]) for i in range(len(cleaned_predictions))]
            avg_metrics[metric_name] = sum(metrics_dict[metric_name]) / len(metrics_dict[metric_name])
        elif metric_name == "llm_score":
            result = metric_func.evaluate(prefixes, cleaned_predictions, cleaned_suffixes)
            if result is None:
                raise RuntimeError(
                    "LLM-as-judge server is not running. Start it in the judge env "
                    "(see README 'LLM-as-judge'):\n"
                    "    source .venv-judge/bin/activate\n"
                    "    bash radgrounder/llm_score/start_gemma3_server.sh\n"
                    "Or omit --eval_llm_score to skip the LLM metric."
                )
            score, scores, reasons = result
            avg_metrics[metric_name] = (score - 1) / 4
            metrics_dict[metric_name] = scores
            metrics_dict["llm_score_reasons"] = reasons
        elif metric_name == "grounding" and "segment" in args.dataset_name:
            final_results = metric_func.generate_dataset_report(grounding_results)
            avg_metrics[metric_name] = final_results["overall_metrics"]
            metrics_dict[metric_name] = final_results["detailed_results"]
            keyword_metrics = metric_func.eval_keyword_metrics(predictions, suffixes)
            avg_metrics.update(keyword_metrics)
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

    try:
        if visualization_samples:
            llm_scores_per_sample = metrics_dict.get("llm_score")
            grounding_details = metrics_dict.get("grounding")
            save_visualization_outputs(visualization_samples, llm_scores_per_sample, grounding_details, run_name, args)
    except Exception as e:
        print(f"Error saving visualization outputs: {e}")
        
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
    if len(run_name) > 100:
        run_name = run_name[:100]  # Truncate if too long
        
    file_name = f"{run_name}_eval_{args.language}_{args.notes}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    output_csv_path = os.path.join(output_dir, file_name) # Add folder path
    results_df.to_csv(output_csv_path, index=False)

    print(f"Evaluation completed. Results saved to {output_csv_path}")

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
    # for metric_name in avg_metrics:
    #     results_log[metric_name] = round(avg_metrics[metric_name], 4)
        
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
            
    if args.dataset_name == "refrad2d_vqa" and args.question_types == "vqa":
        open_closed_scores = calculate_refrad2dvqa_open_closed(metrics_dict, infos)
        results_log.update(open_closed_scores)

    # results_log["language_averaged"] = language_averaged
    results_log["model_path"] = model_path # Add model path
    results_log["csv_path"] = os.path.join(os.path.basename(output_dir), file_name) # Save relative path
    results_log["dataset_name"] = args.dataset_name
    results_log["dataset"] = args.dataset_name
    results_log["selected_dataset"] = args.selected_dataset
    results_log["question_types"] = args.question_types
    results_log["only_segmented"] = args.only_segmented
    results_log["val_dataset_size"] = total_samples
    results_log["body_part"] = args.body_part if hasattr(args, 'body_part') else None
    results_log["notes"] = args.notes
    results_log["prefix_style"] = args.prefix_style
    results_log["language"] = args.language if hasattr(args, 'language') else "all"
    results_log["generate_config"] = generate_config
    results_log["validation_time"] = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    results_log["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    experiment_results.append(results_log)

    with open(experiment_results_path, 'w') as f:
        json.dump(experiment_results, f, indent=4)
    print(f"Experiment results saved to {experiment_results_path}")

    if getattr(args, "use_wandb", False):
        import wandb
        run = wandb.init(project="refrad2d-eval", name=run_name, notes=args.notes, config=results_log)
        for metric_name in avg_metrics:
            if metric_name == "grounding":
                for sub_metric, value in avg_metrics[metric_name].items():
                    metric_key = f"{metric_name}_{sub_metric}"
                    run.summary[metric_key] = results_log[metric_key]
            else:
                value = round(avg_metrics[metric_name], 4)
                run.summary[metric_name] = results_log[metric_name]

    return results_df


def seg_dataset_eval_loop(model, val_dataloader, processor, generate_config, DEVICE, run_name, remove_pattern, metrics, args):
    predictions = []
    prefixes = []
    suffixes = []
    infos = []
    visualization_samples = []
    max_visualizations = getattr(args, "num_visualization_samples", 0)
    global_sample_idx = 0
    grounding_evaluator = metrics["grounding"] if "grounding" in metrics else None
    grounding_results = []
    
    for batch_inputs, batch_prefixes, batch_suffixes, batch_infos, segment_gt, seg_token_pos_gt, mask_labels_gt in tqdm(val_dataloader, desc="Validation Inference"):
        if batch_inputs is None: # Skip empty batches
            continue

        # Generate output for the batch
        # The inputs are already on the correct device and dtype from collate_fn
        for key in batch_inputs:
            if isinstance(batch_inputs[key], torch.Tensor):
                batch_inputs[key] = batch_inputs[key].to(DEVICE)
        outputs, segmentation_logits_and_pos = model.generate(**batch_inputs, **generate_config)

        # Decode predictions for each item in the batch
        input_len = batch_inputs["input_ids"].shape[1]
        # Ensure generated_ids are on CPU for decoding
        generated_ids = outputs[:, input_len:].cpu()
        batch_predictions = processor.batch_decode(generated_ids, skip_special_tokens=False)
        #remove special tokens except segmentation tokens
        for i in range(len(batch_predictions)):
            batch_predictions[i] = re.sub(remove_pattern, "", batch_predictions[i])


        pred_seg_logits, segment_gt_for_samples = organize_segmentation_masks(outputs, segmentation_logits_and_pos, segment_gt, seg_token_pos_gt)

        if max_visualizations > 0 and len(visualization_samples) < max_visualizations:
            collect_segmentation_samples(
                visualization_samples,
                max_visualizations,
                global_sample_idx,
                pred_seg_logits,
                segment_gt_for_samples,
                batch_inputs,
                batch_prefixes,
                batch_predictions,
                batch_suffixes,
            )
        
        if grounding_evaluator is not None:
            grounding_evaluator.evaluate_batch(
                batch_predictions, batch_suffixes, pred_seg_logits, segment_gt_for_samples, grounding_results
            )

        # print(f"Batch predictions: {batch_predictions}")
        prefixes.extend(batch_prefixes)
        suffixes.extend(batch_suffixes)
        predictions.extend(batch_predictions)
        infos.extend(batch_infos)
        global_sample_idx += len(batch_predictions)
        
    return prefixes, suffixes, predictions, infos, grounding_results, visualization_samples

def other_datasets_eval_loop(model, val_dataloader, processor, generate_config, DEVICE, run_name, remove_pattern, args):
    predictions = []
    prefixes = []
    suffixes = []
    infos = []
    for batch_inputs, batch_prefixes, batch_suffixes, batch_infos in tqdm(val_dataloader, desc="Validation Inference"):
        if batch_inputs is None: # Skip empty batches
            continue

        # Generate output for the batch
        # The inputs are already on the correct device and dtype from collate_fn
        for key in batch_inputs:
            if isinstance(batch_inputs[key], torch.Tensor):
                batch_inputs[key] = batch_inputs[key].to(DEVICE)
        outputs, segmentation_logits_and_pos = model.generate(**batch_inputs, **generate_config)

        # Decode predictions for each item in the batch
        input_len = batch_inputs["input_ids"].shape[1]
        # Ensure generated_ids are on CPU for decoding
        generated_ids = outputs[:, input_len:].cpu()
        batch_predictions = processor.batch_decode(generated_ids, skip_special_tokens=False)
        #remove special tokens except segmentation tokens
        for i in range(len(batch_predictions)):
            batch_predictions[i] = re.sub(remove_pattern, "", batch_predictions[i])

        # print(f"Batch predictions: {batch_predictions}")
        prefixes.extend(batch_prefixes)
        suffixes.extend(batch_suffixes)
        predictions.extend(batch_predictions)
        infos.extend(batch_infos)
        
    return prefixes, suffixes, predictions, infos


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="Path to the model checkpoint (run dir or final_model/)")
    parser.add_argument("--eval_list_path", type=str, default="", help="Comma-separated list of model paths to evaluate")
    parser.add_argument("-esp", "--experiment_save_path", type=str, default="", help="Path to save the experiment results (defaults to RADGROUNDER_RESULTS_JSON)")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, help="Batch size for evaluation") # Added batch_size argument
    parser.add_argument("-n", "--notes", type=str, default="")
    parser.add_argument("-i", "--img_size", type=int, default=224, help="Image size for evaluation")
    parser.add_argument("-l", "--language", type=str, default="all", help="Language for evaluation (en, de, all)")
    parser.add_argument("-d", "--dataset_size", type=int, default=10000, help="Size of the dataset for evaluation")
    parser.add_argument("-m", "--model_id", type=str, default="google/paligemma2-3b-pt-224", help="Model ID for processor")
    parser.add_argument("-c", "--cache_dir", type=str, default="./models/paligemma-3b", help="Cache directory for the model")
    parser.add_argument("-t", "--torch_dtype", type=str, default="bfloat16", help="Torch dtype for the model")
    parser.add_argument("-s", "--seq_len", type=int, default=100, help="Sequence length for evaluation")
    parser.add_argument("-po", "--prefix_style", type=str, default="both", help="Prefix style", choices=["none", "klinische_angaben", "fragestellung", "both", "random"])
    parser.add_argument("-bp", "--body_part", type=str, default="ALL", help="Body part to filter the dataset by", choices=["ALL", "ABDOMEN", "CHEST", "PELVIS"])
    parser.add_argument("-mod", "--modality", type=str, default="all", help="Modality to filter the dataset by", choices=["ct", "mr", "all"])
    parser.add_argument("--dataset_name", type=str, default="refrad2d_segment_merged", help="Name of the dataset to use for evaluation")
    parser.add_argument("--question_types", type=str, default="report", help="Question types- just for logging its always report for detection")
    parser.add_argument("--only_segmented", action="store_true", help="Use only segmented images for evaluation")
    parser.add_argument("--selected_dataset", type=str, default="None", help="Name of the selected dataset for evaluation from the merged dataset", choices=["refrad2d_segment_vqa", "refrad2d_segment"])
    parser.add_argument("--add_other_vqa_datasets", action="store_true", help="Add VQA-RAD and SLAKE VQA datasets to the training")
    parser.add_argument("--eval_llm_score", action="store_true", help="Compute the LLM-as-judge metric (requires the vLLM judge server, see README)")
    parser.add_argument("--use_wandb", action="store_true", help="Log eval results to Weights & Biases (off by default; no login needed)")
    parser.add_argument("--num_visualization_samples", type=int, default=64, help="Number of segmentation examples to save after metrics are computed")
    parser.add_argument(
        "--normalization",
        type=str,
        default=NormalizationType.DATASET_STATS.value,
        choices=[t.value for t in NormalizationType],
        help="Normalization strategy passed to the dataset manager.",
    )
    
    MODEL_ID = "google/paligemma2-3b-pt-224"
    # cache_dir = "./models/paligemma-3b"
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
        "grounding": SegmentGroundingEvaluator()
    }
    if args.eval_llm_score:
        metrics["llm_score"] = LLMScoreServer()
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





