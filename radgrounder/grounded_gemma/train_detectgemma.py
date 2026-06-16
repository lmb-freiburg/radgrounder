import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import argparse

from radgrounder.dataset.dataset_manager import DatasetManager
from radgrounder.dataset.image_preprocessing import NormalizationConfig, NormalizationType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

hash_excluded_keys = [
    "output_dir", "resume_from_checkpoint", "cache_dir", "val_dataset_size", "use_wandb", "wandb_project",
    "logging_steps", "save_steps", "eval_steps", "save_total_limit", "eval_accumulation_steps", "gradient_checkpointing",
    "config_saved_at"
]

def load_config(config_path: str) -> dict:
    """Load training configuration from JSON"""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DetectGemma for medical VQA")
    parser.add_argument("--config", type=str, default=None, help="Path to training config JSON (not required if resuming)")
    parser.add_argument("--output_dir", type=str, default="./runs", help="Output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume training from (e.g., './runs/model_name/checkpoints/checkpoint-1000')")
    
    args = parser.parse_args()
    
    # If resuming from checkpoint, load config from the saved config file
    if args.resume_from_checkpoint:
        run_dir = Path(args.resume_from_checkpoint)
        # The checkpoint structure is: runs/run_name/checkpoints/checkpoint-N
        # So we need to go up two levels to get to the run directory
        config_file = run_dir / "config.json"
        
        if config_file.exists():
            logger.info(f"Loading config from {config_file}")
            config = load_config(str(config_file))
            config["resume_from_checkpoint"] = args.resume_from_checkpoint
            logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        else:
            raise FileNotFoundError(f"Config file not found at {config_file}. Cannot resume without config.")
    else:
        # Normal training flow - config is required
        if not args.config:
            raise ValueError("--config is required when starting new training (not resuming)")
        config = load_config(args.config)
        config["output_dir"] = args.output_dir
        config["resume_from_checkpoint"] = None
    
    device = print_system_info()

    TORCH_DTYPE = torch.bfloat16
    
    # Determine model ID based on image size
    img_size = config.get("image_size", 224)
    if img_size == 224:
        MODEL_ID = config.get("model_id", "google/paligemma2-3b-pt-224")
    elif img_size == 448:
        MODEL_ID = config.get("model_id", "google/paligemma2-3b-pt-448")
    else:
        raise ValueError(f"Image size {img_size} not supported")
    
    model_name = MODEL_ID.split("/")[-1]
    logger.info(f"Using model: {model_name}")
    
    # Create output directory with config hash FIRST (before checking for checkpoints)
    if not config.get("run_name"):
        # Create a hash from config (excluding non-training params)
        config_for_hash = {k: v for k, v in config.items()  if k not in hash_excluded_keys}
        config_str = json.dumps(config_for_hash, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        batch_size = config.get("batch_size", 64)
        grad_accum = config.get("gradient_accumulation_steps", 8)
        effective_bs = batch_size * grad_accum
        notes = config.get("notes", "")
        model_short = model_name.split("-")[0]
        run_name = f"{model_short}_detect_{notes}_bs_{batch_size}x{grad_accum}={effective_bs}_{config_hash}" if notes else f"{model_short}_detect_bs_{batch_size}x{grad_accum}={effective_bs}_{config_hash}"
        config["run_name"] = run_name
        config["run_id"] = config_hash
    else:
        run_name = config["run_name"]
    
    output_dir = Path(config["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Run name: {run_name}")
    
    # Handle resuming from checkpoint
    resume_from_checkpoint = config.get("resume_from_checkpoint", None)
    processor_path = None
    
    if resume_from_checkpoint is None and (output_dir / "final_model").exists():
        logger.info("Final model already exists, skipping training.")
        return
    
    # Check for existing checkpoints and validate them (auto-resume feature)
    if resume_from_checkpoint is None and any(output_dir.glob("checkpoints/checkpoint-*")):
        logger.info("Checking for existing checkpoints to auto-resume...")
        checkpoints = sorted(
            output_dir.glob("checkpoints/checkpoint-*"),
            key=lambda x: int(x.name.split("-")[-1]),
            reverse=True
        )
        
        # Find the latest checkpoint that has model parameters
        for checkpoint in checkpoints:
            # Check for various model file patterns:
            # - pytorch_model.bin (single file)
            # - model.safetensors (single file)
            # - pytorch_model-00001-of-00002.bin (sharded)
            # - model-00001-of-00002.safetensors (sharded)
            has_model = (
                (checkpoint / "pytorch_model.bin").exists() or
                (checkpoint / "model.safetensors").exists() or
                any(checkpoint.glob("pytorch_model-*.bin")) or
                any(checkpoint.glob("model-*.safetensors"))
            )
            if has_model:
                resume_from_checkpoint = str(checkpoint)
                config["resume_from_checkpoint"] = resume_from_checkpoint
                processor_path = Path(resume_from_checkpoint)
                logger.info(f"✓ Found existing checkpoint, will resume from: {resume_from_checkpoint}")
                break
        
        if resume_from_checkpoint is None:
            logger.info("No valid checkpoints found, starting training from scratch")
    elif resume_from_checkpoint:
        # If the user provided a run directory, try to find the latest checkpoint inside it
        checkpoint_path = Path(resume_from_checkpoint)
        search_dir = None
        
        if (checkpoint_path / "checkpoints").exists():
            search_dir = checkpoint_path / "checkpoints"
        elif checkpoint_path.name == "checkpoints" and checkpoint_path.exists():
            search_dir = checkpoint_path
            
        if search_dir:
            checkpoints = sorted(
                search_dir.glob("checkpoint-*"),
                key=lambda x: int(x.name.split("-")[-1]),
                reverse=True
            )
            if checkpoints:
                resume_from_checkpoint = str(checkpoints[0])
                config["resume_from_checkpoint"] = resume_from_checkpoint
                logger.info(f"Auto-resolved latest checkpoint from directory: {resume_from_checkpoint}")

        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        processor_path = Path(resume_from_checkpoint)
    
    # Save config with timestamp (useful for bookkeeping, excluded from hash)
    config["config_saved_at"] = datetime.now().isoformat(timespec="seconds")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load model and processor
    cache_dir = config.get("cache_dir", "../paligemma/models/paligemma-3b")
    
    if resume_from_checkpoint and processor_path:
        # Try to find the processor: first in checkpoint, then parent run directory, then base model
        processor_found = False
        
        if processor_path.exists() and any(processor_path.glob("*processor*")):
            logger.info(f"Loading processor from checkpoint: {processor_path}")
            processor = PaliGemmaProcessor.from_pretrained(str(processor_path), cache_dir=cache_dir, use_fast=True)
            processor_found = True
        else:
            # Try parent run directory
            run_processor_path = processor_path.parent.parent
            if run_processor_path.exists() and any(run_processor_path.glob("*processor*")):
                logger.info(f"Loading processor from run directory: {run_processor_path}")
                processor = PaliGemmaProcessor.from_pretrained(str(run_processor_path), cache_dir=cache_dir, use_fast=True)
                processor_found = True
        
        if not processor_found:
            logger.info(f"Processor not found in checkpoint, loading from base model: {MODEL_ID}")
            processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
        
        # Add special tokens to processor
        new_tokens = ["<p bbox=", "</p>", "id="]
        processor.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        
        # Load model from checkpoint
        logger.info(f"Loading model from checkpoint: {resume_from_checkpoint}")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            resume_from_checkpoint, 
            torch_dtype=TORCH_DTYPE, 
            device_map="auto", 
            attn_implementation='eager'
        )
    else:
        # Load from base model
        logger.info(f"Loading processor and model from: {MODEL_ID}")
        processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
        new_tokens = ["<p bbox=", "</p>", "id="]
        processor.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype=TORCH_DTYPE, 
            device_map="auto", 
            attn_implementation='eager', 
            cache_dir=cache_dir
        )
        
        if config.get("load_siglip_weights", False):
            # Load SigLIP weights into the vision tower
            from radgrounder.grounded_gemma.load_paligemma_w_new_siglip import load_siglip_weights_into_paligemma
            siglip_path = config.get("siglip_model_path", None)
            print("Loading Siglip weights from: ", siglip_path)
            model = load_siglip_weights_into_paligemma(model, siglip_model_path=siglip_path)
    
    # Resize token embeddings to account for special tokens
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # Set trainable layers
    train_encoder = config.get("train_encoder", False)
    train_projector = config.get("train_projector", False)
    set_trainable_layers(model, train_encoder, train_projector)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of model parameters: {total_params}")

    # Extract config parameters
    LEARNING_RATE = config.get("learning_rate", 5e-5)
    SEQLEN = config.get("max_length", 200)
    
    # Handle language config
    language = config.get("language", "all")
    if language == "en":
        language = "english"
    elif language == "de":
        language = "german"
    
    # Handle modality
    modality = config.get("modality", "all")
    if modality != "all":
        modality = modality.lower()
    
    # Normalization config
    norm_cfg = config.get("normalization", "dataset_stats")
    normalization_config = NormalizationConfig(strategy=NormalizationType(norm_cfg))
    
    # Prepare datasets
    logger.info("Preparing datasets using DatasetManager...")
    val_dataset_manager = DatasetManager(
        dataset_name=config.get("dataset_name", "refrad2d_detect_merged"),
        split="val",
        img_size=img_size,
        eval_mode=False,
        language=language,
        augment=False,
        dataset_size=config.get("val_dataset_size", 512),
        max_length=SEQLEN,
        body_part=config.get("body_part", "ALL"),
        modality=modality,
        selected_dataset=config.get("selected_dataset", None),
        prefix_style=config.get("val_prefix_style", "both"),
        tokenizer=processor.tokenizer,
        question_types=config.get("question_types", "all"),
        add_other_vqa_datasets=config.get("add_other_vqa_datasets", False),
        normalization=normalization_config
    )
    train_dataset_manager = DatasetManager(
        dataset_name=config.get("dataset_name", "refrad2d_detect_merged"),
        split="train",
        img_size=img_size,
        eval_mode=False,
        language=language,
        augment=config.get("augment", True),
        dataset_size=config.get("train_dataset_size", None),
        max_length=SEQLEN,
        body_part=config.get("body_part", "ALL"),
        modality=modality,
        selected_dataset=config.get("selected_dataset", None),
        prefix_style=config.get("prefix_style", "random"),
        tokenizer=processor.tokenizer,
        question_types=config.get("question_types", "all"),
        add_other_vqa_datasets=config.get("add_other_vqa_datasets", False),
        normalization=normalization_config
    )

    global get_collate_fn
    get_collate_fn = train_dataset_manager.get_collate_fn
    val_dataset = val_dataset_manager.dataset
    train_dataset = train_dataset_manager.dataset

    TRAIN_EXAMPLES = len(train_dataset)
    BATCH_SIZE_STEP = config.get("batch_size", 64)
    EVAL_BATCH_SIZE = config.get("eval_batch_size", 16)
    GRAD_ACCUM = config.get("gradient_accumulation_steps", 8)
    epochs = config.get("num_epochs", 2)
    
    # Calculate effective batch size and steps
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE_STEP * GRAD_ACCUM
    ONE_EPOCH_STEPS = TRAIN_EXAMPLES // EFFECTIVE_BATCH_SIZE
    TRAIN_STEPS = ONE_EPOCH_STEPS * epochs
    
    # Get save/eval steps from config or calculate
    SAVE_STEPS = config.get("save_steps", int(ONE_EPOCH_STEPS // 20))
    EVAL_STEPS = config.get("eval_steps", SAVE_STEPS)
    SAVE_LIMIT = config.get("save_total_limit", 3)

    # Ensure steps are at least GRAD_ACCUM
    if EVAL_STEPS < GRAD_ACCUM:
        EVAL_STEPS = GRAD_ACCUM * 50

    if SAVE_STEPS < GRAD_ACCUM:
        SAVE_STEPS = GRAD_ACCUM * 50

    logger.info(f"✓ Training samples: {len(train_dataset)}")
    logger.info(f"✓ Validation samples: {len(val_dataset)}")
    logger.info(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")
    logger.info(f"Training steps per epoch: {ONE_EPOCH_STEPS}")
    logger.info(f"Total training steps: {TRAIN_STEPS}")
    logger.info(f"Save steps: {SAVE_STEPS}")
    logger.info(f"Eval steps: {EVAL_STEPS}")
    logger.info(f"Gradient accumulation: {GRAD_ACCUM}")
    logger.info(f"Batch size per step: {BATCH_SIZE_STEP}")

    # Prepare config dict for trainer
    trainer_config = dict(config)
    trainer_config.update({
        "save_steps": SAVE_STEPS,
        "save_limit": SAVE_LIMIT,
        "resume_from_checkpoint": resume_from_checkpoint,
        "eval_steps": EVAL_STEPS,
        "seq_len": SEQLEN,
        "batch_size_step": BATCH_SIZE_STEP,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "output_dir": str(output_dir / "checkpoints"),
        "run_name": run_name,
        "torch_dtype": TORCH_DTYPE,
        "learning_rate": LEARNING_RATE,
        "epochs": epochs,
        "logging_steps": config.get("logging_steps", 20),
        "optim": config.get("optim", "adamw_torch_fused"),
        "warmup_ratio": config.get("warmup_ratio", 0.03),
        "weight_decay": config.get("weight_decay", 0.01),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "num_workers": config.get("num_workers", 4),
        "eval_accumulation_steps": config.get("eval_accumulation_steps", 4),
        "use_wandb": config.get("use_wandb", False),
        "wandb_project": config.get("wandb_project", "radgrounder-detectgemma"),
    })

    model = train_with_sft_trainer(model, train_dataset, val_dataset, processor, trainer_config)

    best_model_path = output_dir / "final_model"
    logger.info(f"Saving final model to {best_model_path}")
    model.save_pretrained(str(best_model_path), safe_serialization=False)
    processor.save_pretrained(str(best_model_path))
    logger.info("✓ Training complete.")


def train_with_sft_trainer(model, train_dataset, val_dataset, processor, config):
    from trl import SFTTrainer, SFTConfig
    
    logger.info("Starting SFT training...")
    
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config.get("wandb_project", "radgrounder"),
            name=config["run_name"],
            config=config
        )
        print(f"WandB Config: {wandb.config}")

    sft_config = SFTConfig(
        output_dir=config["output_dir"],
        run_name=config["run_name"],
        num_train_epochs=config["epochs"],
        max_steps=config.get("max_steps", -1),  # -1 = run full epochs; set small for a smoke test
        per_device_train_batch_size=config["batch_size_step"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        eval_accumulation_steps=config["eval_accumulation_steps"],
        max_length=config["seq_len"],
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        optim=config["optim"],
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_limit"],
        save_safetensors=False,
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        bf16=True,
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to=["wandb"] if use_wandb else [],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        dataloader_num_workers=config["num_workers"],
    )   

    # Create the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=get_collate_fn(processor, config["seq_len"], config["torch_dtype"]),
        processing_class=processor,
        args=sft_config,
    )

    # Fine-tune the model
    logger.info(f"Resuming from checkpoint: {config['resume_from_checkpoint']}")
    trainer.train(resume_from_checkpoint=config["resume_from_checkpoint"])

    if use_wandb:
        wandb.finish()

    return trainer.model

def print_system_info():
    import sys, transformers, tokenizers, accelerate
    logger.info(f'Python Version: {sys.version}')
    logger.info(f'Torch Version: {torch.__version__}')
    logger.info(f'CUDA Available: {torch.cuda.is_available()}')
    logger.info(f'CUDA Device Count: {torch.cuda.device_count()}')
    logger.info(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"}')
    logger.info(f'Transformers Version: {transformers.__version__}')
    logger.info(f'Tokenizers Version: {tokenizers.__version__}')
    logger.info(f'Accelerate Version: {accelerate.__version__}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    os.system("nvidia-smi")
    return device

eval_samples = []
def get_compute_metrics_fn(metrics, processor):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # generated_ids = predictions[:, input_len:]
        predictions = np.where(predictions != -100, predictions, processor.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

        eos_mask = predictions == processor.tokenizer.eos_token_id
        first_indices = np.where(eos_mask.any(axis=1), eos_mask.argmax(axis=1), -1)
        rows, cols = np.indices(predictions.shape)
        mask_before_first = cols < first_indices[:, None] 
        extracted_prediction = predictions.copy()
        extracted_prediction[mask_before_first] = processor.tokenizer.pad_token_id
        decoded_predictions = processor.batch_decode(extracted_prediction, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)


        decoded_predictions_w_spec_token = processor.batch_decode(predictions, skip_special_tokens=False)
        decoded_labels_w_spec_token = processor.batch_decode(labels, skip_special_tokens=False)
        num_eos_tokens = np.sum(predictions == processor.tokenizer.eos_token_id, axis=1)
        print(f"Num end tokens: {num_eos_tokens}")
        # for i, pred in enumerate(predictions):
        #     print(np.where(pred == 1))
        #     print(pred.shape)
        #     print(len(decoded_predictions[i]))
        # print("predictions", batch_predictions)
        # print("labels", batch_labels)

        # print(f"Length of batch_predictions: {len(batch_predictions)}")
        # print(f"Length of batch_labels: {len(batch_labels)}")
        avg_metrics = {}
        log_size = min(len(decoded_predictions), 32)
        logged_samples = []
        for i in range(log_size):
            log_string = f"Label:\n{decoded_labels[i]}\n\nExtracted Prediction:\n{decoded_predictions[i]}"
            log_string += f"\n\nNum end tokens: {num_eos_tokens[i]}"
            log_string += f"\nLabel with special token:\n{decoded_labels_w_spec_token[i]}\n\nPrediction with special token:\n{decoded_predictions_w_spec_token[i]}"
            logged_samples.append(log_string)
        eval_samples.append(logged_samples)
        avg_metrics["valid_samples"] = eval_samples
        # text_table = wandb.Table(columns=["samples"])
        # logged_samples = [f"*Label:\n{batch_labels[i]}\n\nPrediction:\n{batch_predictions[i]}" for i in range(log_size)]
        # text_table.add_data(logged_samples)
        # avg_metrics["valid_samples"] = text_table
        
        for metric_name, metric_func in metrics.items():
            if metric_name == "f1": 
                f1_scores = [metric_func(decoded_predictions[i], decoded_labels[i]) for i in range(len(decoded_predictions))]
                avg_metrics[metric_name] = sum(f1_scores) / len(f1_scores)
            elif hasattr(metric_func, 'update') and hasattr(metric_func, 'compute'): # Check if it's an object like NgramMetric
                    metric_func.update(decoded_predictions, decoded_labels) 
                    # For simplicity, let's compute here; better approach might be needed for some metrics
                    avg_metrics[metric_name] = metric_func.compute() # Assuming compute gives the score
            else:
                # Fallback or error for unknown metric types
                print(f"Warning: Unknown metric type for {metric_name}")
                avg_metrics[metric_name] = 0.0
        return avg_metrics

    return compute_metrics

def set_trainable_layers(model, train_encoder, train_projector):
    # Set the model trainable parameters based on the command line arguments
    for param in model.vision_tower.parameters():
        param.requires_grad = train_encoder

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = train_projector

    for param in model.language_model.parameters():
        param.requires_grad = True

    logger.info("Reporting trainable layers:")
    frozen_layers_count = 0
    trainable_layers_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_layers_count += 1
        else:
            trainable_layers_count += 1

    logger.info("Training modules:")
    logger.info(f"  - Vision tower: {train_encoder}")
    logger.info(f"  - Multi-modal projector: {train_projector}")
    logger.info(f"  - Language model: True")
    logger.info(f"Total frozen layers: {frozen_layers_count}")
    logger.info(f"Total trainable layers: {trainable_layers_count}")

if __name__ == "__main__":
    main()

