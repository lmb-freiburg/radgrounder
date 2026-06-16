import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from typing import List, Optional, Literal
import torch.nn.functional as F
import re
import math
import sys
import tqdm
import pandas as pd
from pathlib import Path

from radgrounder.grounded_gemma.utils import read_dicom_as_numpy
from radgrounder.grounded_gemma.augmentations import build_augmentation_pipeline
from radgrounder.dataset.dataset_manager import USE_GROUNDED_REPORT_PROMPT
from radgrounder.dataset.image_preprocessing import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    NormalizationType,
    apply_normalization,
)
from radgrounder.dataset.dataset_utils import (
    clean_punctuation,
    load_snippet2rsopids,
    tokenize_text,
)

GRAYSCALE_PROB = 0.2


class RefRad2DV18(Dataset):
    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train", # split options: "train", "val", "test"
        img_size: int = 224,
        eval_mode: bool = False,
        language: Literal["english", "german", "all"] = "all",
        augment: bool = False,
        dataset_size: Optional[int] = None,
        max_length: int = 100,
        prefix_style: Optional[Literal["none", "klinische_angaben", "fragestellung", "both", "random"]] = None,
        question_types: Literal["all", "vqa", "report"] = "all",
        modality: Literal["all", "ct", "mr"] = "all",
        normalization: Optional[NormalizationConfig] = None,
        prompt_template_type: str = "refrad2d",
    ):
        print("Initializing RefRad2DV18 Dataset")
        modality = modality.lower()
        if modality not in ["all", "ct", "mr"]:
            raise ValueError(f"Invalid modality: {modality}. Must be one of ['all', 'ct', 'mr'].")
        
        print(f"Loading modality: {modality}")
        print("*" * 50)
        self.snippet2rsopids = load_snippet2rsopids(split, modality)
        
        from radgrounder.paths import REFRAD2D_VQA_PARQUET
        merged_vqa_dataset_path = str(REFRAD2D_VQA_PARQUET)
        self.refrad2d_dataset = pd.read_parquet(merged_vqa_dataset_path)
        self.refrad2d_dataset = self.refrad2d_dataset.set_index('rsopid')

        # Split into train/validation based on eval_mode
        print(f"Using split: {split}, eval_mode: {eval_mode}")
        print("Number of unique snippets in this split:", len(self.snippet2rsopids))
        print("Split path:", "from load_snippet2rsopids")
        
        self.normalization = normalization or DEFAULT_NORMALIZATION
        self.prompt_template_type = prompt_template_type
        print("Normalization strategy:", self.normalization.strategy.value, "Prompt template type:", self.prompt_template_type)

        self.split = split
        
        all_rsopids = [rsopid for rsopids in self.snippet2rsopids.values() for rsopid in rsopids]
        self.refrad2d_dataset = self.refrad2d_dataset[self.refrad2d_dataset.index.isin(all_rsopids)]

        from radgrounder.paths import REFRAD2D_DICOM_DIR
        self.slices_path = str(REFRAD2D_DICOM_DIR)
        self.image_shape = (img_size, img_size)
        self.dataset_stats = {"avr_ct_mean": -610.1908535827807, "avr_ct_std": 737.3882466073229,
                                "avr_mr_mean": 141.1154960399739, "avr_mr_std": 255.40429635138338}


        # Truncate the dataset if dataset_size is specified
        if dataset_size is not None:
            if dataset_size > len(self.snippet2rsopids):
                print(f"Dataset size {dataset_size} is larger than the actual dataset size {len(self.snippet2rsopids)}. Using the actual dataset size.")
                dataset_size = len(self.snippet2rsopids)
            
            # all_keys = list(self.snippet2rsopids.keys())
            # selected_keys = all_keys[:dataset_size]
            rng = np.random.default_rng(42)
            selected_keys = rng.choice(list(self.snippet2rsopids.keys()), size=dataset_size, replace=False)
            self.snippet2rsopids = {k: self.snippet2rsopids[k] for k in selected_keys}
    
        self.keys = list(self.snippet2rsopids.keys())
        self.augment = augment
        if self.augment:
            self.augmentation_pipeline = build_augmentation_pipeline()

        self.torch_dtype = torch.bfloat16
        self.eval_mode = eval_mode
        self.language = language
        if self.language not in ["english", "german", "all"]:
            raise ValueError(f"Invalid language: {self.language}. Must be one of ['english', 'german', 'all'].")
        self.max_length = max_length

        self.prefix_style = prefix_style
        self.prefix_options = ["none", "klinische_angaben", "fragestellung", "both"]

        print(f"Prefix style: {self.prefix_style}, Question types: {question_types}")
        self.question_types = ["vqa"]
        if question_types == "all":
            self.question_types = ["vqa", "report"]
        elif question_types == "report":
            self.question_types = ["report"]
            
        if self.eval_mode:
            seed = 42
            rng = np.random.default_rng(seed)
            self.random_choice_function = lambda x: x[rng.integers(len(x))]
        else:
            self.random_choice_function = np.random.choice
    
    def __len__(self):
        return len(self.snippet2rsopids)
    
    def __getitem__(self, idx):
        caption = self.keys[idx]
        if self.eval_mode:
            rsopid = self.snippet2rsopids[caption][0] 
        else:
            rsopid = self.random_choice_function(self.snippet2rsopids[caption])
        data = self.refrad2d_dataset.loc[rsopid]

        # Load image
        image_path = os.path.join(self.slices_path, rsopid[:2], rsopid + ".dcm.zst")
        modality = data["modality"]
        image = self.load_image(image_path, modality)

        image_tensor = image.type(self.torch_dtype)

        data = data.copy()
        data["rsopid"] = rsopid
        prefix, prompt, suffix, language, question_type =  self.load_text(data)
        prefix_full = f"<image>\n{prefix}\n{prompt} "
        info = {}
        if self.eval_mode:
            info = {
                "dicom_path": image_path,
                "modality": modality,
                "language": language,
                "question_type": question_type
            }
        
        return image_tensor, prefix_full, suffix, info, self.eval_mode
    
    def load_image(self, dicom_path, modality, eval_mode=False):
        try:
            dicom_array = read_dicom_as_numpy(dicom_path)
        except Exception as e:
            print(f"Error loading image {dicom_path}: {e}")
            return None

        dicom_array = torch.from_numpy(dicom_array)
        
        if modality == "CT":
            dicom_array = torch.clip(dicom_array, min=-1000) 
        
        dicom_array = dicom_array.unsqueeze(0)  # Add channel dimension
        if self.augment and not eval_mode:
            dicom_array = self.augmentation_pipeline(dicom_array)

        image = self.normalize_image(dicom_array, modality, eval_mode=eval_mode)

        #resize to 224 x 224
        image = F.interpolate(image.unsqueeze(0), size=self.image_shape, mode='bilinear', align_corners=False)
        image = image[0]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image

    def normalize_image(self, image, modality=None, eval_mode=False):
        norm_img = apply_normalization(
            image,
            modality,
            self.normalization,
            dataset_stats=self.dataset_stats,
        )
        
        #select one channel randomly and dublicate it to grayscale if normalization is medgemma and eval_mode is False
        if self.normalization.strategy == NormalizationType.MEDGEMMA and not eval_mode and self.augment:
            if np.random.rand() < GRAYSCALE_PROB:
                channel_idx = self.random_choice_function(norm_img.shape[0])
                norm_img = norm_img[channel_idx, :, :].unsqueeze(0).repeat(3, 1, 1)

        return norm_img

    def load_text(self, data):
        language = self.language
        if language == "all":
            language = self.random_choice_function(["english", "german"])
        # print(data[language]["dicom_vqa"], data[language]["snippet_vqa"])
        quest_type = self.random_choice_function(self.question_types)
        if quest_type == "report":
            prefix, prompt, suffix, question_type = self.load_report(data, language)
        elif quest_type == "vqa":
            prefix, prompt, suffix, question_type = self.load_vqa(data, language)

        # adaptively cut prefix and suffix to fit within max_length
        len_prompt = len(prompt.split())
        effective_max_length = self.max_length - len_prompt
        prefix, suffix = adaptively_cut_text(prefix, suffix, effective_max_length)

        return prefix, prompt, suffix, language, question_type

    def load_report(self, data, language):
        question_type = "report"
        if USE_GROUNDED_REPORT_PROMPT:
            if language == "english":
                prompt = "Caption this image and detect keywords:"
            else:
                prompt = "Beschreibe dieses Bild und detektiere Schlüsselwörter:"
        elif self.prompt_template_type == "medgemma":
            if language == "english":
                prompt = "Describe this image and provide the most likely condition. Keep your answer brief."
            else:
                prompt = "Beschreibe dieses Bild und gib die wahrscheinlichste Diagnose an. Halte deine Antwort kurz."
        else:
            if language == "english":
                prompt = "Caption this image:"
            elif language == "german":
                prompt = "Beschreibe dieses Bild:"

        prefix = self.load_prefix(data, language)
        suffix = self.get_report_content(data[language], "CleanedSentence")

        return prefix, prompt, suffix, question_type

    def load_prefix(self, data, language):
        # Determine which fields to include (none, one, or both)
        if self.prefix_style == "random" or self.prefix_style is None:
            prefix_type = self.random_choice_function(self.prefix_options).item()
        else:
            prefix_type = self.prefix_style
        lan_data = data[language]

        # Build prefix content
        if prefix_type == "none":
            prefix_content = ""
        elif prefix_type == "both":
            content_1 = self.get_report_content(lan_data, "klinische_angaben").strip()
            content_1_name = self.get_content_name("klinische_angaben", language)
            content_2 = self.get_report_content(lan_data, "fragestellung").strip()
            content_2_name = self.get_content_name("fragestellung", language)
            if content_1 == content_2:
                prefix_content = f"{content_1_name} {content_1}".strip()
            else:
                prefix_content = f"{content_1_name} {content_1}\n{content_2_name} {content_2}".strip()
        else:
            prefix_content = self.get_report_content(lan_data, prefix_type).strip()
            content_name = self.get_content_name(prefix_type, language)
            prefix_content = f"{content_name} {prefix_content}" if prefix_content else ""

        return prefix_content

    
    def get_content_name(self, prefix_type, language):
        if prefix_type == "fragestellung":
            if language == "english":
                return "Question:"
            else:
                return "Fragestellung:"
        elif prefix_type == "klinische_angaben":
            if language == "english":
                return "Clinical information:"
            else:
                return "Klinische angaben:"

    def get_report_content(self, lan_data, prefix_type):
        if prefix_type == "fragestellung":
            key = "ReportFragestellungCleaned"
        elif prefix_type == "klinische_angaben":
            key = "ReportKlinischeAngabenCleaned"
        else:
            key = prefix_type  # e.g., "CleanedSentence"

        content = lan_data.get(key, "")
        if content:
            content = clean_punctuation(content)

        return content

    def load_vqa(self, data, language):
        vqa_list = list(data[language]["dicom_vqa"]) + list(data[language]["snippet_vqa"])
        if not vqa_list:
            return None
        
        selected_vqa = self.random_choice_function(vqa_list)
        
        prefix = selected_vqa["question"]
        question_type = selected_vqa["question_type"]
        if question_type == "multiple":
            prefix += "\n" + "\n".join(selected_vqa["choices"])
            
        suffix = selected_vqa["answer"]
        if language == "english":
            prompt = "Answer:"
        else:
            prompt = "Antwort:"

        return prefix, prompt, suffix, question_type



def adaptively_cut_text(prefix: str, suffix: str, effective_max_length: int):
    """
    Adaptively cut prefix and suffix to fit within effective_max_length.
    TODO this would better be done on token level after tokenizing each text.

    Args:
        prefix: The prefix text to potentially cut
        suffix: The suffix text to potentially cut
        effective_max_length: Maximum total length allowed (max_length - prompt_len)

    Returns:
        tuple: (new_prefix, new_suffix) with lengths adjusted to fit within effective_max_length
    """
    prefix_split = prefix.split()
    prefix_len = len(prefix_split)
    suffix_split = suffix.split()
    suffix_len = len(suffix_split)

    if prefix_len + suffix_len > effective_max_length:
        # cut prefix up to half of the maximum length
        num_to_cut = prefix_len + suffix_len - effective_max_length
        new_prefix_len = max(math.floor(effective_max_length / 2), prefix_len - num_to_cut)
        prefix = " ".join(prefix_split[:new_prefix_len])
        remaining_to_cut = new_prefix_len + suffix_len - effective_max_length
        if remaining_to_cut > 0:
            new_suffix_len = suffix_len - remaining_to_cut
            suffix = " ".join(suffix_split[:new_suffix_len])
    return prefix, suffix


def get_collate_fn(processor, seq_len, torch_dtype, return_infos=False):
    def collate_fn(batch):
        images, prefixes, suffixes, infos, eval_mode = zip(*batch)

        output_kwargs = {"text_kwargs": {"max_length": seq_len, "truncation": False,
                                          "return_tensors": "pt", "padding": "longest"}}
        if eval_mode[0]:
            text_inputs = tokenize_text(processor, prefixes, None, output_kwargs)
        else:    
            text_inputs = tokenize_text(processor, prefixes, suffixes, output_kwargs)

        image_inputs = torch.stack(images).to(torch_dtype)
        # image_inputs = torch.stack(images)
        # print(image_inputs.shape)

        inputs = {"pixel_values": image_inputs, **text_inputs}
        if return_infos:
            return inputs, prefixes, suffixes, infos
            
        return inputs    
    return collate_fn

if __name__ == "__main__":
    # Example usage
    from transformers import PaliGemmaProcessor
    MODEL_ID ="google/paligemma2-3b-pt-224"
    cache_dir = "./models/paligemma-3b"
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
    language = "all"  # or "german" or "all"
    dataset = RefRad2DV18(split="train", img_size=224, eval_mode=True, language=language, augment=True,
                         dataset_size=None, max_length=10000, prefix_style="both", question_types="all", modality="MR")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=get_collate_fn(processor, 200, torch.bfloat16, return_infos=True), num_workers=4, pin_memory=True)

    values = 0
    val_squares = 0
    num_elements = 0
    prefix_word_counts = []
    suffix_word_counts = []
    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        # image_tensor, prefix_full, suffix, info, eval_mode = batch
        inputs, prefixes, suffixes, infos = batch
        # pixel_values = inputs["pixel_values"]
        # print(pixel_values.shape)
        # values += pixel_values.sum().item()
        # val_squares += (pixel_values ** 2).sum().item()
        # num_elements += pixel_values.numel()
        # print(inputs.keys())

        for i in range(len(prefixes)):
            # tokenized_prefix = processor.tokenizer(prefixes[i], return_tensors="pt", truncation=False)
            # tokenized_suffix = processor.tokenizer(suffixes[i], return_tensors="pt", truncation=False)
            print("=" * 20)
            print(f"Prefix: {prefixes[i]}")
            print(f"Suffix: {suffixes[i]}")
        #     prefix_word_counts.append(len(tokenized_prefix))
        #     suffix_word_counts.append(len(tokenized_suffix))
            # print(f"Prefix: {prefixes[i]}")
            # print(f"Suffix: {suffixes[i]}")
            # print("*" * 20)
       
  
        exit()
    print("Mean prefix length:", np.mean(prefix_word_counts), "Std:", np.std(prefix_word_counts))
    print("Mean suffix length:", np.mean(suffix_word_counts), "Std:", np.std(suffix_word_counts))
    # avr_mean = values / num_elements if num_elements > 0 else 0
    # avr_std = np.sqrt((val_squares / num_elements) - (avr_mean ** 2)) if num_elements > 0 else 0
    # print(f"Mean: {avr_mean}, Std: {avr_std}")
    # print(f"Total elements: {num_elements}")
    