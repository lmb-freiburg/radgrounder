from random import sample
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import json
import numpy as np
import os
from typing import List, Optional
import torch.nn.functional as F
import re
import math
import sys
import tqdm
import pandas as pd

from radgrounder.dataset.image_preprocessing import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    NormalizationType,
)
from radgrounder.dataset.segmentation.refrad2d_detect import RefRad2DDetect
from radgrounder.dataset.segmentation.refrad2d_detect_vqa import RefRad2DDetectVQA
from radgrounder.dataset.vqa_dataset.refrad2d_v18 import RefRad2DV18
from radgrounder.dataset.vqa_dataset.vqa_rad import VqaRad
from radgrounder.dataset.vqa_dataset.slake_vqa import SlakeVQA
from radgrounder.dataset.dataset_utils import tokenize_text

class RefRad2DDetectMerged(Dataset):
    def __init__(
        self,
        split="train",
        img_size=224,
        eval_mode=False,
        language="all",
        augment=False,
        dataset_size=None,
        max_length=100,
        body_part=None,
        modality=None,
        prefix_style=None,
        only_segmented=False,
        tokenizer=None,
        selected_dataset=None,
        question_types="all",
        add_other_vqa_datasets=False,
        normalization: Optional[NormalizationConfig] = None,
    ):

        self.normalization = normalization or DEFAULT_NORMALIZATION

        if selected_dataset is None:
            self.refrad2d_detect = RefRad2DDetect(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                language=language,
                augment=augment,
                dataset_size=dataset_size,
                max_length=max_length,
                body_part=body_part,
                modality=modality,
                prefix_style=prefix_style,
                only_segmented=only_segmented,
                tokenizer=tokenizer,
                question_types=question_types,
                normalization=self.normalization,
            )
            
            self.refrad2d_detect_vqa = RefRad2DDetectVQA(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                language=language,
                augment=augment,
                dataset_size=dataset_size,
                max_length=max_length,
                tokenizer=tokenizer,
                normalization=self.normalization,
                modality=modality,
            )

            dataset_list = [self.refrad2d_detect, self.refrad2d_detect_vqa]
        
        if selected_dataset == "refrad2d_detect":
            self.refrad2d_detect = RefRad2DDetect(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                language=language,
                augment=augment,
                dataset_size=dataset_size,
                max_length=max_length,
                body_part=body_part,
                modality=modality,
                prefix_style=prefix_style,
                only_segmented=only_segmented,
                tokenizer=tokenizer,
                question_types=question_types,
                normalization=self.normalization,
            )
            dataset_list = [self.refrad2d_detect]
            print("Selected dataset: RefRad2D Detect")
        elif selected_dataset == "refrad2d_detect_vqa":
            self.refrad2d_detect_vqa = RefRad2DDetectVQA(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                language=language,
                augment=augment,
                dataset_size=dataset_size,
                max_length=max_length,
                tokenizer=tokenizer,
                normalization=self.normalization,
                modality=modality,
            )
            dataset_list = [self.refrad2d_detect_vqa]
            print("Selected dataset: RefRad2D Detect VQA")
        elif selected_dataset == "refrad2d_v18":
            self.refrad2d_v18 = RefRad2DV18(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                language=language,
                augment=augment,
                dataset_size=dataset_size,
                prefix_style=prefix_style,
                question_types=question_types,
                max_length=max_length,
                normalization=self.normalization,
            )
            dataset_list = [self.refrad2d_v18]
            print("Selected dataset: RefRad2D V18")
        elif selected_dataset == "vqa_rad":
            self.vqarad = VqaRad(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                augment=augment,
                max_length=max_length,
                normalization=self.normalization,
            )
            dataset_list = [self.vqarad]
            print("Selected dataset: VQA-RAD")
        elif selected_dataset == "slake_vqa":
            self.slake = SlakeVQA(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                augment=augment,
                max_length=max_length,
                normalization=self.normalization,
            )
            dataset_list = [self.slake]
            print("Selected dataset: SLAKE VQA")
        elif selected_dataset == "external_dataset":
            self.vqarad = VqaRad(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                augment=augment,
                max_length=max_length,
                normalization=self.normalization,
            )
            
            self.slake = SlakeVQA(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                augment=augment,
                max_length=max_length,
                normalization=self.normalization,
            )
            dataset_list = [self.vqarad, self.slake]
            print("Selected dataset: External VQA datasets (VQA-RAD and SLAKE)")
        
        if add_other_vqa_datasets:
            self.vqarad = VqaRad(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                augment=augment,
                max_length=100,
                normalization=self.normalization,
            )
        
            self.slake = SlakeVQA(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                augment=augment,
                max_length=100,
                normalization=self.normalization,
            )
            dataset_list += [self.vqarad, self.slake]
            print("Added VQA-RAD and SLAKE VQA datasets to the merged dataset")
            # print("Not added VQA-RAD and SLAKE VQA datasets to the merged dataset")

        self.concat_dataset = ConcatDataset(dataset_list)

        #print the sizes of the datasets
        for dataset in dataset_list:
            print(f"Dataset {dataset.__class__.__name__} size: {len(dataset)}")
        print(f"Combined dataset size: {len(self.concat_dataset)}")

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

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

        inputs = {"pixel_values": image_inputs, **text_inputs}
        if return_infos:
            return inputs, prefixes, suffixes, infos
            
        return inputs    
    return collate_fn

if __name__ == '__main__':
    from transformers import PaliGemmaProcessor
    
    MODEL_ID ="google/paligemma2-3b-pt-224"
    cache_dir = None
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
    
    normalization_config = NormalizationConfig(strategy=NormalizationType.MEDGEMMA)
    
    dataset = RefRad2DDetectMerged(
        split="test",
        img_size=224,
        eval_mode=False,
        language="all",
        augment=True,
        max_length=200,
        body_part="ALL",
        modality="ct",
        prefix_style="random",
        only_segmented=False,
        tokenizer=processor.tokenizer,
        add_other_vqa_datasets=True,
        selected_dataset="refrad2d_detect",
        normalization=normalization_config,
    )
    
    # dataset = RefRad2DDetectMerged(
    #     split="test",
    #     img_size=224,
    #     eval_mode=False,
    #     language="all",
    #     augment=True,
    #     max_length=200,
    #     body_part="ALL",
    #     modality="mr",
    #     prefix_style="random",
    #     only_segmented=False,
    #     tokenizer=processor.tokenizer,
    #     add_other_vqa_datasets=True,
    #     normalization=normalization_config,
    # )
    
    
    exit()
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=get_collate_fn(processor, 200, torch.bfloat16, return_infos=True),
        num_workers=4,
        pin_memory=True
    )
    for i, batch in enumerate(tqdm.tqdm(dataloader, desc="Processing batches")):
        inputs, prefixes, suffixes, infos = batch
        # images = inputs['pixel_values']
        
        # print("Prefix:" + prefixes[i])
        # print("Suffix:" + suffixes[i])
        
        for j in range(len(prefixes)):
            print(f"Batch {i+1}, Sample {j+1}")
            print(f"Prefixes: {prefixes[j]}")
            print(f"Suffixes: {suffixes[j]}")
            break
        # if i > 3: # Check first 3 batches
        #     break
