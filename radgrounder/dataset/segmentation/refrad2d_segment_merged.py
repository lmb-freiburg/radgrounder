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
)
from radgrounder.dataset.segmentation.refrad2d_segment import RefRad2DSegment
from radgrounder.dataset.segmentation.refrad2d_segment_vqa import RefRad2DSegmentVQA
from radgrounder.dataset.vqa_dataset.vqa_rad import VqaRad
from radgrounder.dataset.vqa_dataset.slake_vqa import SlakeVQA

from radgrounder.dataset.segmentation.refrad2d_segment import SEG_START, SEG_END, get_collate_fn


class RefRad2DSegmentMerged(Dataset):
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
        question_types="all",
        only_segmented=False,
        tokenizer=None,
        selected_dataset=None,
        add_other_vqa_datasets=False,
        normalization: Optional[NormalizationConfig] = None,
    ):

        self.normalization = normalization or DEFAULT_NORMALIZATION

        dataset_list = []
        if selected_dataset == "refrad2d_segment" or selected_dataset is None:
            self.refrad2d_segment = RefRad2DSegment(
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
                question_types=question_types,
                normalization=self.normalization,
            )
            dataset_list.append(self.refrad2d_segment)
            print("Selected dataset: RefRad2D Segment")

        
        if selected_dataset == "refrad2d_segment_vqa" or selected_dataset is None:
            self.refrad2d_segment_vqa = RefRad2DSegmentVQA(
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
            dataset_list.append(self.refrad2d_segment_vqa)
            print("Selected dataset: RefRad2D Segment VQA")
        
        if add_other_vqa_datasets:
            self.vqarad = VqaRad(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                augment=augment,
                max_length=max_length,
                return_dummy_mask=True,
                normalization=self.normalization,
            )

            self.slake = SlakeVQA(
                split=split,
                img_size=img_size,
                eval_mode=eval_mode,
                augment=augment,
                max_length=max_length,
                return_dummy_mask=True,
                normalization=self.normalization,
            )
            dataset_list.extend([self.vqarad, self.slake])

        self.concat_dataset = ConcatDataset(dataset_list)

        for dataset in dataset_list:
            print(f"Dataset {dataset.__class__.__name__} size: {len(dataset)}")
        print(f"Combined dataset size: {len(self.concat_dataset)}")
        

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]


if __name__ == '__main__':
    from transformers import PaliGemmaProcessor
    from collections import defaultdict
    from radgrounder.dataset.segmentation.total_segmentator.read_segmentation import visualize_slices_with_segmentation
    
    MODEL_ID ="google/paligemma2-3b-pt-224"
    cache_dir = None
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
    processor.tokenizer.add_special_tokens({'additional_special_tokens': [SEG_START, SEG_END]})
    
    dataset = RefRad2DSegmentMerged(
        split="train",
        img_size=224,
        eval_mode=False,
        language="all",
        augment=True,
        max_length=200,
        body_part="ALL",
        modality="all",
        prefix_style="random",
        only_segmented=False,
        tokenizer=processor.tokenizer,
        dataset_size=None
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=get_collate_fn(processor, 200, torch.bfloat16, return_infos=True),
        num_workers=4,
        pin_memory=True
    )
    
    with open("radgrounder/dataset/segmentation/label_map/merged_label_map.json", "r") as f:
        label_map = json.load(f)
    label_map = {int(v): k for k, v in label_map.items()}
    
    print(f"Pixel values shape: Processing batches...")
    keyword_count_for_the_same_slice = defaultdict(int)
    
    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        inputs, prefixes, suffixes, infos, binary_gt_masks, seg_token_pos, mask_labels = batch
        pixel_values = inputs["pixel_values"]
        print(f"Pixel values shape: {pixel_values.shape}, Segmentation map shape: {binary_gt_masks.shape}")
        if seg_token_pos is not None:
            print(f"seg_token_pos shape: {seg_token_pos.shape}")
        
        for mask_id, (binary_map, seg_pos) in enumerate(zip(binary_gt_masks, seg_token_pos)): 
            i = seg_pos[0]
            # Extract identifier from different sources depending on dataset
            if "dicom_path" in infos[i]:
                # From RefRad2DSegment dataset
                identifier = infos[i]["dicom_path"].split("/")[-1].split(".")[0]
            elif "image_path" in infos[i]:
                # From RefRad2DSegmentVQA dataset
                identifier = infos[i]["image_path"].split("/")[-1].split(".")[0]
            else:
                identifier = f"sample_{i}"
                
            save_path = f"./samples_from_dataloader/segment_merged_{identifier}_{seg_pos[1]}.png"
            image_slice = pixel_values[i].cpu().float().permute(1, 2, 0).numpy()
            binary_seg_slice = binary_map.cpu().int().numpy()

            binary_mask_keyword = infos[i]["binary_mask_keywords"]
            print(f"Binary mask keyword: {binary_mask_keyword}")
            print("mask_label", mask_labels[mask_id])
            class_id = mask_labels[mask_id]

            seg_slice = binary_seg_slice * class_id
            print(f"Image slice shape: {image_slice.shape}, Segmentation slice shape: {seg_slice.shape}")
            unique_labels = np.unique(seg_slice)
            print(f"Unique labels in segmentation slice: {unique_labels}")
            
            # Visualize the slices with segmentation
            visualize_slices_with_segmentation(
                image_slice, 
                seg_slice, 
                label_map, 
                slice_idx=0, 
                modality=infos[i]["modality"], 
                save_path=save_path, 
                caption=suffixes[i]
            )

        # Exit after processing first batch to save samples
        break
