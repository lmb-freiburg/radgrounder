import torch
from torch.utils.data import Dataset, DataLoader
import json
import math
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm

from radgrounder.grounded_gemma.utils import read_dicom_as_numpy
from radgrounder.grounded_gemma.augmentations import build_augmentation_pipeline, build_segment_aug_pipeline
from radgrounder.dataset.image_preprocessing import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    NormalizationType,
    apply_normalization,
)
from radgrounder.dataset.dataset_utils import (
    clean_punctuation,
    load_segmentation_dataset,
    load_snippet2rsopids,
    tokenize_text,
)


SEG_START = "<seg>"
SEG_END = "</seg>"
GRAYSCALE_PROB = 0.2

class RefRad2DSegment(Dataset):
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
        question_types="all",
        normalization: Optional[NormalizationConfig] = None,
    ):
        print("Initializing RefRad2DSegment dataset...")
        self.split = split
        self.eval_mode = eval_mode
        self.normalization = normalization or DEFAULT_NORMALIZATION
        print("Normalization strategy:", self.normalization.strategy.value)

        self._setup_random_generators()

        snippet2rsopids = load_snippet2rsopids(split, modality)

        from radgrounder.paths import REFRAD2D_VQA_PARQUET
        merged_vqa_dataset_path = str(REFRAD2D_VQA_PARQUET)
        refrad2d_dataset = pd.read_parquet(merged_vqa_dataset_path)
        refrad2d_dataset = refrad2d_dataset.set_index('rsopid')

        all_rsopids = {rsopid for rsopids in snippet2rsopids.values() for rsopid in rsopids}
        refrad2d_dataset = refrad2d_dataset[refrad2d_dataset.index.isin(all_rsopids)]
        rsopid_to_data = refrad2d_dataset.to_dict(orient="index")

        self.segmentation_dataset = load_segmentation_dataset(modality)
        print(f"Loaded segmentation dataset with {len(self.segmentation_dataset)} samples.")
        self.segmentation_dataset = {
            k: v for k, v in self.segmentation_dataset.items()
            if len(v["keywords_eng"]) > 0 and len(v["keywords_de"]) > 0
        }
        print(f"Filtered segmentation dataset to {len(self.segmentation_dataset)} samples with keywords.")

        filtered_snippet2rsopids = {}
        self.filtered_dataset = {}
        print(f"Body part filter: {body_part}, Modality filter: {modality}, Language filter: {language}")
        for caption, rsopids in tqdm.tqdm(snippet2rsopids.items(), desc="Filtering dataset"):
            valid_rsopids = []
            for rsopid in rsopids:
                data = rsopid_to_data.get(rsopid)
                if data is None:
                    continue
                if only_segmented and rsopid not in self.segmentation_dataset:
                    continue
                if not self._matches_filters(data, body_part, modality):
                    continue
                valid_rsopids.append(rsopid)
                self.filtered_dataset[rsopid] = data
            if valid_rsopids:
                filtered_snippet2rsopids[caption] = valid_rsopids

        if not filtered_snippet2rsopids:
            raise ValueError("No samples left after applying the provided filters.")

        if dataset_size is not None:
            if dataset_size > len(filtered_snippet2rsopids):
                print(
                    f"Dataset size {dataset_size} is larger than the actual dataset size {len(filtered_snippet2rsopids)}. "
                    "Using the actual dataset size."
                )
                dataset_size = len(filtered_snippet2rsopids)
            selected_keys = list(filtered_snippet2rsopids.keys())[:dataset_size]
            filtered_snippet2rsopids = {k: filtered_snippet2rsopids[k] for k in selected_keys}
            allowed_rsopids = {
                rsopid
                for values in filtered_snippet2rsopids.values()
                for rsopid in values
            }
            self.filtered_dataset = {rsopid: self.filtered_dataset[rsopid] for rsopid in allowed_rsopids}

        self.snippet2rsopids = filtered_snippet2rsopids
        self.keys = list(self.snippet2rsopids.keys())
        print(f"Filtered dataset size after grouping: {len(self.snippet2rsopids)} captions")

        from radgrounder.paths import REFRAD2D_DICOM_DIR
        self.slices_path = str(REFRAD2D_DICOM_DIR)
        self.image_shape = (img_size, img_size)
        self.dataset_stats = {
            "avr_ct_mean": -610.1908535827807,
            "avr_ct_std": 737.3882466073229,
            "avr_mr_mean": 141.1154960399739,
            "avr_mr_std": 255.40429635138338,
        }

        self.augment = augment
        if self.augment:
            self.seg_aug_pipeline = build_segment_aug_pipeline()
            self.default_aug_pipeline = build_augmentation_pipeline()

        self.torch_dtype = torch.bfloat16
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


    def __len__(self):
        return len(self.snippet2rsopids)

  
    
    def __getitem__(self, idx):
        caption = self.keys[idx]
        candidates = self.snippet2rsopids[caption]
        rsopid = candidates[0] if self.eval_mode else self._sample(candidates)

        data = self.filtered_dataset[rsopid]

        # Load image
        dicom_path = os.path.join(self.slices_path, rsopid[:2], rsopid + ".dcm.zst")
        modality = data["modality"]
        if rsopid in self.segmentation_dataset:
            seg_path = self.segmentation_dataset[rsopid]["segment_slice_path"]
            # seg_path = data["segment_slice_path"]
            image, seg_mask = self.load_image_and_seg_mask(dicom_path, seg_path, modality)
            prefix, prompt, suffix, language, question_type, binary_masks, binary_mask_keywords = self.load_segmentation(data, rsopid, seg_mask)
        else:
            binary_mask_keywords = []
            binary_masks = None
            image, _ = self.load_image_and_seg_mask(dicom_path, segment_path=None, modality=modality)
            prefix, prompt, suffix, language, question_type =  self.load_text(data)

        
        image_tensor = image.type(self.torch_dtype)
        prefix_full = f"<image>\n{prefix}\n{prompt} "
        modality = data["modality"]
        
        info = {
            "dicom_path": dicom_path,
            "modality": modality,
            "language": language,
            "question_type": question_type,
            "binary_mask_keywords": binary_mask_keywords,
        }
        
        return image_tensor, binary_masks, prefix_full, suffix, info, self.eval_mode

    def load_image_and_seg_mask(self, dicom_path, segment_path=None, modality=None):
        try:
            dicom_array = read_dicom_as_numpy(dicom_path)
        except Exception as e:
            print(f"Error loading image {dicom_path}: {e}")
            return None

        dicom_array = torch.from_numpy(dicom_array)
        if modality == "CT":
            dicom_array = torch.clip(dicom_array, min=-1000)
        dicom_array = dicom_array.unsqueeze(0)  # Add channel dimension
        # print(f"Loaded image {dicom_path} with shape {dicom_array.shape}")

        seg_mask = None
        if segment_path:
            seg_mask = np.load(segment_path, allow_pickle=True)
            seg_mask = torch.from_numpy(seg_mask).unsqueeze(0).float()
        

        if self.augment:
            if segment_path:
                aug_input = {"image": dicom_array, "seg_mask": seg_mask}
                aug_output = self.seg_aug_pipeline(aug_input)
                dicom_array = aug_output["image"]
                seg_mask = aug_output["seg_mask"]
            else:
                dicom_array = self.default_aug_pipeline(dicom_array)    

        normalized_image = self.normalize_image(dicom_array, modality, eval_mode=self.eval_mode)
        image = normalized_image

        #resize to 224 x 224
        image = F.interpolate(image.unsqueeze(0), size=self.image_shape, mode='bilinear', align_corners=False)
        image = image[0]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        if segment_path:
            seg_mask = F.interpolate(seg_mask.unsqueeze(0), size=self.image_shape, mode='nearest')

        return image, seg_mask

    def normalize_image(self, image, modality=None, eval_mode=False):
        norm_img = apply_normalization(
            image,
            modality,
            self.normalization,
            dataset_stats=self.dataset_stats,
        )

        if self.normalization.strategy == NormalizationType.MEDGEMMA and not eval_mode and self.augment:
            if self._random_float() < GRAYSCALE_PROB:
                channel_idx = self.random_choice_function(np.arange(norm_img.shape[0]))
                norm_img = norm_img[channel_idx, :, :].unsqueeze(0).repeat(3, 1, 1)

        return norm_img

    def load_text(self, data):
        language = self.language
        if language == "all":
            language = self._sample(["english", "german"])
        # print(data[language]["dicom_vqa"], data[language]["snippet_vqa"])
        quest_type = self._sample(self.question_types)
        
        if quest_type == "report":
            prefix, prompt, suffix, question_type = self.load_report(data, language)
        elif quest_type == "vqa":
            prefix, prompt, suffix, question_type = self.load_vqa(data, language)

        # adaptively cut prefix and suffix to fit within max_length
        len_prompt = len(prompt.split())
        effective_max_length = self.max_length - len_prompt
        prefix, suffix = adaptively_cut_text(prefix, suffix, effective_max_length)

        return prefix, prompt, suffix, language, question_type

    def load_segmentation(self, data, rsopid, seg_mask):
        sample = self.segmentation_dataset[rsopid]
        language = self.language
        if language == "all":
            language = self._sample(["english", "german"])

        question_type = "detection"
        if language == "english":
            keywords = sample["keywords_eng"]
            if len(keywords) > 0:
                prompt = "Caption this image and segment keywords:"
            else:
                prompt = "Caption this image:"
        elif language == "german":
            keywords = sample["keywords_de"]
            if len(keywords) > 0:
                prompt = "Beschreibe dieses Bild und segmentiere die Schlüsselwörter:"
            else:
                prompt = "Beschreibe dieses Bild:"

        prefix = self.load_prefix(data, language)
        suffix = self.get_report_content(data[language], "CleanedSentence")

        suffix_w_segment_tokens, segmented_keywords = self.add_segmentation_tokens(suffix, keywords, seg_mask)
        # binary_masks = self.load_binary_segm_maps(sample, segmented_keywords)
        binary_masks = self.convert_mask_to_binary(seg_mask, segmented_keywords)
        # print(f"load_segmentation: Binary masks shape: {binary_masks.shape}, Binary mask keywords: {segmented_keywords}")
        return prefix, prompt, suffix_w_segment_tokens, language, question_type, binary_masks, segmented_keywords

    def add_segmentation_tokens(self, suffix, keywords, seg_mask):
        filtered_keywords = []
        segmented_keywords = []
        # If a keyword is in other keyword, then remove it
        keywords = sorted(keywords, key=lambda x: len(x[0]), reverse=True)
        for i, (keyword, class_name, class_id) in enumerate(keywords):
            if not any(keyword in other_kw[0] for j, other_kw in enumerate(keywords) if i != j):
                filtered_keywords.append((keyword, class_name, class_id))

        for keyword, class_name, class_id in filtered_keywords:
            if keyword in suffix and class_id in seg_mask:
                count = suffix.count(keyword)
                keyword_pattern = fr"\b{re.escape(keyword)}\b"
                suffix = re.sub(keyword_pattern, f"{SEG_START}{keyword}{SEG_END}", suffix)
                for _ in range(count):
                    segmented_keywords.append((keyword, class_id))


        # print(f"Suffix: {suffix}")
        # print(f"Segmented IDs: {segmented_ids}")

        return suffix, segmented_keywords

    # def load_binary_segm_maps(self, sample, segmented_keywords):
    #     segment_path = sample["segment_slice_path"]
    #     seg_map = np.load(segment_path, allow_pickle=True)
    #     #filter the seg_map to only include the segmented ids
    #     # filtered_seg_map = np.vectorize(lambda x: x if x in segmented_ids else 0)(seg_map)
        
    #     seg_map = torch.from_numpy(seg_map).unsqueeze(0).float()
    #     seg_map = F.interpolate(seg_map.unsqueeze(0), size=self.image_shape, mode='nearest')

    #     binary_masks = torch.zeros((len(segmented_keywords), *self.image_shape), dtype=seg_map.dtype)
    #     for i, (keyword, class_id) in enumerate(segmented_keywords):
    #         binary_masks[i] = (seg_map == class_id).float()
    #     # seg_map = seg_map[0].squeeze(0)  # Remove batch and channel dimensions
    #     return binary_masks
    
    def convert_mask_to_binary(self, seg_mask, segmented_keywords):
        binary_masks = torch.zeros((len(segmented_keywords), *self.image_shape), dtype=seg_mask.dtype)
        for i, (keyword, class_id) in enumerate(segmented_keywords):
            binary_masks[i] = (seg_mask == class_id).float()
        return binary_masks

    def load_vqa(self, data, language):
        vqa_list = list(data[language]["dicom_vqa"]) + list(data[language]["snippet_vqa"])
        # print(f"VQA list length: {len(vqa_list)}")
        if not vqa_list:
            return None
        
        selected_vqa = self._sample(vqa_list)
        
        prefix = selected_vqa["question"]
        question_type = selected_vqa["question_type"]
        if question_type == "multiple":
            prefix += "\n" + "\n".join(selected_vqa["choices"])
            
        suffix = selected_vqa["answer"]
        if language == "english":
            prompt = "Answer:"
        elif language == "german":
            prompt = "Antwort:"
        
        return prefix, prompt, suffix, question_type
    
    def load_report(self, data, language):
        question_type = "report"
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
            prefix_type = self._sample(self.prefix_options)
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
            key = prefix_type

        content = lan_data.get(key, "")
        if content:
            content = clean_punctuation(content)

        return content

    def _setup_random_generators(self):
        if self.eval_mode:
            self._rng = np.random.default_rng(42)
            self.random_choice_function = lambda seq: seq[self._rng.integers(len(seq))]
            self._random_float = lambda: self._rng.random()
        else:
            self.random_choice_function = lambda seq: np.random.choice(seq)
            self._random_float = lambda: np.random.rand()

    def _sample(self, options):
        if isinstance(options, set):
            options = list(options)
        choice = self.random_choice_function(options)
        return choice.item() if hasattr(choice, "item") else choice

    def _matches_filters(self, data, body_part, modality):
        if modality and modality.lower() != "all":
            sample_mod = data.get("modality")
            sample_mod = sample_mod.lower() if isinstance(sample_mod, str) else None
            if sample_mod != modality.lower():
                return False

        if body_part and body_part.lower() != "all":
            sample_body_part = data.get("body_part")
            sample_body_part = sample_body_part.lower() if isinstance(sample_body_part, str) else None
            if sample_body_part != body_part.lower():
                return False

        return True

    # def load_vqa(self, data, language):
    #     vqa_list = list(data[language]["dicom_vqa"]) + list(data[language]["snippet_vqa"])
    #     if not vqa_list:
    #         return None
    #     selected_vqa = np.random.choice(vqa_list)
    #     prefix = selected_vqa["question"]
    #     question_type = selected_vqa["question_type"]
    #     if question_type == "multiple":
    #         prefix += "\n" + "\n".join(selected_vqa["choices"])

    #     suffix = selected_vqa["answer"]
    #     prompt = "Answer:"
        
    #     return prefix, prompt, suffix, question_type


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
        # Check if batch contains segmentation masks (6 elements per sample)
        images, binary_masks, prefixes, suffixes, infos, eval_mode = zip(*batch)

        output_kwargs = {"text_kwargs": {"max_length": seq_len, "truncation": False,
                                          "return_tensors": "pt", "padding": "longest"}}
        if eval_mode[0]:
            text_inputs = tokenize_text(processor, prefixes, None, output_kwargs)
            #we need this to prepare the gt data for evaluation
            input_ids = tokenize_text(processor, prefixes, suffixes, output_kwargs)["input_ids"]
        else:
            text_inputs = tokenize_text(processor, prefixes, suffixes, output_kwargs)
            input_ids = text_inputs["input_ids"]

        # seg_start_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_START)
        seg_end_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_END)
        # print(f"Input ids shape: {input_ids.shape}, seg_start_token_id: {seg_start_token_id}, seg_end_token_id: {seg_end_token_id}")
        seg_token_pos = []
        ordered_binary_masks = []
        mask_labels = []
        for i in range(len(input_ids)):
            # Find the start and end token indices for segment tokens
            # start_token_indices = (input_ids[i] == seg_start_token_id).nonzero(as_tuple=True)[0]
            end_token_indices = (input_ids[i] == seg_end_token_id).nonzero(as_tuple=True)[0]
            for mask_idx, end_idx in enumerate(end_token_indices):
                end_idx = end_idx.item()
                seg_token_pos.append((i, end_idx))
                ordered_binary_masks.append(binary_masks[i][mask_idx])
                mask_label = infos[i]["binary_mask_keywords"][mask_idx][1]
                mask_labels.append(mask_label)

        image_inputs = torch.stack(images).to(torch_dtype)
        if len(ordered_binary_masks) == 0:
            binary_masks = torch.empty((0, image_inputs.shape[2], image_inputs.shape[3]), dtype=torch_dtype)
            seg_token_pos = None
        else:
            binary_masks = torch.stack(ordered_binary_masks).to(torch_dtype)
            seg_token_pos = torch.tensor(seg_token_pos)

        if eval_mode[0]:
            inputs = {"pixel_values": image_inputs, "segment_gt": None, "seg_token_pos": None, **text_inputs}
        else:
            inputs = {"pixel_values": image_inputs, "segment_gt": binary_masks, "seg_token_pos": seg_token_pos, **text_inputs}

        if return_infos:
            return inputs, prefixes, suffixes, infos, binary_masks, seg_token_pos, mask_labels

        return inputs    
    return collate_fn


if __name__ == "__main__":
    # Example usage
    from collections import defaultdict
    from transformers import PaliGemmaProcessor
    from radgrounder.dataset.segmentation.total_segmentator.read_segmentation import visualize_slices_with_segmentation
    MODEL_ID ="google/paligemma2-3b-pt-224"
    cache_dir = None
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
    processor.tokenizer.add_special_tokens({'additional_special_tokens': [SEG_START, SEG_END]})
    language = "german"  # or "german" or "all"
    normalization = NormalizationConfig(strategy=NormalizationType.MEDGEMMA)
    dataset = RefRad2DSegment(split="train", img_size=224, eval_mode=True, language=language, augment=False, only_segmented=True, question_types="all",
                         dataset_size=None, max_length=200, prefix_style="none", body_part="ALL", modality="all", normalization=normalization)
    
    g = torch.Generator()
    g.manual_seed(43)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, generator=g, collate_fn=get_collate_fn(processor, 200, torch.bfloat16, return_infos=True), num_workers=4, pin_memory=True)

    import json
    with open("radgrounder/dataset/segmentation/label_map/merged_label_map.json", "r") as f:
        label_map = json.load(f)
    label_map = {int(v): k for k, v in label_map.items()}

    # print(f"label_map: {label_map}")
    values = 0
    val_squares = 0
    num_elements = 0
    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        # image_tensor, prefix_full, suffix, info, eval_mode = batch
        inputs, prefixes, suffixes, infos, binary_gt_masks, seg_token_pos, mask_labels = batch
        pixel_values = inputs["pixel_values"]
        # seg_token_pos = inputs["seg_token_pos"]
        if binary_gt_masks is None or seg_token_pos is None:
            print("No segmentation mask for this sample...")
            print(f"Pixel values shape: {pixel_values.shape})")
            continue
        # print(f"Pixel values shape: {pixel_values.shape}, Segmentation map shape: {binary_gt_masks.shape}, seg_token_pos: {seg_token_pos.shape}")
        # # seg_token_pos_grouped = {}
        # # for i, pos in enumerate(seg_token_pos):
        # #     if pos[0] not in seg_token_pos_grouped:
        # #         seg_token_pos_grouped[pos[0]] = []
        # #     seg_token_pos_grouped[pos[0]].append((i, pos[1]))
        # # print(infos)
        # # mask_labels = inputs["mask_labels"]
        # keyword_count_for_the_same_slice = defaultdict(int)
        # for mask_id, (binary_map, seg_pos) in enumerate(zip(binary_gt_masks, seg_token_pos)): 
        #     i = seg_pos[0]
        #     rsopid = infos[i]["dicom_path"].split("/")[-1].split(".")[0]
        #     save_path = f"./samples_from_dataloader/segment/{rsopid}_{seg_pos[1]}.png"
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     image_slice = pixel_values[i].cpu().float().permute(1, 2, 0).numpy()
        #     binary_seg_slice = binary_map.cpu().int().numpy()

        #     binary_mask_keyword = infos[i]["binary_mask_keywords"]
        #     print(f"Binary mask keyword: {binary_mask_keyword}")
        #     # keyword_idx = keyword_count_for_the_same_slice[rsopid]
        #     # # print(f"Keyword index for the same slice: {keyword_idx}")
        #     # class_id = binary_mask_keyword[keyword_idx][1] if binary_mask_keyword else 1
        #     # keyword_count_for_the_same_slice[rsopid] += 1
        #     # print("extracted class id:", class_id)
        #     print("mask_label", mask_labels[mask_id])
        #     class_id = mask_labels[mask_id]

        #     seg_slice = binary_seg_slice * class_id
        #     print(f"Image slice shape: {image_slice.shape}, Segmentation slice shape: {seg_slice.shape}")
        #     unique_labels = np.unique(seg_slice)
        #     print(f"Unique labels in segmentation slice: {unique_labels}")
        #     # Visualize the slices with segmentation
        #     visualize_slices_with_segmentation(image_slice, seg_slice, label_map, slice_idx=0, modality=infos[i]["modality"],
        #                                        save_path=save_path, caption=suffixes[i])
        
        
        #vizualize combined segmentation for each slice
        # samples_with_masks = []
        # segment_maps = defaultdict(list)
        # for mask_id, (binary_map, seg_pos) in enumerate(zip(binary_gt_masks, seg_token_pos)):
        #     i = seg_pos[0].item()
        #     samples_with_masks.append(i)
        #     slice_name = infos[i]["dicom_path"].split("/")[-1].split(".")[0]
        #     segment_maps[slice_name].append((mask_id, binary_map, seg_pos))

        # for slice_name, masks in segment_maps.items():
        #     first_mask_id, first_binary_map, first_seg_pos = masks[0]
        #     image_index = first_seg_pos[0].item()

        #     combined_seg_slice = torch.zeros_like(first_binary_map, dtype=torch.float32)
        #     for mask_id, binary_map, seg_pos in masks:
        #         class_id = mask_labels[mask_id]
        #         class_value = int(class_id.item()) if hasattr(class_id, "item") else int(class_id)
        #         combined_seg_slice = torch.maximum(
        #             combined_seg_slice,
        #             binary_map.to(dtype=combined_seg_slice.dtype) * class_value,
        #         )

        #     combined_seg_slice = combined_seg_slice.cpu().numpy().astype(int)
        #     image_slice = pixel_values[image_index].cpu().float().permute(1, 2, 0).numpy()
        #     save_path = f"./samples_from_dataloader/segment/{language}/{slice_name}_combined.png"
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     caption = suffixes[image_index]
        #     #example <seg>liver</seg> extract liver
        #     # pattern = r"<seg>(.*?)</seg>"
        #     # regex = r"<.*?>|</.*?>"
        #     # caption = re.sub(regex, "", caption)
        #     print(caption)
        #     visualize_slices_with_segmentation(
        #         image_slice,
        #         combined_seg_slice,
        #         label_map,
        #         slice_idx=0,
        #         title="RefRad2D Grounded Sample",
        #         save_path=save_path,
        #         caption=caption,
        #         prefix=prefixes[image_index]
        #     )



        # exit()