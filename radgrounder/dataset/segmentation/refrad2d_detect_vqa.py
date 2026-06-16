import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from typing import List, Optional
import torch.nn.functional as F
import re
import tqdm

from radgrounder.grounded_gemma.augmentations import build_detect_aug_pipeline, build_augmentation_pipeline
from radgrounder.dataset.image_preprocessing import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    NormalizationType,
    apply_normalization,
)
from radgrounder.dataset.dataset_utils import (
    load_sampled_dataset,
    load_snippet2rsopids,
    tokenize_text,
)

GRAYSCALE_PROB = 0.2

class RefRad2DDetectVQA(Dataset):
    def __init__(
        self,
        split="train",
        img_size=224,
        eval_mode=False,
        language="all",
        augment=False,
        dataset_size=None,
        max_length=100,
        tokenizer=None,
        normalization: Optional[NormalizationConfig] = None,
        modality="ct",
    ):
        print("Initializing RefRad2DDetectVQA dataset")
        self.seed = 42
        self.normalization = normalization or DEFAULT_NORMALIZATION
        print("Normalization strategy:", self.normalization.strategy.value)
        self.eval_mode = eval_mode
        
        self.sampled_dataset = load_sampled_dataset(modality)


        script_dir = os.path.dirname(os.path.abspath(__file__))
        class_name_2_organ_name_path = os.path.join(script_dir, "label_map", "class_2_organ_name_english.json")
        with open(class_name_2_organ_name_path, "r", encoding="utf-8") as f:
            self.class_name_2_organ_name_eng = json.load(f)
        self.organ_name_2_class_name_eng = {v: k for k, v in self.class_name_2_organ_name_eng.items()}

        class_name_2_organ_name_path = os.path.join(script_dir, "label_map", "class_2_organ_name_german.json")
        with open(class_name_2_organ_name_path, "r", encoding="utf-8") as f:
            self.class_name_2_organ_name_german = json.load(f)
        self.organ_name_2_class_name_german = {v: k for k, v in self.class_name_2_organ_name_german.items()}


        template_path = os.path.join(script_dir, "detect_vqa", "vqa_templates_english.json")
        with open(template_path, "r", encoding="utf-8") as f:
            self.question_templates_eng = json.load(f)

        template_path = os.path.join(script_dir, "detect_vqa", "vqa_templates_german.json")
        with open(template_path, "r", encoding="utf-8") as f:
            self.question_templates_german = json.load(f)

        with open(os.path.join(script_dir, "label_map", "merged_label_map.json"), "r", encoding="utf-8") as f:
            self.class_name_2_id = json.load(f)

        self.id_2_class_name = {v: k for k, v in self.class_name_2_id.items()}


        # Split into train/validation/test using snippet-based splits
        print(f"Using split: {split}, eval_mode: {eval_mode}")

        self.split = split
        self.snippet2rsopids = load_snippet2rsopids(split, modality)
        self.filtered_dataset = self._filter_dataset_by_split(self.snippet2rsopids, dataset_size)

        if not self.filtered_dataset:
            raise ValueError(f"No samples available for split '{split}' after filtering.")

        print(f"{self.split} dataset size: {len(self.filtered_dataset)}")

        self.keys = list(self.filtered_dataset.keys())
        self.num_bins = 512
        self.image_shape = (img_size, img_size)    
        self.dataset_stats = {"avr_ct_mean": -610.1908535827807, "avr_ct_std": 737.3882466073229,
                        "avr_mr_mean": 141.1154960399739, "avr_mr_std": 255.40429635138338}
        self.augment = augment
        print(f"augment: {self.augment}")
        if self.augment:
            self.default_augmentation = build_augmentation_pipeline()
            self.detect_augmentation = build_detect_aug_pipeline()

        self.torch_dtype = torch.bfloat16
        self.language = language
        if self.language not in ["english", "german", "all"]:
            raise ValueError(f"Invalid language: {self.language}. Must be one of ['english', 'german', 'all'].")
        self.max_length = max_length
        self._setup_random_generators()
        
        self.replace_braces_pattern = re.compile(r"\{[^}]*\}")
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided in RefRad2DDetectVQA")
        self.tokenizer = tokenizer
    
    

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        slice_name = self.keys[idx]

        sample = self.filtered_dataset[slice_name]

        # Load image
        bboxes, labels = self.load_bbox_and_labels(sample)
        #the augmentation can crop the image thats why first load the image and the augmented bboxes
        image_path = sample["scan_slice_path"]
        modality = sample["modality"]
        image, img_size, bboxes, labels = self.load_image(image_path, bboxes, labels, modality=modality)

        question, answer, language, question_type, detected_bboxes = self.load_detection_qa(bboxes, labels, img_size)
        
        question, answer = self.adaptively_cut_text(question, answer, self.max_length)

        image_tensor = image.type(self.torch_dtype)
        
        prefix_full = f"<image>\n{question} "
        info = {}
        if self.eval_mode:
            info = {
                "image_path": image_path,
                "language": language,
                "question_type": question_type,
                "bboxes": [det_box[0] for det_box in detected_bboxes],
                "labels": [det_box[1] for det_box in detected_bboxes],
                "descrete_bboxes": [det_box[2] for det_box in detected_bboxes]
            }
        
        return image_tensor, prefix_full, answer, info, self.eval_mode

    def load_bbox_and_labels(self, sample):
        bboxes_w_labels = sample["bboxes"]
        labels = [bbox[4] for bbox in bboxes_w_labels]
        labels = np.array(labels, dtype=np.int64)
        bboxes = [[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in bboxes_w_labels]
        bboxes = np.array(bboxes, dtype=np.float32)
        return bboxes, labels

    def load_image(self, slice_path, bboxes=None, labels=None, modality=None):

        slice_img = np.load(slice_path)
        # slice_img = slice_img.T

        slice_img = torch.from_numpy(slice_img)
        if modality == "CT":
            slice_img = torch.clip(slice_img, min=-1000)
        slice_img = slice_img.unsqueeze(0)  # Add channel dimension

        if self.augment:
            if bboxes is not None and labels is not None:
                data = {"image": slice_img, "boxes": bboxes, "labels": labels}
                aug_data = self.detect_augmentation(data)
                slice_img = aug_data["image"]
                bboxes = aug_data["boxes"]
                labels = aug_data["labels"]
            else:
                slice_img = self.default_augmentation(slice_img)

        slice_img = self.normalize_image(slice_img, modality=modality, eval_mode=self.eval_mode)

        #Clipping the values to be between -2 and 2
        org_img_size = slice_img.shape[-2:]
        
        #resize to 224 x 224
        image = F.interpolate(slice_img.unsqueeze(0), size=self.image_shape, mode='bilinear', align_corners=False)
        image = image[0]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if bboxes is not None and bboxes.shape[0] > 0:
            # print(image.shape[2], org_img_size[1])
            scale_x = image.shape[2] / org_img_size[1]
            scale_y = image.shape[1] / org_img_size[0]
            bboxes *= np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
            bboxes = bboxes.astype(int)

        return image, org_img_size, bboxes, labels

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

    def load_detection_qa(self, bboxes, labels, img_size):
        language = self.language
        if language == "all":
            language = self._sample(["english", "german"])

        if language == "english":
            class_name_2_organ_name = self.class_name_2_organ_name_eng
            organ_name_2_class_name = self.organ_name_2_class_name_eng
            question_templates = self.question_templates_eng
        else:
            class_name_2_organ_name = self.class_name_2_organ_name_german
            organ_name_2_class_name = self.organ_name_2_class_name_german
            question_templates = self.question_templates_german
        
        segmented_class_names = []
        segmented_organ_names = []
        labels = [int(label) for label in labels]
        for label in labels:
            class_name = self.id_2_class_name[label]
            segmented_class_names.append(class_name)
            organ_name = class_name_2_organ_name[class_name]
            segmented_organ_names.append(organ_name)

        # print(f"Segmented Class Names: {segmented_class_names}")
        # print(f"Segmented Organ Names: {segmented_organ_names}")
        selected_template = self._sample(question_templates)
        # print(selected_template)
        question_type = selected_template["question_type"]
        question_temp = selected_template["question"]
        answer_temp = selected_template["answer"]
        all_organ_names = set(class_name_2_organ_name.values())
        detected_bboxes = []

        force_negative_response = len(segmented_organ_names) == 0 or bboxes is None or len(bboxes) == 0

        if "negative" in question_type or force_negative_response:
            negative_organ_names = all_organ_names - set(segmented_organ_names)
            selected_negative_organ = self._sample(list(negative_organ_names))
            question = self.replace_braces(question_temp, selected_negative_organ)
            answer = self.replace_braces(answer_temp, selected_negative_organ)
        elif "affirmative" in question_type:
            affirmative_organ_names = set(segmented_organ_names)
            selected_affirmative_organ = self._sample(list(affirmative_organ_names))
            question = self.replace_braces(question_temp, selected_affirmative_organ)
            class_name = organ_name_2_class_name.get(selected_affirmative_organ, "unknown")
            bbox, label = self.find_bbox_for_organ(class_name, bboxes, labels)
            # print(f"{bbox=} {label=}")
            organ_with_det_tokens, descrete_bbox = self.convert_bbox_to_detection_token(bbox, label, selected_affirmative_organ)
            answer = self.replace_braces(answer_temp, organ_with_det_tokens)
            detected_bboxes.append((bbox, label, descrete_bbox))
        else:
            organs_with_det_tokens = []
            for class_name in segmented_class_names:
                bbox, label = self.find_bbox_for_organ(class_name, bboxes, labels)
                organ_name = class_name_2_organ_name.get(class_name, class_name)
                if bbox is not None:

                    organ_with_det_tokens, descrete_bbox = self.convert_bbox_to_detection_token(bbox, label, organ_name)
                    detected_bboxes.append((bbox, label, descrete_bbox))
                    
                else:
                    organ_with_det_tokens = organ_name
                organs_with_det_tokens.append(organ_with_det_tokens)
                
            # print(f"{detected_bboxes=}")
            # display_sliced_img_w_bbox(scan_slice_path, detected_bboxes)
            organ_list = ", ".join(organs_with_det_tokens)
            # print(organ_list)
            question = question_temp
            answer = self.replace_braces(answer_temp, organ_list)


        return question, answer, language, question_type, detected_bboxes


    def replace_braces(self, template, value):
        return self.replace_braces_pattern.sub(value, template)

    def find_bbox_for_organ(self, organ_name, bboxes, labels):
        organ_id = self.class_name_2_id.get(organ_name, -1)
        for i, bbox in enumerate(bboxes):
            id = labels[i]
            if organ_id == id:
                return bbox, id
        return None, None
    
    def convert_bbox_to_detection_token(self, bbox, label, organ_name):
        
        # print(img_size)
        normalized_bbox = [
            bbox[0] / self.image_shape[1],  # x_min
            bbox[1] / self.image_shape[0],  # y_min
            bbox[2] / self.image_shape[1],  # x_max
            bbox[3] / self.image_shape[0],  # y_max
        ]
        # print(normalized_bbox)

        #make sure the bbox is between 0 and 1
        normalized_bbox = [
            np.clip(coord, 0, 1) for coord in normalized_bbox
        ]

        descrete_bbox = [
            int(normalized_bbox[0] * self.num_bins),
            int(normalized_bbox[1] * self.num_bins),
            int(normalized_bbox[2] * self.num_bins),
            int(normalized_bbox[3] * self.num_bins)
        ]
        # print("bbox", bbox, "descrete_bbox", descrete_bbox, "image size:", img_size)
        
        x1, y1, x2, y2 = descrete_bbox
        label_token = f"<seg{label:03d}>"
        start_token = f"<p bbox=<loc{x1:04d}><loc{y1:04d}><loc{x2:04d}><loc{y2:04d}> id={label_token}>"
        end_token = "</p>"
        return f"{start_token}{organ_name}{end_token}", descrete_bbox

    def adaptively_cut_text(self, prefix: str, suffix: str, effective_max_length: int):
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

        tokenized_prefix = self.tokenizer.encode(prefix)
        tokenized_suffix = self.tokenizer.encode(suffix)
        pref_len = len(tokenized_prefix)
        suff_len = len(tokenized_suffix)
        if pref_len + suff_len > effective_max_length:
            max_length_for_suffix = effective_max_length - pref_len
            tokenized_suffix = tokenized_suffix[:max_length_for_suffix]
            tokenized_prefix = tokenized_prefix[:effective_max_length]
            prefix = self.tokenizer.decode(tokenized_prefix)
            suffix = self.tokenizer.decode(tokenized_suffix)

            # find the last </p> in suffix and discard the rest of the text
            last_p_index = suffix.rfind("</p>")
            if last_p_index != -1:
                suffix = suffix[:last_p_index + 4]
            

        return prefix, suffix

    def _setup_random_generators(self):
        if self.eval_mode:
            self._rng = np.random.default_rng(self.seed)
            self.random_choice_function = lambda seq: seq[self._rng.integers(len(seq))]
            self._random_float = lambda: float(self._rng.random())
        else:
            self.random_choice_function = lambda seq: seq[int(np.random.randint(len(seq)))]
            self._random_float = lambda: float(np.random.random())

    def _sample(self, options):
        if isinstance(options, set):
            options = list(options)
        choice = self.random_choice_function(options)
        return choice.item() if hasattr(choice, "item") else choice

    def _filter_dataset_by_split(self, snippet2rsopids, dataset_size):
        allowed_rsopids = {rsopid for rsopids in snippet2rsopids.values() for rsopid in rsopids}
        rsopid_to_slice_names = {}
        for slice_name, sample in self.sampled_dataset.items():
            rsopid = sample.get("rsopid")
            if rsopid is None or rsopid not in allowed_rsopids:
                continue
            rsopid_to_slice_names.setdefault(rsopid, []).append(slice_name)

        ordered_slice_names = []
        for rsopids in snippet2rsopids.values():
            for rsopid in rsopids:
                ordered_slice_names.extend(rsopid_to_slice_names.get(rsopid, []))

        # Remove potential duplicates while preserving order
        seen = set()
        unique_slice_names = []
        for name in ordered_slice_names:
            if name not in seen:
                seen.add(name)
                unique_slice_names.append(name)

        if dataset_size is not None:
            if dataset_size > len(unique_slice_names):
                print(
                    f"Dataset size {dataset_size} is larger than the filtered dataset size {len(unique_slice_names)}. "
                    "Using the actual dataset size."
                )
                dataset_size = len(unique_slice_names)
            unique_slice_names = unique_slice_names[:dataset_size]

        return {k: self.sampled_dataset[k] for k in unique_slice_names}


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
        # seg_map = torch.stack(seg_map).to(torch_dtype)
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
    from radgrounder.dataset.segmentation.total_segmentator.read_segmentation import visualize_slices_with_segmentation
    MODEL_ID ="google/paligemma2-3b-pt-224"
    cache_dir = None
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
    # encoded = processor.tokenizer("This is a test </p>", return_tensors="pt")
    # decoded = processor.tokenizer.decode(encoded["input_ids"][0])
    # print(f"Decoded: {decoded}")
    
    
    language = "all"  # or "german" or "all"
    normalization = NormalizationConfig(strategy=NormalizationType.MEDGEMMA)
    dataset = RefRad2DDetectVQA(split="train", dataset_size=None, img_size=224, eval_mode=True, language=language, augment=True, max_length=200,
                               tokenizer=processor.tokenizer, normalization=normalization, modality="all")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=get_collate_fn(processor, 200, torch.bfloat16, return_infos=True), num_workers=4, pin_memory=True)

    import json
    with open("radgrounder/dataset/segmentation/label_map/merged_label_map.json", "r") as f:
        label_map = json.load(f)
    label_map = {int(v): k for k, v in label_map.items()}

    from radgrounder.grounded_gemma.detection_augmentations import plot_img_with_boxes
    from matplotlib import pyplot as plt
    # print(f"label_map: {label_map}")
    values = 0
    val_squares = 0
    num_elements = 0
    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        # image_tensor, prefix_full, suffix, info, eval_mode = batch
        inputs, prefixes, suffixes, infos = batch 
        for i in range(len(suffixes)):
            print(f"Prefix: {prefixes[i]}")
            print(f"Suffix: {suffixes[i]}")
            # print(f"Info: {infos[i]}")
            bboxes = infos[i]["bboxes"]
            labels = infos[i]["labels"]
            descrete_bboxes = infos[i]["descrete_bboxes"]
            # print(f"BBoxes: {bboxes}")
            # print(f"Labels: {labels}")
            img = inputs["pixel_values"][i]
            img = img.type(torch.float32)
            img = img.permute(1, 2, 0).cpu().numpy()
            # print(img.shape)
            
            # print("BBoxes:", bboxes)
            # print("Descrete BBoxes:", descrete_bboxes)
            # renormalized_bboxes = [224 * np.array(bbox) / 512 for bbox in descrete_bboxes]
            # plot_img_with_boxes(img, renormalized_bboxes, labels, f"./samples_from_dataloader/detect_vqa_image_{i}.png")

            # Find all matches in the suffix string
            suffix = suffixes[i]
            matches = re.finditer(r"<p bbox=((<loc\d{4}>){4}) id=<seg(\d{3})>>", suffix)

            bbox_list = []
            label_list = []
            for match in matches:
                bbox_tokens = re.findall(r"<loc(\d{4})>", match.group(1))
                bbox = [int(224 * int(token) / 512) for token in bbox_tokens]
                label = int(match.group(3))
                bbox_list.append(bbox)
                label_list.append(label)
            print(f"Sample {i} -----------------------")
            print("Extracted bboxes:", bbox_list)
            print("Extracted labels:", label_list)
            print("Suffix:", suffix)
            save_path = f"./samples_from_dataloader/detect_vqa/{language}/detect_vqa_image_{i}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plot_img_with_boxes(
                img,
                bbox_list,
                label_list,
                save_path=save_path,
                title=f"Detection G-VQA RefRad2D Sample",
                caption=suffix,
                prefix=prefixes[i] if prefixes else None,
                font_size=18,
                label_map=label_map,
                show=False,
            )

        exit()




        # values += pixel_values.sum().item()
        # val_squares += (pixel_values ** 2).sum().item()
        # num_elements += pixel_values.numel()
        # print(inputs.keys())

        # for i in range(len(prefixes)):
        #     print(f"Prefix: {prefixes[i]}")
        #     print(f"Suffix: {suffixes[i]}")
        #     print("*" * 20)
       
        # for i in range(len(prefix_full)):
        #     print(f"Image shape: {image_tensor[i].shape}\nPrefix: {prefix_full[i]}\nSuffix: {suffix[i]}")
        #     q_type = info["question_type"][i]
        #     print(f"Info: {q_type}")
        # exit()
    # avr_mean = values / num_elements if num_elements > 0 else 0
    # avr_std = np.sqrt((val_squares / num_elements) - (avr_mean ** 2)) if num_elements > 0 else 0
    # print(f"Mean: {avr_mean}, Std: {avr_std}")
    # print(f"Total elements: {num_elements}")