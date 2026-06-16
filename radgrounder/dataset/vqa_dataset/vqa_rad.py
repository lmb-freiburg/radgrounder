import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import torch.nn.functional as F

import tqdm
import sys
from typing import Optional

from radgrounder.grounded_gemma.augmentations import build_augmentation_pipeline
from radgrounder.dataset.image_preprocessing import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    NormalizationType,
    apply_normalization,
)

from PIL import Image


VQA_RAD_DATASET_MEAN = (0.3090255, 0.3090255, 0.3090255)
VQA_RAD_DATASET_STD = (0.2935555, 0.2935555, 0.2935555)

def apply_prompt_template(question: str, template_type: str) -> str:
    if template_type == "medgemma":
        # prompt = f"Given this radiology image, which can be a frontal chest X-ray, a single slice head or abdominal CT or MR image, provide a very short, definitive, and concise answer (if possible, a single word) to the following question: "
        prompt = "You may write out your argument before stating your final very short,\
definitive, and concise answer (if possible, a single word or the letter corresponding to your answer\
choice) X in the format 'Final Answer: X': "
        full_prompt = f"<image>\n{question}\n{prompt}"
    else:
        full_prompt = f"<image>\n{question}\nAnswer: "
    return full_prompt

class VqaRad(Dataset):
    def __init__(
        self,
        split="train",
        img_size=224,
        eval_mode=True,
        augment=False,
        max_length=100,
        question_types="all",
        return_dummy_mask=False,
        normalization: Optional[NormalizationConfig] = None,
        prompt_template_type: str = "refrad2d",
    ):
        print("Initializing VQA-RAD Dataset")
        self.normalization = normalization or DEFAULT_NORMALIZATION
        self.prompt_template_type = prompt_template_type
        print("Normalization strategy:", self.normalization.strategy.value)
        print("Prompt template type:", self.prompt_template_type)
        

        if split == "val":
            split = "test"
            
        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'test'].")

        from radgrounder.paths import VQA_RAD_ROOT, VQA_RAD_SPLIT_DIR
        self.image_folder = os.path.join(str(VQA_RAD_ROOT), "images")
        # Split JSONs ship with the repo (data_splits/vqa_rad/); images come from VQA_RAD_ROOT.
        dataset_path = os.path.join(str(VQA_RAD_SPLIT_DIR), f"{split}_fixed_split.json")
        with open(dataset_path, "r") as f:
            vqa_dataset = json.load(f)

        # Optionally filter by open/closed question type (VQA-RAD `q_type` field).
        # Any other value (e.g. "all", "vqa") keeps the full set (combined).
        if question_types in ("vqa_open", "vqa_closed"):
            want = "open" if question_types == "vqa_open" else "closed"
            vqa_dataset = [
                item for item in vqa_dataset
                if str(item.get("q_type", "")).strip().lower() == want
            ]

        self.vqa_dataset = vqa_dataset
        self.split = split
        self.eval_mode = eval_mode
        self.torch_dtype = torch.bfloat16
        self.max_length = max_length
        self.return_dummy_mask = return_dummy_mask
        self.channel_stats = {"mean": VQA_RAD_DATASET_MEAN, "std": VQA_RAD_DATASET_STD}

        print(f"Using split: {split}, eval_mode: {eval_mode}")
        print(f"Dataset size: {len(self.vqa_dataset)}")

        self.image_shape = (img_size, img_size)

        self.augment = augment
        if self.augment and not self.eval_mode:
            self.augmentation_pipeline = build_augmentation_pipeline(pixel_offset=0.1)


    def __len__(self):
        return len(self.vqa_dataset)

    def __getitem__(self, idx):
        # print(self.vqa_dataset[idx])
        item = self.vqa_dataset[idx]

        image_name = item["image_name"]
        image_path = os.path.join(self.image_folder, image_name)
        image = self.load_image(image_path)

        if self.augment and not self.eval_mode:
            image = self.augmentation_pipeline(image)
            
        
        image = F.interpolate(image.unsqueeze(0), size=self.image_shape, mode='bilinear', align_corners=False)
        image = image[0]

        image_tensor = image.type(self.torch_dtype)

        info = item
        info["image_path"] = image_path

        prefix = item["question"]
        template_type = self.prompt_template_type
        prefix = apply_prompt_template(item["question"], template_type=template_type)
        suffix = item["answer"]
        info["system_instruction"] = "You are an expert radiologist."
        
        if self.return_dummy_mask:
            dummy_mask = None
            return image_tensor, dummy_mask, prefix, suffix, info, self.eval_mode

        return image_tensor, prefix, suffix, info, self.eval_mode

    def load_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.image_shape)
                img = torch.from_numpy(np.array(img))
                image = img.permute(2, 0, 1).float()
                image = image / 255.0
                image = apply_normalization(
                    image,
                    modality=None,
                    config=self.normalization,
                    channel_stats=self.channel_stats,
                )
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

 

        return image


if __name__ == "__main__":
    norm_config = NormalizationConfig(NormalizationType.DATASET_STATS)
    dataset = VqaRad(split="test", img_size=224, eval_mode=False, augment=True, max_length=100, normalization=norm_config)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    print("Size of dataset:", len(dataset))
    # total_sum = 0.0
    # total_count = 0
    # for batch in tqdm.tqdm(dataloader):
    #     images, questions, answers, infos, eval_modes = batch
    #     print(images.shape)
    #     # print("Questions:", questions)
    #     # print("Answers:", answers)
    #     total_sum += images.sum().item()
    #     total_count += images.numel()
    #     # For std calculation
    #     if total_count == images.numel():
    #         sum_squared = (images ** 2).sum().item()
    #     else:
    #         sum_squared += (images ** 2).sum().item()

    # mean = total_sum / total_count
    # std = (sum_squared / total_count - mean ** 2) ** 0.5
    # print(f"Mean: {mean}, Std: {std}")