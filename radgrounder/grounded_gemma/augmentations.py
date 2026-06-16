from monai.transforms import (
    Compose,
    RandScaleCrop,
    RandAdjustContrast,
    RandShiftIntensity
)

from .detection_augmentations import RandCropImageBoxesd
import numpy as np
import torch



def build_augmentation_pipeline(pixel_level=True, pixel_offset=20.0):
    augmentations = [            
            RandScaleCrop(
                roi_scale=0.9,
                random_center=True,
                random_size=True
            ),]
    if pixel_level:
        augmentations.extend([

            RandAdjustContrast(prob=0.3, gamma=(0.7, 1.2)),
            RandShiftIntensity(prob=0.5, offsets=pixel_offset),
        ])

    return Compose(augmentations)

def build_detect_aug_pipeline(pixel_level=True, pixel_offset=20.0):
    from monai.transforms import ( ToTensord,
    RandAdjustContrastd, RandShiftIntensityd
    )
    augmentations = [
            RandCropImageBoxesd(
                keys=["image"],
                box_key="boxes",
                label_key="labels",
                crop_rel_range=(0.9, 1),
                random_center=True,
                prob=1
            ),
    ]

    if pixel_level:
        augmentations.extend([
            # 4. Intensity transforms (apply **only to the image**)
            RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=(0.7, 1.2)
            ),
            RandShiftIntensityd(
                keys=["image"],
                prob=0.5,
                offsets=pixel_offset
            ),

            ToTensord(keys=["image", "boxes", "labels"])
        ])
    else:
        augmentations.append(ToTensord(keys=["image", "boxes", "labels"]))
    return Compose(augmentations)

def build_segment_aug_pipeline(pixel_level=True):
    from monai.transforms import ( ToTensord,
    RandAdjustContrastd, RandShiftIntensityd, RandScaleCropd
    )
    augmentations = [
        RandScaleCropd(
            keys=["image", "seg_mask"],
            roi_scale=0.9,       # Minimum crop size: 90% of original dimensions
            max_roi_scale=1.0,     # Maximum crop size: 100% of original dimensions
            random_center=True,
        ),
    ]

    if pixel_level:
        augmentations.extend([
            # 4. Intensity transforms (apply **only to the image**)
            RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=(0.7, 1.2)
            ),
            RandShiftIntensityd(
                keys=["image"],
                prob=0.5,
                offsets=20.0
            ),

            ToTensord(keys=["image", "seg_mask"])
        ])
    else:
        augmentations.append(ToTensord(keys=["image", "seg_mask"]))
    return Compose(augmentations)

if __name__ == "__main__":
    from radgrounder.dataset.dataset_manager import DatasetManager
    from PIL import Image
    from radgrounder.grounded_gemma.utils import read_dicom_as_numpy
    import os
    # Load a DICOM image
    dataset_manager = DatasetManager("refrad2d_detect", eval_mode=True, augment=True, only_segmented=True)
    dataset = dataset_manager.dataset
    # image, prefix, suffix = dataset[0]

    # pipeline = build_augmentation_pipeline()
    folder_name = "augmented_images"
    os.makedirs(folder_name, exist_ok=True)

    num_samples = 10
    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples} " + "=" * 20)
        random_idx = np.random.randint(0, len(dataset))
        augmented_image, prefix, suffix, info, _ = dataset[random_idx]
        # print(np.min(image), np.max(image))
        dicom_path = info["dicom_path"]
        print(dicom_path)
        image = read_dicom_as_numpy(dicom_path)
        # image_tensor = torch.tensor(image).unsqueeze(0)  # -> (1, 1, H, W)
        # print("Tensor shape:", image_tensor.shape)
        # augmented_image = pipeline(image_tensor)
        # print("Augmented shape:", augmented_image.shape)

        # Convert augmented image to PIL
        print("Augmented image shape:", augmented_image.shape)
        augmented_image_np = augmented_image.squeeze().to(torch.float16).detach().cpu().numpy()  # (H, W)
        augmented_image_np = np.transpose(augmented_image_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        print("Augmented min:", augmented_image_np.min(), "max:", augmented_image_np.max())
        augmented_image_uint8 = (255 * (augmented_image_np - augmented_image_np.min()) / (augmented_image_np.max() - augmented_image_np.min())).astype(np.uint8)
        aug_pil = Image.fromarray(augmented_image_uint8)

        # Convert original image to PIL
        image_min = image.min()
        image_max = image.max()
        print("Original min:", image_min, "max:", image_max)
        original_image_uint8 = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
        print("Original min:", original_image_uint8.min(), "max:", original_image_uint8.max())
        orig_pil = Image.fromarray(original_image_uint8.squeeze())

        # Save individually (optional)
        orig_pil.save(os.path.join(folder_name, f"image_{i}_original.png"))
        aug_pil.save(os.path.join(folder_name, f"image_{i}_augmented.png"))

        # Combine side by side
        # aug_pil_resized = aug_pil.resize(orig_pil.size, resample=Image.BILINEAR)

        # Create a new blank image to hold both side by side
        width, height = orig_pil.size
        combined = Image.new("L", (2 * width, height))  # Grayscale mode 'L'

        # Paste images
        combined.paste(orig_pil, (0, 0))
        combined.paste(aug_pil, (width, 0))

        # 💾 Save final combined image
        combined.save(os.path.join(folder_name, f"image_{i}_side_by_side.png"))


