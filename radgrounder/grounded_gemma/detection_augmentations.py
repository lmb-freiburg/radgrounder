"""Example augmentation pipeline using MONAI applying the SAME random crop
to both the image and its bounding boxes.

This file defines:
 1. A custom Randomizable MapTransform `RandCropImageBoxesd` that:
       - samples a crop window (roi_size) inside the spatial size of the image
       - crops the image tensor
       - adjusts (and clips) all bounding boxes to the new cropped coordinates
       - drops boxes completely outside the crop
 2. A demo creating a dummy (1, H, W) image with a white rectangle + box.
 3. Visualization of original vs augmented (cropped) result with boxes.

Bounding box format assumed: [xmin, ymin, xmax, ymax] in pixel coordinates.
Channel-first image shape: (C, H, W)
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Dict, Any

from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    MapTransform,
    Randomizable,
    RandAdjustContrastd,
    RandShiftIntensityd,
)
from monai.utils import ensure_tuple


class RandCropImageBoxesd(Randomizable, MapTransform):
    """Random spatial crop with size sampled as a percentage of image dimensions.

    Instead of passing a fixed `roi_size`, you specify a percentage range for height
    and width. The same random crop is applied to the image and bounding boxes.

    Args:
        keys: list of keys (first key must be the image)
        box_key: key holding (N,4) boxes in [xmin, ymin, xmax, ymax]
        label_key: key with (N,) labels (filtered alongside boxes)
        crop_rel_range: tuple specifying ( (min_h, max_h), (min_w, max_w) ) where each value
                        is in (0,1]. If a single 2-tuple is given, it's applied to both dims.
                        Example: ((0.6,0.9),(0.6,0.9)) means sample height percent p_h ~ U[0.6,0.9]
                        and width percent p_w ~ U[0.6,0.9].
        keep_aspect: if True, the same sampled percentage is used for both H and W (uses the
                     height range); width range is ignored.
        random_center: if True choose random top-left; else center crop.
        min_box_coverage: minimum intersection area in pixels to keep a box.
        allow_empty: allow crops with zero boxes (otherwise one retry performed).
        prob: probability of applying the crop. If the random draw fails, image & boxes
              are returned unchanged (aside from casting) and `crop_start` is None.
    """

    def __init__(
        self,
        keys: KeysCollection,
        box_key: str = "boxes",
        label_key: str | None = "labels",
        crop_rel_range: Sequence[Sequence[float]] | Sequence[float] = ((0.7, 0.9), (0.7, 0.9)),
        keep_aspect: bool = False,
        random_center: bool = True,
        min_box_coverage: int = 1,
        allow_empty: bool = True,
        prob: float = 1.0,
    ) -> None:
        super().__init__(keys)
        self.box_key = box_key
        self.label_key = label_key
        self.prob = float(prob)
        if not (0.0 <= self.prob <= 1.0):
            raise ValueError("prob must be in [0,1]")
        # Normalize crop_rel_range
        crr = ensure_tuple(crop_rel_range)
        if len(crr) == 2 and all(isinstance(v, (float, int)) for v in crr):
            # single (min,max) -> apply to both dims
            self.crop_rel_range = ((float(crr[0]), float(crr[1])), (float(crr[0]), float(crr[1])))
        elif len(crr) == 2 and all(hasattr(v, "__len__") for v in crr):
            self.crop_rel_range = (
                (float(crr[0][0]), float(crr[0][1])),
                (float(crr[1][0]), float(crr[1][1])),
            )
        else:
            raise ValueError("crop_rel_range must be ((min_h,max_h),(min_w,max_w)) or (min,max)")
        for rng in self.crop_rel_range:
            if not (0 < rng[0] <= rng[1] <= 1.0):
                raise ValueError("Each relative range must satisfy 0 < min <= max <= 1")
        self.keep_aspect = keep_aspect
        self.random_center = random_center
        self.min_box_coverage = int(min_box_coverage)
        self.allow_empty = allow_empty
        self._crop_start: Tuple[int, int] | None = None
        self._roi_size: Tuple[int, int] | None = None  # actual sampled size per call

    def randomize(self, img_spatial: Tuple[int, int]) -> None:
        H, W = img_spatial
        (h_min, h_max), (w_min, w_max) = self.crop_rel_range
        if self.keep_aspect:
            p = self.R.uniform(h_min, h_max)
            ph, pw = p, p
        else:
            ph = self.R.uniform(h_min, h_max)
            pw = self.R.uniform(w_min, w_max)
        rh = int(round(H * ph))
        rw = int(round(W * pw))
        rh = max(1, min(rh, H))
        rw = max(1, min(rw, W))
        self._roi_size = (rh, rw)
        if self.random_center:
            y0 = self.R.randint(0, H - rh + 1)
            x0 = self.R.randint(0, W - rw + 1)
        else:
            y0 = (H - rh) // 2
            x0 = (W - rw) // 2
        self._crop_start = (y0, x0)

    def _apply_crop(self, image: torch.Tensor) -> torch.Tensor:
        if self._crop_start is None:
            raise RuntimeError("Crop start not randomized")
        y0, x0 = self._crop_start
        if self._roi_size is None:
            raise RuntimeError("ROI size not set; randomize() not called")
        rh, rw = self._roi_size
        return image[..., y0 : y0 + rh, x0 : x0 + rw]

    def _adjust_boxes(
        self, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # (filtered_boxes, keep_mask)
        """Clip and translate boxes into cropped coordinate system.
        Returns adjusted boxes and mask of which boxes kept.
        """
        if self._crop_start is None:
            raise RuntimeError("Crop start not randomized")
        y0, x0 = self._crop_start
        if self._roi_size is None:
            raise RuntimeError("ROI size not set; randomize() not called")
        rh, rw = self._roi_size
        crop_xmin, crop_ymin = x0, y0
        crop_xmax, crop_ymax = x0 + rw, y0 + rh

        if boxes.size == 0:
            return boxes.copy(), np.zeros((0,), dtype=bool)

        # boxes: [xmin, ymin, xmax, ymax]
        xmin = boxes[:, 0]
        ymin = boxes[:, 1]
        xmax = boxes[:, 2]
        ymax = boxes[:, 3]

        # Intersection with crop
        ixmin = np.maximum(xmin, crop_xmin)
        iymin = np.maximum(ymin, crop_ymin)
        ixmax = np.minimum(xmax, crop_xmax)
        iymax = np.minimum(ymax, crop_ymax)

        inter_w = np.clip(ixmax - ixmin, a_min=0, a_max=None)
        inter_h = np.clip(iymax - iymin, a_min=0, a_max=None)
        inter_area = inter_w * inter_h

        keep = inter_area >= self.min_box_coverage

        # Clip & translate kept boxes
        kept_boxes = np.stack([ixmin, iymin, ixmax, iymax], axis=1)[keep]
        # Translate to new origin
        kept_boxes[:, [0, 2]] -= crop_xmin
        kept_boxes[:, [1, 3]] -= crop_ymin

        # Ensure within [0, roi_size]
        kept_boxes[:, 0::2] = np.clip(kept_boxes[:, 0::2], 0, rw)
        kept_boxes[:, 1::2] = np.clip(kept_boxes[:, 1::2], 0, rh)

        return kept_boxes.astype(boxes.dtype), keep

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        image = d[self.keys[0]]  # expecting first key is image
        if isinstance(image, np.ndarray):
            image_t = torch.from_numpy(image)
        else:
            image_t = image  # assume torch.Tensor (C,H,W)
        if image_t.ndim != 3:
            raise ValueError("Image must be (C,H,W)")

        H, W = image_t.shape[-2:]

        # Decide whether to apply
        if self.R.random() > self.prob:
            # No-op: still ensure numpy vs tensor consistency
            d["crop_start"] = None
            d["roi_size"] = (H, W)
            return d

        # Apply crop logic (with one optional retry if not allow_empty)
        for attempt in (0, 1):
            self.randomize((H, W))
            boxes_arr = d.get(self.box_key, np.zeros((0, 4), dtype=np.float32))
            boxes_np = np.asarray(boxes_arr)
            adj_boxes, keep_mask = self._adjust_boxes(boxes_np)
            if (adj_boxes.size > 0) or self.allow_empty or attempt == 1:
                break

        cropped = self._apply_crop(image_t)

        # Update dict with cropped image
        d[self.keys[0]] = cropped if isinstance(image, torch.Tensor) else cropped.numpy()
        d[self.box_key] = adj_boxes
        if self.label_key and self.label_key in d:
            labels = np.asarray(d[self.label_key])
            d[self.label_key] = labels[keep_mask]
        d["crop_start"] = self._crop_start
        d["roi_size"] = self._roi_size
        return d


def _draw_boxes(ax, boxes: np.ndarray, labels: np.ndarray | None = None):
    """Draw boxes with optional class-based coloring.

    A small color palette is used cyclically if labels provided.
    """
    if boxes is None or len(boxes) == 0:
        return
    palette = ["lime", "yellow", "cyan", "magenta", "orange", "red"]
    for i, b in enumerate(boxes):
        xmin, ymin, xmax, ymax = b
        w = xmax - xmin
        h = ymax - ymin
        if labels is not None and i < len(labels):
            color = palette[int(labels[i]) % len(palette)]
            label_text = str(int(labels[i]))
        else:
            color = "r"
            label_text = ""
        rect = plt.Rectangle((xmin, ymin), w, h, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        if label_text:
            ax.text(xmin + 2, ymin + 10, label_text, color=color, fontsize=9, weight="bold")

def plot_img_with_boxes(
    img,
    bboxes,
    labels,
    save_path=None,
    title=None,
    caption=None,
    prefix=None,
    font_size=25,
    label_map=None,
    show=True,
):
    """
    Visualize image slice with bounding boxes and label legend, similar to visualize_slices_with_segmentation.
    Args:
        img: 2D image array
        bboxes: (N,4) array of [xmin, ymin, xmax, ymax]
        labels: (N,) array of int labels
        save_path: path to save output
        title: plot title
        caption: optional caption text
        prefix: optional prefix text
        font_size: font size for title/prefix/caption
        label_map: dict mapping label int to name
        show: whether to show plot
    """
    img_norm = (img - np.min(img)) / (np.ptp(img) + 1e-6)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_norm, cmap="gray")
    # Draw bounding boxes and label numbers
    # palette = ["lime", "yellow", "cyan", "magenta", "orange", "red", "blue", "purple"]
    cmap = plt.get_cmap("tab20")
    legend_patches = []
    if bboxes is not None and len(bboxes) > 0:
        for i, box in enumerate(bboxes):
            xmin, ymin, xmax, ymax = box
            w, h = xmax - xmin, ymax - ymin
            label = labels[i] if labels is not None and i < len(labels) else None
            color = cmap(int(label) % 20) if label is not None else "r"
            rect = plt.Rectangle((xmin, ymin), w, h, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            # Draw label number at top-left corner
            ax.text(
                xmin + 2, ymin + 2, str(int(label)) if label is not None else "",
                color="white", fontsize=18, ha="left", va="top",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
            )
            # Build legend patch
            if label_map:
                name = label_map.get(int(label), "Unknown")
            else:
                name = str(label) if label is not None else "Unknown"
            patch = plt.Line2D([0], [0], color=color, lw=4, label=f"{label}: {name}")
            legend_patches.append(patch)
    if title:
        ax.set_title(title, fontdict={"fontsize": font_size})
    ax.axis("off")

    # Add prefix (prompt) text
    if prefix:
        prefix = "Prompt: " + prefix
        text_obj = ax.text(
            1.03, 0.98, prefix,
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=font_size,
            wrap=True,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
            clip_on=False,
            verticalalignment='top',
            horizontalalignment='left',
            multialignment='left'
        )
        text_obj._get_wrap_line_width = lambda : 2000

    # Add caption if provided
    if caption:
        import re
        from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox

        # --- This part for parsing the caption remains the same ---
        segments = []
        last_idx = 0
        # Use your existing regex to parse the caption into colored segments
        for match in re.finditer(r"<p bbox=((<loc\d{4}>){4}) id=<seg(\d{3})>>(.*?)</p>", caption):
            start, end = match.span()
            keyword = match.group(4)
            label = int(match.group(3))
            color = cmap(label % 20)
            if start > last_idx:
                segments.append((caption[last_idx:start], None))
            segments.append((keyword, color))
            last_idx = end
        if last_idx < len(caption):
            segments.append((caption[last_idx:], None))
        
        cleaned_segments = []
        for seg, color in segments:
            if color is None:
                seg = re.sub(r"<p [^>]*>(.*?)</p>", r"\1", seg)
            cleaned_segments.append((seg, color))


        # --- Start of wrapping logic with bolding for colored text ---

        # Max characters per line (approximate; you may need to tune this value)
        max_chars_per_line = 40 

        vbox_lines = []
        
        # Start the first line with a bold "Caption: " prefix
        prefix_area = TextArea("Caption: ", textprops=dict(color="black", fontsize=font_size, weight="bold"))
        current_line_text_areas = [prefix_area]
        current_line_char_count = len("Caption: ")

        for seg, color in cleaned_segments:
            if not seg:
                continue
            
            # Split each segment by words to handle wrapping gracefully
            words = seg.split() 
            for word in words:
                # --- KEY CHANGE HERE ---
                # Set text properties: if a color exists, set weight to 'bold'
                text_props = {
                    "color": color if color else "black",
                    "fontsize": font_size,
                    "weight": "bold" if color else "normal" 
                }
                word_area = TextArea(word, textprops=text_props)
                word_len = len(word)

                # If adding the new word exceeds the line limit, pack the current line and start a new one
                if current_line_char_count > 0 and (current_line_char_count + word_len > max_chars_per_line):
                    vbox_lines.append(HPacker(children=current_line_text_areas, pad=0, sep=5, align="left"))
                    current_line_text_areas = [word_area] # Start a new line
                    current_line_char_count = word_len
                else:
                    # Otherwise, add the word to the current line
                    current_line_text_areas.append(word_area)
                    current_line_char_count += word_len + 1 # Add 1 to account for a space

        # Add the last remaining line
        if current_line_text_areas:
            vbox_lines.append(HPacker(children=current_line_text_areas, pad=0, sep=5, align="left"))

        # Pack all horizontal lines vertically, ensuring they are left-aligned
        vbox = VPacker(children=vbox_lines, pad=0, sep=5, align="left")
        
        # Place the AnnotationBbox below the image
        ab = AnnotationBbox(
            vbox,
            (1.02, 0.5),
            xycoords='axes fraction',
            boxcoords="axes fraction",
            frameon=True,
            bboxprops=dict(facecolor='lightgray', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
            pad=0.2,
            box_alignment=(0, 0.5), 
        )
        ax.add_artist(ab)


    # Add legend
    if legend_patches:
        plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(-0.55, 1), fontsize="xx-large", borderaxespad=0, frameon=True)
    # plt.tight_layout()
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved bounding box overlay with label numbers to {save_path}")


def demo(seed: int = 42, save_path: str = "augmentation_example.png"):
    # Create dummy image and a couple of boxes
    H, W = 256, 256
    image = np.zeros((1, H, W), dtype=np.float32)
    # Three overlapping boxes (classes 1,2,3)
    boxes = np.array([
        [40, 60, 170, 190],    # box 1
        [100, 100, 200, 220],  # box 2 overlaps with box1
        [130, 50, 210, 140],   # box 3 overlaps with box1 & box2
    ], dtype=np.float32)
    # Fill boxes with white so visually apparent
    for (xmin, ymin, xmax, ymax) in boxes.astype(int):
        image[0, ymin:ymax, xmin:xmax] = 1.0

    image = torch.from_numpy(image) 
    data = {"image": image, "boxes": boxes, "labels": np.array([1, 2, 3], dtype=np.int64)}

    # Compose pipeline (add more transforms here if desired)
    # Use relative crop between 60% and 85% of each dimension
    crop_transform = RandCropImageBoxesd(
        keys=["image"],
        box_key="boxes",
        label_key="labels",
        crop_rel_range=(0.8, 1),
        random_center=True,
    )
    # Set random state for reproducibility
    # crop_transform.set_random_state(seed=seed)
    pipeline = Compose([
        crop_transform,
        # Intensity augmentations (applied only to image)
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.2)),
        RandShiftIntensityd(keys=["image"], offsets=20.0, prob=0.5),
    ])

    augmented = pipeline(data)

    aug_image = augmented["image"]  # tensor or ndarray
    if isinstance(aug_image, torch.Tensor):
        aug_image_np = aug_image.numpy()
    else:
        aug_image_np = aug_image
    # Prepare for plotting (C,H,W) -> (H,W)
    orig_img_vis = image[0]
    aug_img_vis = aug_image_np[0]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(orig_img_vis, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Original")
    _draw_boxes(axs[0], boxes, labels=data["labels"])
    axs[0].axis("off")

    axs[1].imshow(aug_img_vis, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Cropped")
    _draw_boxes(axs[1], augmented["boxes"], labels=augmented.get("labels"))
    axs[1].axis("off")

    fig.suptitle(f"Rand Crop start={augmented.get('crop_start')}, roi={augmented.get('roi_size')}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved figure to {save_path}")
    # Optionally show
    # plt.show()

    print("Original boxes:\n", boxes)
    print("Augmented boxes:\n", augmented["boxes"])


if __name__ == "__main__":
    demo()

