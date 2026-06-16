import numpy as np
import os
import textwrap

import matplotlib.pyplot as plt
import re
from radgrounder.dataset.segmentation.refrad2d_segment import SEG_START, SEG_END
from highlight_text import ax_text


FONT_SIZE = 12


def plot_masks_on_image(image: np.ndarray, masks: list, keywords: list, keyword_colors: dict, ax=None):
    """
    Plot masks on an image with appropriate colors.
    
    Args:
        image (np.ndarray): The input image (H, W, 3) or (H, W).
        masks (list): List of binary masks to overlay.
        keywords (list): List of keywords corresponding to each mask.
        keyword_colors (dict): Dictionary mapping keywords to colors.
    """
    if ax is None:
        ax = plt.gca()

    # Display the base image
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    
    if masks is not None:
        if keywords and keyword_colors:
            # Plot each mask with its keyword color
            for i, (mask, keyword) in enumerate(zip(masks, keywords)):
                color = keyword_colors.get(keyword.lower(), (1, 0, 0, 0.5))  # fallback to semi-transparent red
                # Create a color mask
                color_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
                color_mask[..., :3] = color[:3]  # RGB
                color_mask[..., 3] = mask * 0.5  # Alpha
                ax.imshow(color_mask)
        else:
            # Fallback: plot all masks in jet colormap
            combined_masks = np.clip(np.sum(masks, axis=0), 0, 1)
            ax.imshow(combined_masks, cmap='jet', alpha=0.5)

    ax.axis('off')

def display_colored_text(ax, text, keywords, keyword_colors, *, wrap_width: int = 40):
    """Display text with colored keywords using highlight_text package."""
    # Keep text axes detached from image axes so long descriptions never shrink the image.
    ax.set_axis_off()
    x_pos = 0.5
    y_pos = 1.0
    text_kwargs = {"transform": ax.transAxes, "clip_on": False}

    if not text:
        return

    if not keywords or not keyword_colors:
        # No keywords to color, display normally
        wrapped_text = "\n".join(textwrap.wrap(text, width=wrap_width))
        ax.text(
            x_pos,
            y_pos,
            wrapped_text,
            ha="center",
            va="top",
            fontsize=FONT_SIZE,
            wrap=True,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
            **text_kwargs,
        )
        return
    
    # Process text to create highlight_text format
    processed_text = text
    highlight_textprops = []
    
    # Replace segmentation tags with highlight_text delimiters and build textprops
    for keyword in keywords:
        if keyword.lower() in keyword_colors:
            color = keyword_colors[keyword.lower()]
            # Convert matplotlib color to a format suitable for highlight_text
            if isinstance(color, (list, tuple, np.ndarray)) and len(color) >= 3:
                # RGB tuple - convert to hex
                hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
            else:
                # Assume it's already a valid matplotlib color
                hex_color = color
            
            pattern = rf"{re.escape(SEG_START)}{re.escape(keyword)}{re.escape(SEG_END)}"
            count = len(re.findall(pattern, processed_text, flags=re.IGNORECASE))
            underscored_keyword = keyword.replace(" ", "_")
            processed_text = re.sub(pattern, f"<{underscored_keyword}>", processed_text, flags=re.IGNORECASE)
            for _ in range(count):
                highlight_textprops.append({"color": hex_color, "weight": "bold"})
    
    processed_text = processed_text.replace(SEG_START, "").replace(SEG_END, "")
    wrapped_text = "\n".join(textwrap.wrap(processed_text, width=wrap_width))
    wrapped_text = wrapped_text.replace("_", " ")

    try:
        ax_text(
            x=x_pos,
            y=y_pos,
            s=wrapped_text,
            highlight_textprops=highlight_textprops,
            ha="center",
            va="top",
            fontsize=FONT_SIZE,
            ax=ax,
            **text_kwargs,
        )
    except Exception as e:
        # Fallback to regular text if highlight_text fails
        print(f"highlight_text failed: {e}, falling back to regular text")
        wrapped_text = "\n".join(textwrap.wrap(text, width=wrap_width))
        ax.text(
            x_pos,
            y_pos,
            wrapped_text,
            ha="center",
            va="top",
            fontsize=FONT_SIZE,
            wrap=True,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
            **text_kwargs,
        )

def display_info_panel(ax, prefix_text: str = None, metrics_text: str = None, *, wrap_width: int = 38):
    """Render prefix and metrics strings into colored boxes on the provided axis."""
    ax.set_axis_off()
    if not prefix_text and not metrics_text:
        return

    y_pos = 0.95
    line_spacing = 0.055

    def _wrap_text(text: str, *, preserve_newlines: bool):
        if not text:
            return ""
        if preserve_newlines:
            wrapped_lines = []
            for line in text.split("\n"):
                line_wrapped = textwrap.wrap(line, width=wrap_width) or [""]
                wrapped_lines.extend(line_wrapped)
            return "\n".join(wrapped_lines)
        return "\n".join(textwrap.wrap(text, width=wrap_width))

    def _draw_box(text: str, facecolor: str):
        nonlocal y_pos
        preserve_newlines = "\n" in text
        wrapped = _wrap_text(text, preserve_newlines=preserve_newlines)
        line_count = wrapped.count("\n") + 1
        ax.text(
            0.02,
            y_pos,
            wrapped,
            ha="left",
            va="top",
            fontsize=FONT_SIZE,
            wrap=True,
            transform=ax.transAxes,
            bbox=dict(
                facecolor=facecolor,
                alpha=0.95,
                edgecolor='lightgrey',
                boxstyle='round,pad=0.4',
            ),
        )
        y_pos -= line_count * line_spacing + 0.08

    if prefix_text:
        _draw_box(prefix_text, '#eef3ff')
    if metrics_text:
        _draw_box(metrics_text, '#fff4dc')


def visualize_and_save_segmentation(image: np.ndarray, 
                                   pred_masks: list, 
                                   gt_masks: list, 
                                   pred_text: str = None,
                                   gt_text: str = None,
                                   prefix: str = None,
                                   metrics: dict = None,
                                   save_path: str = None):
    """
    Visualize the image, predicted binary segmentation masks, and ground truth masks.
    Overlay all predicted masks and all ground truth masks on the image.
    Save the visualization to the specified path.

    Args:
        image (np.ndarray): The input image (H, W, 3) or (H, W).
        pred_masks (np.ndarray): Array of predicted binary masks (N, H, W).
        gt_masks (np.ndarray): Array of ground truth binary masks (N, H, W).
        save_path (str): Path to save the visualization.
    """
    fig = plt.figure(figsize=(12, 5.5))
    grid_spec = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 0.8],
        height_ratios=[2.6, 2.1],
        wspace=0.28,
        hspace=0.18,
    )
    pred_ax = fig.add_subplot(grid_spec[0, 0])
    gt_ax = fig.add_subplot(grid_spec[0, 1])
    info_ax = fig.add_subplot(grid_spec[:, 2])
    pred_text_ax = fig.add_subplot(grid_spec[1, 0])
    gt_text_ax = fig.add_subplot(grid_spec[1, 1])
    keyword_colors = {}
    pred_keywords = []
    gt_keywords = []
    if pred_text and gt_text:
        keyword_pattern = re.compile(rf"{SEG_START}(.*?){SEG_END}", re.IGNORECASE)
        pred_keywords = keyword_pattern.findall(pred_text)
        gt_keywords = keyword_pattern.findall(gt_text)
        #assign nice colors to each keyword
        unique_keywords = set(pred_keywords) | set(gt_keywords)
        cmap = plt.get_cmap("tab20b", max(20, len(unique_keywords)))
        for i, keyword in enumerate(unique_keywords):
            keyword_colors[keyword.lower()] = cmap(i % cmap.N)

    #normalize the image
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Overlay all predicted masks and annotations
    plot_masks_on_image(image, pred_masks, pred_keywords, keyword_colors, ax=pred_ax)
    pred_ax.set_title('Prediction')
    if pred_text is not None:
        display_colored_text(pred_text_ax, pred_text, pred_keywords, keyword_colors)
    else:
        pred_text_ax.set_axis_off()

    # Overlay all ground truth masks and annotations
    plot_masks_on_image(image, gt_masks, gt_keywords, keyword_colors, ax=gt_ax)
    gt_ax.set_title('GT')
    if gt_text is not None:
        display_colored_text(gt_text_ax, gt_text, gt_keywords, keyword_colors)
    else:
        gt_text_ax.set_axis_off()

    info_prefix = None
    info_metrics = None
    if prefix:
        prefix_clean = prefix.strip()
        if prefix_clean:
            info_prefix = "Prompt: " + prefix_clean
    if metrics:
        metric_parts = []
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                formatted_value = f"{float(value):.2f}"
            except (ValueError, TypeError):
                formatted_value = str(value)
            metric_parts.append(f"{key}: {formatted_value}")
        if metric_parts:
            info_metrics = "Metrics:\n" + "\n".join(metric_parts)

    if info_prefix or info_metrics:
        display_info_panel(info_ax, info_prefix, info_metrics)
    else:
        info_ax.set_axis_off()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    
if __name__ == "__main__":
    # Example usage
    # Create a dummy example
    image = np.zeros((224, 224, 3), dtype=np.uint8)  # Random RGB image
    # Extract keywords from pred_text and gt_text
    keyword_pattern = re.compile(rf"{SEG_START}(.*?){SEG_END}", re.IGNORECASE)
    # pred_text = "Abdomen: <seg>Leber</seg>, Pfortader und <seg>Leber</seg>venen: Hypodense Läsion in <seg>Segment II</seg>, am ehesten Mehrverfettung."
    # gt_text = "<seg>Lebermetastasen</seg> in <seg>Segment II</seg> mit 3,4 × 3,2 cm, in Segment V mit 3,1 × 2,7 cm, in <seg>Segment VI</seg> mit 6,6 × 4,6 cm"
    pred_text= "Abdomen: <seg>Liver</seg>, <seg>portal vein</seg> and <seg>hepatic veins</seg>: Hypodense <seg>liver lesions</seg>, exemplarily in <seg>segment II</seg> with 12 mm, most likely cystic."
    gt_text = "This image shows several visible anatomical structures, including: <seg>spleen</seg>, <seg>right kidney</seg>, <seg>left kidney</seg>, <seg>liver</seg>, <seg>stomach</seg>, <seg>pancreas</seg>, <seg>left adrenal gland</seg>, <seg>colon</seg>, <seg>twelfth thoracic vertebra (T12)</seg>, <seg>aorta</seg>, <seg>inferior vena cava</seg>."                                          
    pred_keywords = keyword_pattern.findall(pred_text)
    gt_keywords = keyword_pattern.findall(gt_text)

    # Create a mask for each keyword (dummy masks for demonstration)
    # Define fixed positions for each keyword (top-left corner of the square)
    keyword_positions = {
        "Liver": (10, 10),
        "portal vein": (60, 10),
        "hepatic veins": (110, 10),
        "liver lesions": (160, 10),
        "segment II": (60, 60),
        "spleen": (10, 60),
        "right kidney": (60, 60),
        "left kidney": (110, 60),
        "liver": (160, 60),
        "stomach": (10, 110),
        "pancreas": (60, 110),
        "left adrenal gland": (110, 110),
        "colon": (160, 110),
        "twelfth thoracic vertebra (T12)": (10, 160),
        "aorta": (110, 160),
        "inferior vena cava": (160, 160),
    }
    def create_square_mask(keyword, size=224, square_size=32):
        mask = np.zeros((size, size), dtype=np.uint8)
        if keyword in keyword_positions:
            x, y = keyword_positions[keyword]
            mask[y:y+square_size, x:x+square_size] = 1
        return mask

    def build_masks(keyword_list):
        masks = [create_square_mask(k) for k in keyword_list]
        if masks:
            return np.stack(masks)
        return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)

    pred_masks = build_masks(pred_keywords)
    gt_masks = build_masks(gt_keywords)

    prefix = "Example prompt describing the clinical question for the segmentation"
    metrics = {"LLMScore": 0.83, "mIoU": 0.47, "G-IoU": 0.52}

    visualize_and_save_segmentation(
        image,
        pred_masks,
        gt_masks,
        pred_text,
        gt_text,
        prefix=prefix,
        metrics=metrics,
        save_path="test_output/test_visualization.png",
    )

    # Extended example to verify long text keeps image size stable
    long_pred_sections = [
        "This extensive report covers the appearance of <seg>Liver</seg> parenchyma wrapping around the <seg>portal vein</seg> while tracking <seg>hepatic veins</seg> back to the atrium.",
        "Radiologists note subtle <seg>liver lesions</seg> concentrated within <seg>segment II</seg> and referencing surrounding <seg>spleen</seg> and <seg>right kidney</seg> landmarks for orientation.",
        "Further description reviews the <seg>left kidney</seg>, anterior <seg>stomach</seg>, contouring <seg>pancreas</seg>, and resilient <seg>left adrenal gland</seg> tissue.",
        "The narrative continues with bowel detail along the <seg>colon</seg> and axial alignment at the <seg>twelfth thoracic vertebra (T12)</seg>.",
        "Finally the vascular map reiterates patency of the <seg>aorta</seg> and <seg>inferior vena cava</seg> as contrasted blood pools circulate." 
    ]
    long_gt_sections = [
        "Ground truth confirms crisp visualization of <seg>Liver</seg>, <seg>spleen</seg>, and <seg>right kidney</seg> contours without focal defects.",
        "Annotated educators mark the <seg>left kidney</seg>, interposed <seg>stomach</seg>, graceful <seg>pancreas</seg>, and balanced <seg>left adrenal gland</seg> glands despite motion.",
        "Bowel loops within the <seg>colon</seg> and osseous landmarks at the <seg>twelfth thoracic vertebra (T12)</seg> remain aligned.",
        "Major vessels including the <seg>aorta</seg> and <seg>inferior vena cava</seg> display expected caliber and enhancement, guiding consistent interpretation." 
    ]
    long_pred_text = " ".join(long_pred_sections * 3)
    long_gt_text = " ".join(long_gt_sections * 3)

    long_pred_keywords = keyword_pattern.findall(long_pred_text)
    long_gt_keywords = keyword_pattern.findall(long_gt_text)
    long_pred_masks = build_masks(long_pred_keywords)
    long_gt_masks = build_masks(long_gt_keywords)

    long_prefix = "Long-form prompt: describe every abdominal structure in exhaustive prose for multidisciplinary review." * 2
    long_metrics = {"LLMScore": 0.91, "mIoU": 0.63, "G-IoU": 0.58}

    visualize_and_save_segmentation(
        image,
        long_pred_masks,
        long_gt_masks,
        long_pred_text,
        long_gt_text,
        prefix=long_prefix,
        metrics=long_metrics,
        save_path="test_output/test_visualization_long_text.png",
    )