import numpy as np
import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from highlight_text import ax_text


def parse_detection_text(text: str):
    """
    Parse detection text to extract bounding boxes and labels.
    
    Args:
        text (str): Text containing detection annotations in format:
                   <p bbox=<loc0001><loc0002><loc0003><loc0004> id=<seg001>keyword</p>
    
    Returns:
        list: List of tuples (bbox, keyword) where bbox is [x1, y1, x2, y2] in normalized coordinates
        list: List of keywords found in the text
    """
    # Pattern to match detection annotations
    pattern = r'<p bbox=<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})> id=<seg(\d{3})>>(.*?)</p>'
    matches = re.findall(pattern, text)
    
    bboxes_and_keywords = []
    keywords = []
    ids = []
    
    for match in matches:
        x1, y1, x2, y2, seg_id, keyword = match
        # Convert from discrete coordinates (0-1023) to normalized (0-1)
        bbox = [
            int(x1) / 512.0,
            int(y1) / 512.0, 
            int(x2) / 512.0,
            int(y2) / 512.0
        ]
        bboxes_and_keywords.append((bbox, keyword.strip()))
        keywords.append(keyword.strip())
        ids.append(int(seg_id))
    
    return bboxes_and_keywords, keywords, ids


def plot_bboxes_on_image(image: np.ndarray, bboxes_and_keywords: list, keyword_colors: dict):
    """
    Plot bounding boxes on an image with appropriate colors.
    
    Args:
        image (np.ndarray): The input image (H, W, 3) or (H, W).
        bboxes_and_keywords (list): List of tuples (bbox, keyword) where bbox is [x1, y1, x2, y2] 
                                   in normalized coordinates.
        keyword_colors (dict): Dictionary mapping keywords to colors.
    """
    # Improve image contrast using the same method as plot_img_with_boxes
    display_image = (image - np.min(image)) / (np.ptp(image) + 1e-6)
    
    # Display the base image
    if display_image.ndim == 2:
        plt.imshow(display_image, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(display_image)
    
    h, w = image.shape[:2]
    ax = plt.gca()
    
    # Use a color palette similar to detection_augmentations.py if no keyword color is found
    default_palette = ["lime", "yellow", "cyan", "magenta", "orange", "red"]
    text_poses = []
    for i, (bbox, keyword) in enumerate(bboxes_and_keywords):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = bbox
        x1_px, y1_px, x2_px, y2_px = x1 * w, y1 * h, x2 * w, y2 * h
        
        # Get color for this keyword, fallback to palette
        if keyword.lower() in keyword_colors:
            color = keyword_colors[keyword.lower()]
            # Convert matplotlib color to hex if needed
            if isinstance(color, (list, tuple, np.ndarray)) and len(color) >= 3:
                color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        else:
            color = default_palette[i % len(default_palette)]
        
        # Calculate width and height
        box_w = x2_px - x1_px
        box_h = y2_px - y1_px
        
        # Create rectangle patch with better visibility
        rect = patches.Rectangle(
            (x1_px, y1_px), 
            box_w, 
            box_h,
            linewidth=1.2, 
            edgecolor=color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with better positioning and visibility
        if keyword:
            text_pos = (x1_px, y1_px - 5)  # Position above the box
            #if the distance to other text positions is too close, adjust
            for other_pos in text_poses:
                distance =  (text_pos[1] - other_pos[1])**2 + (text_pos[0] - other_pos[0])**2
                if distance < 25:  # If too close vertically
                    text_pos = (text_pos[0], other_pos[1] - 15)  # Adjust position
            text_poses.append(text_pos)
            ax.text(
                *text_pos,
                keyword,
                color=color,
                fontsize=6,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', boxstyle='round,pad=0.1')
            )


def display_colored_text_with_bboxes(ax, text: str, bboxes_and_keywords: list, keyword_colors: dict, x=0.5, y=-0.01, text_width=50):
    """
    Display text with colored keywords and remove bbox annotations for cleaner display.
    
    Args:
        ax: Matplotlib axis to draw on
        text (str): Text containing detection annotations
        bboxes_and_keywords (list): List of tuples (bbox, keyword)
        keyword_colors (dict): Dictionary mapping keywords to colors
        x, y: Position for text placement
    """
    # Clean text by removing detection annotations but keeping keywords
    clean_text = text
    keywords = [keyword for _, keyword in bboxes_and_keywords]
    
    # Clean keywords (remove any unwanted prefixes like '>')
    cleaned_keywords = []
    for keyword in keywords:
        clean_keyword = keyword.strip().lstrip('>')  # Remove leading '>' and whitespace
        cleaned_keywords.append(clean_keyword)
    
    if not cleaned_keywords or not keyword_colors:
        # No keywords to color, display normally
        # Remove detection annotations
        patterns_to_clean = [
            r'<p bbox=<loc\d{4}><loc\d{4}><loc\d{4}><loc\d{4}> id=<seg\d{3}>([^<]+)</p>',
            r'<p[^>]*>([^<]+)</p>'  # Fallback pattern
        ]
        
        for pattern in patterns_to_clean:
            clean_text = re.sub(pattern, r'\1', clean_text)
        
        wrapped_text = "\n".join(textwrap.wrap(clean_text, width=50))
        ax.text(
            x, y, wrapped_text,
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=10,
            wrap=True,
            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.2'),
            clip_on=False,
        )
        return
    
    # Process text to create highlight_text format
    processed_text = clean_text
    highlight_textprops = []
    
    # Replace detection annotations with highlight_text delimiters and build textprops
    for original_keyword, clean_keyword in zip(keywords, cleaned_keywords):
        if clean_keyword.lower() in keyword_colors:
            color = keyword_colors[clean_keyword.lower()]
            # Convert matplotlib color to a format suitable for highlight_text
            if isinstance(color, (list, tuple, np.ndarray)) and len(color) >= 3:
                # RGB tuple - convert to hex
                hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
            else:
                # Assume it's already a valid matplotlib color
                hex_color = color
            
            # Try multiple patterns to match the detection annotations
            patterns_to_try = [
                r'<p bbox=<loc\d{4}><loc\d{4}><loc\d{4}><loc\d{4}> id=<seg\d{3}>>' + re.escape(original_keyword) + r'</p>',
                r'<p[^>]*>(' + re.escape(original_keyword) + r')</p>',
                re.escape(original_keyword)  # Direct keyword match as fallback
            ]
            
            count = 0
            for pattern in patterns_to_try:
                matches = re.findall(pattern, processed_text, flags=re.IGNORECASE)
                if matches:
                    count = len(matches)
                    underscored_keyword = clean_keyword.replace(" ", "_")
                    processed_text = re.sub(pattern, f"<{underscored_keyword}>", processed_text, flags=re.IGNORECASE)
                    break
            
            # Add textprops for this keyword
            for _ in range(count):
                highlight_textprops.append({"color": hex_color, "weight": "bold"})
    
    # Clean up any remaining detection annotations
    cleanup_patterns = [
        r'<p bbox=<loc\d{4}><loc\d{4}><loc\d{4}><loc\d{4}> id=<seg\d{3}>([^<]+)</p>',
        r'<p[^>]*>([^<]+)</p>'
    ]
    
    for pattern in cleanup_patterns:
        processed_text = re.sub(pattern, r'\1', processed_text)
    
    # Wrap the processed text
    wrapped_text = "\n".join(textwrap.wrap(processed_text, width=text_width))
    wrapped_text = wrapped_text.replace("_", " ")
    
    # Use highlight_text to display colored text
    try:
        # Simple fallback approach for highlight_text
        if highlight_textprops:
            ax_text(
                x=x, y=250, 
                s=wrapped_text,
                highlight_textprops=highlight_textprops,
                transform=ax.transAxes,
                fontsize=10,
                ax=ax
            )
        else:
            raise Exception("No highlight textprops, using fallback")
    except Exception as e:
        # Fallback to regular text if highlight_text fails
        # Clean text for fallback
        clean_fallback_text = clean_text
        for pattern in cleanup_patterns:
            clean_fallback_text = re.sub(pattern, r'\1', clean_fallback_text)
        wrapped_text = "\n".join(textwrap.wrap(clean_fallback_text, width=50))
        ax.text(
            x, y, wrapped_text,
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=10,
            wrap=True,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
            clip_on=False,
        )


def visualize_and_save_detection(image: np.ndarray, 
                                pred_text: str = None,
                                gt_text: str = None,
                                save_path: str = None,
                                prefix: str = None,
                                show_original: bool = True):
    """
    Visualize the image with predicted and ground truth detection results.
    Shows bounding boxes overlaid on images and highlighted text below.

    Args:
        image (np.ndarray): The input image (H, W, 3) or (H, W).
        pred_text (str): Predicted text with detection annotations.
        gt_text (str): Ground truth text with detection annotations.
        save_path (str): Path to save the visualization.
    """
    plt.figure(figsize=(12, 4))
    
    # Parse detection annotations from text
    pred_bboxes_keywords, pred_keywords = [], []
    gt_bboxes_keywords, gt_keywords = [], []
    
    if pred_text:
        pred_bboxes_keywords, pred_keywords, pred_ids = parse_detection_text(pred_text)
    if gt_text:
        gt_bboxes_keywords, gt_keywords, gt_ids = parse_detection_text(gt_text)

    # Assign colors to keywords (clean them first)
    keyword_colors = {}
    # Clean keywords for color assignment
    cleaned_pred_keywords = [k.strip().lstrip('>') for k in pred_keywords]
    cleaned_gt_keywords = [k.strip().lstrip('>') for k in gt_keywords]
    unique_keywords = set(cleaned_pred_keywords) | set(cleaned_gt_keywords)
    cmap = plt.get_cmap("tab20")
    for i, keyword in enumerate(unique_keywords):
        keyword_colors[keyword.lower()] = cmap(i % 20)

    # Don't normalize here - let plot_bboxes_on_image handle it for better contrast
    show_pred = pred_text is not None and len(pred_text) > 0
    show_gt = gt_text is not None and len(gt_text) > 0
    num_imgs = int(show_original) + int(show_pred) + int(show_gt)
    disp_idx = 1
    
    # Add prefix as main title if provided
    if prefix:
        plt.suptitle(prefix, fontsize=10)
    
    # Show original image
    if show_original:
        plt.subplot(1, num_imgs, disp_idx)
        disp_idx += 1
        # Use the same normalization as plot_img_with_boxes for consistency
        display_image = (image - np.min(image)) / (np.ptp(image) + 1e-6)
        if display_image.ndim == 2:
            plt.imshow(display_image, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(display_image)
        plt.title('Original Image')
        plt.axis('off')

    if show_pred:
        # Show predictions with bounding boxes
        ax = plt.subplot(1, num_imgs, disp_idx)
        disp_idx += 1
        plot_bboxes_on_image(image, pred_bboxes_keywords, keyword_colors)
        plt.title('Predictions')
        plt.axis('off')
        display_colored_text_with_bboxes(ax, pred_text, pred_bboxes_keywords, keyword_colors)
            

    if show_gt:
        # Show ground truth with bounding boxes
        ax = plt.subplot(1, num_imgs, disp_idx)
        disp_idx += 1
        plot_bboxes_on_image(image, gt_bboxes_keywords, keyword_colors)
        plt.title('Ground Truth')
        plt.axis('off')
        display_colored_text_with_bboxes(ax, gt_text, gt_bboxes_keywords, keyword_colors)
    

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        # Save with higher DPI and better quality like in detection_augmentations.py
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    # Example usage
    # Create a dummy example
    image = np.random.rand(224, 224, 3)  # Random RGB image
    
    # Example detection texts with bbox annotations
    pred_text = ("Abdomen: <p bbox=<loc0010><loc0010><loc0100><loc0100> id=<seg001>>liver</p>, "
                "<p bbox=<loc0200><loc0050><loc0300><loc0150> id=<seg002>>kidney</p> and "
                "<p bbox=<loc0400><loc0200><loc0500><loc0300> id=<seg003>>spleen</p> visible.")

    gt_text = ("This image shows <p bbox=<loc0015><loc0015><loc0095><loc0095> id=<seg001>>liver</p>, "
              "<p bbox=<loc0205><loc0055><loc0295><loc0145> id=<seg002>>kidney</p>, "
              "<p bbox=<loc0405><loc0205><loc0495><loc0295> id=<seg003>>spleen</p> and "
              "<p bbox=<loc0100><loc0300><loc0200><loc0400> id=<seg004>>stomach</p>.")

    visualize_and_save_detection(image, pred_text, gt_text, save_path="test_output/test_detection_visualization.png")
    print("Detection visualization saved to test_output/test_detection_visualization.png")
