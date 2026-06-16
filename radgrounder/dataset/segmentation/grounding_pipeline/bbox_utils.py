"""Bounding-box extraction from 2D multi-class segmentation slices.

The core ``extract_all_bboxes`` turns a labelled 2D slice into per-class boxes
``(x1, y1, x2, y2, class_id)`` via connected components. ``cv2`` and ``matplotlib``
are imported lazily inside the functions, so this module imports fine without the
optional ``grounding`` extra installed (only calling the functions needs it).
"""

import numpy as np


def extract_bboxes_from_mask(binary_mask):
    """Bounding boxes for the connected components of a single binary mask.

    Args:
        binary_mask: 2D ``np.ndarray``, non-zero where the class is present.

    Returns:
        List of ``(x1, y1, x2, y2)`` tuples, one per connected component.
    """
    import cv2

    bboxes = []
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    for i in range(1, num_labels):  # skip background (label 0)
        x, y, w, h, _area = stats[i]
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def extract_all_bboxes(segmentation_slice, merge_classes=False):
    """Bounding boxes for every class present in a 2D segmentation slice.

    Args:
        segmentation_slice: 2D ``np.ndarray`` of integer class ids (0 = background).
        merge_classes: If True, return a single box per class spanning all of its
            components; otherwise one box per connected component.

    Returns:
        List of ``(x1, y1, x2, y2, class_id)`` int tuples.
    """
    segmentation_bin = segmentation_slice.astype(np.uint8)
    all_bboxes = []

    for class_id in np.unique(segmentation_bin):
        if class_id == 0:  # skip background
            continue
        class_mask = (segmentation_bin == class_id).astype(np.uint8)
        for x1, y1, x2, y2 in extract_bboxes_from_mask(class_mask):
            all_bboxes.append((int(x1), int(y1), int(x2), int(y2), int(class_id)))

    if merge_classes:
        merged = {}
        for x1, y1, x2, y2, class_id in all_bboxes:
            merged.setdefault(class_id, []).append((x1, y1, x2, y2))
        all_bboxes = [
            (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
                class_id,
            )
            for class_id, boxes in merged.items()
        ]

    return all_bboxes


def plot_segmentation_with_bboxes(
    image_slice, segmentation_slice, bboxes, label_map=None, slice_idx=0, save_path=None
):
    """Overlay a segmentation mask and its bounding boxes on the image (debug aid).

    Optional visualization; needs ``matplotlib`` (part of the ``grounding`` extra).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap

    label_map = label_map or {}
    image_normalized = (image_slice - np.min(image_slice)) / (np.ptp(image_slice) + 1e-6)

    unique_labels = np.unique(segmentation_slice)
    unique_labels = unique_labels[unique_labels != 0]
    if len(unique_labels) == 0:
        print("No segmentation labels found (only background)")
        return

    base_cmap = plt.colormaps.get_cmap("tab20")
    cmap = ListedColormap(base_cmap.colors[: len(unique_labels)])
    label_to_color = {int(lbl): cmap(i) for i, lbl in enumerate(unique_labels)}

    overlay = np.zeros(segmentation_slice.shape + (4,))
    for lbl, color in label_to_color.items():
        overlay[segmentation_slice == lbl] = color

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image_normalized, cmap="gray")
    ax.imshow(overlay, alpha=0.5)
    for x1, y1, x2, y2, class_id in bboxes:
        color = label_to_color.get(class_id, "red")
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
        )
        ax.text(
            x1,
            y1 - 5,
            f"{class_id}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
        )
    ax.set_title(f"Segmentation + bounding boxes (slice {slice_idx})")
    ax.axis("off")
    legend = [
        mpatches.Patch(color=color, label=f"{lbl}: {label_map.get(lbl, f'class {lbl}')}")
        for lbl, color in label_to_color.items()
    ]
    plt.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved overlay to {save_path}")
    plt.show()
