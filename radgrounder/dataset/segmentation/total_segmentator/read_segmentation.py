import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.patches as mpatches
from scipy.ndimage import center_of_mass
import xmltodict
import os
from nibabel.orientations import io_orientation, axcodes2ornt, apply_orientation, aff2axcodes


def rotate_scan(scan):
    data = np.transpose(scan, (1, 0, 2))
    data = np.flip(data, axis=0)
    return data

def load_segmentation(segment_path, orientation="LPI", return_label_map=False):
    seg_scan = nib.load(f"{segment_path}/seg_scan_clean.nii.gz")
    #get the header
    # print(label_map)
    seg_data = seg_scan.get_fdata()

    if orientation:
        target_ornt = axcodes2ornt((orientation[0], orientation[1], orientation[2]))
        # Orientation transform from current to target
        affine = seg_scan.affine
        # axcode = aff2axcodes(affine)
        # print(f"Current orientation: {axcode}, Target orientation: {orientation}")
        ornt = io_orientation(affine)
        transform = nib.orientations.ornt_transform(ornt, target_ornt)

        # Apply transform to data
        reoriented_data = apply_orientation(seg_data, transform)
        seg_data = np.transpose(reoriented_data, (1, 0, 2))

    if return_label_map:
        label_map = get_label_map_from_header(seg_scan)
        return seg_data, label_map

    return seg_data


def visualize_slices_with_segmentation(image_slice, seg_slice, label_map, slice_idx=None, title=None, save_path=None, caption=None,
                                       prefix=None, font_size=25):
     # Normalize image slice for display (0–1)
    image_slice_norm = (image_slice - np.min(image_slice)) / (np.ptp(image_slice) + 1e-6)

    # Build color map for segmentation
    unique_labels = np.unique(seg_slice)
    unique_labels = unique_labels[unique_labels != 0]  # exclude background

    base_cmap = plt.colormaps.get_cmap('tab20')
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(base_cmap.colors[:len(unique_labels)])
    label_to_color = {int(label): cmap(i) for i, label in enumerate(unique_labels)}

    # Build RGBA image for segmentation overlay
    rgba_overlay = np.zeros(seg_slice.shape + (4,))
    for label, color in label_to_color.items():
        mask = seg_slice == label
        rgba_overlay[mask] = color

    # Plot image + segmentation overlay
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_slice_norm, cmap='gray')
    ax.imshow(rgba_overlay, alpha=0.5)
    if title:
        ax.set_title(title, fontdict={"fontsize":font_size})
    ax.axis('off')

    # Add label numbers at center of mass
    for label in unique_labels:
        mask = seg_slice == label
        if np.any(mask):
            yx = center_of_mass(mask)
            if yx:  # (y, x) from center_of_mass
                y, x = yx
                ax.text(x, y, str(int(label)), color='white', fontsize=10, ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='circle'))
    # Build legend
    legend_patches = []
    for label, color in label_to_color.items():
        # label += 1
        name = label_map.get(label, "Unknown")
        patch = mpatches.Patch(color=color, label=f"{label}: {name}")
        legend_patches.append(patch)


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
        caption = "Caption: " + caption
        text_obj = ax.text(
            1.03, 0.4, caption,
            ha="left", va="center",
            transform=ax.transAxes,
            fontsize=font_size,
            wrap=True,
            bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
            clip_on=False,
            verticalalignment='center',
            horizontalalignment='left',
            multialignment='left'
        )
        
        text_obj._get_wrap_line_width = lambda : 2000

    # Add legend
    plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(-0.55, 1), fontsize="xx-large", borderaxespad=0, frameon=True)
    # plt.tight_layout()
    plt.show()
    if save_path:
        # save_name = os.path.basename(save_path)
        # save_name = f"{save_name}_slice_{slice_idx}.png"
        # save_path = os.path.join(f"{os.path.dirname(__file__)}/imgs", save_name)
        # Save output
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved segmentation overlay with label numbers to {save_path}")


def visualize_segmentation_with_labels(segment_path, slice_idx=None, save_path=None):
    # Load segmentation
    seg, label_map = load_segmentation(segment_path)

    # Load metadata
    with open(f"{segment_path}/segment_info.json", "r") as f:
        segment_info = json.load(f)

    # Load input image
    input_path = segment_info["scan_path"]
    scan = nib.load(input_path).get_fdata()
    scan = rotate_scan(scan)

    # Load label map
    modality = segment_info.get("modality", "CT")
    # labels_path = "mr_labels.json" if modality == "MR" else "ct_labels.json"
    # with open(labels_path, "r") as f:
    #     label_map = json.load(f)

    # Select middle axial slice
    if slice_idx is None:
        slice_idx = seg.shape[2] // 2

    seg_slice = seg[:, :, slice_idx]

    image_slice = scan[:, :, slice_idx]
    visualize_slices_with_segmentation(image_slice, seg_slice, label_map, slice_idx=slice_idx, modality=modality, save_path=save_path)

def get_label_map_from_header(seg):
    ext_header = seg.header.extensions[0].get_content()
    ext_header = xmltodict.parse(ext_header)
    ext_header = ext_header["CaretExtension"]["VolumeInformation"]["LabelTable"]["Label"]
    label_map = {int(label["@Key"]): label["#text"] for label in ext_header}
    return label_map

if __name__ == "__main__":
    dir = "./segmentations/9d/"
    path_list = os.listdir(dir)
    num_plot = 20
    for p in path_list[:num_plot]:
        path = os.path.join(dir, p)
        visualize_segmentation_with_labels(path)
    # visualize_segmentation_with_labels(path)
    # visualize_segmentation_with_labels(<path>
