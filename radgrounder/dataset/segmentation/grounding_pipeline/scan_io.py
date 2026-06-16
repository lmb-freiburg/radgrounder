"""Minimal NIfTI scan I/O for the grounding pipeline.

Self-contained helpers (nibabel + numpy only) for loading a 3D scan with a known
orientation and extracting a 2D slice that matches how the corresponding DICOM
slice was oriented. Vendored from the authors' internal ``medv`` utilities so the
release has no private dependency.
"""

import numpy as np
import nibabel as nib
from nibabel.orientations import (
    aff2axcodes,
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
)


def read_scan_from_file(nii_file, orientation="LPS", normalize="none", verbose=False):
    """Read a NIfTI scan and return ``(data, affine, nibabel_image)``.

    Args:
        nii_file: Path to the ``.nii`` / ``.nii.gz`` file.
        orientation: Target axis orientation. ``"LPS"`` matches how DICOMs (and
            most public 2D datasets) are oriented: an axial slice with the liver
            (patient right) on the left and the spine (posterior) at the bottom.
            ``"RAS"`` is nibabel's world coordinate system. ``None`` skips
            reorientation.
        normalize: ``"ctrate"`` clips Hounsfield units to [-1000, 1000] and scales
            to [-1, 1]; ``None`` / ``"none"`` leaves the data untouched. For
            segmentation label volumes always use ``"none"``.
        verbose: Print orientation/shape debug info.

    Returns:
        Tuple of (data ``np.ndarray``, reoriented affine, nibabel image).
    """
    scan = nib.load(str(nii_file))
    # get_fdata applies the stored slope/intercept scaling automatically.
    odata = scan.get_fdata(dtype=np.float32)
    affine = scan.affine  # voxel -> world transform
    axcodes = aff2axcodes(affine)

    if orientation:
        target_ornt = axcodes2ornt((orientation[0], orientation[1], orientation[2]))
        ornt = io_orientation(affine)
        transform = nib.orientations.ornt_transform(ornt, target_ornt)
        new_affine = affine @ inv_ornt_aff(transform, scan.shape)
        data = apply_orientation(odata, transform)
        if verbose:
            print(
                f"Reoriented {axcodes} -> {orientation}; "
                f"shape {odata.shape} -> {data.shape}"
            )
    else:
        data = odata
        new_affine = affine

    if normalize == "ctrate":
        data = np.clip(data, -1000, 1000) / 1000.0
    elif normalize in (None, "none"):
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize!r}")

    return data, new_affine, scan


def get_slice_from_scan(scanarr, slice_axis, slice_idx):
    """Extract the 2D slice from an LPS-oriented scan matching the DICOM slice.

    Handles the three DICOM pixel orderings the dataset uses and transposes the
    result so the slice is ``(rows, cols)`` like the DICOM, not ``(x, y)``.

    Args:
        scanarr: 3D array from :func:`read_scan_from_file` (orientation ``"LPS"``).
        slice_axis: 0, 1, or 2 — the axis along which the slice was taken.
        slice_idx: Index of the slice along ``slice_axis``.

    Returns:
        2D ``np.ndarray`` slice (rows, cols).
    """
    if slice_axis == 0:
        # PIR dicom, LPS scan: slice L, keep P, invert I->S (idx already inverted).
        scan_slice = scanarr[slice_idx, :, ::-1]
    elif slice_axis == 1:
        # LIP dicom, LPS scan: slice P, keep L, invert S->I.
        scan_slice = scanarr[:, slice_idx, ::-1]
    elif slice_axis == 2:
        # LPS dicom, LPS scan: no inversion needed.
        scan_slice = scanarr[:, :, slice_idx]
    else:
        raise ValueError(f"Unexpected slice axis: {slice_axis}")

    # dicom is (rows, cols) while scan is (x, y, z), so transpose the slice.
    return scan_slice.T
