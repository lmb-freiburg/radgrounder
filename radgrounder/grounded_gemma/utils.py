import pydicom
import zstandard as zstd
import os
import io
from pathlib import Path
import numpy as np
from typing import Union
import json
import matplotlib.pyplot as plt


STATS_V6 = {
    "avr_ct_mean": -610.1908535827807,
    "avr_ct_std": 737.3882466073229,
    "avr_mr_mean": 141.1154960399739,
    "avr_mr_std": 255.40429635138338,
}

def read_dicom_zst(input_zst_path: Union[str, Path]) -> pydicom.Dataset:
    """Read a dicom .zst file and return the loaded dicom."""
    compressed_data = Path(input_zst_path).read_bytes()
    decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)
    dcm = pydicom.dcmread(io.BytesIO(decompressed_data))
    return dcm

def read_dicom_as_numpy(dicom_path: Union[str, Path]) -> np.ndarray:
    """Read a dicom file and return the pixel array after applying rescale slope and intercept."""
    dcm = read_dicom_zst(dicom_path)
    arr = dcm.pixel_array.astype(np.float32)
    rescale_slope = getattr(dcm, 'RescaleSlope', 1)
    rescale_intercept = getattr(dcm, 'RescaleIntercept', 0)
    pixel_array = arr * rescale_slope + rescale_intercept
    return pixel_array

def read_from_rsopid(rsopid: str, base_dir: Union[str, Path] = os.environ.get("REFRAD2D_DATA_DIR", "")) -> np.ndarray:
    """Read a DICOM file from the given rsopid and return the pixel array."""
    file_path = Path(base_dir) / "slices/dicoms_anon" / rsopid[:2] / f"{rsopid}.dcm.zst"
    if not file_path.exists():
        raise FileNotFoundError(f"DICOM file for RSOPID {rsopid} not found at {file_path}")
    
    return read_dicom_as_numpy(file_path)


def read_scan(scan_rserid: str, base_dir: Union[str, Path] = os.environ.get("REFRAD2D_DATA_DIR", ""), orientation: str = "LPI", normalize=True) -> np.ndarray:
    import nibabel as nib
    from nibabel.orientations import io_orientation, axcodes2ornt, inv_ornt_aff, apply_orientation, aff2axcodes

    base_dir = Path(os.environ.get("REFRAD2D_DATA_DIR", ""))

    info_file = base_dir / "scans/scan_downloads" / scan_rserid[:2] / scan_rserid / "scan_clean.json"
    with open(info_file) as f:
        info = json.load(f)
    modality = info["Modality"]
    mean = STATS_V6[f"avr_{modality.lower()}_mean"]
    std = STATS_V6[f"avr_{modality.lower()}_std"]
    
    nii_file = base_dir / "scans/scan_downloads" / scan_rserid[:2] / scan_rserid / "scan_clean.nii.gz"
    scan = nib.load(nii_file)
    odata = scan.get_fdata()

    affine = scan.affine
    axcodes = aff2axcodes(affine)
    if orientation:
        target_ornt = axcodes2ornt((orientation[0], orientation[1], orientation[2]))

        # Orientation transform from current to target
        ornt = io_orientation(affine)
        transform = nib.orientations.ornt_transform(ornt, target_ornt)

        # Apply transform to data
        odata = apply_orientation(odata, transform)
        data = np.transpose(odata, (1, 0, 2))

    if normalize:
        data = (data - mean) / std  # should be mean 0 std 1 range -2 to 2
        datamin, datamax = -2, 2
        data = np.clip(data, datamin, datamax)

    return data, scan.header, info, axcodes


def display_scan(scan_rserid, save=False):
    data, header, json_info, axcodes = read_scan(scan_rserid)
    slices = []
    nx = 16
    for ii, i in enumerate(np.linspace(0, data.shape[2], nx, endpoint=False)):
        if ii == 0 or ii == nx - 1:
            continue
        i = round(i)
        sliceh = data[:, :, i]
        # plt.imshow(slice)  # , cmap='gray')
        # # plt.title(f'Slice {slice_index}')
        # plt.axis('off')
        # plt.show()
        slices.append(sliceh)
    plt.figure(figsize=(18, 6))
    slices0 = np.concatenate(slices[:nx//2-1], axis=1)
    slices1 = np.concatenate(slices[nx//2-1:], axis=1)
    slices = np.concatenate((slices0, slices1), axis=0)
    print(slices.min(), slices.max())
    plt.imshow(slices, cmap='gray')
    plt.axis('off')
    plt.show()
    #save image
    if save:
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/scan_{scan_rserid}.png", dpi=500)

    return data, header, json_info


def read_snippet(rsopid: str, base_dir: Union[str, Path] = os.environ.get("REFRAD2D_DICOM_DIR", ""), verbose=False) -> np.ndarray:
    file = Path(base_dir) / rsopid[:2] / f"{rsopid}.dcm.zst"
    dcm = read_dicom_zst(file)
    rescale_slope = getattr(dcm, 'RescaleSlope', 1)
    rescale_intercept = getattr(dcm, 'RescaleIntercept', 0)

    if verbose:
        print(f"Rescale Slope: {rescale_slope}")
        print(f"Rescale Intercept: {rescale_intercept}")    
    modality = dcm.get("Modality").lower()
    if modality not in ["ct", "mr"]:
        modality = "ct"  # default to CT if modality is not recognized

    mean = STATS_V6[f"avr_{modality}_mean"]
    std = STATS_V6[f"avr_{modality}_std"]
    arr = dcm.pixel_array.astype(np.float32)
    rescale_slope = getattr(dcm, 'RescaleSlope', 1)
    rescale_intercept = getattr(dcm, 'RescaleIntercept', 0)
    arr = arr * rescale_slope + rescale_intercept
    arr = (arr - mean) / std
    arr = np.clip(arr, -2, 2)
    return arr

def display_snippet(rsopid, nmin=None, nmax=None, verbose=True, save=False):
    arr = read_snippet(rsopid, verbose=verbose)

    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap='gray')
    plt.axis('off')
    plt.title(f"Slice min {arr.min():.1f} {arr.max():.1f}")
    plt.show()
    if save:
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/slice_{rsopid}.png", dpi=500)

    return arr

def get_slice_path(rsopid: str, base_dir: Union[str, Path] = os.environ.get("REFRAD2D_DATA_DIR", "")) -> Path:
    """Get the path to the DICOM file for a given rsopid."""
    return Path(base_dir) / "slices/dicoms_anon" / rsopid[:2] / f"{rsopid}.dcm.zst"

def get_slice_plane(file: Union[str, Path]) -> str:
    """Get the image plane from the DICOM metadata."""
    accepted_anatomical_planes = {"AXIAL":"AXIAL", "MIP_COR": "CORONAL", "MIP_SAG": "SAGITTAL"}
    dcm = read_dicom_zst(file)
    image_type = dcm.get("ImageType", "None")
    if isinstance(image_type, str):
        image_type = [image_type]
    image_type_str = ", ".join(str(item) for item in image_type)
    anatomical_plane = None
    for plane in accepted_anatomical_planes:
        if plane in image_type_str:
            anatomical_plane = accepted_anatomical_planes[plane]
            break
    return anatomical_plane, image_type_str

def display_report_row(row):
    print(f"<b>Klinische Angaben:</b> {row['ReportKlinischeAngabenCleaned']}")
    print(f"<b>Fragestellung:</b> {row['ReportFragestellungCleaned']}")
    print(f"<b>Protocol:</b> {row['ProtocolName']}")
    print(f"<b>Befund:</b> {row['ReportBefundCleaned']}")
    print(f"<b>Beurteilung:</b> {row['ReportBeurteilungCleaned']}")