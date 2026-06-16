"""Normalization utilities for RefRad2D datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import torch


class NormalizationType(Enum):
    """Supported normalization strategies."""

    MIN_MAX = "min_max"
    DATASET_STATS = "dataset_stats"
    MEDGEMMA = "medgemma"


@dataclass(frozen=True)
class NormalizationConfig:
    """Encapsulates the selected normalization strategy."""

    strategy: NormalizationType = NormalizationType.DATASET_STATS


DEFAULT_NORMALIZATION = NormalizationConfig()

MEDGEMMA_CT_WINDOWS: Iterable[tuple[float, float]] = (
    (2250.0, -100.0),
    (350.0, 40.0),
    (80.0, 40.0),
)

MEDGEMMA_WINDOW_NAMES = [
    "LUNG",
    "SOFT_TISSUE",
    "BRAIN",
]


def normalize_min_max(image: torch.Tensor, modality: str | None = None) -> torch.Tensor:
    """Min-max normalize the image after modality-specific clipping."""

    if modality == "CT":
        image = torch.clamp(image, min=-1000.0, max=3000.0)
    elif modality == "MR":
        image = torch.clamp(image, min=0.0, max=4096.0)
    else:
        image = torch.clamp(image, min=-1000.0, max=3000.0)

    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val + 1e-5)


def normalize_dataset_stats(
    image: torch.Tensor,
    modality: str | None,
    dataset_stats: Mapping[str, float],
) -> torch.Tensor:
    """Normalize using dataset statistics and clamp to [-2, 2]."""

    if modality == "CT":
        mean = dataset_stats.get("avr_ct_mean")
        std = dataset_stats.get("avr_ct_std")
    elif modality == "MR":
        mean = dataset_stats.get("avr_mr_mean")
        std = dataset_stats.get("avr_mr_std")
    else:
        mean = dataset_stats.get("avr_mr_mean")
        std = dataset_stats.get("avr_mr_std")

    if mean is None or std is None:
        raise ValueError("Dataset statistics must include modality-specific mean and std.")

    normalized = (image - mean) / (std + 1e-8)
    return torch.clamp(normalized, min=-2, max=2)


def normalize_channel_stats(
    image: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Channel-wise normalization for RGB images."""

    mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype).view(-1, 1, 1)
    return (image - mean_tensor) / (std_tensor + 1e-8)


def _apply_window(image: torch.Tensor, *, width: float, level: float) -> torch.Tensor:
    lower = level - width / 2.0
    upper = level + width / 2.0
    windowed = torch.clamp(image, min=lower, max=upper)
    windowed = (windowed - lower) / (upper - lower + 1e-5)
    return windowed * 2.0 - 1.0


def normalize_medgemma(image: torch.Tensor, modality: str | None) -> torch.Tensor:
    """Apply MedGemma windowing for CT images, fall back to symmetric scaling otherwise."""

    if modality == "CT":
        channels = [
            _apply_window(image, width=width, level=level)
            for width, level in MEDGEMMA_CT_WINDOWS
        ]
        return torch.cat(channels, dim=0)

    min_val = image.min()
    max_val = image.max()
    scaled = (image - min_val) / (max_val - min_val + 1e-5)
    return scaled * 2.0 - 1.0


def apply_normalization(
    image: torch.Tensor,
    modality: str | None,
    config: NormalizationConfig,
    *,
    dataset_stats: Mapping[str, float] | None = None,
    channel_stats: Mapping[str, Sequence[float]] | None = None,
) -> torch.Tensor:
    """Dispatch normalization based on the provided configuration."""

    if config.strategy == NormalizationType.MIN_MAX:
        return normalize_min_max(image, modality)

    if config.strategy == NormalizationType.DATASET_STATS:
        if channel_stats is not None:
            mean = channel_stats.get("mean")
            std = channel_stats.get("std")
            if mean is None or std is None:
                raise ValueError("Channel statistics must include 'mean' and 'std'.")
            return normalize_channel_stats(image, mean, std)
        if dataset_stats is None:
            raise ValueError("Dataset statistics are required for dataset-stats normalization.")
        return normalize_dataset_stats(image, modality, dataset_stats)

    if config.strategy == NormalizationType.MEDGEMMA:
        return normalize_medgemma(image, modality)

    raise ValueError(f"Unsupported normalization strategy: {config.strategy}")
