"""
RadGrounder: Medical Vision-Language Model for Radiology Report Generation

A comprehensive medical vision-language model (VLM) designed for generating 
radiology reports from CT and MRI images. This project leverages state-of-the-art 
vision-language models, including PaliGemma and other architectures, fine-tuned 
on the RefRad2D dataset for German and English radiology report generation.
"""

__version__ = "0.1.0"
__author__ = "RadGrounder Team"

# Subpackages (dataset, grounded_gemma, llm_score) are imported explicitly by callers
# to avoid heavy import-time side effects.

VERSION = __version__

__all__ = ["__version__", "VERSION"]
