#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Models Module for BraTS Segmentation.

This module contains the refactored network components with configuration-driven architecture
and clean code practices. All components are designed to work with the configuration system.
"""

# Main network architecture
from .network_backbone import Waveformer, create_waveformer, ProjectionHead, ChannelCalibration

# Transformer components
from .waveformer import MultiscaleTransformer

# Helper components
from .wave_helper import (
    Block, PatchMerging, PatchMergingV2, CCF_FFN, Mlp, 
    WaveletTransform3D, DWConv, OverlapPatchEmbed, PatchEmbed, 
    PosCNN, ProjectionUpsample
)

# Upsampling components
from .idwt_upsample import UnetrIDWTBlock as IDWTBlock, HFRefinementRes
from .attention import Attention 

# Version and module info
__version__ = "1.0.0"
__author__ = "BraTS Team"
__description__ = "Refactored network models with configuration support"

# Main exports for easy access
__all__ = [
    # Main architecture
    "Waveformer",
    "create_waveformer",
    "ProjectionHead", 
    "ChannelCalibration",
    
    # Transformer
    "MultiscaleTransformer",
    
    # Helper components
    "Block",
    "PatchMerging", 
    "PatchMergingV2",
    "CCF_FFN",
    "Mlp",
    "WaveletTransform3D",
    "DWConv",
    "OverlapPatchEmbed",
    "PatchEmbed",
    "PosCNN",
    "ProjectionUpsample",
    
    # Upsampling
    "IDWTBlock",
    "HFRefinementRes",
    
    # Multi-scale attention
    "Attention",
] 