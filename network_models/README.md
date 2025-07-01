# Network Models Module

This module contains the refactored network architecture components extracted from `msHead_3D` for better organization and maintainability.

## Overview

The `network_models` module provides a clean, modular architecture for 3D medical image segmentation using Multi-Scale Head Attention networks. It includes:

- **MSHEAD_ATTN**: Main segmentation network
- **MultiscaleTransformer**: Multi-Resolution Attention Transformer backbone
- **Helper Components**: Various building blocks for the network architecture
- **Configuration System**: Centralized configuration management

## Module Structure

```
network_models/
├── __init__.py              # Module exports
├── network_backbone.py      # Main MSHEAD_ATTN network
├── mra_transformer.py       # Multi-Resolution Attention Transformer
├── mra_helper.py           # Helper components and utilities
├── multi_scale_head.py     # Window attention mechanism
├── idwt_upsample.py        # Inverse Discrete Wavelet Transform upsampling
└── README.md               # This file
```

## Key Components

### 1. MSHEAD_ATTN (Main Network)

The primary segmentation network that combines:

- Multi-scale transformer backbone
- IDWT upsampling for high-frequency details
- Learnable projection upsampling
- Channel calibration (optional)

**Usage:**

```python
from network_models import MSHEAD_ATTN, create_mshead_model

# Create model from configuration
model = create_mshead_model(network_config)

# Or create directly
model = MSHEAD_ATTN(
    img_size=(128, 128, 128),
    patch_size=2,
    in_chans=4,
    out_chans=4,
    depths=[2, 2, 2, 2],
    feat_size=[48, 96, 192, 384],
    num_heads=[3, 6, 12, 24],
    drop_path_rate=0.1
)
```

### 2. MultiscaleTransformer (Backbone)

The MultiscaleTransformer serves as the backbone for the Waveformer architecture:

```python
from network_models import MultiscaleTransformer

# Create transformer backbone
backbone = MultiscaleTransformer(
    img_size=(128, 128, 128),
    patch_size=2,
    in_chans=4,
    num_classes=4,
    embed_dims=[48, 96, 192, 384],
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    drop_path_rate=0.1,
    mlp_ratios=[4, 4, 4, 4],
    decom_levels=[3, 2, 1, 0],
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    attn_drop_rate=0,
    drop_rate=0
)
```
