#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Resolution Attention Transformer for 3D Medical Image Segmentation.
Refactored version with configuration-driven architecture and clean code practices.
"""

import math
import time
import sys
import os
from functools import partial
from typing import List, Tuple, Optional, Dict, Any
from ptflops import get_model_complexity_info

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.layers import trunc_normal_

# Import custom PatchEmbed instead of MONAI's
from monai.networks.blocks import PatchEmbed
from monai.utils import optional_import
rearrange, _ = optional_import("einops", name="rearrange")

from .wave_helper import Block, PatchMerging


class MultiscaleTransformer(nn.Module):
    """Multiscale Transformer (formerly MRATransformer)."""
    
    def __init__(self, img_size=(128, 128, 128), patch_size=2, in_chans=4, num_classes=4, 
                 embed_dims=[48, 96, 192, 384], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], 
                 decom_levels = [3,2,1,0],  multi_scale_attention=True,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, patch_norm=False, depths=[2, 2, 2, 2],
                 network_config=None):
        """
        Initialize MultiscaleTransformer.
        
        Args:
            img_size: Input image size (D, H, W)
            patch_size: Patch size for embedding
            in_chans: Number of input channels
            num_classes: Number of output classes
            embed_dims: Embedding dimensions for each stage
            num_heads: Number of attention heads for each stage
            mlp_ratios: MLP ratios for each stage
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QK attention
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate for stochastic depth
            spatial_dims: Number of spatial dimensions
            norm_layer: Normalization layer
            patch_norm: Whether to use patch normalization
            depths: Number of transformer blocks per stage
            network_config: Network configuration dictionary
        """
        super().__init__()
        
        self.network_config = network_config or {}
        self.num_classes = num_classes
        self.depths = depths
        self.patch_norm = patch_norm
        self.patch_size = patch_size
        self.img_size = img_size
        self.levels = decom_levels
        self.multi_scale_attention = multi_scale_attention
        
        # Get configuration parameters
        # patch_embed_config = self.network_config.get('patch_embed', {})
        # block_config = self.network_config.get('block', {})
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=len(self.img_size),
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # Stage 1
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], 
                num_heads=num_heads[0], 
                mlp_ratio=mlp_ratios[0], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[cur + i], 
                norm_layer=norm_layer, 
                level=self.levels[0],
                ms_attention=self.multi_scale_attention,
                img_size=(img_size[0] // 2, img_size[1] // 2, img_size[2] // 2),
                network_config=self.network_config
            ) for i in range(depths[0])
        ])
        self.downsample_1 = PatchMerging(dim=embed_dims[0], norm_layer=norm_layer, spatial_dims=len(img_size))
        cur += depths[0]

        # Stage 2
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], 
                num_heads=num_heads[1], 
                mlp_ratio=mlp_ratios[1], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[cur + i], 
                norm_layer=norm_layer, 
                level=self.levels[1],
                ms_attention=self.multi_scale_attention,
                img_size=(img_size[0] // 4, img_size[1] // 4, img_size[2] // 4),
                network_config=self.network_config
            ) for i in range(depths[1])
        ])
        self.downsample_2 = PatchMerging(dim=embed_dims[1], norm_layer=norm_layer, spatial_dims=len(img_size))
        cur += depths[1]

        # Stage 3
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], 
                num_heads=num_heads[2], 
                mlp_ratio=mlp_ratios[2], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[cur + i], 
                norm_layer=norm_layer, 
                level=self.levels[2],
                ms_attention=self.multi_scale_attention,
                img_size=(img_size[0] // 8, img_size[1] // 8, img_size[2] // 8),
                network_config=self.network_config
            ) for i in range(depths[2])
        ])
        self.downsample_3 = PatchMerging(dim=embed_dims[2], norm_layer=norm_layer, spatial_dims=len(img_size))
        cur += depths[2]

        # Stage 4
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], 
                num_heads=num_heads[3], 
                mlp_ratio=mlp_ratios[3], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[cur + i], 
                norm_layer=norm_layer, 
                level=self.levels[3],
                ms_attention=self.multi_scale_attention,
                img_size=(img_size[0] // 16, img_size[1] // 16, img_size[2] // 16),
                network_config=self.network_config
            ) for i in range(depths[3])
        ])
        cur += depths[3]

        self.apply(self._init_weights)

    def proj_out(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Project output features.
        
        Args:
            x: Input tensor
            normalize: Whether to apply normalization
            
        Returns:
            Projected tensor
        """
        if normalize:
            x_shape = x.shape
            ch = int(x_shape[1])
            if len(x_shape) == 5:
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def _init_weights(self, m: nn.Module):
        """Initialize weights for the model."""
        init_config = self.network_config.get('initialization', {})
        weight_std = init_config.get('weight_std', 0.02)
        bias_constant = init_config.get('bias_constant', 0.0)
        layer_norm_weight = init_config.get('layer_norm_weight', 1.0)
        layer_norm_bias = init_config.get('layer_norm_bias', 0.0)
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=weight_std)
            if m.bias is not None:
                init.constant_(m.bias, bias_constant)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, layer_norm_bias)
            init.constant_(m.weight, layer_norm_weight)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained: str):
        """Initialize weights from pretrained model."""
        if isinstance(pretrained, str):
            self.load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')
    
    def load_dualpath_model(self, model: nn.Module, model_file: str):
        """Load dual-path model weights."""
        t_start = time.time()
        
        if isinstance(model_file, str):
            raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            if 'model' in raw_state_dict.keys():
                raw_state_dict = raw_state_dict['model']
        else:
            raw_state_dict = model_file

        t_ioend = time.time()
        model.load_state_dict(raw_state_dict, strict=False)
        t_end = time.time()
        
        # Log loading time if logger is available
        if hasattr(self, 'logger'):
            self.logger.info(f"Load model, Time usage:\n\tIO: {t_ioend - t_start}, initialize parameters: {t_end - t_ioend}")

    def forward_features(self, x_rgb: torch.Tensor, normalize: bool = True) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through feature extraction layers.
        
        Args:
            x_rgb: Input tensor of shape (B, C, D, H, W)
            normalize: Whether to normalize output features
            
        Returns:
            Tuple of (outputs, high-frequency outputs)
        """
        outs = []
        outs_hf = []

        # print(f'input: {x_rgb.shape}')
        outs = []
        outs_hf = []
        # print(f'x_rgb:{x_rgb.dtype}')
        B, C, D, H, W = x_rgb.shape
        
        ######## Patch Embedding
        x0 = self.patch_embed(x_rgb)                # B, c, d, h, w         
        x0 = self.pos_drop(x0)
        
        
        # Stage 1
        x1 = rearrange(x0, "b c d h w -> b d h w c")
        for blk in self.block1:
            x1, x_h = blk(x1, )
        x1_out = rearrange(x1, "b d h w c -> b c d h w")
        x1_out = self.proj_out(x1_out, normalize)
        outs.append(x1_out)
        outs_hf.append(x_h if x_h is not None else ())

        # Stage 2
        x2 = self.downsample_1(x1)
        for blk in self.block2:
            x2, x_h = blk(x2)
        x2_out = rearrange(x2, "b d h w c -> b c d h w")
        x2_out = self.proj_out(x2_out, normalize)
        outs.append(x2_out)
        outs_hf.append(x_h if x_h is not None else ())

        # Stage 3
        x3 = self.downsample_2(x2)
        for blk in self.block3:
            x3, x_h = blk(x3)
        x3_out = rearrange(x3, "b d h w c -> b c d h w")
        x3_out = self.proj_out(x3_out, normalize)
        outs.append(x3_out)
        outs_hf.append(x_h if x_h is not None else ())

        # Stage 4
        x4 = self.downsample_3(x3)
        for blk in self.block4:
            x4 = blk(x4)
        if isinstance(x4, tuple):
            x4 = x4[0]
        x4_out = rearrange(x4, "b d h w c -> b c d h w")
        x4_out = self.proj_out(x4_out, normalize)
        outs.append(x4_out)

        return outs, outs_hf

    def forward(self, x_rgb: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x_rgb: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Tuple of (outputs, high-frequency outputs)
        """
        return self.forward_features(x_rgb)

    def flops(self) -> int:
        """Calculate FLOPs for the model."""
        flops = 0
        # Note: This is a placeholder. Implement actual FLOP calculation if needed.
        return flops


if __name__ == "__main__":
    # Test configuration
    B = 2
    C = 4
    D = 128
    H = 128
    W = 128
    
    # Create model using MultiscaleTransformer directly
    backbone = MultiscaleTransformer(
        img_size=(D, H, W),
        patch_size=2,
        num_classes=4,
        in_chans=C,
        embed_dims=[48, 96, 192, 384],
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.1
    )
    
    # Test forward pass
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    rgb = torch.randn(B, C, D, H, W).to(device)
    backbone = backbone.to(device)
    
    outputs, outputs_hf = backbone(rgb)
    
    # Print output shapes
    for i, out in enumerate(outputs):
        print(f'Stage {i}: {out.size()}')
    
    # Calculate parameters
    total_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")