#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Scale Head Attention Network for 3D Medical Image Segmentation.
Refactored version with configuration-driven architecture and clean code practices.
"""

import sys
import os
from typing import Tuple, Union
from functools import partial

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from ptflops import get_model_complexity_info

from monai.networks.nets import UNETR, SwinUNETR
from monai.networks.blocks.dynunet_block import UnetOutBlock, get_conv_layer
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from lib.models.tools.module_helper import ModuleHelper
from .waveformer import MultiscaleTransformer
from .idwt_upsample import UnetrIDWTBlock
from .wave_helper import ProjectionUpsample


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, dim_in: int, proj_dim: int = 256, proj: str = 'convmlp', bn_type: str = 'torchbn'):
        """
        Initialize projection head.
        
        Args:
            dim_in: Input dimension
            proj_dim: Projection dimension
            proj: Projection type ('linear' or 'convmlp')
            bn_type: Batch normalization type
        """
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )
        else:
            raise ValueError(f"Unknown projection type: {proj}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.normalize(self.proj(x), p=2, dim=1)


class ChannelCalibration(nn.Module):
    """SENet-style channel calibration with normalization."""
    
    def __init__(self, in_channels: int = 384, reduction_ratio: int = 4, norm_layer: type = nn.BatchNorm3d):
        """
        Initialize channel calibration module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel squeeze
            norm_layer: Normalization layer type
        """
        super(ChannelCalibration, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        # Dimensionality Reduction and Expansion
        self.reduce = nn.Conv3d(in_channels, reduced_channels, kernel_size=1)
        self.norm_reduce = norm_layer(reduced_channels)

        self.conv = nn.Conv3d(reduced_channels, reduced_channels, kernel_size=3, padding=1)
        self.norm_conv = norm_layer(reduced_channels)

        self.expand = nn.Conv3d(reduced_channels, in_channels, kernel_size=1)
        self.norm_expand = norm_layer(in_channels)

        # Squeeze-and-Excitation (Global Attention)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, in_channels)

        # Residual connection
        self.residual = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_channels, D, H, W)
            
        Returns:
            Output tensor of shape (B, in_channels, D, H, W)
        """
        identity = self.residual(x)

        # Dimensionality reduction and spatial refinement
        x = self.relu(self.norm_reduce(self.reduce(x)))
        x = self.relu(self.norm_conv(self.conv(x)))
        x = self.norm_expand(self.expand(x))

        # Squeeze-and-Excitation
        b, c, _, _, _ = x.shape
        se = self.global_pool(x).view(b, c)
        se = F.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        se = se.view(b, c, 1, 1, 1)
        x = x * se

        return self.relu(x + identity)


class Waveformer(nn.Module):
    """WaveFormer Network for 3D Medical Image Segmentation."""

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: int = 2,
        in_chans: int = 1,
        out_chans: int = 13,
        depths: list = None,
        feat_size: list = None,
        num_heads: list = None,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
        use_checkpoint: bool = False,
        network_config: dict = None
    ) -> None:
        """
        Initialize Waveformer network.
        
        Args:
            img_size: Input image size (D, H, W)
            patch_size: Patch size for embedding
            in_chans: Number of input channels
            out_chans: Number of output channels
            depths: Number of transformer blocks per stage
            feat_size: Feature dimensions for each stage
            num_heads: Number of attention heads per stage
            drop_path_rate: Drop path rate for stochastic depth
            layer_scale_init_value: Layer scale initialization value
            hidden_size: Hidden size for transformer
            norm_name: Normalization layer name
            conv_block: Whether to use convolutional blocks
            res_block: Whether to use residual blocks
            spatial_dims: Number of spatial dimensions
            use_checkpoint: Whether to use gradient checkpointing
            network_config: Network configuration dictionary (optional)
        """
        super().__init__()

        # Set default values if not provided
        depths = depths or [2, 2, 2, 2]
        feat_size = feat_size or [48, 96, 192, 384]
        num_heads = num_heads or [3, 6, 12, 24]

        # Store configuration
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.spatial_dims = spatial_dims
        

        self.network_config = network_config or {}
        self.transformer_config = self.network_config.get('transformer', {})
        self.hf_refinement = self.transformer_config.get('hf_refinement', False)
        
        # Create output indices
        self.out_indice = list(range(len(self.depths)))

        # Initialize waveformer encoder
        self._init_waveformer_encoder()
        
        # Initialize encoder blocks
        self._init_residual_blocks(norm_name, res_block)
        
        # Initialize decoder blocks
        self._init_decoder_blocks(norm_name, res_block)
        
        # Initialize output layer
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.feat_size[0], 
            out_channels=self.out_chans
        )

    def _init_waveformer_encoder(self):
        """Initialize waveformer encoder."""
        # Use transformer config if available, otherwise use default values
        
        
        self.waveformer_encoder = MultiscaleTransformer(
            img_size=self.img_size,
            in_chans=self.in_chans,
            patch_size=self.patch_size,
            num_classes=self.out_chans,
            embed_dims=self.transformer_config.get('embed_dims', self.feat_size),
            depths=self.transformer_config.get('depths', self.depths),
            num_heads=self.transformer_config.get('num_heads', self.num_heads),
            drop_path_rate=self.transformer_config.get('drop_path_rate', self.drop_path_rate),
            mlp_ratios=self.transformer_config.get('mlp_ratios', [4, 4, 4, 4]),
            decom_levels=self.transformer_config.get('decom_levels', [3, 2, 1, 0]),
            multi_scale_attention=self.transformer_config.get('multi_scale_attention', True),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate=0,
            drop_rate=0,
            network_config=self.network_config,
        )

    def _init_residual_blocks(self, norm_name: str, res_block: bool):
        """Initialize encoder blocks with hardcoded values like original."""
        # Encoder blocks - using hardcoded values like original
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # Channel calibration - using hardcoded values like original
        self.encoder10 = ChannelCalibration(
            in_channels=self.feat_size[3],
            reduction_ratio=4,
            norm_layer=nn.InstanceNorm3d
        )

    def _init_decoder_blocks(self, norm_name: str, res_block: bool):
        """Initialize decoder blocks with hardcoded values like original."""
        # IDWT decoder blocks - using hardcoded values like original
        
        self.decoder4 = UnetrIDWTBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            stage=1,
            hf_refinement = self.hf_refinement,
            wavelet='db1',
            kernel_size=3,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder3 = UnetrIDWTBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[1],
            stage=2,
            hf_refinement=self.hf_refinement,
            wavelet='db1',
            kernel_size=3,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2 = UnetrIDWTBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[0],
            stage=3,
            hf_refinement=self.hf_refinement,
            wavelet='db1',
            kernel_size=3,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # Learnable upsampling - using hardcoded values like original
        self.learnable_up4 = ProjectionUpsample(
            in_channels=self.feat_size[2], 
            out_channels=self.feat_size[0], 
            stride=4, 
            residual=True, 
            use_double_conv=True
        )
        self.learnable_up3 = ProjectionUpsample(
            in_channels=self.feat_size[1], 
            out_channels=self.feat_size[0], 
            stride=2, 
            residual=True
        )
        
        # Final decoder - using hardcoded values like original
        self.decoder1 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[0] * 3,  # 3 concatenated features
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

    def _get_norm_layer(self, norm_name: str) -> type:
        """Get normalization layer class from name."""
        norm_layers = {
            'LayerNorm': nn.LayerNorm,
            'BatchNorm3d': nn.BatchNorm3d,
            'InstanceNorm3d': nn.InstanceNorm3d,
            'GroupNorm': nn.GroupNorm
        }
        
        if norm_name not in norm_layers:
            raise ValueError(f"Unknown normalization layer: {norm_name}")
        
        return norm_layers[norm_name]

    # def proj_feat(self, x: torch.Tensor, hidden_size: int, feat_size: Tuple[int, int, int]) -> torch.Tensor:
    #     """Project features to the correct shape."""
    #     new_view = (x.size(0), *feat_size, hidden_size)
    #     x = x.view(new_view)
    #     new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
    #     x = x.permute(new_axes).contiguous()
    #     return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Get transformer outputs
        outs, outs_hf = self.waveformer_encoder(x_in)
        
        # Residual Layers on features
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(outs[0])
        enc2 = self.encoder3(outs[1])
        enc3 = self.encoder4(outs[2])

        # Channel Calibration
        dec5 = self.encoder10(outs[3])

        # Decoder
        dec4 = self.decoder4(dec5, enc3, outs_hf[-1])
        dec3 = self.decoder3(dec5, enc2, outs_hf[-2])
        dec2 = self.decoder2(dec5, enc1, outs_hf[-3])

        # Learnable upsampling
        dec4_upsampled = self.learnable_up4(dec4)
        dec3_upsampled = self.learnable_up3(dec3)

        # Fuse all decoder features
        combined = torch.cat([dec4_upsampled, dec3_upsampled, dec2], dim=1)  # Concatenate along channel dimension
        dec1 = self.decoder1(combined, enc0)
        
        return self.out(dec1)


def create_waveformer(network_config: dict) -> Waveformer:
    """
    Create Waveformer model from configuration.
    
    Args:
        network_config: Network configuration dictionary
        
    Returns:
        Waveformer: Initialized model
    """
    return Waveformer(
        img_size=network_config['img_size'],
        patch_size=network_config['patch_size'],
        in_chans=network_config['in_chans'],
        out_chans=network_config['out_chans'],
        depths=network_config['depths'],
        feat_size=network_config['embed_dims'],
        num_heads=network_config['num_heads'],
        drop_path_rate=network_config['drop_path_rate'],
        use_checkpoint=network_config.get('use_checkpoint', False),
        network_config=network_config
    )


if __name__ == "__main__":
    # Test configuration
    B = 1
    C = 4
    D = 128
    H = 128
    W = 128
    num_classes = 4
    img_size = (D, H, W)
    
    # Create model
    model = Waveformer(
        img_size=(D, H, W),
        patch_size=2,
        in_chans=C,
        out_chans=num_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.1,
        use_checkpoint=False,
    )
    
    model.cuda()
    x = torch.randn(B, C, D, H, W).cuda()
    outputs = model(x)
    print(f'Output shape: {outputs.shape}')

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # # Calculate FLOPs
    # macs, params = get_model_complexity_info(
    #     model, (C, D, H, W), 
    #     as_strings=True, 
    #     print_per_layer_stat=True, 
    #     verbose=True
    # )