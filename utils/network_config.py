#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Network configuration utility for the BraTS training project.
Handles configuration parsing, validation, and provides default values.
Simplified to match the original code structure.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from functools import partial


class NetworkConfig:
    """Network configuration handler with validation and default values."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize network configuration from config dictionary.
        
        Args:
            config: Configuration dictionary containing network settings
        """
        self.config = config.get('network', {})
        self._validate_config()
    
    def _validate_config(self):
        """Validate the network configuration."""
        required_keys = ['model_type', 'in_channels', 'out_channels', 'img_size']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required network configuration key: {key}")
        
        # Validate transformer configuration
        if 'transformer' not in self.config:
            raise ValueError("Missing transformer configuration")
        
        transformer = self.config['transformer']
        required_transformer_keys = ['embed_dims', 'depths', 'num_heads']
        for key in required_transformer_keys:
            if key not in transformer:
                raise ValueError(f"Missing required transformer configuration key: {key}")
        
        # Validate that lists have the same length
        embed_dims = transformer['embed_dims']
        depths = transformer['depths']
        num_heads = transformer['num_heads']
        
        if not (len(embed_dims) == len(depths) == len(num_heads)):
            raise ValueError("embed_dims, depths, and num_heads must have the same length")
    
    @property
    def model_type(self) -> str:
        """Get model type."""
        return self.config['model_type']
    
    @property
    def in_channels(self) -> int:
        """Get input channels."""
        return self.config['in_channels']
    
    @property
    def out_channels(self) -> int:
        """Get output channels."""
        return self.config['out_channels']
    
    @property
    def img_size(self) -> Tuple[int, int, int]:
        """Get image size as tuple."""
        img_size = self.config['img_size']
        return tuple(img_size) if isinstance(img_size, list) else img_size
    
    @property
    def patch_size(self) -> int:
        """Get patch size."""
        return self.config.get('patch_size', 2)
    
    @property
    def spatial_dims(self) -> int:
        """Get spatial dimensions."""
        return self.config.get('spatial_dims', 3)
    
    @property
    def hidden_size(self) -> int:
        """Get hidden size."""
        return self.config.get('hidden_size', 768)
    
    @property
    def layer_scale_init_value(self) -> float:
        """Get layer scale initialization value."""
        return self.config.get('layer_scale_init_value', 1e-6)
    
    @property
    def conv_block(self) -> bool:
        """Get conv block setting."""
        return self.config.get('conv_block', True)
    
    @property
    def res_block(self) -> bool:
        """Get res block setting."""
        return self.config.get('res_block', True)
    
    @property
    def use_checkpoint(self) -> bool:
        """Get use checkpoint setting."""
        return self.config.get('use_checkpoint', False)
    
    @property
    def transformer_config(self) -> Dict[str, Any]:
        """Get transformer configuration."""
        transformer = self.config['transformer']
        return {
            'embed_dims': transformer['embed_dims'],
            'depths': transformer['depths'],
            'num_heads': transformer['num_heads'],
            'mlp_ratios': transformer.get('mlp_ratios', [4, 4, 4, 4]),
            'decom_levels': transformer.get('decom_levels', [3, 2, 1, 0]),
            'qkv_bias': transformer.get('qkv_bias', True),
            'qk_scale': transformer.get('qk_scale', None),
            'drop_rate': transformer.get('drop_rate', 0.0),
            'attn_drop_rate': transformer.get('attn_drop_rate', 0.0),
            'drop_path_rate': transformer.get('drop_path_rate', 0.1),
            'patch_norm': transformer.get('patch_norm', False),
            'norm_layer': self._get_norm_layer(transformer.get('norm_layer', 'LayerNorm')),
            'norm_eps': transformer.get('norm_eps', 1e-6)
        }
    
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
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for model initialization."""
        base_kwargs = {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'in_chans': self.in_channels,
            'spatial_dims': self.spatial_dims,
            'hidden_size': self.hidden_size,
            'layer_scale_init_value': self.layer_scale_init_value,
            'conv_block': self.conv_block,
            'res_block': self.res_block,
            'use_checkpoint': self.use_checkpoint,
            'network_config': self.config,
            **self.transformer_config
        }
        
        # Add Waveformer-specific parameters only if this is a Waveformer model
        if self.model_type == "Waveformer":
            base_kwargs.update({
                'out_chans': self.out_channels,
            })
        
        return base_kwargs
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"NetworkConfig(model_type={self.model_type}, in_channels={self.in_channels}, out_channels={self.out_channels}, img_size={self.img_size})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__()


def get_network_config(config: Dict[str, Any]) -> NetworkConfig:
    """
    Get network configuration from main config.
    
    Args:
        config: Main configuration dictionary
        
    Returns:
        NetworkConfig: Network configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    return NetworkConfig(config)


def create_norm_layer(norm_name: str, num_features: int, eps: float = 1e-6) -> nn.Module:
    """
    Create normalization layer.
    
    Args:
        norm_name: Name of normalization layer
        num_features: Number of features
        eps: Epsilon value for normalization
        
    Returns:
        nn.Module: Normalization layer
    """
    if norm_name.lower() == 'instance':
        return nn.InstanceNorm3d(num_features, eps=eps, affine=True)
    elif norm_name.lower() == 'batch':
        return nn.BatchNorm3d(num_features, eps=eps)
    elif norm_name.lower() == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=num_features, eps=eps)
    elif norm_name.lower() == 'layer':
        return nn.LayerNorm(num_features, eps=eps)
    else:
        raise ValueError(f"Unknown normalization layer: {norm_name}")


def get_activation_layer(activation_name: str) -> nn.Module:
    """
    Get activation layer.
    
    Args:
        activation_name: Name of activation function
        
    Returns:
        nn.Module: Activation layer
    """
    activation_layers = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'leaky_relu': nn.LeakyReLU,
        'silu': nn.SiLU,
        'swish': nn.SiLU
    }
    
    if activation_name.lower() not in activation_layers:
        raise ValueError(f"Unknown activation layer: {activation_name}")
    
    return activation_layers[activation_name.lower()]() 