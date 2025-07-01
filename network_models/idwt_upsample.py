from typing import Sequence, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer


### Residual HF Refinement Block (Filters HF Before IDWT)
class HFRefinementRes(nn.Module):
    """Residual HF Refinement Block for filtering high-frequency coefficients before IDWT."""
    
    def __init__(self, in_channels, init_alpha=0.3, network_config=None):
        """
        Initialize HF refinement block.
        
        Args:
            in_channels: Number of input channels
            init_alpha: Initial alpha value for refinement
            network_config: Network configuration dictionary
        """
        super().__init__()
        
        self.network_config = network_config or {}
        hf_config = self.network_config.get('hf_refinement', {})
        
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True)
        self.norm = nn.InstanceNorm3d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=True)
        
        if hf_config.get('use_sigmoid', True):
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, x):
        """Forward pass."""
        refined = self.conv1(x)
        refined = self.norm(refined)
        refined = self.relu(refined)
        refined = self.conv2(refined)
        
        if self.sigmoid is not None:
            refined = self.sigmoid(refined)
        
        out = x * refined
        return out


class UnetrIDWTBlock(nn.Module):
    """
    Inverse Discrete Wavelet Transform (IDWT) Upsampling Block for UNETR.
    Uses HF refinement before IDWT to filter noise and enhance edges.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        stage: int,
        hf_refinement: bool,
        wavelet: str,
        kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
        network_config: Dict[str, Any] = None
    ) -> None:
        """
        Initialize IDWT block.
        
        Args:
            spatial_dims: Number of spatial dimensions
            in_channels: Number of input channels
            out_channels: Number of output channels
            stage: Stage number for HF refinement
            wavelet: Wavelet type (e.g., 'db1', 'haar')
            kernel_size: Convolution kernel size
            norm_name: Normalization type
            res_block: Whether to use residual block
            network_config: Network configuration dictionary
        """
        super(UnetrIDWTBlock, self).__init__()
        
        self.network_config = network_config or {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet = wavelet
        self.hf_refinement = hf_refinement


        # HF refinement blocks
        if self.hf_refinement:
            self.hf_ref = []
            for _ in range(stage):
                self.hf_ref.append(
                    HFRefinementRes(
                        in_channels // pow(2, stage), 
                        network_config=self.network_config
                    )
                )
            self.hf_ref = nn.ModuleList(self.hf_ref)

        # Convolution for Low-Frequency (LF) components
        self.conv_lf_block = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
            is_transposed=False,
        )

        # Select residual or basic block
        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                out_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip, hf_coeffs):
        """
        Forward pass.
        Args:
            inp: Low-frequency input from previous layer.
            skip: Skip connection from encoder.
            hf_coeffs: High-frequency coefficients from encoder.

        Returns:
            Refined and reconstructed feature map.
        """
        inp = self.conv_lf_block(inp)

        # **HF Refinement BEFORE IDWT**

        if self.hf_refinement:
            hf_coeffs = tuple(
                {key: self.hf_ref[i](hf_dict[key]) for key in hf_dict} for i, hf_dict in enumerate(hf_coeffs)
            )

        # Use filtered hf_coeffs
        inp_tuple = (inp,) + hf_coeffs
        out = ptwt.waverec3(inp_tuple, wavelet=self.wavelet)  # IDWT Reconstruction

        # **Fuse reconstructed features with skip connection**
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)

        return out