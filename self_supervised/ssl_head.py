import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import ViT
from monai.utils import ensure_tuple_rep


class SSLViT(nn.Module):
    """
    A self-supervised Vision Transformer (ViT) that uses:
      1) Reconstruction head: decodes feature maps to the original volume.
      2) Contrastive head: projects the global representation to a latent space
         for contrastive learning.

    No rotation head is included, matching the approach from the survival
    prediction paper that has only reconstruction + contrastive tasks.
    """

    def __init__(
        self,
        args,
        upsample_mode: str = "vae",
        hidden_size: int = 768,
        projection_size: int = 256,
    ):
        """
        Args:
            args: config/namespace with fields:
                - in_channels (int): number of input MRI channels
                - spatial_dims (int): 2 or 3 (likely 3 for volumetric data)
                - img_size (tuple or int): size of input images (H, W, D)
                - patch_size (int or tuple): size of each patch
                - mlp_dim (int): MLP dimension inside ViT
                - num_layers (int): number of transformer blocks
                - num_heads (int): attention heads
                - drop_rate (float): dropout rate
            upsample_mode: how to upsample for reconstruction: "vae" | "deconv" | "large_kernel_deconv"
            hidden_size: dimension of ViT embeddings
            projection_size: dimension of the contrastive embedding
        """
        super().__init__()

        # Ensure patch_size is a tuple of length = spatial_dims
        patch_size = ensure_tuple_rep(args.patch_size, args.spatial_dims)
        if isinstance(args.img_size, int):
            img_size = ensure_tuple_rep(args.img_size, args.spatial_dims)
        else:
            img_size = args.img_size

        # ---------------------
        # 1) ViT Encoder
        # ---------------------
        self.vit = ViT(
            in_channels=args.in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            pos_embed="conv",      # or "perceptron" if desired
            classification=False,  # we want feature maps, not a classification token
            dropout_rate=args.drop_rate,
            spatial_dims=args.spatial_dims,
        )

        # ---------------------
        # 2) Contrastive Head
        # ---------------------
        # We'll do a simple projection from [B, hidden_size] to [B, projection_size].
        self.proj_contrastive = nn.Linear(hidden_size, projection_size)

        # ---------------------
        # 3) Reconstruction Decoder
        # ---------------------
        # We assume the feature map shape is [B, hidden_size, H', W', D'].
        # We'll reconstruct back to [B, in_channels, H, W, D].
        if upsample_mode == "large_kernel_deconv":
            self.decoder = nn.ConvTranspose3d(
                hidden_size, args.in_channels,
                kernel_size=(32, 32, 32), stride=(32, 32, 32)
            )
        elif upsample_mode == "deconv":
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 2, 2),
                nn.ConvTranspose3d(hidden_size // 2, hidden_size // 4, 2, 2),
                nn.ConvTranspose3d(hidden_size // 4, hidden_size // 8, 2, 2),
                nn.ConvTranspose3d(hidden_size // 8, hidden_size // 16, 2, 2),
                nn.ConvTranspose3d(hidden_size // 16, args.in_channels, 2, 2),
            )
        elif upsample_mode == "vae":
            self.decoder = nn.Sequential(
                nn.Conv3d(hidden_size, hidden_size // 2, 3, stride=1, padding=1),
                nn.InstanceNorm3d(hidden_size // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),

                nn.Conv3d(hidden_size // 2, hidden_size // 4, 3, stride=1, padding=1),
                nn.InstanceNorm3d(hidden_size // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),

                nn.Conv3d(hidden_size // 4, hidden_size // 8, 3, stride=1, padding=1),
                nn.InstanceNorm3d(hidden_size // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),

                nn.Conv3d(hidden_size // 8, hidden_size // 16, 3, stride=1, padding=1),
                nn.InstanceNorm3d(hidden_size // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),

                nn.Conv3d(hidden_size // 16, hidden_size // 16, 3, stride=1, padding=1),
                nn.InstanceNorm3d(hidden_size // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),

                nn.Conv3d(hidden_size // 16, args.in_channels, kernel_size=1, stride=1),
            )
        else:
            raise ValueError(f"Unknown upsample mode: {upsample_mode}")

    def forward(self, x):
        """
        Forward pass:
          1) Pass augmented input 'x' through ViT -> feature maps
          2) Global average pool for contrastive embeddings
          3) Decode feature maps to reconstruct the input
        Returns:
          embeddings [B, projection_size]  - for contrastive
          rec_volume [B, in_channels, H, W, D] - reconstruction
        """
        # 1) ViT: [B, hidden_size, H', W', D']
        feat_map = self.vit(x)
        B, C, H, W, D = feat_map.shape

        # 2) Contrastive Embeddings via global avg-pool
        #    Flatten the spatial dims and average: shape = [B, C]
        pooled = feat_map.view(B, C, -1).mean(dim=2)
        embeddings = self.proj_contrastive(pooled)

        # 3) Reconstruction
        rec_volume = self.decoder(feat_map)  # shape -> [B, in_channels, H, W, D]

        return embeddings, rec_volume
