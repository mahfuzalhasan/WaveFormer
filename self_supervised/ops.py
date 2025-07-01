# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from numpy.random import randint


def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=args.local_rank
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x


def rot_rand(args, x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
    return x_aug


def augment_context_restoration(x, num_swaps=3, max_patch_fraction=0.2):
    """
    Randomly selects up to 'num_swaps' patches in the same volume and swaps them.
    This is a simplified context restoration augmentation.

    Args:
        x: torch.Tensor of shape [C, H, W, D]
        num_swaps (int): how many patch-swaps to perform
        max_patch_fraction (float): maximum fraction of x's dimension for patch size
    Returns:
        x_aug: The same tensor with some patches swapped in-place
    """
    # x shape: (channels, height, width, depth)
    # We'll do in-place modifications, so clone first if you need to preserve original
    c, h, w, z = x.shape

    # Convert to CPU if needed for convenience (optional).
    device = x.device

    # For each swap:
    for _ in range(num_swaps):
        # Random patch size
        ph = randint(1, int(h * max_patch_fraction))
        pw = randint(1, int(w * max_patch_fraction))
        pd = randint(1, int(z * max_patch_fraction))

        # Pick two random positions:
        #   (r1,c1,s1) is top-left of patch1
        #   (r2,c2,s2) is top-left of patch2
        r1 = randint(0, h - ph)
        c1 = randint(0, w - pw)
        s1 = randint(0, z - pd)

        r2 = randint(0, h - ph)
        c2 = randint(0, w - pw)
        s2 = randint(0, z - pd)

        # Swap the contents of these two patches
        patch1 = x[:, r1:r1+ph, c1:c1+pw, s1:s1+pd].clone()
        patch2 = x[:, r2:r2+ph, c2:c2+pw, s2:s2+pd].clone()

        x[:, r1:r1+ph, c1:c1+pw, s1:s1+pd] = patch2
        x[:, r2:r2+ph, c2:c2+pw, s2:s2+pd] = patch1

    return x
