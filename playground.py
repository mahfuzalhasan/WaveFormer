import torch.nn as nn    
import torch
from torch.nn import functional as F
import os

B, C, D, H, W = 2, 128, 8, 8, 8
x = torch.randn(B, C, D, H, W)

y = F.adaptive_avg_pool3d(x, output_size=(4,4,4)) + F.adaptive_max_pool3d(x, output_size=(4,4,4))
print(y.shape)