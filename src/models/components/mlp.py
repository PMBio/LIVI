import math
from typing import List, Optional

import torch
import torch.nn as nn


def create_mlp(
    input_size: int,
    output_size: int,
    hidden_dims: List[int],
    layer_norm: bool,
    device: str,
) -> nn.Sequential:
    dims = [input_size] + hidden_dims
    layers = nn.Sequential()
    for i in range(len(dims) - 1):
        layers += nn.Sequential(
            nn.Linear(dims[i], dims[i + 1], device=device),
            nn.ReLU(),
            nn.LayerNorm(dims[i + 1]) if layer_norm else nn.Identity(),
        )
    has_hidden = len(hidden_dims) != 0
    layers.append(nn.Linear(dims[-1], output_size, bias=has_hidden, device=device))
    return layers


def init_mlp(mlp: nn.Sequential, generator: Optional[torch.Generator] = None) -> None:
    # He Uniform initialization for the MLP weights
    for layer in mlp:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu", generator=generator)
            # Initialise bias same way as in pytorch v2.3: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
