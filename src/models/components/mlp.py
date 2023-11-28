from typing import List

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
    layers.append(nn.Linear(dims[-1], output_size, device=device))
    return layers
