"""Robust normal distribution."""

from torch.distributions import Normal


class RobustNormal(Normal):
    def __init__(self, loc, scale, validate_args=None, eps=1e-4):
        super().__init__(loc=loc, scale=scale + eps, validate_args=validate_args)
