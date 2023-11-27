import sys
from itertools import chain
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import pytorch_lightning as pl

from src.models.components.mlp import create_mlp
from src.models.components.distributions import RobustNormal
from src.models.vae import VAE, Encoder, VAEBatch, NormalDecoderBatch

class LIVI(VAE):
    """The Latent Interaction Variational Inference (LIVI) model.

    LIVI is a linearly decoded VAE with a conditional latent space:

    z | y = z_base + Az_base * U_context(y) + V_persistent(y)

    where z_base is independent of y. A is an (optional) linear transformation
    to model interactions between latent dimensions and U_context and V_persistent
    are linear models.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        layer_norm: bool,
        likelihood: str = "nb",
        transform_base: bool = True,
    ):
        """Initializes LIVI.

        Args:
            x_dim: Dimensionality of input data.
            z_dim: Dimensionality of continuous latent space.
            y_dim: Dimensionality of discrete latent space.
            encoder_hidden_dims: List of hidden dimensions for each encoder layer.
            learning_rate: Learning rate.
            layer_norm: Whether to use layer normalization.
            likelihood: Likelihood model to use, one of "normal" or "poisson".
            transform_base: Whether to apply linear transformation to the base
                latent space before computing dynamic effects.
        """
        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=[],
            learning_rate=learning_rate,
            layer_norm=layer_norm,
            likelihood=likelihood,
        )

        self.y_dim = y_dim

        # embeddings for cell-state-spacific and persistent effects
        self.U_context = nn.Embedding(y_dim, z_dim)
        self.V_persistent = nn.Embedding(y_dim, z_dim)

        self.transform_base = transform_base
        if self.transform_base:
            # used to transform z before multiplication with U_context
            self.A = nn.Parameter(torch.randn(z_dim, z_dim))

        self.save_hyperparameters()

    def transform_latent(self, z: torch.Tensor, y: torch.Tensor):
        zA = z
        if self.transform_base:
            zA = z @ self.A
        return z + zA * self.U_context(y) + self.V_persistent(y)


class LIVIadv(LIVI):
    """LIVI with adversarial loss.

    Variant of LIVI that includes an adversarial loss for explicitly removing label information
    from the base latent space z_base.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        layer_norm: bool,
        likelihood: str = "nb",
        transform_base: bool = True,
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
    ):
        """Initializes LIVI.

        Args:
            x_dim: Dimensionality of input data.
            z_dim: Dimensionality of continuous latent space.
            y_dim: Dimensionality of discrete latent space.
            encoder_hidden_dims: List of hidden dimensions for each encoder layer.
            learning_rate: Learning rate.
            layer_norm: Whether to use layer normalization.
            likelihood: Likelihood model to use, one of "normal" or "poisson".
            adversarial_weight: If > 0, add adversarial loss to remove individual effects.
            adversary_hidden_dims: List of hidden dimensions for each adversary layer.
            adversary_learning_rate: Learning rate for adversary.
            adversary_steps: Number of steps to train adversary for every step of VAE.
            transform_base: Whether to apply linear transformation to the base latent space before computing dynamic effects.
        """
        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            layer_norm=layer_norm,
            likelihood=likelihood,
            transform_base=transform_base,
        )

        self.adversary_learning_rate = adversary_learning_rate
        self.adversary_steps = adversary_steps

        # set up adversary
        self.adversary_weight = adversary_weight
        self.adversary = create_mlp(
            input_size=z_dim,
            output_size=y_dim,
            hidden_dims=adversary_hidden_dims,
            layer_norm=False,
        )
        self.save_hyperparameters()

    def step(self, batch, batch_idx, optimizer_idx=0, mode="train"):
        """Performs a single training or validation step."""
        x, y, size_factor = self.prepare_batch(batch)

        z_dist = self(x)
        z = z_dist.rsample()
        loss = F.cross_entropy(self.adversary(z), y)
        logs = {f"{mode}/adversary_loss": loss.item()}

        if optimizer_idx == 0:
            elbo = self.compute_elbo(z_dist, x, y, size_factor)
            logs[f"{mode}/elbo"] = elbo.item()
            loss = -elbo - self.adversary_weight * loss
            logs[f"{mode}/vae_loss"] = loss.item()
            logs["hp_metric"] = loss.item()

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Performs a single training step."""
        return self.step(batch, batch_idx, optimizer_idx, mode="train")

    def validation_step(self, batch, batch_idx, optimizer_idx):
        """Performs a single validation step."""
        return self.step(batch, batch_idx, optimizer_idx, mode="val")

    def configure_optimizers(self):
        """Configures optimizer."""
        params = chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.U_context.parameters(),
            self.V_persistent.parameters(),
        )
        if self.transform_base:
            params = chain(params, [self.A])

        optim_vae = torch.optim.Adam(
            params,
            lr=self.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.adversary_learning_rate,
        )
        return (
            {
                "optimizer": optim_vae,
                "frequency": 1,
            },
            {
                "optimizer": optim_adversary,
                "frequency": self.adversary_steps,
            },
        )

