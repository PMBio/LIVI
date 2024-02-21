import sys
from itertools import chain
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.distributions import RobustNormal
from src.models.components.mlp import create_mlp
from src.models.vae import VAE, Encoder, NegativeBinomialDecoderBatchSex


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
        device: str = "cuda",
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
        """
        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=[],
            learning_rate=learning_rate,
            layer_norm=layer_norm,
            likelihood=likelihood,
            device=device,
        )

        self.y_dim = y_dim

        # embeddings for cell-state-spacific and persistent effects
        self.U_context = nn.Embedding(y_dim, z_dim, device=device)
        self.V_persistent = nn.Embedding(y_dim, z_dim, device=device)

        self.pretrain_mode = False

        self.save_hyperparameters()

    def set_pretrain_mode(self, mode: bool):
        """Set VAE pretrain mode.

        If True, label information is ignored.
        """
        self.pretrain_mode = mode
        if mode:
            # freeze parameters
            self.U_context.requires_grad = False
            self.V_persistent.requires_grad = False
        else:
            # unfreeze parameters
            self.U_context.requires_grad = True
            self.V_persistent.requires_grad = True

    def transform_latent(self, z: torch.Tensor, y: torch.Tensor):
        if self.pretrain_mode:
            return z
        else:
            return z + z * self.U_context(y) + self.V_persistent(y)


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
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        device: str = "cuda",
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
        """
        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            layer_norm=layer_norm,
            likelihood=likelihood,
            device=device,
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
            device=device,
        )

        self.automatic_optimization = False

        self.save_hyperparameters()

    def set_pretrain_mode(self, mode: bool):
        """Set VAE pretrain mode.

        If True, label information is ignored.
        """
        super().set_pretrain_mode(mode)
        if mode:
            self.adversary.requires_grad = False
        else:
            self.adversary.requires_grad = True

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, size_factor = self.prepare_batch(batch)
        optim_vae, optim_adversary = self.optimizers()

        train_adversary = batch_idx % self.adversary_steps == 0
        train_adversary = train_adversary & (not self.pretrain_mode)
        train_adversary = train_adversary * (mode == "train")

        z_dist = self(x)
        if self.pretrain_mode:
            # no adversary signal
            loss = torch.zeros([1], device=self.device)
        else:
            z = z_dist.rsample()
            loss = F.cross_entropy(self.adversary(z), y)
        logs = {f"{mode}/adversary_loss": loss.item()}

        if train_adversary:
            # train adversary
            optim_adversary.zero_grad()
            self.manual_backward(loss)
            optim_adversary.step()
        else:
            # train vae
            elbo = self.compute_elbo(z_dist, x, y, size_factor)
            logs[f"{mode}/elbo"] = elbo.item()
            loss = -elbo - self.adversary_weight * loss
            logs[f"{mode}/livi_loss"] = loss.item()
            logs["hp_metric"] = loss.item()

            if mode == "train":
                optim_vae.zero_grad()
                self.manual_backward(loss)
                optim_vae.step()

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        self.step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        """Configures optimizer."""
        params = chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.U_context.parameters(),
            self.V_persistent.parameters(),
        )

        optim_vae = torch.optim.Adam(
            params,
            lr=self.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.adversary_learning_rate,
        )
        return optim_vae, optim_adversary


class LIVIadvBatchSex(LIVIadv):
    """Variant of LIVI that learns two additional correction terms per gene, one to account for the
    effect of the experimental batch and another one to account for the effect of the donor sex.

    The gene-level correction terms are added to the decoder output.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        exbatch_dim: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        layer_norm: bool,
        donor_sex_dim: int = 2,
        likelihood: str = "nb_batch_sex",
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        device: str = "cuda",
    ):
        """Initializes LIVI.

        Args:
            x_dim: Dimensionality of input data.
            z_dim: Dimensionality of cell-state latent space.
            exbatch_dim: Number of experimental batches in the dataset.
            donor_sex_dim: Number of potential donor sexes in the dataset.
            y_dim: Number of individuals in the dataset.
            encoder_hidden_dims: List of hidden dimensions for each encoder layer.
            learning_rate: Learning rate.
            layer_norm: Whether to use layer normalization.
            likelihood: Likelihood model to use. Defaults to one of "nb_batch_sex", which means Negative-Binomial likelihood
                        with batch and sex correction after decoding.
            adversarial_weight: If > 0, add adversarial loss to remove individual effects from the cell-state latent space.
            adversary_hidden_dims: List of hidden dimensions for each adversary layer.
            adversary_learning_rate: Learning rate for adversary.
            adversary_steps: Number of steps to train adversary for every step of VAE.
            device: Accelerator to use for training.
        """
        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            layer_norm=layer_norm,
            likelihood=likelihood,
            adversary_weight=adversary_weight,
            adversary_hidden_dims=adversary_hidden_dims,
            adversary_learning_rate=adversary_learning_rate,
            adversary_steps=adversary_steps,
            device=device,
        )

        self.y_dim = y_dim

        # self.decoder = NegativeBinomialDecoderBatchSex(
        #     z_dim=z_dim,
        #     x_dim=x_dim,
        #     decoder_hidden_dims=[],
        #     layer_norm=layer_norm,
        #     device=device,
        # )

        # Sex and batch correction per gene
        if donor_sex_dim is not None:
            self.sex_effect = nn.Embedding(donor_sex_dim, x_dim, device=device)
        else:
            self.sex_effect = None
        self.batch_effect = nn.Embedding(exbatch_dim, x_dim, device=device)

        self.save_hyperparameters()

    def prepare_batch(self, batch):
        x = batch["x"]
        y = batch["y"]
        dsex = None if self.hparams.donor_sex_dim is None else batch["dsex"]
        eb = batch["eb"]
        size_factor = batch["size_factor"]
        return x, y, dsex, eb, size_factor

    def compute_elbo(
        self,
        z_dist: torch.distributions.Distribution,
        x: torch.Tensor,
        y: torch.Tensor,
        dsex: torch.Tensor,
        eb: torch.Tensor,
        size_factor: torch.Tensor = None,
    ):
        """Computes evidence lower bound (ELBO).

        Args:
            z_dist: Variational distribution.
            x: Input data.
            y: Donor IDs.
            dsex: Donor sex.
            eb: Experimental batch IDs.
            size_factor: Size factor to correct for gene count differences.

        Returns:
            Mean evidence lower bound (ELBO).
        """
        z = z_dist.rsample()
        # z = nn.Softmax(dim=1)(z)
        z_combined = self.transform_latent(z, y)
        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()

        log_lik = (
            self.decoder(
                z=z_combined,
                size_factor=size_factor,
                batch_effect=self.batch_effect(eb),
                donor_sex_effect=self.sex_effect(dsex) if dsex is not None else None,
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div

    def step(self, batch, batch_idx, optimizer_idx=0, mode="train"):
        """Performs a single training or validation step."""
        x, y, donor_sex, exp_batch_ids, size_factor = self.prepare_batch(batch)

        z_dist = self(x)
        z = z_dist.rsample()
        loss = F.cross_entropy(self.adversary(z), y)
        logs = {f"{mode}/adversary_loss": loss.item()}

        if optimizer_idx == 0:
            elbo = self.compute_elbo(z_dist, x, y, donor_sex, exp_batch_ids, size_factor)
            logs[f"{mode}/elbo"] = elbo.item()
            loss = -elbo - self.adversary_weight * loss
            logs[f"{mode}/total_loss"] = loss.item()
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
            self.batch_effect.parameters(),
            self.sex_effect.parameters(),
        )

        optim_vae = torch.optim.Adam(
            params,
            lr=self.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.hparams.adversary_learning_rate,
        )
        return (
            {
                "optimizer": optim_vae,
                "frequency": 1,
            },
            {
                "optimizer": optim_adversary,
                "frequency": self.hparams.adversary_steps,
            },
        )
