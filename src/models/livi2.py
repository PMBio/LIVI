from itertools import chain
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.distributions import RobustNormal
from src.models.components.mlp import create_mlp
from src.models.vae import Encoder, LIVI2_Decoder


class LIVI2(pl.LightningModule):
    """LIVI with separate decoders for cell-state and genetic factors.

    Each cell-state has its own genetic factors (hierarchical model).
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_gxc_factors: int,
        n_persistent_factors: int,
        exbatch_dim: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        layer_norm: bool,
        donor_sex_dim: int = 2,
        pretrain_vae: bool = True,
        pretrain_G: bool = False,
        l1_weight: float = 0.001,
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
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_gxc_factors = n_gxc_factors
        self.n_persistent_factors = n_persistent_factors
        # self.pretrain_mode = pretrain_vae
        # self.pretrain_G = pretrain_G

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=layer_norm,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))

        self.adversary_learning_rate = adversary_learning_rate
        self.adversary_steps = adversary_steps
        # Set up adversary
        self.adversary_weight = adversary_weight
        self.adversary = create_mlp(
            input_size=z_dim,
            output_size=y_dim,
            hidden_dims=adversary_hidden_dims,
            layer_norm=False,
            device=device,
        )

        # hierarchical model
        self.U_context = nn.Embedding(y_dim, z_dim * n_gxc_factors, device=device)
        self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)

        # decoder_kwargs = {"n_gxc_factors": n_gxc_factors,
        #                   "n_persistent_factors": n_persistent_factors,
        #                   "pretrain_VAE": pretrain_vae,
        #                   "pretrain_G": pretrain_vae,
        #                   "batch_norm": False}

        self.decoder = LIVI2_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=layer_norm,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=pretrain_vae,
            pretrain_G=pretrain_G,
            device=device,
        )

        # Sex and batch correction per gene
        if donor_sex_dim is not None:
            self.sex_effect = nn.Embedding(donor_sex_dim, x_dim, device=device)
        else:
            self.sex_effect = None
        self.batch_effect = nn.Embedding(exbatch_dim, x_dim, device=device)

        self.automatic_optimization = False
        self.save_hyperparameters()

    def set_pretrain_mode(self, mode: bool):
        """Set VAE pretrain mode.

        If True, discriminator and genetic embeddings are not optimised.
        """
        self.pretrain_mode = mode
        self.decoder.pretrain_VAE = mode
        if mode:
            # freeze parameters
            self.adversary.requires_grad = False
            self.U_context.requires_grad = False
            self.V_persistent.requires_grad = False
            self.decoder.persistent_decoder[0].weight.requires_grad = False
        else:
            # unfreeze parameters
            self.adversary.requires_grad = True
            self.V_persistent.requires_grad = True
            self.decoder.persistent_decoder[0].weight.requires_grad = True

    def set_pretrain_G_mode(self, mode: bool):
        """Set persistent (global) genetic effects pretrain mode.

        If True, the context-specific genetic effects are not learned.
        """
        self.pretrain_G = mode
        self.decoder.pretrain_G = mode
        if mode:
            # freeze parameters
            self.U_context.requires_grad = False
            self.decoder.context_decoder[0].weight.requires_grad = False
        else:
            # unfreeze parameters
            self.U_context.requires_grad = True
            self.decoder.context_decoder[0].weight.requires_grad = True

    def prepare_batch(self, batch):
        x = None
        y = None
        dsex = None
        eb = None
        size_factor = None

        if self.hparams.donor_sex_dim is not None:
            x, y, dsex, eb, size_factor = batch
        else:
            x, y, eb, size_factor = batch

        return x, y, dsex, eb, size_factor

    def forward(self, x: torch.Tensor):
        """Encodes data into cell-state latent space."""
        return self.encoder(x)

    def get_prior(self) -> tdist.Distribution:
        """Constructs zbase prior for given batch shape."""
        return tdist.Independent(tdist.Normal(self.z_prior_loc, self.z_prior_scale), 1)

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
        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()

        z_interaction = self.U_context(y) * z.repeat_interleave(self.n_gxc_factors, dim=1)

        log_lik = (
            self.decoder(
                z=z,
                GxC=z_interaction,
                persistent_G=self.V_persistent(y),
                size_factor=size_factor,
                batch_effect=self.batch_effect(eb),
                donor_sex_effect=self.sex_effect(dsex) if dsex is not None else None,
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, donor_sex, exp_batch_ids, size_factor = self.prepare_batch(batch)
        optim_vae, optim_adversary = self.optimizers()

        train_adversary = batch_idx % self.adversary_steps == 0
        train_adversary = train_adversary & (not self.pretrain_mode)
        train_adversary = train_adversary * (mode == "train")

        z_dist = self(x)

        if self.pretrain_mode:
            # no adversary signal
            adversarial_loss = torch.zeros([1], device=self.device)
        else:
            z = z_dist.rsample()
            adversarial_loss = F.cross_entropy(self.adversary(z), y)
        logs = {f"{mode}/adversarial_loss": adversarial_loss.item()}

        if train_adversary:
            # train adversary
            optim_adversary.zero_grad()
            self.manual_backward(adversarial_loss)
            optim_adversary.step()
        else:
            # train vae
            elbo = self.compute_elbo(z_dist, x, y, donor_sex, exp_batch_ids, size_factor)
            logs[f"{mode}/elbo"] = elbo.item()
            if self.pretrain_G:
                l1_loss_context = torch.zeros([1], device=self.device)
                if self.pretrain_mode:
                    l1_loss_persistent = torch.zeros([1], device=self.device)
                else:
                    l1_loss_persistent = self.hparams.l1_weight * torch.linalg.norm(
                        torch.cat([p for p in self.decoder.persistent_decoder.parameters()]), ord=1
                    )
            else:
                l1_loss_context = self.hparams.l1_weight * torch.linalg.norm(
                    torch.cat([p for p in self.decoder.context_decoder.parameters()]), ord=1
                )
                l1_loss_persistent = self.hparams.l1_weight * torch.linalg.norm(
                    torch.cat([p for p in self.decoder.persistent_decoder.parameters()]), ord=1
                )
            logs[f"{mode}/L1_penalty_context"] = l1_loss_context.item()
            logs[f"{mode}/L1_penalty_persistent"] = l1_loss_persistent.item()
            loss = (
                -elbo
                - self.adversary_weight * adversarial_loss
                + l1_loss_context
                + l1_loss_persistent
            )
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
        return self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        return self.step(batch, batch_idx, mode="val")

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
            lr=self.hparams.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.hparams.adversary_learning_rate,
        )

        return optim_vae, optim_adversary
