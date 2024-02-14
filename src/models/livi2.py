from itertools import chain
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.mlp import create_mlp
from src.models.vae import Encoder, LIVI2_Decoder


class LIVI2_experimental(pl.LightningModule):
    """LIVI with flexibility to add only context-specific or context-specific and persistent
    genetic effects and separate decoders and impose structure in the context-specific effects."""

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
        warmup_epochs_vae: int = 10,
        warmup_epochs_G: int = 10,
        hierarchical_model: bool = True,
        donor_sex_dim: int = 2,
        l1_weight: float = 0.001,
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
    ):
        """Initializes LIVI.

        Parameters
        ----------
            x_dim (int): Dimensionality of input data.
            z_dim (int): Dimensionality of cell-state latent space.
            exbatch_dim (int): Number of experimental batches in the dataset.
            donor_sex_dim (int): Number of potential donor sexes in the dataset.
            y_dim (int): Number of individuals in the dataset.
            encoder_hidden_dims (List[int]): List of hidden dimensions for each encoder layer.
            layer_norm (bool): Whether to apply layer normalisation to the encoder.
            hierarchical_model (bool): Whether the context-specific genetic factors have fixed assignments to cell-state factors or the assignments are learned.
            warmup_epochs_vae (int): Initially train only the VAE part of the model for `warmup_epochs_vae` epochs.
            warmup_epochs_G (int): Initially learn only the persistent genetic effects for `warmup_epochs_G` epochs. Applied only after the VAE warm-up is finished.
            learning_rate (float): Learning rate.
            l1_weight (float): Weight of the L1 penalty, which is applied on the genetic decoders.
            batch_norm_decoder (bool): Whether to apply batch normalisation to the combined decoder output.
            adversarial_weight (float): If > 0, add adversarial loss to remove individual effects from the cell-state latent space.
            adversary_hidden_dims (List[int]): List of hidden dimensions for each adversary layer.
            adversary_learning_rate (float): Learning rate for adversary.
            adversary_steps (int): Number of steps to train adversary for every step of VAE.
            device (str): Accelerator to use for training.
        """
        super().__init__()

        self.save_hyperparameters()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_gxc_factors = n_gxc_factors
        self.n_persistent_factors = n_persistent_factors
        self.warmup_epochs_G = 0 if self.n_persistent_factors == 0 else warmup_epochs_G
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.pretrain_G_mode = True if self.pretrain_mode or self.warmup_epochs_G > 0 else False

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=True,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))

        self.adversary_learning_rate = adversary_learning_rate
        self.adversary_steps = adversary_steps
        # Set up adversary
        self.adversary = create_mlp(
            input_size=z_dim,
            output_size=y_dim,
            hidden_dims=adversary_hidden_dims,
            layer_norm=False,
            device=device,
        )

        if n_gxc_factors != 0:
            if hierarchical_model:
                # hierarchical model with fixed design matrix A
                self.U_context = nn.Embedding(y_dim, z_dim * n_gxc_factors, device=device)
            else:
                # learnable design matrix A
                self.bernoulli_logits = nn.Parameter(
                    torch.rand(z_dim, n_gxc_factors, device=device)
                )
                self.U_context = nn.Embedding(y_dim, n_gxc_factors, device=device)
        if n_persistent_factors != 0:
            self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)

        self.decoder = LIVI2_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            pretrain_G=self.pretrain_G_mode,
            batch_norm=batch_norm_decoder,
            hierarchical_decoder=hierarchical_model,
            device=device,
        )

        # Sex and batch correction per gene
        if donor_sex_dim is not None:
            self.sex_effect = nn.Embedding(donor_sex_dim, x_dim, device=device)
        else:
            self.sex_effect = None
        self.batch_effect = nn.Embedding(exbatch_dim, x_dim, device=device)

        self.set_pretrain_mode(self.pretrain_mode)
        self.set_pretrain_G_mode(self.pretrain_G_mode)

        self.automatic_optimization = False

    def set_pretrain_mode(self, mode: bool):
        """Set VAE pretrain mode.

        If True, discriminator and genetic embeddings are not optimised.
        """
        self.pretrain_mode = mode
        self.decoder.pretrain_VAE = mode
        if mode:
            # freeze parameters
            for p in self.adversary.parameters():
                p.requires_grad = False
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = False
                self.decoder.context_decoder[0].weight.requires_grad = False
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = False
                self.decoder.persistent_decoder[0].weight.requires_grad = False
            if not self.hparams.hierarchical_model:
                self.bernoulli_logits.requires_grad = False
        else:
            # unfreeze parameters
            for p in self.adversary.parameters():
                p.requires_grad = True
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = True
                self.decoder.persistent_decoder[0].weight.requires_grad = True

    def set_pretrain_G_mode(self, mode: bool):
        """Set persistent (global) genetic effects pretrain mode.

        If True, the context-specific genetic effects are not learned.
        """
        self.pretrain_G_mode = mode
        self.decoder.pretrain_G = mode
        if mode:
            # freeze parameters
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = False
                self.decoder.context_decoder[0].weight.requires_grad = False
                if not self.hparams.hierarchical_model:
                    self.bernoulli_logits.requires_grad = False
        else:
            # unfreeze parameters
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = True
                self.decoder.context_decoder[0].weight.requires_grad = True
                if not self.hparams.hierarchical_model:
                    self.bernoulli_logits.requires_grad = True

    def prepare_batch(self, batch):
        x = batch["x"]
        y = batch["y"]
        dsex = None if self.hparams.donor_sex_dim is None else batch["dsex"]
        eb = batch["eb"]
        size_factor = batch["size_factor"]
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

        Parameters
        ----------
            z_dist: Variational distribution.
            x: Input data.
            y: Donor IDs.
            dsex: Donor sex.
            eb: Experimental batch IDs.
            size_factor: Size factor to correct for gene count differences.

        Returns
        -------
            Mean evidence lower bound (ELBO).
        """
        z = z_dist.rsample()
        # z = nn.Softmax(dim=1)(z)
        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()

        if self.hparams.hierarchical_model and self.n_gxc_factors != 0:
            z_interaction = self.U_context(y) * z.repeat_interleave(self.n_gxc_factors, dim=1)
        elif not self.hparams.hierarchical_model and self.n_gxc_factors != 0:
            A = torch.distributions.RelaxedBernoulli(
                temperature=0.01, logits=self.bernoulli_logits
            ).rsample()
            # Straight-through trick
            A_hard = torch.where(A < 0.5, 0, 1).to(torch.float)
            A = A_hard.detach() - A.detach() + A
            z_interaction = (z @ A) * self.U_context(y)
        else:
            z_interaction = None

        log_lik = (
            self.decoder(
                z=z,
                GxC=z_interaction,
                persistent_G=self.V_persistent(y) if self.n_persistent_factors != 0 else None,
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
            if self.pretrain_G_mode or self.n_gxc_factors == 0:
                l1_loss_context = torch.zeros([1], device=self.device)
            else:
                l1_loss_context = self.hparams.l1_weight * torch.linalg.norm(
                    torch.cat([p for p in self.decoder.context_decoder.parameters()]), ord=1
                )
            if self.pretrain_mode or self.n_persistent_factors == 0:
                l1_loss_persistent = torch.zeros([1], device=self.device)
            else:
                l1_loss_persistent = self.hparams.l1_weight * torch.linalg.norm(
                    torch.cat([p for p in self.decoder.persistent_decoder.parameters()]), ord=1
                )
            logs[f"{mode}/L1_penalty_context"] = l1_loss_context.item()
            logs[f"{mode}/L1_penalty_persistent"] = l1_loss_persistent.item()
            loss = (
                -elbo
                - self.hparams.adversary_weight * adversarial_loss
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

    def predict(self, x, y, exp_batch_ids):
        """Model inference. Get latent space and individual embeddings for the input data.

        Parameters
        ----------
            x (torch.Tensor): Input gene expression vector per cell.
            y (torch.Tensor): ID of the individual the cell is derived from.
            exp_batch_ids (torch.Tensor): ID of the experimental the cell belongs to.

        Returns
        -------
        Inference results (Dict[str,torch.Tensor])
            'cell-state_latent' (torch.Tensor): Cell state latent space.
            'base_decoder' (torch.Tensor): Gene loadings for the cell-state decoder.
            'batch_embedding' (torch.Tensor): Learned embedding of technical batch.
            'context_effects' (torch.Tensor): Learned embedding of context-specific individual effects, if applicable.
            'context_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
            'Bernoulli_logits' (torch.Tensor): Learned Bernoulli logits for the assignment of context-specific factors to cell-state factors, if not a fixed hierarchical model.
            'persistent_effects' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
            'persistent_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
        """

        with torch.no_grad():
            self.eval()

            z = self(x).rsample()
            cell_state_decoder = self.decoder.mean[0].weight
            batch_embedding = (self.batch_effect(exp_batch_ids),)
            if self.n_gxc_factors != 0:
                U = self.U_context(y)
                context_decoder = self.decoder.context_decoder[0].weight
            else:
                U = None
                context_decoder = None
            if self.n_persistent_factors != 0:
                V = self.V_persistent(y)
                persistent_decoder = self.decoder.persistent_decoder[0].weight
            else:
                V = None
                persistent_decoder = None

        return {
            "cell-state_latent": z,
            "cell-state_decoder": cell_state_decoder,
            "batch_embedding": batch_embedding,
            "CxG_effects": U,
            "CxG_decoder": context_decoder,
            "Bernoulli_logits": (
                self.bernoulli_logits if not self.hparams.hierarchical_model else None
            ),
            "persistent_effects": V,
            "persistent_decoder": persistent_decoder,
        }

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for VAE and persistent G
        has been reached."""

        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            print("VAE pretraining completed")

        if self.current_epoch == self.hparams.warmup_epochs_vae + self.warmup_epochs_G:
            print("Start learning GxC effects")
            self.set_pretrain_G_mode(False)

    def configure_optimizers(self):
        """Configures optimizer."""

        params = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
            {"params": self.batch_effect.parameters()},
            {"params": self.sex_effect.parameters()},
        ]
        if self.n_gxc_factors != 0:
            params.append({"params": self.U_context.parameters()})
        if self.n_persistent_factors != 0:
            params.append({"params": self.V_persistent.parameters()})
        if not self.hparams.hierarchical_model:
            params.append({"params": self.bernoulli_logits})

        optim_vae = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.hparams.adversary_learning_rate,
        )

        return optim_vae, optim_adversary


class LIVI2_freeze(LIVI2_experimental):
    """LIVI model where additionally the VAE and adversary parameters are frozen after pre-training is completed.
    Then only the individual embeddings are learned.
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
        warmup_epochs_vae: int = 30,
        warmup_epochs_G: int = 10,
        train_epochs_adversary: int = 20,
        hierarchical_model: bool = True,
        donor_sex_dim: int = 2,
        l1_weight: float = 0.001,
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
    ):
        """Initializes LIVI.

        Parameters
        ----------
            x_dim (int): Dimensionality of input data.
            z_dim (int): Dimensionality of cell-state latent space.
            exbatch_dim (int): Number of experimental batches in the dataset.
            donor_sex_dim (int): Number of potential donor sexes in the dataset.
            y_dim (int): Number of individuals in the dataset.
            encoder_hidden_dims (List[int]): List of hidden dimensions for each encoder layer.
            hierarchical_model (bool): Whether the context-specific genetic factors have fixed assignments to
                cell-state factors (`True`) or the assignments are learned (`False`).
            warmup_epochs_vae (int): Initially train only the VAE part of the model for `warmup_epochs_vae` number of epochs.
            warmup_epochs_G (int): Initially learn only the persistent genetic effects for `warmup_epochs_G` number of epochs.
                Applied only after the VAE warm-up is finished.
            train_epochs_adversary (int): Train adversasial network together with the VAE for `train_epochs_adversary`
                number of epochs before freezing their parameters.
            learning_rate (float): Learning rate.
            l1_weight (float): Weight of the L1 penalty, which is applied on the genetic decoders.
            batch_norm_decoder (bool): Whether to apply batch normalisation to the combined decoder output.
            adversarial_weight (float): If > 0, add adversarial loss to remove individual effects from the cell-state latent space.
            adversary_hidden_dims (List[int]): List of hidden dimensions for each adversary layer.
            adversary_learning_rate (float): Learning rate for adversary.
            adversary_steps (int): Number of steps to train adversary for every step of VAE.
            device (str): Accelerator to use for training.
        """
        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            exbatch_dim=exbatch_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            warmup_epochs_vae=warmup_epochs_vae,
            warmup_epochs_G=warmup_epochs_G,
            hierarchical_model=hierarchical_model,
            donor_sex_dim=donor_sex_dim,
            l1_weight=l1_weight,
            adversary_weight=adversary_weight,
            adversary_hidden_dims=adversary_hidden_dims,
            adversary_learning_rate=adversary_learning_rate,
            adversary_steps=adversary_steps,
            batch_norm_decoder=batch_norm_decoder,
            device=device,
        )

        self.frozen = False
        self.save_hyperparameters()

        self.automatic_optimization = False

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, donor_sex, exp_batch_ids, size_factor = self.prepare_batch(batch)
        optim_vae, optim_adversary = self.optimizers()

        train_adversary = batch_idx % self.adversary_steps == 0
        train_adversary = train_adversary & (not self.pretrain_mode) & (not self.frozen)
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
            if self.pretrain_G_mode or self.n_gxc_factors == 0:
                l1_loss_context = torch.zeros([1], device=self.device)
            else:
                l1_loss_context = self.hparams.l1_weight * torch.linalg.norm(
                    torch.cat([p for p in self.decoder.context_decoder.parameters()]), ord=1
                )
            if self.pretrain_mode or self.n_persistent_factors == 0:
                l1_loss_persistent = torch.zeros([1], device=self.device)
            else:
                l1_loss_persistent = self.hparams.l1_weight * torch.linalg.norm(
                    torch.cat([p for p in self.decoder.persistent_decoder.parameters()]), ord=1
                )
            logs[f"{mode}/L1_penalty_context"] = l1_loss_context.item()
            logs[f"{mode}/L1_penalty_persistent"] = l1_loss_persistent.item()
            loss = (
                -elbo
                - self.hparams.adversary_weight * adversarial_loss
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

    def set_pretrain_mode(self, mode: bool):
        """Set VAE pretrain mode.

        If True, discriminator and genetic embeddings are not optimised.
        """
        self.pretrain_mode = mode
        self.decoder.pretrain_VAE = mode
        if mode:
            # freeze parameters
            for p in self.adversary.parameters():
                p.requires_grad = False
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = False
                self.decoder.context_decoder[0].weight.requires_grad = False
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = False
                self.decoder.persistent_decoder[0].weight.requires_grad = False
            if not self.hparams.hierarchical_model:
                self.bernoulli_logits.requires_grad = False
        else:
            # unfreeze adversary
            for p in self.adversary.parameters():
                p.requires_grad = True

    def freeze_vae(self, mode: bool):
        """Freezes VAE, discriminator and covariate embeddings parameters, after the number of VAE
        and discriminator warm-up epochs has been completed."""

        self.frozen = mode
        if mode:
            # freeze VAE etc.
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.decoder.mean[0].weight.requires_grad = False
            self.decoder.log_total_count.requires_grad = False
            # freeze adversary
            for p in self.adversary.parameters():
                p.requires_grad = False
            # freeze covariate embeddings
            for p in self.batch_effect.parameters():
                p.requires_grad = False
            for p in self.sex_effect.parameters():
                p.requires_grad = False
            # unfreeze persistent genetic embedding, if applicable
            # Note: the behavior of cell-state-specific genetic embedding is dictated
            # by the `set_pretrain_G_mode` function
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = True
                self.decoder.persistent_decoder[0].weight.requires_grad = True
        else:
            # train VAE
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.decoder.mean[0].weight.requires_grad = True
            self.decoder.log_total_count.requires_grad = True
            # train adversary
            for p in self.adversary.parameters():
                p.requires_grad = True
            # train covariate embeddings
            for p in self.batch_effect.parameters():
                p.requires_grad = True
            for p in self.sex_effect.parameters():
                p.requires_grad = True
            # freeze genetic embeddings
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = False
                self.decoder.context_decoder[0].weight.requires_grad = False
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = False
                self.decoder.persistent_decoder[0].weight.requires_grad = True

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE, discriminator
        and persistent G has been reached."""
        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            print("VAE pretraining completed.")

        if (
            self.current_epoch
            == self.hparams.warmup_epochs_vae + self.hparams.train_epochs_adversary
        ):
            self.freeze_vae(True)
            print("VAE and Adversary training completed.")

        if (
            self.current_epoch
            == self.hparams.warmup_epochs_vae
            + self.hparams.train_epochs_adversary
            + self.warmup_epochs_G
        ):
            print("Start learning GxC effects.")
            self.set_pretrain_G_mode(False)
