from typing import List, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.livi_decoder import (
    LIVI_Decoder,
    LIVI_Decoder_GT_PCs,
    LIVI_Decoder_Normal,
)
from src.models.components.mlp import create_mlp, init_mlp
from src.models.livi import LIVI
from src.models.vae import Encoder, NegativeBinomialDecoderCovars


class LIVIcis_same_decoder(pl.LightningModule):
    """LIVI with flexibility to add only context-specific or context-specific and persistent
    genetic effect.

    Cell state and genetic latent factors are decoded together.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_cis_snps: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 30,
        covariates_dims: Optional[List[int]] = None,
        A_weight: float = 0.001,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        initialise_training_mode: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_DxC_factors = n_DxC_factors
        # Enable checkpointing after VAE + Dis training is completed and 5 epochs after U,V,A training has started
        self.checkpointing_epoch = warmup_epochs_vae + 5
        self.frozen = False

        self.save_hyperparameters()

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=True,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))

        self.decoder = NegativeBinomialDecoderCovars(
            z_dim=n_DxC_factors if n_DxC_factors > 0 else z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            batch_norm=batch_norm_decoder,
            device=device,
        )

        self.A = nn.Parameter(torch.randn(z_dim, n_DxC_factors, device=device))
        self.D_context = nn.Embedding(y_dim, n_DxC_factors, device=device)

        if n_cis_snps != 0:
            self.SNP_gene_effect = nn.Parameter(
                torch.randn(z_dim, n_cis_snps, x_dim, device=device)
            )  # Learn SNP-gene effect for each cell-state

        # Covariate (e.g. experimental batch) correction per gene
        if self.hparams.covariates_dims is not None:
            self.covariate_effect = nn.Embedding(
                sum(self.hparams.covariates_dims), x_dim, device=device
            )
        else:
            self.covariate_effect = None

        self.automatic_optimization = False

    def prepare_batch(self, batch):
        x = batch["x"]
        y = batch["y"]
        if self.hparams.covariates_dims is not None:
            assert len(self.hparams.covariates_dims) == len(
                batch["covariates"]
            ), "Number of covariates different than the number of covariates in data module."
            covariates = batch["covariates"]
        else:
            covariates = None
        size_factor = batch["size_factor"]
        known_cis_associations = None if self.hparams.n_cis_snps == 0 else batch["known_cis"]
        cell_gt = None if self.hparams.n_cis_snps == 0 else batch["GT_cells"]

        return x, y, covariates, size_factor, known_cis_associations, cell_gt

    def transform_latent(self, z: torch.Tensor, y: torch.Tensor):
        if self.n_DxC_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.D_context(y)
            z_combined = z_interaction
        else:
            z_interaction = None
            A = None
            z_combined = z

        return z_combined, A

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
        covariates: Optional[List[torch.Tensor]] = None,
        size_factor: Optional[torch.Tensor] = None,
        snp_gene_mask: Optional[torch.Tensor] = None,
        cell_snp_mask: Optional[torch.Tensor] = None,
    ):
        """Computes evidence lower bound (ELBO).

        Parameters
        ----------
            z_dist: Variational distribution.
            x: Input data.
            y: Donor IDs.
            covariates: Cell/donor covariates (e.g. technical batch ID or sex).
            size_factor: Size factor to correct for gene count differences.

        Returns
        -------
            Mean evidence lower bound (ELBO).
        """
        z = z_dist.rsample()
        z = nn.Softmax(dim=1)(z)

        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()
        z_combined, A = self.transform_latent(z=z, y=y)

        if (
            snp_gene_mask is not None
            and cell_snp_mask is not None
            and self.hparams.n_cis_snps != 0
        ):
            known_cis_effect = (
                snp_gene_mask.resize(1, self.hparams.n_cis_snps, self.x_dim) * self.SNP_gene_effect
            )  # z x SNPs x genes
            celltype_known_cis_effect = torch.einsum(
                "ij,kjl->kil", cell_snp_mask, known_cis_effect
            )  # z x cells x genes (i cell, j SNP, k dim, l gene)
            celltype_known_cis_effect = torch.einsum(
                "ik,kil->il", z, celltype_known_cis_effect
            )  # cells x genes
        else:
            celltype_known_cis_effect = None

        if covariates is not None:
            covariate_effect = torch.zeros_like(x)
            for covar in range(len(covariates)):
                covar_indices = covariates[covar]
                # increase the indices by the number of categories of the previous covariate(s)
                embedding_indices = covar_indices + sum(self.hparams.covariates_dims[:covar])
                covariate_effect += self.covariate_effect(embedding_indices)
            if celltype_known_cis_effect is not None:
                covariate_effect = covariate_effect + celltype_known_cis_effect
        else:
            covariate_effect = None

        log_lik = (
            self.decoder(
                z=z_combined,
                size_factor=size_factor,
                covariate_effect=covariate_effect,
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div, A

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask = self.prepare_batch(batch)
        optim_vae = self.optimizers()

        z_dist = self(x)

        elbo, A = self.compute_elbo(
            z_dist, x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask
        )
        logs = {f"{mode}/elbo": elbo.item()}
        if self.n_DxC_factors == 0 or self.hparams.A_weight == 0.0:
            loss_A = torch.zeros([1], device=self.device)
        else:
            A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
            loss_A = self.hparams.A_weight * A_penalty.sum()

        logs[f"{mode}/penalty_A"] = loss_A.item()
        loss = -elbo + loss_A
        loss = -elbo
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

    def predict(self, x, y):
        """Model inference. Get latent space and individual embeddings for the input data.

        Parameters
        ----------
            x (torch.Tensor): Input gene expression vector per cell.
            y (torch.Tensor): ID of the individual the cell is derived from.

        Returns
        -------
        Inference results (Dict[str,torch.Tensor])
            'cell-state_latent' (torch.Tensor): Cell state latent space.
            'base_decoder' (torch.Tensor): Gene loadings for the cell-state decoder.
            'D_embedding' (torch.Tensor): Learned embedding of context-specific individual effects, if applicable.
            'DxC_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
            'assignment_matrix' (torch.Tensor): Learned assignment matrix of U factors to cell-state factors.
            'V_embedding' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
            'V_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
        """

        with torch.no_grad():
            self.eval()

            z = self(x).rsample()
            z_softmax = nn.Softmax(dim=1)(z)
            z_combined, _ = self.transform_latent(z=z_softmax, y=y)
            decoder = self.decoder.mean[0].weight
            if self.n_DxC_factors != 0:
                D = self.D_context(y)
            else:
                D = None

        return {
            "cell-state_latent": z,
            "decoder": decoder,
            "D_embedding": D,
            "assignment_matrix": self.A,
            "cis_SNP_effect": self.SNP_gene_effect if self.hparams.n_cis_snps != 0 else None,
        }

    def on_save_checkpoint(self, checkpoint: dict):
        # Save model's pretraining and frozen state
        attributes_to_save = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k not in list(self.__dict__["_hparams"].keys())
            and "G_gen" not in k
        }
        checkpoint["model_attributes"] = attributes_to_save

    def on_load_checkpoint(self, checkpoint: dict):
        # Restore model's attributes
        model_attributes = checkpoint.get("model_attributes", {})
        for attr_name, attr_value in model_attributes.items():
            setattr(self, attr_name, attr_value)

    def configure_optimizers(self):
        """Configures optimizer."""

        params = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]
        if self.covariate_effect is not None:
            params.append({"params": self.covariate_effect.parameters()})
        if self.n_DxC_factors != 0:
            params.append({"params": self.D_context.parameters()})
            params.append({"params": self.A})
        if self.hparams.n_cis_snps != 0:
            params.append({"params": self.SNP_gene_effect})

        optim_vae = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
        )

        return optim_vae


class LIVI_cis_with_adversary(pl.LightningModule):
    """LIVI model accounting for cis genetic effects."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_persistent_factors: int,
        n_cis_snps: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        cell_state_cis: bool = True,
        warmup_epochs_vae: int = 60,
        warmup_epochs_G: int = 0,
        train_epochs_adversary: int = 30,
        covariates_dims: Optional[List[int]] = None,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        adversary_weight: float = 10,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
        initialise_training_mode: bool = True,
    ):

        super().__init__()

        self.save_hyperparameters()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_DxC_factors = n_DxC_factors
        self.n_persistent_factors = n_persistent_factors
        self.warmup_epochs_G = 0 if self.n_persistent_factors == 0 else warmup_epochs_G

        self.train_epochs_adversary = train_epochs_adversary if adversary_weight > 0 else 0
        self.adversary_steps = adversary_steps if adversary_weight > 0 else 0
        self.adversary_learning_rate = adversary_learning_rate

        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_DxC_mode = False if self.pretrain_mode or self.n_DxC_factors == 0 else True
        self.checkpointing_epoch = (
            warmup_epochs_vae + self.warmup_epochs_G + self.train_epochs_adversary + 5
        )

        self.frozen = False
        self.frozen_dis = False

        if self.hparams.genetics_seed is not None:
            self.G_gen = torch.Generator(device=device)
            self.G_gen.manual_seed(genetics_seed)
        else:
            self.G_gen = None

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=True,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))

        self.decoder = LIVI_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_DxC=self.train_DxC_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_generator=self.G_gen,
        )

        self.A = nn.Parameter(
            torch.randn(z_dim, n_DxC_factors, device=device, generator=self.G_gen)
        )
        self.D_context = nn.Embedding(y_dim, n_DxC_factors, device=device)
        # if self.hparams.genetics_seed is not None:
        #     self.init_individual_embedding(self.D_context, self.hparams.genetics_seed)
        nn.init.normal_(self.D_context.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

        if n_persistent_factors != 0:
            self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)
            # if self.hparams.genetics_seed is not None:
            #     self.init_individual_embedding(self.V_persistent, self.hparams.genetics_seed)

        if n_cis_snps != 0:
            if cell_state_cis:
                self.SNP_gene_effect = nn.Parameter(
                    torch.randn(z_dim, n_cis_snps, x_dim, device=device, generator=self.G_gen)
                )  # Learn SNP-gene effect for each cell-state
            else:
                self.SNP_gene_effect = nn.Parameter(
                    torch.randn(n_cis_snps, x_dim, device=device, generator=self.G_gen)
                )

        # Covariate (e.g. experimental batch) correction per gene
        if self.hparams.covariates_dims is not None:
            self.covariate_effect = nn.Embedding(
                sum(self.hparams.covariates_dims), x_dim, device=device
            )
        else:
            self.covariate_effect = None

        self.adversary = create_mlp(
            input_size=z_dim,
            output_size=y_dim,
            hidden_dims=adversary_hidden_dims,
            layer_norm=False,
            device=device,
        )

        self.automatic_optimization = False

        # self.set_pretrain_mode(self.pretrain_mode)
        # self.set_train_V_mode(self.train_V_mode)
        # self.set_train_DxC_mode(self.train_DxC_mode)
        if initialise_training_mode:
            self.initialise_model()

    def initialise_model(self):
        self.set_pretrain_mode(self.pretrain_mode)
        self.set_train_V_mode(self.train_V_mode)
        self.set_train_DxC_mode(self.train_DxC_mode)

    # def init_individual_embedding(self, embedding, seed):
    #     # Save the current random state
    #     current_state = torch.get_rng_state()
    #     torch.manual_seed(seed)
    #     # Initialize the embedding using the specified random seed
    #     embedding.reset_parameters()
    #     # Restore the original random state
    #     torch.set_rng_state(current_state)

    def prepare_batch(self, batch):
        x = batch["x"]
        y = batch["y"]
        if self.hparams.covariates_dims is not None:
            assert len(self.hparams.covariates_dims) == len(
                batch["covariates"]
            ), "Number of covariates different than the number of covariates in data module."
            covariates = batch["covariates"]
        else:
            covariates = None
        size_factor = batch["size_factor"]
        known_cis_associations = None if self.hparams.n_cis_snps == 0 else batch["known_cis"]
        cell_gt = None if self.hparams.n_cis_snps == 0 else batch["GT_cells"]

        return x, y, covariates, size_factor, known_cis_associations, cell_gt

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
        covariates: Optional[List[torch.Tensor]] = None,
        size_factor: Optional[torch.Tensor] = None,
        snp_gene_mask: Optional[torch.Tensor] = None,
        cell_snp_mask: Optional[torch.Tensor] = None,
    ):
        """Computes evidence lower bound (ELBO).

        Parameters
        ----------
            z_dist: Variational distribution.
            x: Input data.
            y: Donor IDs.
            covariates: Cell/donor covariates (e.g. technical batch ID or sex).
            size_factor: Size factor to correct for gene count differences.

        Returns
        -------
            Mean evidence lower bound (ELBO).
        """
        z = z_dist.rsample()
        z = nn.Softmax(dim=1)(z)

        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()

        if self.n_DxC_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.D_context(y)
        else:
            z_interaction = None
            A = None

        if covariates is not None:
            covariate_effect = torch.zeros_like(x)
            for covar in range(len(covariates)):
                covar_indices = covariates[covar]
                # increase the indices by the number of categories of the previous covariate(s)
                embedding_indices = covar_indices + sum(self.hparams.covariates_dims[:covar])
                covariate_effect += self.covariate_effect(embedding_indices)
        else:
            covariate_effect = None

        if (
            snp_gene_mask is not None
            and cell_snp_mask is not None
            and self.hparams.n_cis_snps != 0
        ):
            if self.hparams.cell_state_cis:
                known_cis_effect = (
                    snp_gene_mask.resize(1, self.hparams.n_cis_snps, self.x_dim)
                    * self.SNP_gene_effect
                )  # z x SNPs x genes
                celltype_known_cis_effect = torch.einsum(
                    "ij,kjl->kil", cell_snp_mask, known_cis_effect
                )  # z x cells x genes
                celltype_known_cis_effect = torch.einsum(
                    "ik,kil->il", z, celltype_known_cis_effect
                )  # cells x genes
            else:
                known_cis_effect = snp_gene_mask * self.SNP_gene_effect  # SNPs x genes
                known_cis_effect = (
                    cell_snp_mask @ known_cis_effect
                )  # cells x genes: mean cis-SNP effect on each gene
                # Make the cis effect cell-state-specific by multiplying with the reconstructed cell-state GEX
                # (=> genes essentially define the cell-state; if a gene is less relevant for the given cells
                # (i.e. closer to 0), then the cis effect will become close to zero as well)
                y_c = F.softmax(self.decoder.mean(z), dim=-1)
                celltype_known_cis_effect = (
                    known_cis_effect * y_c
                )  # celltype_known_cis_effect = known_cis_effect
        else:
            celltype_known_cis_effect = None

        log_lik = (
            self.decoder(
                z=z,
                DxC=z_interaction,
                persistent_G=self.V_persistent(y) if self.n_persistent_factors != 0 else None,
                size_factor=size_factor,
                covariate_effect=covariate_effect,
                known_cis_effect=celltype_known_cis_effect,
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div, A

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask = self.prepare_batch(batch)
        optim_vae, optim_adversary = self.optimizers()

        if self.adversary_steps > 0:
            train_adversary = batch_idx % self.adversary_steps == 0
            train_adversary = train_adversary & (not self.pretrain_mode) & (not self.frozen_dis)
            train_adversary = train_adversary * (mode == "train")
        else:
            train_adversary = False

        z_dist = self(x)

        if self.pretrain_mode or self.frozen_dis or not train_adversary:
            # no adversary signal
            adversarial_loss = torch.zeros([1], device=self.device)
        else:
            z = z_dist.rsample()
            # z = nn.Softmax(dim=1)(z)
            adversarial_loss = F.cross_entropy(self.adversary(z), y)
        logs = {f"{mode}/adversarial_loss": adversarial_loss.item()}

        if train_adversary:
            # train adversary
            optim_adversary.zero_grad()
            self.manual_backward(adversarial_loss)
            optim_adversary.step()
        else:
            # train vae
            elbo, A = self.compute_elbo(
                z_dist, x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask
            )
            logs[f"{mode}/elbo"] = elbo.item()
            if not self.train_DxC_mode or self.n_DxC_factors == 0:
                l1_loss_DxC = torch.zeros([1], device=self.device)
                loss_A = torch.zeros([1], device=self.device)
            else:
                l1_loss_DxC = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.DxC_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
                A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
                loss_A = self.hparams.A_weight * A_penalty.sum()

            logs[f"{mode}/L1_penalty_DxC"] = l1_loss_DxC.item()
            logs[f"{mode}/penalty_A"] = loss_A.item()
            loss = -elbo - self.hparams.adversary_weight * adversarial_loss + l1_loss_DxC + loss_A
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

    def predict(self, x, y):
        """Model inference. Get latent space and individual embeddings for the input data.

        Parameters
        ----------
            x (torch.Tensor): Input gene expression vector per cell.
            y (torch.Tensor): ID of the individual the cell is derived from.

        Returns
        -------
        Inference results (Dict[str,torch.Tensor])
            'cell-state_latent' (torch.Tensor): Cell state latent space.
            'base_decoder' (torch.Tensor): Gene loadings for the cell-state decoder.
            'D_embedding' (torch.Tensor): Learned embedding of context-specific individual effects, if applicable.
            'DxC_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
            'assignment_matrix' (torch.Tensor): Learned assignment matrix of U factors to cell-state factors.
            'V_embedding' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
            'V_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
        """

        with torch.no_grad():
            self.eval()

            z = self(x).rsample()
            cell_state_decoder = self.decoder.mean[0].weight
            if self.n_DxC_factors != 0:
                D = self.D_context(y)
                DxC_decoder = self.decoder.DxC_decoder[0].weight
            else:
                D = None
                DxC_decoder = None
            if self.n_persistent_factors != 0:
                V = self.V_persistent(y)
                V_decoder = self.decoder.persistent_decoder[0].weight
            else:
                V = None
                V_decoder = None

        return {
            "cell-state_latent": z,
            "cell-state_decoder": cell_state_decoder,
            "D_embedding": D,
            "DxC_decoder": DxC_decoder,
            "assignment_matrix": self.A,
            "V_embedding": V,
            "V_decoder": V_decoder,
            "cis_SNP_effect": self.SNP_gene_effect if self.hparams.n_cis_snps != 0 else None,
        }

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
            self.set_train_DxC_mode(False)
            self.set_train_V_mode(False)
        else:
            # unfreeze adversary
            for p in self.adversary.parameters():
                p.requires_grad = True

    def set_train_V_mode(self, mode: bool):
        self.train_V_mode = mode
        self.decoder.train_V = mode
        if mode:
            # Train V
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = True
                self.decoder.persistent_decoder[0].weight.requires_grad = True
        else:
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = False
                self.decoder.persistent_decoder[0].weight.requires_grad = False

    def set_train_DxC_mode(self, mode: bool):
        self.train_DxC_mode = mode
        self.decoder.train_DxC = mode
        if mode:
            if self.n_DxC_factors != 0:
                for p in self.D_context.parameters():
                    p.requires_grad = True
                self.decoder.DxC_decoder[0].weight.requires_grad = True
                self.A.requires_grad_(True)
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad_(True)
        else:
            if self.n_DxC_factors != 0:
                for p in self.D_context.parameters():
                    p.requires_grad = False
                self.decoder.DxC_decoder[0].weight.requires_grad = False
                self.A.requires_grad_(False)
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad_(False)

    def freeze_vae(self, mode: bool):
        """Freezes VAE and covariate embeddings parameters, after the number of VAE and
        discriminator warm-up epochs has been completed."""

        self.frozen = mode
        if mode:
            # freeze VAE etc.
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.decoder.mean[0].weight.requires_grad = False
            self.decoder.log_total_count.requires_grad = (
                False  # set to true to retrain total_count
            )
            # freeze covariate embeddings, if applicable
            if self.covariate_effect is not None:
                for p in self.covariate_effect.parameters():
                    p.requires_grad = False
        else:
            # train VAE
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.decoder.mean[0].weight.requires_grad = True
            self.decoder.log_total_count.requires_grad = True
            # train covariate embeddings, if applicable
            if self.covariate_effect is not None:
                for p in self.covariate_effect.parameters():
                    p.requires_grad = True
            # freeze genetic embeddings
            self.set_train_DxC_mode(False)
            self.set_train_V_mode(False)

    def freeze_adversary(self, mode: bool):
        """Freezes discriminator parameters, after the number of VAE and discriminator warm-up
        epochs has been completed."""
        self.frozen_dis = mode
        # freeze adversary
        if mode:
            for p in self.adversary.parameters():
                p.requires_grad = False
        else:
            # train adversary
            for p in self.adversary.parameters():
                p.requires_grad = True
            # freeze genetic embeddings
            self.set_train_DxC_mode(False)
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = False

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE, discriminator
        and persistent G has been reached."""
        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            #  self.set_train_V_mode(True) # train V with Dis
            print("VAE pretraining completed.")
            print("Start training Adversary.")

        if self.current_epoch == self.hparams.warmup_epochs_vae + self.train_epochs_adversary:
            self.freeze_vae(True)  # Freeze VAE when starting training V
            print("Freeze VAE parameters.")
            self.set_train_V_mode(True)
            print("Start training V.")

        if (
            self.current_epoch
            == self.hparams.warmup_epochs_vae + self.train_epochs_adversary + self.warmup_epochs_G
        ):
            # self.set_train_V_mode(False) # Freeze V
            print("Pretraining completed.")
            self.freeze_adversary(True)
            print("Start learning DxC effects.")
            self.set_train_DxC_mode(True)

    def on_save_checkpoint(self, checkpoint: dict):
        # Save model's pretraining and frozen state
        attributes_to_save = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k not in list(self.__dict__["_hparams"].keys())
            and "G_gen" not in k
        }
        checkpoint["model_attributes"] = attributes_to_save

    def on_load_checkpoint(self, checkpoint: dict):
        # Restore model's attributes
        model_attributes = checkpoint.get("model_attributes", {})
        for attr_name, attr_value in model_attributes.items():
            setattr(self, attr_name, attr_value)

    def configure_optimizers(self):
        """Configures optimizer."""

        params = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]
        if self.covariate_effect is not None:
            params.append({"params": self.covariate_effect.parameters()})
        if self.n_DxC_factors != 0:
            params.append({"params": self.D_context.parameters()})
            params.append({"params": self.A})
        if self.n_persistent_factors != 0:
            params.append({"params": self.V_persistent.parameters()})
        if self.hparams.n_cis_snps != 0:
            params.append({"params": self.SNP_gene_effect})

        optim_vae = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.hparams.adversary_learning_rate,
        )

        return optim_vae, optim_adversary


class LIVI_WO_freezing(LIVI_cis_with_adversary):
    """Modifies the `LIVI_cis_with_adversary` class to learn genetic (donor) effects without
    freezing the VAE and Discriminator modules."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_persistent_factors: int,
        exbatch_dim: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 60,
        train_epochs_adversary: int = 30,
        donor_sex_dim: int = 2,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
    ):

        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            n_cis_snps=0,
            exbatch_dim=exbatch_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            warmup_epochs_vae=warmup_epochs_vae,
            warmup_epochs_G=0,
            train_epochs_adversary=train_epochs_adversary,
            donor_sex_dim=donor_sex_dim,
            l1_weight=l1_weight,
            A_weight=l1_weight,
            adversary_weight=adversary_weight,
            adversary_hidden_dims=adversary_hidden_dims,
            adversary_learning_rate=adversary_learning_rate,
            adversary_steps=adversary_steps,
            batch_norm_decoder=batch_norm_decoder,
            device=device,
            genetics_seed=genetics_seed,
        )

        self.checkpointing_epoch = warmup_epochs_vae + self.train_epochs_adversary
        self.frozen = False

    def pretrain_Dis(self, mode: bool):
        """Pretrain VAE and discriminator WO the individual embeddings."""
        if mode:
            # freeze parameters
            for p in self.adversary.parameters():
                p.requires_grad = True
            if self.n_DxC_factors != 0:
                for p in self.D_context.parameters():
                    p.requires_grad = False
                self.decoder.CxG_decoder[0].weight.requires_grad = False
                self.A.requires_grad = False
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = False
                self.decoder.persistent_decoder[0].weight.requires_grad = False
        else:
            # unfreeze individual embeddings
            if self.n_DxC_factors != 0:
                for p in self.D_context.parameters():
                    p.requires_grad = True
                self.decoder.CxG_decoder[0].weight.requires_grad = True
                self.A.requires_grad = True
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = True
                self.decoder.persistent_decoder[0].weight.requires_grad = True

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
            # z = nn.Softmax(dim=1)(z)
            adversarial_loss = F.cross_entropy(self.adversary(z), y)
        logs = {f"{mode}/adversarial_loss": adversarial_loss.item()}

        if train_adversary:
            # train adversary
            optim_adversary.zero_grad()
            self.manual_backward(adversarial_loss)
            optim_adversary.step()
        else:
            # train vae
            elbo, A = self.compute_elbo(z_dist, x, y, donor_sex, exp_batch_ids, size_factor)
            logs[f"{mode}/elbo"] = elbo.item()
            if self.n_DxC_factors == 0:
                l1_loss_DxC = torch.zeros([1], device=self.device)
                l1_loss_A = torch.zeros([1], device=self.device)
            else:
                l1_loss_DxC = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.CxG_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
                A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
                l1_loss_A = self.hparams.A_weight * A_penalty.sum()
            if self.n_persistent_factors == 0:
                l1_loss_persistent = torch.zeros([1], device=self.device)
            else:
                l1_loss_persistent = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.persistent_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
            logs[f"{mode}/L1_penalty_DxC"] = l1_loss_DxC.item()
            logs[f"{mode}/penalty_A"] = l1_loss_A.item()
            logs[f"{mode}/L1_penalty_persistent"] = l1_loss_persistent.item()
            loss = (
                -elbo
                - self.hparams.adversary_weight * adversarial_loss
                + l1_loss_DxC
                + l1_loss_A
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

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE, discriminator
        and persistent G has been reached."""
        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            print("VAE pretraining completed.")

        if self.current_epoch == self.hparams.warmup_epochs_vae + self.train_epochs_adversary:
            self.pretrain_Dis(False)
            print("VAE and Adversary pretraining completed. Start learning individual effects.")


class LIVI_train_Dis_with_UV_WO_freezing(LIVI_WO_freezing):
    """Modifies the `LIVI_WO_freezing` class to remove warm-up training for the Discriminator, i.e.
    the Discriminator is trained simultaneously with the individual embeddings, without pretraining
    only with the VAE."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_persistent_factors: int,
        exbatch_dim: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 60,
        donor_sex_dim: int = 2,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
    ):

        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            exbatch_dim=exbatch_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            warmup_epochs_vae=warmup_epochs_vae,
            train_epochs_adversary=0,
            donor_sex_dim=donor_sex_dim,
            l1_weight=l1_weight,
            A_weight=A_weight,
            adversary_weight=adversary_weight,
            adversary_hidden_dims=adversary_hidden_dims,
            adversary_learning_rate=adversary_learning_rate,
            adversary_steps=adversary_steps,
            batch_norm_decoder=batch_norm_decoder,
            device=device,
            genetics_seed=genetics_seed,
        )

        self.checkpointing_epoch = warmup_epochs_vae + 1
        self.frozen = False

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE has been
        reached."""
        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            self.pretrain_Dis(False)
            print(
                "VAE pretraining completed.\nStart training adversary and individual embeddings."
            )


class LIVI_cis_efficient(pl.LightningModule):
    """LIVI model accounting for cell-state-specific cis genetic effects in a memory efficint
    manner."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_persistent_factors: int,
        n_cis_eqtls: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 60,
        warmup_epochs_G: int = 0,
        covariates_dims: Optional[List[int]] = None,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
        initialise_training_mode: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters()

        if self.hparams.genetics_seed is not None:
            self.G_gen = torch.Generator(device=device)
            self.G_gen.manual_seed(genetics_seed)
        else:
            self.G_gen = None

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_DxC_factors = n_DxC_factors
        self.n_persistent_factors = n_persistent_factors
        self.warmup_epochs_G = 0 if self.n_persistent_factors == 0 else warmup_epochs_G
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_DxC_mode = False if self.pretrain_mode or self.n_DxC_factors == 0 else True
        self.checkpointing_epoch = warmup_epochs_vae + self.warmup_epochs_G + 5
        self.frozen = False

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=True,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))

        self.decoder = LIVI_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_DxC=self.train_DxC_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_generator=self.G_gen,
        )

        self.A = nn.Parameter(
            torch.randn(z_dim, n_DxC_factors, device=device, generator=self.G_gen)
        )
        self.D_context = nn.Embedding(y_dim, n_DxC_factors, device=device)
        nn.init.normal_(self.D_context.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

        if n_persistent_factors != 0:
            self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)

        # Covariate (e.g. experimental batch) correction per gene
        if self.hparams.covariates_dims is not None:
            self.covariate_effect = nn.Embedding(
                sum(self.hparams.covariates_dims), x_dim, device=device
            )
        else:
            self.covariate_effect = None

        if n_cis_eqtls != 0:
            self.SNP_gene_effect = nn.Parameter(
                torch.randn(self.z_dim, n_cis_eqtls, device=device, generator=self.G_gen)
            )
            self.nonzero_indices_set = False

        self.automatic_optimization = False

        if initialise_training_mode:
            self.initialise_model()

    def initialise_model(self):
        self.set_pretrain_mode(self.pretrain_mode)
        self.set_train_V_mode(self.train_V_mode)
        self.set_train_DxC_mode(self.train_DxC_mode)

    def prepare_batch(self, batch):
        x = batch["x"]
        y = batch["y"]
        if self.hparams.covariates_dims is not None:
            assert len(self.hparams.covariates_dims) == len(
                batch["covariates"]
            ), "Number of covariates different than the number of covariates in data module."
            covariates = batch["covariates"]
        else:
            covariates = None
        size_factor = batch["size_factor"]
        known_cis_associations = None if self.hparams.n_cis_eqtls == 0 else batch["known_cis"]
        cell_gt = None if self.hparams.n_cis_eqtls == 0 else batch["GT_cells"]

        if not self.nonzero_indices_set and known_cis_associations is not None:
            self.nonzero_indices = known_cis_associations.nonzero(as_tuple=True)
            self.nonzero_indices_set = True
            assert (
                self.nonzero_indices[0].numel() == self.hparams.n_cis_eqtls
            ), "The number of known cis-eQTLs in the DataModule does not match the number of known cis-eQTLs in the model args."

        return x, y, covariates, size_factor, known_cis_associations, cell_gt

    def compute_elbo(
        self,
        z_dist: torch.distributions.Distribution,
        x: torch.Tensor,
        y: torch.Tensor,
        covariates: Optional[List[torch.Tensor]] = None,
        size_factor: Optional[torch.Tensor] = None,
        snp_gene_mask: Optional[torch.Tensor] = None,
        cell_snp_mask: Optional[torch.Tensor] = None,
    ):
        """Computes evidence lower bound (ELBO).

        Parameters
        ----------
            z_dist: Variational distribution.
            x: Input data.
            y: Donor IDs.
            covariates: Cell/donor covariates (e.g. technical batch ID or sex).
            size_factor: Size factor to correct for gene count differences.

        Returns
        -------
            Mean evidence lower bound (ELBO).
        """
        z = z_dist.rsample()
        z = nn.Softmax(dim=1)(z)

        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()

        if self.n_DxC_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.D_context(y)
        else:
            z_interaction = None
            A = None

        if covariates is not None:
            covariate_effect = torch.zeros_like(x)
            for covar in range(len(covariates)):
                covar_indices = covariates[covar]
                # increase the indices by the number of categories of the previous covariate(s)
                embedding_indices = covar_indices + sum(self.hparams.covariates_dims[:covar])
                covariate_effect += self.covariate_effect(embedding_indices)
        else:
            covariate_effect = None

        if (
            snp_gene_mask is not None
            and cell_snp_mask is not None
            and self.hparams.n_cis_eqtls != 0
        ):

            row_indices, col_indices = self.nonzero_indices

            # Select only eSNPs
            cell_snp_mask = cell_snp_mask[:, row_indices]  # cells x non-zero
            # Cell-state-specific cis effect
            celltype_cis_effect = z @ self.SNP_gene_effect  # cells x non-zero
            # Element-wise multiplication with the GT
            celltype_cis_effect = cell_snp_mask * celltype_cis_effect  # cells x non-zero
            # Scatter back to cells x genes
            celltype_known_cis_effect = torch.zeros(
                (celltype_cis_effect.size(0), self.x_dim), device=self.hparams.device
            )
            celltype_known_cis_effect.scatter_add_(
                dim=1,
                index=col_indices.unsqueeze(0).expand(celltype_cis_effect.shape),
                src=celltype_cis_effect.to(torch.float),
            )
        else:
            celltype_known_cis_effect = None

        log_lik = (
            self.decoder(
                z=z,
                DxC=z_interaction,
                persistent_G=self.V_persistent(y) if self.n_persistent_factors != 0 else None,
                size_factor=size_factor,
                covariate_effect=covariate_effect,
                known_cis_effect=celltype_known_cis_effect,
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div, A

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask = self.prepare_batch(batch)

        optim_vae = self.optimizers()

        z_dist = self(x)

        elbo, A = self.compute_elbo(
            z_dist, x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask
        )
        logs = {f"{mode}/elbo": elbo.item()}
        if not self.train_DxC_mode or self.n_DxC_factors == 0:
            l1_loss_DxC = torch.zeros([1], device=self.device)
            loss_A = torch.zeros([1], device=self.device)
        else:
            l1_loss_DxC = self.hparams.l1_weight * torch.linalg.vector_norm(
                torch.cat([p for p in self.decoder.DxC_decoder.parameters()]),
                ord=1,
                dim=(0, 1),
            )
            A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
            loss_A = self.hparams.A_weight * A_penalty.sum()

        logs[f"{mode}/L1_penalty_context"] = l1_loss_DxC.item()
        logs[f"{mode}/penalty_A"] = loss_A.item()
        loss = -elbo + l1_loss_DxC + loss_A
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

    def set_pretrain_mode(self, mode: bool):
        """Set VAE pretrain mode.

        If True, discriminator and genetic embeddings are not optimised.
        """
        self.pretrain_mode = mode
        self.decoder.pretrain_VAE = mode
        if mode:
            # freeze parameters
            self.set_train_DxC_mode(False)
            self.set_train_V_mode(False)

    def set_train_V_mode(self, mode: bool):
        self.train_V_mode = mode
        self.decoder.train_V = mode
        if mode:
            # Train V
            if self.n_persistent_factors != 0:
                self.V_persistent.requires_grad_(True)
                self.decoder.persistent_decoder.requires_grad_(True)
        else:
            if self.n_persistent_factors != 0:
                self.V_persistent.requires_grad_(False)
                self.decoder.persistent_decoder.requires_grad_(False)

    def set_train_DxC_mode(self, mode: bool):
        self.train_DxC_mode = mode
        self.decoder.train_DxC = mode
        if mode:
            if self.n_DxC_factors != 0:
                self.D_context.requires_grad_(True)
                self.decoder.DxC_decoder.requires_grad_(True)
                self.A.requires_grad_(True)
            if self.hparams.n_cis_eqtls != 0:
                self.SNP_gene_effect.requires_grad_(True)
        else:
            if self.n_DxC_factors != 0:
                self.D_context.requires_grad_(False)
                self.decoder.DxC_decoder.requires_grad_(False)
                self.A.requires_grad_(False)
            if self.hparams.n_cis_eqtls != 0:
                self.SNP_gene_effect.requires_grad_(False)

    def freeze_vae(self, mode: bool):
        """Freezes VAE and covariate embeddings parameters, after the number of VAE and
        discriminator warm-up epochs has been completed."""

        self.frozen = mode
        if mode:
            # freeze VAE etc.
            self.encoder.requires_grad_(False)
            self.decoder.mean.requires_grad_(False)
            # # Retrain total_count
            self.decoder.log_total_count.requires_grad_(False)
            # freeze covariate embeddings, if applicable
            if self.covariate_effect is not None:
                self.covariate_effect.requires_grad_(False)
        else:
            # train VAE
            self.encoder.requires_grad_(True)
            self.decoder.mean.requires_grad_(True)
            self.decoder.log_total_count.requires_grad_(True)
            # train covariate embeddings, if applicable
            if self.covariate_effect is not None:
                self.covariate_effect.requires_grad_(True)
            # freeze genetic embeddings
            self.set_train_DxC_mode(False)
            self.set_train_V_mode(False)

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE, discriminator
        and persistent G has been reached."""
        print(f"VAE frozen: {self.frozen}")
        print(f"Training V: {self.train_V_mode}")
        print(f"Training DxC: {self.train_DxC_mode}")
        print(f"DxC decoder requires grad: {self.decoder.DxC_decoder[0].weight.requires_grad}")
        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            print("VAE pretraining completed.")
            self.freeze_vae(True)  # Freeze VAE when starting training V
            print("Freeze VAE parameters.")
            self.set_train_V_mode(True)
            print("Start training V.")

        if self.current_epoch == self.hparams.warmup_epochs_vae + self.warmup_epochs_G:
            # self.freeze_vae(True) # Freeze VAE after starting training V
            # self.set_train_V_mode(False) # Freeze V
            print("Pretraining completed.")
            print("Start learning DxC effects.")
            self.set_train_DxC_mode(True)

    def on_save_checkpoint(self, checkpoint: dict):
        # Save model's pretraining and frozen state
        attributes_to_save = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k not in list(self.__dict__["_hparams"].keys())
            and "G_gen" not in k
        }
        checkpoint["model_attributes"] = attributes_to_save

    def on_load_checkpoint(self, checkpoint: dict):
        # Restore model's attributes
        model_attributes = checkpoint.get("model_attributes", {})
        for attr_name, attr_value in model_attributes.items():
            setattr(self, attr_name, attr_value)

    def predict(self, x, y):
        """Model inference. Get latent space and individual embeddings for the input data.

        Parameters
        ----------
            x (torch.Tensor): Input gene expression vector per cell.
            y (torch.Tensor): ID of the individual the cell is derived from.

        Returns
        -------
        Inference results (Dict[str,torch.Tensor])
            'cell-state_latent' (torch.Tensor): Cell state latent space.
            'base_decoder' (torch.Tensor): Gene loadings for the cell-state decoder.
            'D_embedding' (torch.Tensor): Learned embedding of context-specific individual effects, if applicable.
            'DxC_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
            'assignment_matrix' (torch.Tensor): Learned assignment matrix of U factors to cell-state factors.
            'V_embedding' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
            'V_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
        """

        with torch.no_grad():
            self.eval()

            z = self(x).rsample()
            cell_state_decoder = self.decoder.mean[0].weight
            if self.n_DxC_factors != 0:
                D = self.D_context(y)
                DxC_decoder = self.decoder.DxC_decoder[0].weight
            else:
                D = None
                DxC_decoder = None
            if self.n_persistent_factors != 0:
                V = self.V_persistent(y)
                V_decoder = self.decoder.persistent_decoder[0].weight
            else:
                V = None
                V_decoder = None

            if self.hparams.n_cis_eqtls != 0:
                cell_state_cis_effect = torch.zeros(
                    self.z_dim, self.hparams.n_cis_eqtls, self.x_dim
                )
                cell_state_cis_effect[:, self.nonzero_indices[0], self.nonzero_indices[1]] = (
                    self.SNP_gene_effect
                )
            else:
                cell_state_cis_effect = None

        return {
            "cell-state_latent": z,
            "cell-state_decoder": cell_state_decoder,
            "D_embedding": D,
            "DxC_decoder": DxC_decoder,
            "assignment_matrix": self.A,
            "V_embedding": V,
            "V_decoder": V_decoder,
            "cis_SNP_effect": cell_state_cis_effect,
        }

    def configure_optimizers(self):
        """Configures optimizer."""

        params = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]
        if self.covariate_effect is not None:
            params.append({"params": self.covariate_effect.parameters()})
        if self.n_DxC_factors != 0:
            params.append({"params": self.D_context.parameters()})
            params.append({"params": self.A})
        if self.n_persistent_factors != 0:
            params.append({"params": self.V_persistent.parameters()})
        if self.hparams.n_cis_snps != 0:
            params.append({"params": self.SNP_gene_effect})

        optim_vae = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
        )

        return optim_vae


class LIVI_GT_PCs(LIVI):
    """LIVI model accounting for cis genetic effects and global genetic effects captured by top
    genotype PCs."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_persistent_factors: int,
        n_cis_snps: int,
        n_gt_pcs: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 60,
        warmup_epochs_G: int = 0,
        covariates_dims: Optional[List[int]] = None,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
        initialise_training_mode: bool = True,
    ):
        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            n_cis_snps=n_cis_snps,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            warmup_epochs_vae=warmup_epochs_vae,
            warmup_epochs_G=warmup_epochs_G,
            covariates_dims=covariates_dims,
            l1_weight=l1_weight,
            A_weight=A_weight,
            batch_norm_decoder=batch_norm_decoder,
            device=device,
            genetics_seed=genetics_seed,
            initialise_training_mode=False,
        )

        self.save_hyperparameters()

        self.decoder = LIVI_Decoder_GT_PCs(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_DxC=self.train_DxC_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_generator=self.G_gen,
        )

        if n_gt_pcs != 0:
            # self.gt_pcs_effect = create_mlp(
            #     input_size=n_gt_pcs,
            #     output_size=self.x_dim,
            #     hidden_dims=[],
            #     layer_norm=False,
            #     device=device,
            # )
            self.gt_pcs_effect = nn.Parameter(
                torch.randn(z_dim, n_gt_pcs, x_dim, device=device, generator=self.G_gen)
            )

        self.automatic_optimization = False

        if initialise_training_mode:
            self.initialise_model()

    def initialise_model(self):
        self.set_pretrain_mode(self.pretrain_mode)
        self.set_train_V_mode(self.train_V_mode)
        self.set_train_DxC_mode(self.train_DxC_mode)

    def prepare_batch(self, batch):
        x = batch["x"]
        y = batch["y"]
        if self.hparams.covariates_dims is not None:
            assert len(self.hparams.covariates_dims) == len(
                batch["covariates"]
            ), "Number of covariates different than the number of covariates in data module."
            covariates = batch["covariates"]
        else:
            covariates = None
        size_factor = batch["size_factor"]
        known_cis_associations = None if self.hparams.n_cis_snps == 0 else batch["known_cis"]
        cell_gt = None if self.hparams.n_cis_snps == 0 else batch["GT_cells"]
        gt_pcs = None if self.hparams.n_gt_pcs == 0 else batch["GT_PCs"]

        return x, y, covariates, size_factor, known_cis_associations, cell_gt, gt_pcs

    def compute_elbo(
        self,
        z_dist: torch.distributions.Distribution,
        x: torch.Tensor,
        y: torch.Tensor,
        covariates: Optional[List[torch.Tensor]] = None,
        size_factor: Optional[torch.Tensor] = None,
        snp_gene_mask: Optional[torch.Tensor] = None,
        cell_snp_mask: Optional[torch.Tensor] = None,
        gt_pcs: Optional[torch.Tensor] = None,
    ):
        """Computes evidence lower bound (ELBO).

        Parameters
        ----------
            z_dist: Variational distribution.
            x: Input data.
            y: Donor IDs.
            covariates: Cell/donor covariates (e.g. technical batch ID or sex).
            size_factor: Size factor to correct for gene count differences.

        Returns
        -------
            Mean evidence lower bound (ELBO).
        """
        z = z_dist.rsample()
        z = nn.Softmax(dim=1)(z)

        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()

        if self.n_DxC_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.D_context(y)
        else:
            z_interaction = None
            A = None

        if covariates is not None:
            covariate_effect = torch.zeros_like(x)
            for covar in range(len(covariates)):
                covar_indices = covariates[covar]
                # increase the indices by the number of categories of the previous covariate(s)
                embedding_indices = covar_indices + sum(self.hparams.covariates_dims[:covar])
                covariate_effect += self.covariate_effect(embedding_indices)
        else:
            covariate_effect = None

        if (
            snp_gene_mask is not None
            and cell_snp_mask is not None
            and self.hparams.n_cis_snps != 0
        ):
            known_cis_effect = (
                snp_gene_mask.resize(1, self.hparams.n_cis_snps, self.x_dim) * self.SNP_gene_effect
            )  # z x SNPs x genes
            celltype_known_cis_effect = torch.einsum(
                "ij,kjl->kil", cell_snp_mask, known_cis_effect
            )  # z x cells x genes (i cell, j SNP, k dim, l gene)
            celltype_known_cis_effect = torch.einsum(
                "ik,kil->il", z, celltype_known_cis_effect
            )  # cells x genes

        if self.hparams.n_gt_pcs != 0 and gt_pcs is not None:
            # genotype_pcs_effect = self.gt_pcs_effect(gt_pcs)
            genotype_pcs_effect = gt_pcs @ self.gt_pcs_effect  # z x cells x genes
            genotype_pcs_effect = torch.einsum(
                "ik,kil->il", z, genotype_pcs_effect
            )  # cells x genes (i cell, k dim, l gene)

        log_lik = (
            self.decoder(
                z=z,
                DxC=z_interaction,
                persistent_G=self.V_persistent(y) if self.n_persistent_factors != 0 else None,
                size_factor=size_factor,
                covariate_effect=covariate_effect,
                known_cis_effect=(
                    celltype_known_cis_effect
                    if snp_gene_mask is not None and self.hparams.n_cis_snps != 0
                    else None
                ),
                gt_pcs_effect=(
                    genotype_pcs_effect
                    if self.hparams.n_gt_pcs != 0 and gt_pcs is not None
                    else None
                ),
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div, A

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask, gt_pcs = self.prepare_batch(
            batch
        )
        optim_vae = self.optimizers()

        z_dist = self(x)

        elbo, A = self.compute_elbo(
            z_dist, x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask, gt_pcs
        )
        logs = {f"{mode}/elbo": elbo.item()}
        if not self.train_DxC_mode or self.n_DxC_factors == 0:
            l1_loss_DxC = torch.zeros([1], device=self.device)
            loss_A = torch.zeros([1], device=self.device)
        else:
            l1_loss_DxC = self.hparams.l1_weight * torch.linalg.vector_norm(
                torch.cat([p for p in self.decoder.DxC_decoder.parameters()]),
                ord=1,
                dim=(0, 1),
            )
            A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
            loss_A = self.hparams.A_weight * A_penalty.sum()

        logs[f"{mode}/L1_penalty_DxC"] = l1_loss_DxC.item()
        logs[f"{mode}/penalty_A"] = loss_A.item()
        loss = -elbo + l1_loss_DxC + loss_A
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

    def set_train_DxC_mode(self, mode: bool):
        self.train_DxC_mode = mode
        self.decoder.train_DxC = mode
        if mode:
            if self.n_DxC_factors != 0:
                self.D_context.requires_grad_(True)
                self.decoder.DxC_decoder.requires_grad_(True)
                self.A.requires_grad_(True)
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad_(True)
            if self.hparams.n_gt_pcs != 0:
                self.gt_pcs_effect.requires_grad_(True)
        else:
            if self.n_DxC_factors != 0:
                self.D_context.requires_grad_(False)
                self.decoder.DxC_decoder.requires_grad_(False)
                self.A.requires_grad_(False)
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad_(False)
            if self.hparams.n_gt_pcs != 0:
                self.gt_pcs_effect.requires_grad_(False)

    def configure_optimizers(self):
        """Configures optimizer."""

        params = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]
        if self.covariate_effect is not None:
            params.append({"params": self.covariate_effect.parameters()})
        if self.n_DxC_factors != 0:
            params.append({"params": self.D_context.parameters()})
            params.append({"params": self.A})
        if self.n_persistent_factors != 0:
            params.append({"params": self.V_persistent.parameters()})
        if self.hparams.n_cis_snps != 0:
            params.append({"params": self.SNP_gene_effect})
        # if self.hparams.n_gt_pcs != 0:
        #     params.append({"params": self.gt_pcs_effect.parameters()})
        if self.hparams.n_gt_pcs != 0:
            params.append({"params": self.gt_pcs_effect})

        optim_vae = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
        )

        return optim_vae


class LIVI_cis_gen_adversary_U(pl.LightningModule):
    """LIVI model accounting for cis genetic effects."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_persistent_factors: int,
        n_cis_snps: int,
        n_cohorts: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 60,
        # train_epochs_adversary: int = 20,
        covariates_dims: Optional[List[int]] = None,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
    ):

        super().__init__()

        self.save_hyperparameters()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_DxC_factors = n_DxC_factors
        self.n_persistent_factors = n_persistent_factors
        # self.warmup_epochs_G = 0 if self.n_persistent_factors == 0 else warmup_epochs_G
        # self.train_epochs_adversary = train_epochs_adversary if adversary_weight > 0 else 0
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_DxC_mode = False if self.pretrain_mode or self.n_DxC_factors == 0 else True
        # self.checkpointing_epoch = (
        #     warmup_epochs_vae + self.warmup_epochs_G + self.train_epochs_adversary + 5
        # )
        self.checkpointing_epoch = warmup_epochs_vae + 5

        self.frozen = False

        if self.hparams.genetics_seed is not None:
            self.G_gen = torch.Generator(device=device)
            self.G_gen.manual_seed(genetics_seed)
        else:
            self.G_gen = None

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=True,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))

        self.decoder = LIVI_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_DxC=self.train_DxC_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_generator=self.G_gen,
        )

        self.adversary_learning_rate = adversary_learning_rate
        self.adversary_steps = adversary_steps
        # Set up adversary
        self.adversary = create_mlp(
            input_size=n_DxC_factors,
            output_size=n_cohorts,
            hidden_dims=adversary_hidden_dims,
            layer_norm=False,
            device=device,
        )
        init_mlp(self.adversary, generator=self.G_gen)

        self.A = nn.Parameter(
            torch.randn(z_dim, n_DxC_factors, device=device, generator=self.G_gen)
        )
        self.D_context = nn.Embedding(y_dim, n_DxC_factors, device=device)
        if self.hparams.genetics_seed is not None:
            nn.init.normal_(self.D_context.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

        if n_persistent_factors != 0:
            self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)

        # Covariate (e.g. experimental batch) correction per gene
        if self.hparams.covariates_dims is not None:
            self.covariate_effect = nn.Embedding(
                sum(self.hparams.covariates_dims), x_dim, device=device
            )
        else:
            self.covariate_effect = None

        if n_cis_snps != 0:
            # self.SNP_gene_effect = nn.Parameter(
            #     torch.randn(n_cis_snps, x_dim, device=device, generator=self.G_gen)
            # )
            self.SNP_gene_effect = nn.Parameter(
                torch.randn(z_dim, n_cis_snps, x_dim, device=device, generator=self.G_gen)
            )

        self.automatic_optimization = False

        self.set_pretrain_mode(self.pretrain_mode)
        self.set_train_V_mode(self.train_V_mode)
        self.set_train_DxC_mode(self.train_DxC_mode)

    def prepare_batch(self, batch):
        x = batch["x"]
        y = batch["y"]
        cohort_id = batch["cohort_id"]
        if self.hparams.covariates_dims is not None:
            assert len(self.hparams.covariates_dims) == len(
                batch["covariates"]
            ), "Number of covariates different than the number of covariates in data module."
            covariates = batch["covariates"]
        else:
            covariates = None
        size_factor = batch["size_factor"]
        known_cis_associations = None if self.hparams.n_cis_snps == 0 else batch["known_cis"]
        cell_gt = None if self.hparams.n_cis_snps == 0 else batch["GT_cells"]

        return x, y, cohort_id, covariates, size_factor, known_cis_associations, cell_gt

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
        covariates: Optional[List[torch.Tensor]] = None,
        size_factor: Optional[torch.Tensor] = None,
        snp_gene_mask: Optional[torch.Tensor] = None,
        cell_snp_mask: Optional[torch.Tensor] = None,
    ):
        """Computes evidence lower bound (ELBO).

        Parameters
        ----------
            z_dist: Variational distribution.
            x: Input data.
            y: Donor IDs.
            covariates: Cell/donor covariates (e.g. technical batch ID or sex).
            size_factor: Size factor to correct for gene count differences.

        Returns
        -------
            Mean evidence lower bound (ELBO).
        """
        z = z_dist.rsample()
        z = nn.Softmax(dim=1)(z)

        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()

        if self.n_DxC_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.D_context(y)
        else:
            z_interaction = None
            A = None

        if covariates is not None:
            covariate_effect = torch.zeros_like(x)
            for covar in range(len(covariates)):
                covar_indices = covariates[covar]
                # increase the indices by the number of categories of the previous covariate(s)
                embedding_indices = covar_indices + sum(self.hparams.covariates_dims[:covar])
                covariate_effect += self.covariate_effect(embedding_indices)
        else:
            covariate_effect = None

        if (
            snp_gene_mask is not None
            and cell_snp_mask is not None
            and self.hparams.n_cis_snps != 0
        ):
            # known_cis_effect = snp_gene_mask * self.SNP_gene_effect  # SNPs x genes
            # known_cis_effect = (
            #     cell_snp_mask @ known_cis_effect
            # )  # cells x genes: mean cis-SNP effect on each gene
            # # Make the cis effect cell-state-specific by multiplying with the reconstructed cell-state GEX (=> genes essentially define the cell-state; if a gene is less relevant for the given cells (i.e. closer to 0), then the cis effect will become close to zero as well)
            # y_c = F.softmax(self.decoder.mean(z), dim=-1)
            # celltype_known_cis_effect = (
            #     known_cis_effect * y_c
            # )  # celltype_known_cis_effect = known_cis_effect
            known_cis_effect = (
                snp_gene_mask.resize(1, self.hparams.n_cis_snps, self.x_dim) * self.SNP_gene_effect
            )  # z x SNPs x genes
            celltype_known_cis_effect = torch.einsum(
                "ij,kjl->kil", cell_snp_mask, known_cis_effect
            )  # z x cells x genes
            celltype_known_cis_effect = torch.einsum(
                "ik,kil->il", z, celltype_known_cis_effect
            )  # cells x genes

        log_lik = (
            self.decoder(
                z=z,
                DxC=z_interaction,
                persistent_G=self.V_persistent(y) if self.n_persistent_factors != 0 else None,
                size_factor=size_factor,
                covariate_effect=covariate_effect,
                known_cis_effect=(
                    celltype_known_cis_effect
                    if snp_gene_mask is not None and self.hparams.n_cis_snps != 0
                    else None
                ),
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div, A

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, cohort_id, covariates, size_factor, snp_gene_mask, cell_snp_mask = (
            self.prepare_batch(batch)
        )
        optim_vae, optim_adversary = self.optimizers()

        train_adversary = batch_idx % self.adversary_steps == 0
        train_adversary = train_adversary & self.train_DxC_mode & (not self.pretrain_mode)
        train_adversary = train_adversary * (mode == "train")

        z_dist = self(x)

        if self.pretrain_mode or not self.train_DxC_mode:
            # no adversary signal
            adversarial_loss = torch.zeros([1], device=self.device)
        else:
            U_y = self.D_context(y)
            adversarial_loss = F.cross_entropy(self.adversary(U_y), cohort_id)
        logs = {f"{mode}/adversarial_loss": adversarial_loss.item()}

        if train_adversary:
            # train adversary
            optim_adversary.zero_grad()
            self.manual_backward(adversarial_loss)
            optim_adversary.step()
        else:
            # train vae
            elbo, A = self.compute_elbo(
                z_dist, x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask
            )
            logs[f"{mode}/elbo"] = elbo.item()
            if not self.train_DxC_mode or self.n_DxC_factors == 0:
                l1_loss_DxC = torch.zeros([1], device=self.device)
                loss_A = torch.zeros([1], device=self.device)
            else:
                l1_loss_DxC = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.DxC_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
                A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
                loss_A = self.hparams.A_weight * A_penalty.sum()

            logs[f"{mode}/L1_penalty_DxC"] = l1_loss_DxC.item()
            logs[f"{mode}/penalty_A"] = loss_A.item()
            loss = -elbo - self.hparams.adversary_weight * adversarial_loss + l1_loss_DxC + loss_A
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

    def predict(self, x, y):
        """Model inference. Get latent space and individual embeddings for the input data.

        Parameters
        ----------
            x (torch.Tensor): Input gene expression vector per cell.
            y (torch.Tensor): ID of the individual the cell is derived from.

        Returns
        -------
        Inference results (Dict[str,torch.Tensor])
            'cell-state_latent' (torch.Tensor): Cell state latent space.
            'base_decoder' (torch.Tensor): Gene loadings for the cell-state decoder.
            'U_embedding' (torch.Tensor): Learned embedding of context-specific individual effects, if applicable.
            'DxC_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
            'assignment_matrix' (torch.Tensor): Learned assignment matrix of U factors to cell-state factors.
            'V_embedding' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
            'V_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
        """

        with torch.no_grad():
            self.eval()

            z = self(x).rsample()
            cell_state_decoder = self.decoder.mean[0].weight
            if self.n_DxC_factors != 0:
                U = self.D_context(y)
                DxC_decoder = self.decoder.DxC_decoder[0].weight
            else:
                U = None
                DxC_decoder = None
            if self.n_persistent_factors != 0:
                V = self.V_persistent(y)
                V_decoder = self.decoder.persistent_decoder[0].weight
            else:
                V = None
                V_decoder = None

        return {
            "cell-state_latent": z,
            "cell-state_decoder": cell_state_decoder,
            "U_embedding": U,
            "DxC_decoder": DxC_decoder,
            "assignment_matrix": self.A,
            "V_embedding": V,
            "V_decoder": V_decoder,
            "cis_SNP_effect": self.SNP_gene_effect if self.hparams.n_cis_snps != 0 else None,
        }

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
            self.set_train_DxC_mode(False)
            self.set_train_V_mode(False)
        else:
            # unfreeze adversary
            for p in self.adversary.parameters():
                p.requires_grad = True
            # unfreeze U etc
            self.set_train_DxC_mode(True)
            # unfreeze V
            self.set_train_V_mode(True)

    def set_train_V_mode(self, mode: bool):
        self.train_V_mode = mode
        self.decoder.train_V = mode
        if mode:
            # Train V
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = True
                self.decoder.persistent_decoder[0].weight.requires_grad = True
        else:
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = False
                self.decoder.persistent_decoder[0].weight.requires_grad = False

    def set_train_DxC_mode(self, mode: bool):
        self.train_DxC_mode = mode
        self.decoder.train_DxC = mode
        if mode:
            if self.n_DxC_factors != 0:
                for p in self.D_context.parameters():
                    p.requires_grad = True
                self.decoder.DxC_decoder[0].weight.requires_grad = True
                self.A.requires_grad = True
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = True
            for p in self.adversary.parameters():
                p.requires_grad = True
        else:
            if self.n_DxC_factors != 0:
                for p in self.D_context.parameters():
                    p.requires_grad = False
                self.decoder.DxC_decoder[0].weight.requires_grad = False
                self.A.requires_grad = False
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = False
            for p in self.adversary.parameters():
                p.requires_grad = False

    def freeze_vae(self, mode: bool):
        """Freezes VAE and covariate embeddings parameters, after the number of VAE and
        discriminator warm-up epochs has been completed."""

        self.frozen = mode
        if mode:
            # freeze VAE etc.
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.decoder.mean[0].weight.requires_grad = False
            # # Retrain total_count
            self.decoder.log_total_count.requires_grad = False
            # freeze covariate embeddings, if applicable
            if self.covariate_effect is not None:
                for p in self.covariate_effect.parameters():
                    p.requires_grad = False
        else:
            # train VAE
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.decoder.mean[0].weight.requires_grad = True
            self.decoder.log_total_count.requires_grad = True
            # train covariate embeddings, if applicable
            if self.covariate_effect is not None:
                for p in self.covariate_effect.parameters():
                    p.requires_grad = True
            # freeze genetic embeddings and adversary
            for p in self.adversary.parameters():
                p.requires_grad = False
            self.set_train_DxC_mode(False)
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = False
            self.set_train_V_mode(False)

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE, discriminator
        and persistent G has been reached."""
        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            self.freeze_vae(True)
            print("VAE pretraining completed.")
            print("Start leanring genetic effects.")

    def on_save_checkpoint(self, checkpoint: dict):
        # Save model's pretraining and frozen state
        attributes_to_save = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k not in list(self.__dict__["_hparams"].keys())
            and "G_gen" not in k
        }
        checkpoint["model_attributes"] = attributes_to_save

    def on_load_checkpoint(self, checkpoint: dict):
        # Restore model's attributes
        model_attributes = checkpoint.get("model_attributes", {})
        for attr_name, attr_value in model_attributes.items():
            setattr(self, attr_name, attr_value)

    def configure_optimizers(self):
        """Configures optimizer."""

        params = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]
        if self.covariate_effect is not None:
            params.append({"params": self.covariate_effect.parameters()})
        if self.n_DxC_factors != 0:
            params.append({"params": self.D_context.parameters()})
            params.append({"params": self.A})
        if self.n_persistent_factors != 0:
            params.append({"params": self.V_persistent.parameters()})
        if self.hparams.n_cis_snps != 0:
            params.append({"params": self.SNP_gene_effect})

        optim_vae = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.hparams.adversary_learning_rate,
        )

        return optim_vae, optim_adversary


class LIVI_cis_warmup_U(pl.LightningModule):
    """LIVI model accounting for cis genetic effects."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_persistent_factors: int,
        exbatch_dim: int,
        n_cis_snps: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 60,
        warmup_epochs_V: int = 0,
        warmup_epochs_U: int = 0,
        train_epochs_adversary: int = 20,
        donor_sex_dim: int = 2,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
    ):

        super().__init__()

        self.save_hyperparameters()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_DxC_factors = n_DxC_factors
        self.n_persistent_factors = n_persistent_factors
        self.warmup_epochs_V = 0 if self.n_persistent_factors == 0 else warmup_epochs_V
        self.warmup_epochs_U = 0 if self.n_DxC_factors == 0 else warmup_epochs_U
        self.train_epochs_adversary = train_epochs_adversary if adversary_weight > 0 else 0
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_CxG_mode = False if self.pretrain_mode or self.n_DxC_factors == 0 else True
        self.checkpointing_epoch = (
            warmup_epochs_vae
            + self.train_epochs_adversary
            + self.warmup_epochs_V
            + self.warmup_epochs_U
            + 5
        )
        self.frozen = False

        if self.hparams.genetics_seed is not None:
            self.G_gen = torch.Generator(device=device)
            self.G_gen.manual_seed(genetics_seed)
        else:
            self.G_gen = None

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=True,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))

        self.decoder = LIVI_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_CxG=self.train_CxG_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_generator=self.G_gen,
        )

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

        self.A = nn.Parameter(
            torch.randn(z_dim, n_DxC_factors, device=device, generator=self.G_gen)
        )
        self.D_context = nn.Embedding(y_dim, n_DxC_factors, device=device)
        if self.hparams.genetics_seed is not None:
            nn.init.normal_(self.D_context.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

        if n_persistent_factors != 0:
            self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)
            if self.hparams.genetics_seed is not None:
                nn.init.normal_(
                    self.V_persistent.weight.data, mean=0.0, std=1.0, generator=self.G_gen
                )

        if n_cis_snps != 0:
            self.SNP_gene_effect = nn.Parameter(
                torch.randn(n_cis_snps, x_dim, device=device, generator=self.G_gen)
            )
            # self.SNP_gene_effect = nn.Parameter(torch.randn(z_dim, n_cis_snps, x_dim, device=device)) # Learn SNP-gene effect for each cell-state

        # Sex and batch correction per gene
        if donor_sex_dim is not None:
            self.sex_effect = nn.Embedding(donor_sex_dim, x_dim, device=device)
        else:
            self.sex_effect = None
        self.batch_effect = nn.Embedding(exbatch_dim, x_dim, device=device)

        self.automatic_optimization = False

        self.set_pretrain_mode(self.pretrain_mode)
        self.set_train_V_mode(self.train_V_mode)
        self.set_learn_CxG_mode(self.train_CxG_mode)

    def prepare_batch(self, batch):
        x = batch["x"]
        y = batch["y"]
        dsex = None if self.hparams.donor_sex_dim is None else batch["dsex"]
        eb = batch["eb"]
        size_factor = batch["size_factor"]
        known_cis_associations = None if self.hparams.n_cis_snps == 0 else batch["known_cis"]
        cell_gt = None if self.hparams.n_cis_snps == 0 else batch["GT_cells"]

        return x, y, dsex, eb, size_factor, known_cis_associations, cell_gt

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
        eb: torch.Tensor,
        dsex: Optional[torch.Tensor] = None,
        size_factor: Optional[torch.Tensor] = None,
        snp_gene_mask: Optional[torch.Tensor] = None,
        cell_snp_mask: Optional[torch.Tensor] = None,
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
        z = nn.Softmax(dim=1)(z)

        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()

        if self.n_DxC_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.D_context(y)
        else:
            z_interaction = None

        if (
            snp_gene_mask is not None
            and cell_snp_mask is not None
            and self.hparams.n_cis_snps != 0
        ):
            known_cis_effect = snp_gene_mask * self.SNP_gene_effect  # SNPs x genes
            known_cis_effect = (
                cell_snp_mask @ known_cis_effect
            )  # cells x genes: mean cis-SNP effect on each gene
            # Make the cis effect cell-state-specific by multiplying with the reconstructed cell-state GEX (=> genes essentially define the cell-state; if a gene is less relevant for the given cells (i.e. closer to 0), then the cis effect will become close to zero as well)
            y_c = F.softmax(self.decoder.mean(z), dim=-1)
            celltype_known_cis_effect = (
                known_cis_effect * y_c
            )  # celltype_known_cis_effect = known_cis_effect
            # known_cis_effect = snp_gene_mask.resize(1, self.hparams.n_cis_snps, self.x_dim) * self.SNP_gene_effect # z x SNPs x genes
            # celltype_known_cis_effect = torch.einsum("ij,kjl->kil", cell_snp_mask, known_cis_effect) # z x cells x genes
            # celltype_known_cis_effect = torch.einsum("ik,kil->il", z, celltype_known_cis_effect) # cells x genes

        log_lik = (
            self.decoder(
                z=z,
                DxC=z_interaction,
                persistent_G=self.V_persistent(y) if self.n_persistent_factors != 0 else None,
                size_factor=size_factor,
                batch_effect=self.batch_effect(eb),
                donor_sex_effect=self.sex_effect(dsex) if dsex is not None else None,
                known_cis_effect=(
                    celltype_known_cis_effect
                    if snp_gene_mask is not None and self.hparams.n_cis_snps != 0
                    else None
                ),
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div, A

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, donor_sex, exp_batch_ids, size_factor, snp_gene_mask, cell_snp_mask = (
            self.prepare_batch(batch)
        )
        optim_vae, optim_adversary = self.optimizers()

        train_adversary = batch_idx % self.adversary_steps == 0
        train_adversary = train_adversary & (not self.pretrain_mode) & (not self.frozen)
        train_adversary = train_adversary * (mode == "train")

        z_dist = self(x)

        if self.pretrain_mode or self.frozen:
            # no adversary signal
            adversarial_loss = torch.zeros([1], device=self.device)
        else:
            z = z_dist.rsample()
            # z = nn.Softmax(dim=1)(z)
            adversarial_loss = F.cross_entropy(self.adversary(z), y)
        logs = {f"{mode}/adversarial_loss": adversarial_loss.item()}

        if train_adversary:
            # train adversary
            optim_adversary.zero_grad()
            self.manual_backward(adversarial_loss)
            optim_adversary.step()
        else:
            # train vae
            elbo, A = self.compute_elbo(
                z_dist, x, y, exp_batch_ids, donor_sex, size_factor, snp_gene_mask, cell_snp_mask
            )
            logs[f"{mode}/elbo"] = elbo.item()
            if not self.train_CxG_mode or self.n_DxC_factors == 0:
                l1_loss_DxC = torch.zeros([1], device=self.device)
                loss_A = torch.zeros([1], device=self.device)
            else:
                l1_loss_DxC = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.CxG_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
                A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
                loss_A = self.hparams.A_weight * A_penalty.sum()
            # if not self.train_V_mode or self.n_persistent_factors == 0:
            #     l1_loss_persistent = torch.zeros([1], device=self.device)
            # else:
            #     l1_loss_persistent = self.hparams.l1_weight * torch.linalg.vector_norm(
            #         torch.cat([p for p in self.decoder.persistent_decoder.parameters()]),
            #         ord=1,
            #         dim=(0, 1),
            #     )

            logs[f"{mode}/L1_penalty_DxC"] = l1_loss_DxC.item()
            logs[f"{mode}/penalty_A"] = loss_A.item()
            # logs[f"{mode}/L1_penalty_persistent"] = l1_loss_persistent.item()
            loss = (
                -elbo
                - self.hparams.adversary_weight * adversarial_loss
                + l1_loss_DxC
                + loss_A
                #   + l1_loss_persistent
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
            'U_embedding' (torch.Tensor): Learned embedding of context-specific individual effects, if applicable.
            'CxG_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
            'assignment_matrix' (torch.Tensor): Learned assignment matrix of U factors to cell-state factors.
            'V_embedding' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
            'V_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
        """

        with torch.no_grad():
            self.eval()

            z = self(x).rsample()
            cell_state_decoder = self.decoder.mean[0].weight
            batch_embedding = (self.batch_effect(exp_batch_ids),)
            if self.n_DxC_factors != 0:
                U = self.D_context(y)
                CxG_decoder = self.decoder.CxG_decoder[0].weight
            else:
                U = None
                CxG_decoder = None
            if self.n_persistent_factors != 0:
                V = self.V_persistent(y)
                V_decoder = self.decoder.persistent_decoder[0].weight
            else:
                V = None
                V_decoder = None

        return {
            "cell-state_latent": z,
            "cell-state_decoder": cell_state_decoder,
            "batch_embedding": batch_embedding,
            "U_embedding": U,
            "CxG_decoder": CxG_decoder,
            "assignment_matrix": self.A,
            "V_embedding": V,
            "V_decoder": V_decoder,
            "cis_SNP_effect": self.SNP_gene_effect if self.hparams.n_cis_snps != 0 else None,
        }

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
            self.set_learn_CxG_mode(False)
            self.set_train_V_mode(False)
        else:
            # unfreeze adversary
            for p in self.adversary.parameters():
                p.requires_grad = True

    def set_train_V_mode(self, mode: bool):
        self.train_V_mode = mode
        self.decoder.train_V = mode
        if mode:
            # Train V
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = True
                self.decoder.persistent_decoder[0].weight.requires_grad = True
        else:
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = False
                self.decoder.persistent_decoder[0].weight.requires_grad = False

    def set_learn_CxG_mode(self, mode: bool):
        self.train_CxG_mode = mode
        self.decoder.train_CxG = mode
        if mode:
            if self.n_DxC_factors != 0:
                for p in self.D_context.parameters():
                    p.requires_grad = True
                self.decoder.CxG_decoder[0].weight.requires_grad = True
                self.A.requires_grad = True
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = True
        else:
            if self.n_DxC_factors != 0:
                for p in self.D_context.parameters():
                    p.requires_grad = False
                self.decoder.CxG_decoder[0].weight.requires_grad = False
                self.A.requires_grad = False
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = False

    def freeze_vae(self, mode: bool):
        """Freezes VAE, discriminator and covariate embeddings parameters, after the number of VAE
        and discriminator warm-up epochs has been completed."""

        self.frozen = mode
        if mode:
            # freeze VAE etc.
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.decoder.mean[0].weight.requires_grad = False
            # # Retrain total_count
            self.decoder.log_total_count.requires_grad = False
            # freeze adversary
            for p in self.adversary.parameters():
                p.requires_grad = False
            # freeze covariate embeddings
            for p in self.batch_effect.parameters():
                p.requires_grad = False
            for p in self.sex_effect.parameters():
                p.requires_grad = False
            # # unfreeze persistent genetic embedding, if applicable
            # self.train_V(True)
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
            self.set_learn_CxG_mode(False)
            self.set_train_V_mode(False)

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE, discriminator
        and persistent G has been reached."""
        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            #  self.set_train_V_mode(True) # train V with Dis
            print("VAE pretraining completed.")
            print("Start training Adversary.")

        if self.current_epoch == self.hparams.warmup_epochs_vae + self.train_epochs_adversary:
            self.set_train_V_mode(True)
            print("Start training V.")

        if (
            self.current_epoch
            == self.hparams.warmup_epochs_vae + self.train_epochs_adversary + self.warmup_epochs_V
        ):
            print("Start learning CxG effects.")
            self.set_learn_CxG_mode(True)

        if (
            self.current_epoch
            == self.hparams.warmup_epochs_vae
            + self.train_epochs_adversary
            + self.warmup_epochs_V
            + self.warmup_epochs_U
        ):
            self.freeze_vae(True)
            print("Pretraining completed.")

    def on_save_checkpoint(self, checkpoint: dict):
        # Save model's pretraining and frozen state
        attributes_to_save = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k not in list(self.__dict__["_hparams"].keys())
            and "G_gen" not in k
        }
        checkpoint["model_attributes"] = attributes_to_save

    def on_load_checkpoint(self, checkpoint: dict):
        # Restore model's attributes
        model_attributes = checkpoint.get("model_attributes", {})
        for attr_name, attr_value in model_attributes.items():
            setattr(self, attr_name, attr_value)

    def configure_optimizers(self):
        """Configures optimizer."""

        params = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
            {"params": self.batch_effect.parameters()},
            {"params": self.sex_effect.parameters()},
        ]
        if self.n_DxC_factors != 0:
            params.append({"params": self.D_context.parameters()})
            params.append({"params": self.A})
        if self.n_persistent_factors != 0:
            params.append({"params": self.V_persistent.parameters()})
        if self.hparams.n_cis_snps != 0:
            params.append({"params": self.SNP_gene_effect})

        optim_vae = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.hparams.adversary_learning_rate,
        )

        return optim_vae, optim_adversary


class LIVI_cis_Normal(LIVI):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_DxC_factors: int,
        n_persistent_factors: int,
        n_cis_snps: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        cell_state_cis: bool = True,
        warmup_epochs_vae: int = 60,
        warmup_epochs_G: int = 0,
        train_epochs_adversary: int = 20,
        covariates_dims: Optional[List[int]] = None,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        adversary_weight: float = 1.0,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
        initialise_training_mode: bool = True,
    ):

        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            n_cis_snps=n_cis_snps,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            cell_state_cis=cell_state_cis,
            warmup_epochs_vae=warmup_epochs_vae,
            warmup_epochs_G=warmup_epochs_G,
            covariates_dims=covariates_dims,
            l1_weight=l1_weight,
            A_weight=A_weight,
            batch_norm_decoder=batch_norm_decoder,
            device=device,
            genetics_seed=genetics_seed,
            initialise_training_mode=initialise_training_mode,
        )

        self.decoder = LIVI_Decoder_Normal(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_DxC_factors=n_DxC_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_DxC=self.train_DxC_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_generator=self.G_gen,
        )

        if initialise_training_mode:
            self.initialise_model()

    def freeze_vae(self, mode: bool):
        """Freezes VAE and covariate embeddings parameters, after the number of VAE and
        discriminator warm-up epochs has been completed."""

        self.frozen = mode
        if mode:
            # freeze VAE etc.
            self.encoder.requires_grad_(False)
            self.decoder.mean.requires_grad_(False)
            # # Retrain total_count
            self.decoder.log_scale.requires_grad_(False)
            # freeze covariate embeddings, if applicable
            if self.covariate_effect is not None:
                self.covariate_effect.requires_grad_(False)
        else:
            # train VAE
            self.encoder.requires_grad_(True)
            self.decoder.mean.requires_grad_(True)
            self.decoder.log_scale.requires_grad_(True)
            # train covariate embeddings, if applicable
            if self.covariate_effect is not None:
                self.covariate_effect.requires_grad_(True)
            # freeze genetic embeddings
            self.set_train_DxC_mode(False)
            self.set_train_V_mode(False)
