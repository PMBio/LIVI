from typing import List, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.mlp import create_mlp
from src.models.livi import LIVI
from src.models.vae import Encoder, LIVIcis_Decoder, LIVIcis_Decoder_gen


class LIVI_WO_freezing(LIVI):
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
        warmup_epochs_vae: int = 60,
        train_epochs_adversary: int = 30,
        donor_sex_dim: int = 2,
        l1_weight: float = 0.001,
        l1_weight_A: float = 0.001,
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
            z_dim=x_dim,
            y_dim=y_dim,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            exbatch_dim=exbatch_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            warmup_epochs_vae=warmup_epochs_vae,
            warmup_epochs_G=0,
            train_epochs_adversary=train_epochs_adversary,
            donor_sex_dim=donor_sex_dim,
            l1_weight=l1_weight,
            l1_weight_A=l1_weight_A,
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
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = False
                self.decoder.CxG_decoder[0].weight.requires_grad = False
                self.A.requires_grad = False
            if self.n_persistent_factors != 0:
                for p in self.V_persistent.parameters():
                    p.requires_grad = False
                self.decoder.persistent_decoder[0].weight.requires_grad = False
        else:
            # unfreeze individual embeddings
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
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
            if self.n_gxc_factors == 0:
                l1_loss_context = torch.zeros([1], device=self.device)
                l1_loss_A = torch.zeros([1], device=self.device)
            else:
                l1_loss_context = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.CxG_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
                A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
                l1_loss_A = self.hparams.l1_weight_A * A_penalty.sum()
            if self.n_persistent_factors == 0:
                l1_loss_persistent = torch.zeros([1], device=self.device)
            else:
                l1_loss_persistent = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.persistent_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
            logs[f"{mode}/L1_penalty_context"] = l1_loss_context.item()
            logs[f"{mode}/L1_penalty_A"] = l1_loss_A.item()
            logs[f"{mode}/L1_penalty_persistent"] = l1_loss_persistent.item()
            loss = (
                -elbo
                - self.hparams.adversary_weight * adversarial_loss
                + l1_loss_context
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
        warmup_epochs_vae: int = 60,
        donor_sex_dim: int = 2,
        l1_weight: float = 0.001,
        l1_weight_A: float = 0.001,
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
            z_dim=x_dim,
            y_dim=y_dim,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            exbatch_dim=exbatch_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            learning_rate=learning_rate,
            warmup_epochs_vae=warmup_epochs_vae,
            train_epochs_adversary=0,
            donor_sex_dim=donor_sex_dim,
            l1_weight=l1_weight,
            l1_weight_A=l1_weight_A,
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


class LIVI_cis_with_adversary(pl.LightningModule):
    """LIVI model accounting for cis genetic effects."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_gxc_factors: int,
        n_persistent_factors: int,
        n_cis_snps: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
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
    ):

        super().__init__()
        # x_dim=x_dim,
        # z_dim=x_dim,
        # y_dim=y_dim,
        # n_gxc_factors=n_gxc_factors,
        # n_persistent_factors=n_persistent_factors,
        # exbatch_dim=exbatch_dim,
        # encoder_hidden_dims=encoder_hidden_dims,
        # learning_rate=learning_rate,
        # warmup_epochs_vae=warmup_epochs_vae,
        # warmup_epochs_G=warmup_epochs_G,
        # train_epochs_adversary=train_epochs_adversary,
        # donor_sex_dim=donor_sex_dim,
        # l1_weight=l1_weight,
        # l1_weight_A=l1_weight_A,
        # adversary_weight=adversary_weight,
        # adversary_hidden_dims=adversary_hidden_dims,
        # adversary_learning_rate=adversary_learning_rate,
        # adversary_steps=adversary_steps,
        # batch_norm_decoder=batch_norm_decoder,
        # device=device,
        # genetics_seed=genetics_seed,

        self.save_hyperparameters()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_gxc_factors = n_gxc_factors
        self.n_persistent_factors = n_persistent_factors
        self.warmup_epochs_G = 0 if self.n_persistent_factors == 0 else warmup_epochs_G
        self.train_epochs_adversary = train_epochs_adversary if adversary_weight > 0 else 0
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_GxC_mode = False if self.pretrain_mode or self.n_gxc_factors == 0 else True
        self.checkpointing_epoch = (
            warmup_epochs_vae + self.warmup_epochs_G + self.train_epochs_adversary + 5
        )

        self.frozen = False
        self.frozen_dis = False

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=True,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))

        self.decoder = LIVIcis_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_GxC=self.train_GxC_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_seed=self.hparams.genetics_seed,
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
            torch.randn(z_dim, n_gxc_factors, device=device)  # , generator=self.G_gen)
        )
        self.U_context = nn.Embedding(y_dim, n_gxc_factors, device=device)
        if self.hparams.genetics_seed is not None:
            self.init_individual_embedding(self.U_context, self.hparams.genetics_seed)
        # nn.init.normal_(self.U_context.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

        if n_persistent_factors != 0:
            self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)
            # if self.hparams.genetics_seed is not None:
            #     self.init_individual_embedding(self.V_persistent, self.hparams.genetics_seed)

        if n_cis_snps != 0:
            # self.SNP_gene_effect = nn.Parameter(torch.randn(n_cis_snps, x_dim, device=device))
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

        self.set_pretrain_mode(self.pretrain_mode)
        self.set_train_V_mode(self.train_V_mode)
        self.set_train_GxC_mode(self.train_GxC_mode)

    def init_individual_embedding(self, embedding, seed):
        # Save the current random state
        current_state = torch.get_rng_state()
        torch.manual_seed(seed)
        # Initialize the embedding using the specified random seed
        embedding.reset_parameters()
        # Restore the original random state
        torch.set_rng_state(current_state)

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

        if self.n_gxc_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.U_context(y)
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
                GxC=z_interaction,
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
        x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask = self.prepare_batch(batch)
        optim_vae, optim_adversary = self.optimizers()

        train_adversary = batch_idx % self.adversary_steps == 0
        train_adversary = train_adversary & (not self.pretrain_mode) & (not self.frozen_dis)
        train_adversary = train_adversary * (mode == "train")

        z_dist = self(x)

        if self.pretrain_mode or self.frozen_dis:
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
            if not self.train_GxC_mode or self.n_gxc_factors == 0:
                l1_loss_context = torch.zeros([1], device=self.device)
                loss_A = torch.zeros([1], device=self.device)
            else:
                l1_loss_context = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.GxC_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
                A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
                loss_A = self.hparams.A_weight * A_penalty.sum()

            logs[f"{mode}/L1_penalty_context"] = l1_loss_context.item()
            logs[f"{mode}/penalty_A"] = loss_A.item()
            loss = (
                -elbo - self.hparams.adversary_weight * adversarial_loss + l1_loss_context + loss_A
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
            'GxC_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
            'assignment_matrix' (torch.Tensor): Learned assignment matrix of U factors to cell-state factors.
            'V_embedding' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
            'V_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
        """

        with torch.no_grad():
            self.eval()

            z = self(x).rsample()
            cell_state_decoder = self.decoder.mean[0].weight
            if self.n_gxc_factors != 0:
                U = self.U_context(y)
                GxC_decoder = self.decoder.GxC_decoder[0].weight
            else:
                U = None
                GxC_decoder = None
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
            "GxC_decoder": GxC_decoder,
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
            self.set_train_GxC_mode(False)
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

    def set_train_GxC_mode(self, mode: bool):
        self.train_GxC_mode = mode
        self.decoder.train_GxC = mode
        if mode:
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = True
                self.decoder.GxC_decoder[0].weight.requires_grad = True
                self.A.requires_grad = True
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = True
        else:
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = False
                self.decoder.GxC_decoder[0].weight.requires_grad = False
                self.A.requires_grad = False
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = False

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
            # freeze genetic embeddings
            self.set_train_GxC_mode(False)
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = False
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
            self.set_train_GxC_mode(False)
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
            self.set_train_V_mode(True)
            print("Start training V.")
            self.freeze_vae(True)  # Freeze VAE when starting training V
            print("Freeze VAE parameters.")

        if (
            self.current_epoch
            == self.hparams.warmup_epochs_vae + self.train_epochs_adversary + self.warmup_epochs_G
        ):
            # self.freeze_vae(True) # Freeze VAE after starting training V
            # self.set_train_V_mode(False) # Freeze V
            print("Pretraining completed.")
            self.freeze_adversary(True)
            print("Start learning CxG effects.")
            self.set_train_GxC_mode(True)

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
        if self.n_gxc_factors != 0:
            params.append({"params": self.U_context.parameters()})
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


class LIVI_cis(pl.LightningModule):
    """LIVI model accounting for cell-state-specific cis genetic effects."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_gxc_factors: int,
        n_persistent_factors: int,
        n_cis_snps: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 30,
        warmup_epochs_G: int = 0,
        covariates_dims: Optional[List[int]] = None,
        l1_weight: float = 0.001,
        A_weight: float = 0.001,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
    ):

        super().__init__()

        self.save_hyperparameters()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_gxc_factors = n_gxc_factors
        self.n_persistent_factors = n_persistent_factors
        self.warmup_epochs_G = 0 if self.n_persistent_factors == 0 else warmup_epochs_G
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_GxC_mode = False if self.pretrain_mode or self.n_gxc_factors == 0 else True
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

        self.decoder = LIVIcis_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_GxC=self.train_GxC_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_seed=self.hparams.genetics_seed,
        )

        self.A = nn.Parameter(
            torch.randn(z_dim, n_gxc_factors, device=device)  # , generator=self.G_gen)
        )
        self.U_context = nn.Embedding(y_dim, n_gxc_factors, device=device)
        if self.hparams.genetics_seed is not None:
            self.init_individual_embedding(self.U_context, self.hparams.genetics_seed)
        # nn.init.normal_(self.U_context.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

        if n_persistent_factors != 0:
            self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)
            # if self.hparams.genetics_seed is not None:
            #     self.init_individual_embedding(self.V_persistent, self.hparams.genetics_seed)

        if n_cis_snps != 0:
            # self.SNP_gene_effect = nn.Parameter(torch.randn(n_cis_snps, x_dim, device=device))
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

        self.set_pretrain_mode(self.pretrain_mode)
        self.set_train_V_mode(self.train_V_mode)
        self.set_train_GxC_mode(self.train_GxC_mode)

    def init_individual_embedding(self, embedding, seed):
        # Save the current random state
        current_state = torch.get_rng_state()
        torch.manual_seed(seed)
        # Initialize the embedding using the specified random seed
        embedding.reset_parameters()
        # Restore the original random state
        torch.set_rng_state(current_state)

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

        if self.n_gxc_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.U_context(y)
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
            )  # z x cells x genes (i cell, j SNP, k dim, l gene)
            celltype_known_cis_effect = torch.einsum(
                "ik,kil->il", z, celltype_known_cis_effect
            )  # cells x genes

        log_lik = (
            self.decoder(
                z=z,
                GxC=z_interaction,
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
        x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask = self.prepare_batch(batch)
        optim_vae = self.optimizers()

        z_dist = self(x)

        elbo, A = self.compute_elbo(
            z_dist, x, y, covariates, size_factor, snp_gene_mask, cell_snp_mask
        )
        logs = {f"{mode}/elbo": elbo.item()}
        if not self.train_GxC_mode or self.n_gxc_factors == 0:
            l1_loss_context = torch.zeros([1], device=self.device)
            loss_A = torch.zeros([1], device=self.device)
        else:
            l1_loss_context = self.hparams.l1_weight * torch.linalg.vector_norm(
                torch.cat([p for p in self.decoder.GxC_decoder.parameters()]),
                ord=1,
                dim=(0, 1),
            )
            A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
            loss_A = self.hparams.A_weight * A_penalty.sum()

        logs[f"{mode}/L1_penalty_context"] = l1_loss_context.item()
        logs[f"{mode}/penalty_A"] = loss_A.item()
        loss = -elbo + l1_loss_context + loss_A
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
            'GxC_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
            'assignment_matrix' (torch.Tensor): Learned assignment matrix of U factors to cell-state factors.
            'V_embedding' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
            'V_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
        """

        with torch.no_grad():
            self.eval()

            z = self(x).rsample()
            cell_state_decoder = self.decoder.mean[0].weight
            if self.n_gxc_factors != 0:
                U = self.U_context(y)
                GxC_decoder = self.decoder.GxC_decoder[0].weight
            else:
                U = None
                GxC_decoder = None
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
            "GxC_decoder": GxC_decoder,
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
            self.set_train_GxC_mode(False)
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

    def set_train_GxC_mode(self, mode: bool):
        self.train_GxC_mode = mode
        self.decoder.train_GxC = mode
        if mode:
            if self.n_gxc_factors != 0:
                self.U_context.requires_grad_(True)
                self.decoder.GxC_decoder.requires_grad_(True)
                self.A.requires_grad_(True)
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad_(True)
        else:
            if self.n_gxc_factors != 0:
                self.U_context.requires_grad_(False)
                self.decoder.GxC_decoder.requires_grad_(False)
                self.A.requires_grad_(False)
            if self.hparams.n_cis_snps != 0:
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
            self.set_train_GxC_mode(False)
            self.set_train_V_mode(False)

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE, discriminator
        and persistent G has been reached."""
        print(f"VAE frozen: {self.frozen}")
        print(f"Training V: {self.train_V_mode}")
        print(f"Training CxG: {self.train_GxC_mode}")
        print(f"GxC decoder requires grad: {self.decoder.GxC_decoder[0].weight.requires_grad}")
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
            print("Start learning CxG effects.")
            self.set_train_GxC_mode(True)

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
        if self.n_gxc_factors != 0:
            params.append({"params": self.U_context.parameters()})
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


class LIVI_cis_gen(LIVI_cis):
    """LIVI model accounting for cis genetic effects."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_gxc_factors: int,
        n_persistent_factors: int,
        n_cis_snps: int,
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
    ):

        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            n_gxc_factors=n_gxc_factors,
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
        )

        self.save_hyperparameters()

        if self.hparams.genetics_seed is not None:
            self.G_gen = torch.Generator(device=device)
            self.G_gen.manual_seed(genetics_seed)
        else:
            self.G_gen = None

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_gxc_factors = n_gxc_factors
        self.n_persistent_factors = n_persistent_factors
        self.warmup_epochs_G = 0 if self.n_persistent_factors == 0 else warmup_epochs_G
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_GxC_mode = False if self.pretrain_mode or self.n_gxc_factors == 0 else True
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

        self.decoder = LIVIcis_Decoder_gen(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_GxC=self.train_GxC_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_generator=self.G_gen,
        )

        self.A = nn.Parameter(
            torch.randn(z_dim, n_gxc_factors, device=device, generator=self.G_gen)
        )
        self.U_context = nn.Embedding(y_dim, n_gxc_factors, device=device)
        if self.hparams.genetics_seed is not None:
            nn.init.normal_(self.U_context.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

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
        self.set_train_GxC_mode(self.train_GxC_mode)


class LIVI_cis_warmup_U(pl.LightningModule):
    """LIVI model accounting for cis genetic effects."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_gxc_factors: int,
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
        # x_dim=x_dim,
        # z_dim=x_dim,
        # y_dim=y_dim,
        # n_gxc_factors=n_gxc_factors,
        # n_persistent_factors=n_persistent_factors,
        # exbatch_dim=exbatch_dim,
        # encoder_hidden_dims=encoder_hidden_dims,
        # learning_rate=learning_rate,
        # warmup_epochs_vae=warmup_epochs_vae,
        # warmup_epochs_G=warmup_epochs_G,
        # train_epochs_adversary=train_epochs_adversary,
        # donor_sex_dim=donor_sex_dim,
        # l1_weight=l1_weight,
        # l1_weight_A=l1_weight_A,
        # adversary_weight=adversary_weight,
        # adversary_hidden_dims=adversary_hidden_dims,
        # adversary_learning_rate=adversary_learning_rate,
        # adversary_steps=adversary_steps,
        # batch_norm_decoder=batch_norm_decoder,
        # device=device,
        # genetics_seed=genetics_seed,

        self.save_hyperparameters()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.n_gxc_factors = n_gxc_factors
        self.n_persistent_factors = n_persistent_factors
        self.warmup_epochs_V = 0 if self.n_persistent_factors == 0 else warmup_epochs_V
        self.warmup_epochs_U = 0 if self.n_gxc_factors == 0 else warmup_epochs_U
        self.train_epochs_adversary = train_epochs_adversary if adversary_weight > 0 else 0
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_CxG_mode = False if self.pretrain_mode or self.n_gxc_factors == 0 else True
        self.checkpointing_epoch = (
            warmup_epochs_vae
            + self.train_epochs_adversary
            + self.warmup_epochs_V
            + self.warmup_epochs_U
            + 5
        )
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

        self.decoder = LIVIcis_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_V=self.train_V_mode,
            train_CxG=self.train_CxG_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_seed=self.hparams.genetics_seed,
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
            torch.randn(z_dim, n_gxc_factors, device=device)  # , generator=self.G_gen)
        )
        self.U_context = nn.Embedding(y_dim, n_gxc_factors, device=device)
        if self.hparams.genetics_seed is not None:
            self.init_individual_embedding(self.U_context, self.hparams.genetics_seed)
        # nn.init.normal_(self.U_context.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

        if n_persistent_factors != 0:
            self.V_persistent = nn.Embedding(y_dim, n_persistent_factors, device=device)
            if self.hparams.genetics_seed is not None:
                self.init_individual_embedding(self.V_persistent, self.hparams.genetics_seed)

        if n_cis_snps != 0:
            self.SNP_gene_effect = nn.Parameter(torch.randn(n_cis_snps, x_dim, device=device))
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

    def init_individual_embedding(self, embedding, seed):
        # Save the current random state
        current_state = torch.get_rng_state()
        torch.manual_seed(seed)
        # Initialize the embedding using the specified random seed
        embedding.reset_parameters()
        # Restore the original random state
        torch.set_rng_state(current_state)

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

        if self.n_gxc_factors != 0:
            A = torch.sigmoid(self.A)
            z_interaction = (z @ A) * self.U_context(y)
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
                GxC=z_interaction,
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
            if not self.train_CxG_mode or self.n_gxc_factors == 0:
                l1_loss_context = torch.zeros([1], device=self.device)
                loss_A = torch.zeros([1], device=self.device)
            else:
                l1_loss_context = self.hparams.l1_weight * torch.linalg.vector_norm(
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

            logs[f"{mode}/L1_penalty_context"] = l1_loss_context.item()
            logs[f"{mode}/penalty_A"] = loss_A.item()
            # logs[f"{mode}/L1_penalty_persistent"] = l1_loss_persistent.item()
            loss = (
                -elbo
                - self.hparams.adversary_weight * adversarial_loss
                + l1_loss_context
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
            if self.n_gxc_factors != 0:
                U = self.U_context(y)
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
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = True
                self.decoder.CxG_decoder[0].weight.requires_grad = True
                self.A.requires_grad = True
            if self.hparams.n_cis_snps != 0:
                self.SNP_gene_effect.requires_grad = True
        else:
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
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
        if self.n_gxc_factors != 0:
            params.append({"params": self.U_context.parameters()})
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
