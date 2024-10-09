from itertools import chain
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.mlp import create_mlp
from src.models.vae import Encoder, LIVI_Decoder


class LIVI(pl.LightningModule):
    """LIVI with flexibility to add only context-specific or context-specific and persistent
    genetic effects and separate decoders and impose structure in the context-specific effects."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        n_gxc_factors: int,
        n_persistent_factors: int,
        encoder_hidden_dims: List[int],
        learning_rate: float,
        warmup_epochs_vae: int = 60,
        warmup_epochs_G: int = 0,
        train_epochs_adversary: int = 30,
        covariates_dims: Optional[List[int]] = None,
        l1_weight: float = 0.001,
        l1_weight_A: float = 0.001,
        adversary_weight: float = 10,
        adversary_hidden_dims: List[int] = [256, 256],
        adversary_learning_rate: float = 1e-4,
        adversary_steps: int = 2,
        batch_norm_decoder: bool = False,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
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
        self.train_epochs_adversary = train_epochs_adversary if adversary_weight > 0 else 0
        self.pretrain_mode = True if warmup_epochs_vae > 0 else False
        self.train_V_mode = False if self.pretrain_mode or self.n_persistent_factors == 0 else True
        self.train_GxC_mode = False if self.pretrain_mode or self.n_gxc_factors == 0 else True
        # Enable checkpointing after VAE + Dis training is completed and 5 epochs after U,V,A training has started
        self.checkpointing_epoch = (
            warmup_epochs_vae + self.warmup_epochs_G + self.train_epochs_adversary + 5
        )
        self.frozen = False

        self.save_hyperparameters()

        # self.G_gen = torch.Generator(device=device)
        # self.G_gen.manual_seed(genetics_seed)

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
            # learnable factor assignment matrix A; init from Normal
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
            # nn.init.normal_(self.V_persistent.weight.data, mean=0.0, std=1.0, generator=self.G_gen)

        self.decoder = LIVI_Decoder(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=[],
            layer_norm=True,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=self.pretrain_mode,
            train_GxC=self.train_GxC_mode,
            train_V=self.train_V_mode,
            batch_norm=batch_norm_decoder,
            device=device,
            genetics_seed=self.hparams.genetics_seed,
        )

        # Covariate (e.g. experimental batch) correction per gene
        if self.hparams.covariates_dims is not None:
            self.covariate_effect = nn.Embedding(
                sum(self.hparams.covariates_dims), x_dim, device=device
            )
        else:
            self.covariate_effect = None

        self.set_pretrain_mode(self.pretrain_mode)
        self.set_train_V_mode(self.train_V_mode)
        self.set_train_GxC_mode(self.train_GxC_mode)

        self.automatic_optimization = False

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

        return x, y, covariates, size_factor

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

        log_lik = (
            self.decoder(
                z=z,
                size_factor=size_factor,
                GxC=z_interaction,
                persistent_G=self.V_persistent(y) if self.n_persistent_factors != 0 else None,
                covariate_effect=covariate_effect,
            )
            .log_prob(x)
            .mean()
        )

        return log_lik - kl_div, A

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, covariates, size_factor = self.prepare_batch(batch)
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
            elbo, A = self.compute_elbo(z_dist, x, y, covariates, size_factor)
            logs[f"{mode}/elbo"] = elbo.item()
            if not self.train_GxC_mode or self.n_gxc_factors == 0:
                l1_loss_context = torch.zeros([1], device=self.device)
                l1_loss_A = torch.zeros([1], device=self.device)
            else:
                l1_loss_context = self.hparams.l1_weight * torch.linalg.vector_norm(
                    torch.cat([p for p in self.decoder.GxC_decoder.parameters()]),
                    ord=1,
                    dim=(0, 1),
                )
                A_penalty = (A * (1 - A)).pow(2)  # forces entries of A to be towards 0 or 1
                l1_loss_A = self.hparams.l1_weight_A * A_penalty.sum()
            if not self.frozen or self.n_persistent_factors == 0:
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
            'GxC_decoder' (torch.Tensor): Gene loadings for the context-specific individual effects decoder, if applicable.
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
        }

    def set_pretrain_mode(self, mode: bool):
        """Set VAE pretrain mode.

        If True, discriminator and genetic embeddings are not optimised.
        """
        self.pretrain_mode = mode
        self.decoder.pretrain_VAE = mode
        if mode:
            # freeze the rest of the model parameters
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
        else:
            if self.n_gxc_factors != 0:
                for p in self.U_context.parameters():
                    p.requires_grad = False
                self.decoder.GxC_decoder[0].weight.requires_grad = False
                self.A.requires_grad = False

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
            # freeze covariate embeddings if applicable
            if self.covariate_effect is not None:
                for p in self.covariate_effect.parameters():
                    p.requires_grad = False
            # freeze adversary
            for p in self.adversary.parameters():
                p.requires_grad = False
        else:
            # train VAE
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.decoder.mean[0].weight.requires_grad = True
            self.decoder.log_total_count.requires_grad = True
            # train covariate embeddings if applicable
            if self.covariate_effect is not None:
                for p in self.covariate_effect.parameters():
                    p.requires_grad = True
            # train adversary
            for p in self.adversary.parameters():
                p.requires_grad = True
            # freeze genetic embeddings
            self.set_train_GxC_mode(False)
            self.set_train_V_mode(False)

    def on_train_epoch_end(self):
        """After each epoch checks whether the number of warm-up epochs for the VAE, discriminator
        and persistent G has been reached."""
        if self.current_epoch == self.hparams.warmup_epochs_vae:
            self.set_pretrain_mode(False)
            print("VAE pretraining completed.")

        if self.current_epoch == self.hparams.warmup_epochs_vae + self.train_epochs_adversary:
            self.freeze_vae(True)
            print("VAE and Adversary training completed.")
            self.set_train_V_mode(True)

        if (
            self.current_epoch
            == self.hparams.warmup_epochs_vae + self.train_epochs_adversary + self.warmup_epochs_G
        ):
            print("Pretraining completed. Start learning CxG effects.")
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

        optim_vae = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
        )
        optim_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.hparams.adversary_learning_rate,
        )

        return optim_vae, optim_adversary
