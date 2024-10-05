"""Base class for LIVI model."""

from math import sqrt
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.distributions import RobustNormal
from src.models.components.mlp import create_mlp, init_mlp


class Encoder(nn.Module):
    """Encoder module for Variational autoencoder (VAE) model."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        encoder_hidden_dims: List[int],
        layer_norm: bool,
        device: str = "cuda",
    ):
        """Initialize module.

        Args:
            x_dim: Data dimension.
            z_dim: Latent dimension.
            encoder_hidden_dims: Number of hidden nodes for each layer.
            layer_norm: Use layer norm.
        """
        super().__init__()

        self.device = device

        self.net = create_mlp(
            input_size=x_dim,
            output_size=encoder_hidden_dims[-1],
            hidden_dims=encoder_hidden_dims[:-1],
            layer_norm=layer_norm,
            device=self.device,
        )
        # map to mean and diagonal covariance of Gaussian
        self.mean = nn.Linear(encoder_hidden_dims[-1], z_dim, device=self.device)
        self.log_scale = nn.Linear(encoder_hidden_dims[-1], z_dim, device=self.device)

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        mean = self.mean(x)
        log_scale = self.log_scale(x)
        return tdist.Independent(RobustNormal(mean, log_scale.exp()), 1)


class NormalDecoder(nn.Module):
    """Decoder module for autoencoder model with Gaussian likelihood."""

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        decoder_hidden_dims: List[int],
        layer_norm: bool,
        device: str = "cuda",
    ):
        """Initialize module.

        Args:
            z_dim: Latent dimension.
            x_dim: Data dimension.
            decoder_hidden_dims: Number of hidden nodes for each layer.
            layer_norm: Use layer norm.
        """
        super().__init__()

        self.device = device

        self.mean = create_mlp(
            input_size=z_dim,
            output_size=x_dim,
            hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            device=self.device,
        )
        self.log_scale = nn.Parameter(torch.ones(1, device=self.device) * 0.1, requires_grad=True)

    def forward(self, z: torch.Tensor):
        mean = self.mean(z)
        return tdist.Independent(RobustNormal(mean, self.log_scale.exp()), 1)


class NegativeBinomialDecoder(nn.Module):
    """Decoder module autoencoder model with Negative Binomial likelihood."""

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        decoder_hidden_dims: List[int],
        layer_norm: bool,
        device: str = "cuda",
    ):
        """Initialize module.

        Args:
            z_dim: Latent dimension.
            x_dim: Data dimension.
            decoder_hidden_dims: Number of hidden nodes for each layer.
            layer_norm: Use layer norm.
        """
        super().__init__()

        self.device = device

        self.mean = create_mlp(
            input_size=z_dim,
            output_size=x_dim,
            hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            device=self.device,
        )
        self.log_total_count = torch.nn.Parameter(torch.ones(x_dim, device=self.device))

    def forward(self, z: torch.Tensor, size_factor: torch.Tensor):
        total_count = self.log_total_count.exp()
        mean = F.softmax(self.mean(z), dim=-1) * size_factor
        probs = mean / (mean + total_count)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


class NegativeBinomialDecoderBatchSex(nn.Module):
    """Decoder module with Negative Binomial likelihood and batch and sex effect correction."""

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        decoder_hidden_dims: List[int],
        layer_norm: bool,
        batch_norm: bool = False,
        device: str = "cuda",
    ):
        """Initialize module.

        Args:
            z_dim: Latent dimension.
            x_dim: Data dimension.
            decoder_hidden_dims: Number of hidden nodes for each layer.
            layer_norm: Use layer norm.
        """
        super().__init__()

        self.z_dim = z_dim
        self.x_dim = x_dim
        self.batch_norm = batch_norm
        self.device = device

        self.mean = create_mlp(
            input_size=z_dim,
            output_size=x_dim,
            hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            device=self.device,
        )
        self.log_total_count = torch.nn.Parameter(torch.ones(x_dim, device=self.device))

    def forward(
        self,
        z: torch.Tensor,
        size_factor: torch.Tensor,
        batch_effect: Union[torch.Tensor, None],
        donor_sex_effect: Union[torch.Tensor, None],
    ) -> tdist.Distribution:
        total_count = self.log_total_count.exp()
        decoder_out = self.mean(z)
        if batch_effect is not None:
            decoder_out = decoder_out + batch_effect
        if donor_sex_effect is not None:
            decoder_out = decoder_out + donor_sex_effect
        if self.batch_norm:
            decoder_out = nn.BatchNorm1d(self.x_dim, device=self.device)(decoder_out)
        mean = F.softmax(decoder_out, dim=-1) * size_factor
        probs = mean / (mean + total_count)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


class LIVI_Decoder(nn.Module):
    """Decoder module with Negative Binomial likelihood and batch and sex effect correction.

    This module encompasses separate (linear) decoders for cell-state and genetic factors.
    """

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        decoder_hidden_dims: List[int],
        layer_norm: bool,
        n_gxc_factors: int,
        n_persistent_factors: int,
        pretrain_VAE: bool = True,
        train_GxC: bool = False,
        train_V: bool = False,
        batch_norm: bool = True,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
    ):
        """Initialize module.

        Args:
            z_dim: Latent dimension.
            x_dim: Data dimension.
            decoder_hidden_dims: Number of hidden nodes for each layer.
            layer_norm: Use layer norm.
        """
        super().__init__()

        self._x_dim = x_dim
        self._z_dim = z_dim
        self._num_gxc_factors = n_gxc_factors
        self._num_persistent_factors = n_persistent_factors
        self._pretrain_vae = pretrain_VAE
        self._train_V = train_V
        self._train_GxC = train_GxC
        self.batch_norm = batch_norm
        self.device = device
        self.G_seed = genetics_seed

        self.mean = create_mlp(
            input_size=self._z_dim,
            output_size=self._x_dim,
            hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            device=self.device,
        )
        self.log_total_count = torch.nn.Parameter(torch.ones(x_dim, device=self.device))

        if self._num_gxc_factors != 0:
            self.GxC_decoder = create_mlp(
                input_size=self._num_gxc_factors,
                output_size=self._x_dim,
                hidden_dims=[],
                layer_norm=layer_norm,
                device=self.device,
            )
            if self.G_seed is not None:  # Initialise weights with the given random seed
                current_state = torch.get_rng_state()
                torch.manual_seed(self.G_seed)
                self.GxC_decoder[0].reset_parameters()
                torch.set_rng_state(current_state)

        if self._num_persistent_factors != 0:
            self.persistent_decoder = create_mlp(
                input_size=self._num_persistent_factors,
                output_size=self._x_dim,
                hidden_dims=[],
                layer_norm=layer_norm,
                device=self.device,
            )
            if self.G_seed is not None:  # Initialise weights with the given random seed
                current_state = torch.get_rng_state()
                torch.manual_seed(self.G_seed)
                self.persistent_decoder[0].reset_parameters()
                torch.set_rng_state(current_state)

        if self.batch_norm:
            self.BN_decoder = nn.BatchNorm1d(self._x_dim, device=self.device)

    @property
    def pretrain_VAE(self):
        return self._pretrain_vae

    @pretrain_VAE.setter
    def pretrain_VAE(self, mode: bool):
        self._pretrain_vae = mode

    @property
    def train_V(self):
        return self._train_V

    @train_V.setter
    def train_V(self, mode: bool):
        self._train_V = mode

    @property
    def train_GxC(self):
        return self._train_GxC

    @train_GxC.setter
    def train_GxC(self, mode: bool):
        self._train_GxC = mode

    def forward(
        self,
        z: torch.Tensor,
        size_factor: torch.Tensor,
        GxC: Optional[torch.Tensor] = None,
        persistent_G: Optional[torch.Tensor] = None,
        covariate_effect: Optional[torch.Tensor] = None,
    ) -> tdist.Distribution:
        total_count = self.log_total_count.exp()
        decoder_out = self.mean(z)
        if covariate_effect is not None:
            decoder_out = decoder_out + covariate_effect
        if (
            not self.pretrain_VAE
            and self._num_persistent_factors != 0
            and self.train_V
            and persistent_G is not None
        ):
            y_g = self.persistent_decoder(persistent_G)
            decoder_out = decoder_out + y_g
        if (
            not self.pretrain_VAE
            and self._num_gxc_factors != 0
            and self.train_GxC
            and GxC is not None
        ):
            y_gc = self.GxC_decoder(GxC)
            decoder_out = decoder_out + y_gc
        if self.batch_norm:
            decoder_out = self.BN_decoder(decoder_out)
        mean = F.softmax(decoder_out, dim=-1) * size_factor
        probs = mean / (mean + total_count)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


class LIVIcis_Decoder(LIVI_Decoder):
    """Decoder module with Negative Binomial likelihood and batch and sex effect correction.

    This module encompasses separate (linear) decoders for cell-state and genetic factors.
    """

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        decoder_hidden_dims: List[int],
        layer_norm: bool,
        n_gxc_factors: int,
        n_persistent_factors: int,
        pretrain_VAE: bool = True,
        train_V: bool = False,
        train_GxC: bool = False,
        batch_norm: bool = True,
        device: str = "cuda",
        genetics_seed: Optional[int] = None,
    ):
        super().__init__(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=pretrain_VAE,
            train_GxC=train_GxC,
            train_V=train_V,
            batch_norm=batch_norm,
            device=device,
            genetics_seed=genetics_seed,
        )

    @property
    def pretrain_VAE(self):
        return self._pretrain_vae

    @pretrain_VAE.setter
    def pretrain_VAE(self, mode: bool):
        self._pretrain_vae = mode

    @property
    def train_V(self):
        return self._train_V

    @train_V.setter
    def train_V(self, mode: bool):
        self._train_V = mode

    @property
    def train_GxC(self):
        return self._train_GxC

    @train_GxC.setter
    def train_GxC(self, mode: bool):
        self._train_GxC = mode

    def forward(
        self,
        z: torch.Tensor,
        size_factor: torch.Tensor,
        GxC: Optional[torch.Tensor] = None,
        persistent_G: Optional[torch.Tensor] = None,
        covariate_effect: Optional[torch.Tensor] = None,
        known_cis_effect: Optional[torch.Tensor] = None,
    ) -> tdist.Distribution:
        total_count = self.log_total_count.exp()
        decoder_out = self.mean(z)
        if covariate_effect is not None:
            decoder_out = decoder_out + covariate_effect
        if not self.pretrain_VAE and self.train_GxC and known_cis_effect is not None:
            decoder_out = decoder_out + known_cis_effect
        if (
            not self.pretrain_VAE
            and self._num_persistent_factors != 0
            and self.train_V
            and persistent_G is not None
        ):
            y_g = self.persistent_decoder(persistent_G)
            decoder_out = decoder_out + y_g
        if (
            not self.pretrain_VAE
            and self._num_gxc_factors != 0
            and self.train_GxC
            and GxC is not None
        ):
            y_gc = self.GxC_decoder(GxC)
            decoder_out = decoder_out + y_gc
        if self.batch_norm:
            decoder_out = self.BN_decoder(decoder_out)
        mean = F.softmax(decoder_out, dim=-1) * size_factor
        probs = mean / (mean + total_count)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


class LIVIcis_Decoder_gen(nn.Module):
    """Decoder module with Negative Binomial likelihood and batch and sex effect correction.

    This module encompasses separate (linear) decoders for cell-state and genetic factors.
    """

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        decoder_hidden_dims: List[int],
        layer_norm: bool,
        n_gxc_factors: int,
        n_persistent_factors: int,
        pretrain_VAE: bool = True,
        train_V: bool = False,
        train_GxC: bool = False,
        batch_norm: bool = True,
        device: str = "cuda",
        genetics_generator: Optional[torch.Generator] = None,
    ):
        super().__init__()

        self._x_dim = x_dim
        self._z_dim = z_dim
        self._num_gxc_factors = n_gxc_factors
        self._num_persistent_factors = n_persistent_factors
        self._pretrain_vae = pretrain_VAE
        self._train_V = train_V
        self._train_GxC = train_GxC
        self.batch_norm = batch_norm
        self.device = device
        self.genetics_generator = genetics_generator

        self.mean = create_mlp(
            input_size=self._z_dim,
            output_size=self._x_dim,
            hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            device=self.device,
        )
        self.log_total_count = torch.nn.Parameter(torch.ones(x_dim, device=self.device))

        if self._num_gxc_factors != 0:
            self.GxC_decoder = create_mlp(
                input_size=self._num_gxc_factors,
                output_size=self._x_dim,
                hidden_dims=[],
                layer_norm=layer_norm,
                device=self.device,
            )
            if self.genetics_generator is not None:  # Initialise weights with the given generator
                init_mlp(self.GxC_decoder, generator=self.genetics_generator)

        if self._num_persistent_factors != 0:
            self.persistent_decoder = create_mlp(
                input_size=self._num_persistent_factors,
                output_size=self._x_dim,
                hidden_dims=[],
                layer_norm=layer_norm,
                device=self.device,
            )
            # if self.genetics_generator is not None:  # Initialise weights with the given generator
            #     init_mlp(self.persistent_decoder, generator=self.genetics_generator)

        if self.batch_norm:
            self.BN_decoder = nn.BatchNorm1d(self._x_dim, device=self.device)

    @property
    def pretrain_VAE(self):
        return self._pretrain_vae

    @pretrain_VAE.setter
    def pretrain_VAE(self, mode: bool):
        self._pretrain_vae = mode

    @property
    def train_V(self):
        return self._train_V

    @train_V.setter
    def train_V(self, mode: bool):
        self._train_V = mode

    @property
    def train_GxC(self):
        return self._train_GxC

    @train_GxC.setter
    def train_GxC(self, mode: bool):
        self._train_GxC = mode

    def forward(
        self,
        z: torch.Tensor,
        size_factor: torch.Tensor,
        GxC: Optional[torch.Tensor] = None,
        persistent_G: Optional[torch.Tensor] = None,
        covariate_effect: Optional[torch.Tensor] = None,
        known_cis_effect: Optional[torch.Tensor] = None,
    ) -> tdist.Distribution:
        total_count = self.log_total_count.exp()
        decoder_out = self.mean(z)
        if covariate_effect is not None:
            decoder_out = decoder_out + covariate_effect
        if not self.pretrain_VAE and self.train_GxC and known_cis_effect is not None:
            decoder_out = decoder_out + known_cis_effect
        if (
            not self.pretrain_VAE
            and self._num_persistent_factors != 0
            and self.train_V
            and persistent_G is not None
        ):
            y_add = self.persistent_decoder(persistent_G)
            decoder_out = decoder_out + y_add
        if (
            not self.pretrain_VAE
            and self._num_gxc_factors != 0
            and self.train_GxC
            and GxC is not None
        ):
            y_gc = self.GxC_decoder(GxC)
            decoder_out = decoder_out + y_gc
        if self.batch_norm:
            decoder_out = self.BN_decoder(decoder_out)
        mean = F.softmax(decoder_out, dim=-1) * size_factor
        probs = mean / (mean + total_count)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


DECODER_MODELS = {
    "normal": NormalDecoder,
    "nb": NegativeBinomialDecoder,
    "nb_batch_sex": NegativeBinomialDecoderBatchSex,
}


class VAE(pl.LightningModule):
    """Variational autoencoder."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        learning_rate: float,
        layer_norm: bool,
        likelihood: str = "normal",
        device: str = "cuda",
    ):
        """Initializes VAE.

        Args:
            x_dim: Data dimension.
            z_dim: Latent dimension.
            encoder_hidden_dims: List of hidden dimensions for each encoder layer.
            decoder_hidden_dims: List of hidden dimensions for each decoder layer.
            learning_rate: Learning rate.
            layer_norm: Whether to use layer normalization.
            likelihood: Likelihood model to use, one of "normal" or "poisson".
        """

        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.likelihood = likelihood
        self.learning_rate = learning_rate

        self.encoder = Encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            layer_norm=layer_norm,
            device=device,
        )
        self.decoder = DECODER_MODELS[likelihood](
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            device=device,
        )

        self.register_buffer("z_prior_loc", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_scale", torch.ones(self.z_dim))
        self.save_hyperparameters()

    def transform_latent(self, z: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies transformation to latent variable z before decoding.

        May be overwritten by subclasses to transform z using auxiliary information.
        """
        return z

    def get_prior(self) -> tdist.Distribution:
        """Constructs zbase prior for given batch shape."""
        return tdist.Independent(tdist.Normal(self.z_prior_loc, self.z_prior_scale), 1)

    def forward(self, x: torch.Tensor):
        """Encodes data into latent space."""
        return self.encoder(x)

    def compute_elbo(
        self,
        z_dist: torch.distributions.Distribution,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        size_factor: Optional[torch.Tensor] = None,
    ):
        """Computes evidence lower bound (ELBO).

        Args:
            z_dist: Variational distribution.
            x: Input data.
            y: Optional auxiliary information.
            size_factor: Optional size factor.

        Returns:
            Mean evidence lower bound (ELBO).
        """
        z = z_dist.rsample()
        z_combined = self.transform_latent(z, y)

        kl_div = tdist.kl_divergence(z_dist, self.get_prior()).mean()
        if self.likelihood == "nb":
            log_lik = self.decoder(z_combined, size_factor).log_prob(x).mean()
        else:
            log_lik = self.decoder(z_combined).log_prob(x).mean()
        return log_lik - kl_div

    def prepare_batch(self, batch):
        x, y = batch["x"], batch["y"]
        if self.likelihood == "nb":
            size_factor = batch["size_factor"]
        else:
            size_factor = None
            x = torch.log1p(x)
        return x, y, size_factor

    def step(self, batch, batch_idx, mode="train"):
        """Performs a single training or validation step."""
        x, y, size_factor = self.prepare_batch(batch)
        loss = -self.compute_elbo(self(x), x, y, size_factor)
        log_dict = {
            f"{mode}/elbo": -loss.item(),
            "hp_metric": loss.item(),
        }
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        return self.step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        """Configures optimizer."""
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optim
