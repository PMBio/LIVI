from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.distributions import RobustNormal
from src.models.components.mlp import create_mlp, init_mlp


class LIVI_Decoder(nn.Module):
    """Decoder module with Negative Binomial likelihood and correction for known sample covariates
    and cis genetic effects.

    This module encompasses separate (linear) decoders for cell-state and donor factors.
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
        # self.vae_gen = torch.Generator(device=device)
        # self.vae_gen.manual_seed(50)

        self.mean = create_mlp(
            input_size=self._z_dim,
            output_size=self._x_dim,
            hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            device=self.device,
        )
        # init_mlp(self.mean, generator=self.vae_gen)
        self.log_total_count = torch.nn.Parameter(torch.ones(x_dim, device=self.device))

        if self._num_gxc_factors != 0:
            self.DxC_decoder = create_mlp(
                input_size=self._num_gxc_factors,
                output_size=self._x_dim,
                hidden_dims=[],
                layer_norm=layer_norm,
                device=self.device,
            )
            if self.genetics_generator is not None:  # Initialise weights with the given generator
                init_mlp(self.DxC_decoder, generator=self.genetics_generator)

        if self._num_persistent_factors != 0:
            self.persistent_decoder = create_mlp(
                input_size=self._num_persistent_factors,
                output_size=self._x_dim,
                hidden_dims=[],
                layer_norm=layer_norm,
                device=self.device,
            )
            # init_mlp(self.persistent_decoder, generator=self.vae_gen)
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
            y_gc = self.DxC_decoder(GxC)
            decoder_out = decoder_out + y_gc
        if self.batch_norm:
            decoder_out = self.BN_decoder(decoder_out)
        mean = F.softmax(decoder_out, dim=-1) * size_factor
        probs = mean / (mean + total_count)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


class LIVI_Decoder_wo_cis(LIVI_Decoder):
    """Decoder module with Negative Binomial likelihood and correction for known sample covariates.

    This module encompasses separate (linear) decoders for cell-state and donor factors.
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
        genetics_generator: Optional[torch.Generator] = None,
    ):
        """Initialize module.

        Args:
            z_dim: Latent dimension.
            x_dim: Data dimension.
            decoder_hidden_dims: Number of hidden nodes for each layer.
            layer_norm: Use layer norm.
        """
        super().__init__(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=pretrain_VAE,
            train_V=train_V,
            train_GxC=train_GxC,
            batch_norm=batch_norm,
            device=device,
            genetics_generator=genetics_generator,
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
            y_gc = self.DxC_decoder(GxC)
            decoder_out = decoder_out + y_gc
        if self.batch_norm:
            decoder_out = self.BN_decoder(decoder_out)
        mean = F.softmax(decoder_out, dim=-1) * size_factor
        probs = mean / (mean + total_count)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


class LIVI_Decoder_GT_PCs(LIVI_Decoder):
    """Decoder module with Negative Binomial likelihood and correction for known sample covariates,
    cis genetic effects, and global genetic effects.

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
        super().__init__(
            z_dim=z_dim,
            x_dim=x_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            n_gxc_factors=n_gxc_factors,
            n_persistent_factors=n_persistent_factors,
            pretrain_VAE=pretrain_VAE,
            train_V=train_V,
            train_GxC=train_GxC,
            batch_norm=batch_norm,
            device=device,
            genetics_generator=genetics_generator,
        )

    def forward(
        self,
        z: torch.Tensor,
        size_factor: torch.Tensor,
        GxC: Optional[torch.Tensor] = None,
        persistent_G: Optional[torch.Tensor] = None,
        covariate_effect: Optional[torch.Tensor] = None,
        known_cis_effect: Optional[torch.Tensor] = None,
        gt_pcs_effect: Optional[torch.Tensor] = None,
    ) -> tdist.Distribution:
        total_count = self.log_total_count.exp()
        decoder_out = self.mean(z)
        if covariate_effect is not None:
            decoder_out = decoder_out + covariate_effect
        if not self.pretrain_VAE and self.train_GxC and known_cis_effect is not None:
            decoder_out = decoder_out + known_cis_effect
        if not self.pretrain_VAE and self.train_GxC and gt_pcs_effect is not None:
            decoder_out = decoder_out + gt_pcs_effect
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
            y_gc = self.DxC_decoder(GxC)
            decoder_out = decoder_out + y_gc
        if self.batch_norm:
            decoder_out = self.BN_decoder(decoder_out)
        mean = F.softmax(decoder_out, dim=-1) * size_factor
        probs = mean / (mean + total_count)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(total_count).any()

        return tdist.Independent(tdist.NegativeBinomial(total_count=total_count, probs=probs), 1)


class LIVI_Decoder_Normal(nn.Module):
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
        genetics_generator: Optional[torch.Generator] = None,
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
        self.genetics_generator = genetics_generator

        self.mean = create_mlp(
            input_size=self._z_dim,
            output_size=self._x_dim,
            hidden_dims=decoder_hidden_dims,
            layer_norm=layer_norm,
            device=self.device,
        )

        self.log_scale = nn.Parameter(torch.ones(1, device=self.device) * 0.1, requires_grad=True)

        if self._num_gxc_factors != 0:
            self.DxC_decoder = create_mlp(
                input_size=self._num_gxc_factors,
                output_size=self._x_dim,
                hidden_dims=[],
                layer_norm=layer_norm,
                device=self.device,
            )
            if self.genetics_generator is not None:  # Initialise weights with the given generator
                init_mlp(self.DxC_decoder, generator=self.genetics_generator)

        if self._num_persistent_factors != 0:
            self.persistent_decoder = create_mlp(
                input_size=self._num_persistent_factors,
                output_size=self._x_dim,
                hidden_dims=[],
                layer_norm=layer_norm,
                device=self.device,
            )
            # init_mlp(self.persistent_decoder, generator=self.vae_gen)
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
            y_gc = self.DxC_decoder(GxC)
            decoder_out = decoder_out + y_gc
        if self.batch_norm:
            decoder_out = self.BN_decoder(decoder_out)

        # assert not torch.isnan(decoder_out).any()

        return tdist.Independent(RobustNormal(decoder_out, self.log_scale.exp()), 1)
