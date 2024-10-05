import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from pytorch_lightning import LightningDataModule
from scipy.sparse import issparse
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)


class LIVIDataset(Dataset):
    def __init__(
        self,
        adata: Union[str, AnnData],
        y_key: str,
        use_size_factor: bool,
        size_factor_key: Optional[str] = None,
        layer_key: Optional[str] = None,
        covariates_keys: Optional[List[str]] = None,
        known_cis_eqtls: Optional[Union[str, pd.DataFrame]] = None,
        eqtl_genotypes: Optional[Union[str, pd.DataFrame]] = None,
        strict: bool = False,
    ):
        # load anndata
        if isinstance(adata, str):
            self.adata = sc.read_h5ad(adata)
        else:
            self.adata = adata

        if known_cis_eqtls is not None:
            if isinstance(known_cis_eqtls, str):
                self.known_cis_eqtls = pd.read_csv(known_cis_eqtls, sep="\t", index_col=0)
            else:
                self.known_cis_eqtls = known_cis_eqtls
            self.known_cis_eqtls = self.known_cis_eqtls.filter(self.adata.var.index)
        else:
            self.known_cis_eqtls = None

        self.y_key = y_key

        if not use_size_factor and size_factor_key is not None:
            raise ValueError("Set use_size_factor = True when passing log_size_factor_key")
        self.use_size_factor = use_size_factor
        self.size_factor_key = size_factor_key
        self.layer_key = layer_key
        self.y, self.y_index = pd.factorize(
            self.adata.obs[self.y_key], sort=False, use_na_sentinel=False
        )
        if eqtl_genotypes is not None:
            assert (
                self.known_cis_eqtls is not None
            ), "eQTL genotypes provided, but not cis-eQTLs gene associations."
            if isinstance(eqtl_genotypes, str):
                self.GT = pd.read_csv(eqtl_genotypes, sep="\t", index_col=0)
                self.GT = self.GT.loc[self.y_index]
            else:
                self.GT = eqtl_genotypes
                self.GT = self.GT.loc[self.y_index]
            if self.GT.shape[0] == 0:
                raise ValueError(
                    "Individual IDs in the cis-eQTL genotype matrix do not match individual IDs in adata.obs."
                )
            if self.GT.filter(self.known_cis_eqtls.index).shape[1] == 0:
                raise ValueError(
                    "SNP IDs in the cis-eQTL genotype matrix do not match SNP IDs in cis-eQTL associations."
                )
            # Consider only absence/presence of SNP, not dosage ?
            self.GT = self.GT.replace(2, 1)
        else:
            self.GT = None

        self.strict = strict

        if covariates_keys is not None:
            self.covariates = {}
            for covar in covariates_keys:
                self.covariates[covar], _ = pd.factorize(
                    self.adata.obs[covar], sort=False, use_na_sentinel=False
                )
        else:
            self.covariates = None

        if self.use_size_factor:
            if self.size_factor_key is None:
                if self.layer_key is None:
                    X = self.adata.X
                else:
                    X = self.adata.layers[self.layer_key]
                size_factor = X.sum(1)
            else:
                size_factor = self.adata.obs[self.size_factor_key].to_numpy()
            self.size_factor = size_factor

    def __getitem__(self, idx: List[int]) -> Dict[str, np.ndarray]:
        """Get numpy arrays for given indices."""
        data = dict()
        if self.layer_key is None:
            x = self.adata.X
        else:
            x = self.adata.layers[self.layer_key]
        x = x[idx, :]
        if issparse(x):
            x = x.todense()
        x = np.asarray(x)
        if self.strict:
            if np.any(np.not_equal(np.mod(x, 1), 0)) and self.use_size_factor:
                raise TypeError(
                    "LIVI expects raw count data as input, but non-integers were found."
                )
        data["x"] = torch.tensor(x, dtype=torch.float)
        data["y"] = torch.tensor(self.y[idx], dtype=torch.long)
        if self.use_size_factor:
            data["size_factor"] = torch.tensor(
                self.size_factor[idx].reshape(-1, 1), dtype=torch.float
            )

        if self.covariates is not None:
            covars = []
            for indices in self.covariates.values():
                covars.append(torch.tensor(indices[idx], dtype=torch.long))
            data["covariates"] = covars
        if self.known_cis_eqtls is not None:
            data["known_cis"] = torch.from_numpy(self.known_cis_eqtls.to_numpy()).to(torch.long)
        if self.GT is not None:
            data["GT_cells"] = torch.from_numpy(
                self.GT.loc[self.y_index[self.y[idx]]].to_numpy()
            ).to(torch.float)

        return data

    def __len__(self):
        return self.adata.shape[0]


class LIVIDataModule(LightningDataModule):
    def __init__(
        self,
        adata: Union[str, AnnData],
        y_key: str,
        use_size_factor: bool,
        size_factor_key: Optional[str] = None,
        layer_key: Optional[str] = None,
        covariates_keys: Optional[List[str]] = None,
        known_cis_eqtls: Optional[Union[str, pd.DataFrame]] = None,
        eqtl_genotypes: Optional[Union[str, pd.DataFrame]] = None,
        strict: bool = False,
        data_split: List[float] = [0.8],
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        device: Union[torch.device, str] = "cpu",
        **data_loader_kwargs,
    ) -> None:
        """Initialize DataModule.

        Args:
            adata: AnnData object or path to file.
            y_key: Key in adata.obs with labels.
            use_size_factor: Whether to use size factors.
            size_factor_key: Key in adata.obs with pre-computed size factors.
            layer_key: Use data from this layer. If None, use adata.X.
            data_split: Training / validation / test split.
            ...
        """
        super().__init__()
        if len(data_split) not in [1, 2]:
            raise ValueError("data_split has to contain 1 or 2 elements.")
        if sum(data_split) > 1:
            raise ValueError("Sum of split fractions larger than 1.")
        if not use_size_factor and size_factor_key is not None:
            raise ValueError("Set use_size_factor = True when passing log_size_factor_key")

        self.adata = adata
        self.y_key = y_key
        self.use_size_factor = use_size_factor
        self.size_factor_key = size_factor_key
        self.layer_key = layer_key
        self.covariates = covariates_keys
        self.known_cis_eqtls = known_cis_eqtls
        self.eqtl_genotypes = eqtl_genotypes
        self.strict = strict

        self.data_split = data_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.device = device
        self.data_loader_kwargs = data_loader_kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        """Sets up training, validation and test."""
        self.dataset = LIVIDataset(
            adata=self.adata,
            y_key=self.y_key,
            use_size_factor=self.use_size_factor,
            size_factor_key=self.size_factor_key,
            layer_key=self.layer_key,
            covariates_keys=self.covariates,
            known_cis_eqtls=self.known_cis_eqtls,
            eqtl_genotypes=self.eqtl_genotypes,
            strict=self.strict,
        )
        lengths = self._get_splits(len(self.dataset))
        self.train, self.val, self.test = torch.utils.data.random_split(
            self.dataset, lengths, generator=torch.Generator().manual_seed(self.seed)
        )

    def _get_splits(self, len_dataset: int) -> Tuple[int, int, int]:
        """Computes split lengths for train and validation set."""
        train_len = int(len_dataset * self.data_split[0])
        if len(self.data_split) == 1:
            return train_len, len_dataset - train_len, 0

        val_len = int(len_dataset * self.data_split[1])
        return train_len, val_len, len_dataset - (train_len + val_len)

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.train)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset, shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: Optional[bool] = None) -> DataLoader:
        shuffle = self.shuffle if shuffle is None else shuffle
        if shuffle:
            base_sampler = RandomSampler(dataset)
        else:
            base_sampler = SequentialSampler(dataset)
        sampler = BatchSampler(
            sampler=base_sampler, batch_size=self.batch_size, drop_last=self.drop_last
        )
        data_loader_kwargs = copy.copy(self.data_loader_kwargs)
        data_loader_kwargs.update({"sampler": sampler, "batch_size": None})
        return DataLoader(dataset, **data_loader_kwargs)

    def get_num_features(self) -> int:
        """Returns dimension of observed space."""
        return self.dataset.adata.shape[1]
