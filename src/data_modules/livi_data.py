import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from pytorch_lightning import LightningDataModule
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset, TensorDataset


class LIVIDataModule(LightningDataModule):
    def __init__(
        self,
        adata: Union[str, AnnData],
        y_key: str,
        use_size_factor: bool,
        donor_sex_key: Optional[str] = None,
        experimental_batch_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        layer_key: Optional[str] = None,
        data_split: List[float] = [0.8],
        batch_size: int = 128,
        num_workers: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        device: Union[torch.device, str] = "cpu",
        strict: bool = True,
        *args: Any,
        **kwargs: Any,
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
        super().__init__(*args, **kwargs)
        if len(data_split) not in [1, 2]:
            raise ValueError("data_split has to contain 1 or 2 elements.")
        if sum(data_split) > 1:
            raise ValueError("Sum of split fractions larger than 1.")

        # load anndata
        if isinstance(adata, str):
            self.adata = sc.read(adata)
        else:
            self.adata = adata

        self.y_obs_key = y_key

        if not use_size_factor and size_factor_key is not None:
            raise ValueError("Set use_size_factor = True when passing log_size_factor_key")
        self.use_size_factor = use_size_factor
        self.size_factor_key = size_factor_key
        self.donor_sex_key = donor_sex_key
        self.experimental_batch_key = experimental_batch_key
        self.layer_key = layer_key
        self.data_split = data_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.device = device
        self.strict = strict

    def setup(self, stage: Optional[str] = None) -> None:
        """Sets up training, validation and test."""
        if self.layer_key is None:
            X = self.adata.X
        else:
            X = self.adata.layers[self.layer_key]
        if issparse(X):
            X = X.todense()
        X = np.asarray(X)
        if self.strict:
            if np.any(np.not_equal(np.mod(X, 1), 0)) and self.use_size_factor:
                raise TypeError(
                    "LIVI expects raw count data as input, but non-integers were found."
                )
        y, self.y_id = pd.factorize(self.adata.obs[self.y_obs_key])
        tensors = [
            torch.tensor(X, device=self.device).float(),
            torch.tensor(y, device=self.device).long(),
        ]
        if self.donor_sex_key:
            dsex, self.dsex_index = pd.factorize(self.adata.obs[self.donor_sex_key])
            tensors.append(torch.Tensor(dsex, device=self.device).long())
        if self.experimental_batch_key:
            eb, self.eb_index = pd.factorize(self.adata.obs[self.experimental_batch_key])
            tensors.append(torch.tensor(eb, device=self.device).long())

        if self.use_size_factor:
            if self.size_factor_key is not None:
                size_factor = self.adata.obs[self.size_factor_key].to_numpy()
            else:
                # assumes X is not already log transformed
                size_factor = np.sum(X, axis=1)
            tensors.append(torch.tensor(size_factor.reshape(-1, 1), device=self.device).float())

        lengths = self._get_splits(X.shape[0])
        self.dataset = TensorDataset(*tensors)
        self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, lengths)

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
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def get_num_features(self) -> int:
        """Returns dimension of observed space."""
        return self.adata.shape[1]
