import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from pytorch_lightning import LightningDataModule
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset, sampler


class BatchSampler(sampler.Sampler):
    """Custom torch Sampler that returns a list of indices of size batch_size.

    Original: https://github.com/scverse/scvi-tools/blob/2772a09177d4f57be7d6af655f272876af5141af/scvi/dataloaders/_ann_dataloader.py#L16

    Args:
        indices: List of indices to sample from
        batch_size: Batch size of each iteration
        shuffle: If ``True``, shuffles indices before sampling
        drop_last: If int, drops the last batch if its length is less than
            drop_last. If drop_last == True, drops last non-full batch. If
            drop_last == False, iterate over all batches.
    """

    def __init__(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        drop_last: Union[bool, int] = False,
    ):
        self.indices = indices
        self.n_obs = len(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last > batch_size:
            raise ValueError(
                "drop_last can't be greater than batch_size. "
                + f"drop_last is {drop_last} but batch_size is {batch_size}."
            )

        last_batch_len = self.n_obs % self.batch_size
        if (drop_last is True) or (last_batch_len < drop_last):
            drop_last_n = last_batch_len
        elif (drop_last is False) or (last_batch_len >= drop_last):
            drop_last_n = 0
        else:
            raise ValueError("Invalid input for drop_last param. Must be bool or int.")

        self.drop_last_n = drop_last_n

    def __iter__(self):
        if self.shuffle is True:
            idx = torch.randperm(self.n_obs).tolist()
        else:
            idx = torch.arange(self.n_obs).tolist()

        if self.drop_last_n != 0:
            idx = idx[: -self.drop_last_n]

        data_iter = iter(
            [
                self.indices[idx[i : i + self.batch_size]].tolist()
                for i in range(0, len(idx), self.batch_size)
            ]
        )
        return data_iter

    def __len__(self):
        from math import ceil

        if self.drop_last_n != 0:
            length = self.n_obs // self.batch_size
        else:
            length = ceil(self.n_obs / self.batch_size)
        return length


class LIVIDataset(Dataset):
    def __init__(
        self,
        adata: Union[str, AnnData],
        y_key: str,
        use_size_factor: bool,
        size_factor_key: Optional[str] = None,
        donor_sex_key: Optional[str] = None,
        experimental_batch_key: Optional[str] = None,
        layer_key: Optional[str] = None,
        strict: bool = False,
    ):
        # load anndata
        if isinstance(adata, str):
            self.adata = sc.read_h5ad(adata)
        else:
            self.adata = adata

        self.y_key = y_key

        if not use_size_factor and size_factor_key is not None:
            raise ValueError("Set use_size_factor = True when passing log_size_factor_key")
        self.use_size_factor = use_size_factor
        self.size_factor_key = size_factor_key
        self.layer_key = layer_key
        self.y, self.y_index = pd.factorize(self.adata.obs[self.y_key])

        self.donor_sex_key = donor_sex_key
        self.experimental_batch_key = experimental_batch_key
        self.strict = strict

        if self.donor_sex_key:
            self.dsex, self.sex_index = pd.factorize(self.adata.obs[self.donor_sex_key])
        if self.experimental_batch_key:
            self.eb, self.batch_index = pd.factorize(self.adata.obs[self.experimental_batch_key])

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
        data["x"] = torch.tensor(np.asarray(x), dtype=torch.float)
        data["y"] = torch.tensor(self.y[idx], dtype=torch.long)
        if self.use_size_factor:
            data["size_factor"] = torch.tensor(
                self.size_factor[idx].reshape(-1, 1), dtype=torch.float
            )
        if self.donor_sex_key:
            data["dsex"] = torch.tensor(self.dsex[idx], dtype=torch.long)
        if self.experimental_batch_key:
            data["eb"] = torch.tensor(self.eb[idx], dtype=torch.long)
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
        donor_sex_key: Optional[str] = None,
        experimental_batch_key: Optional[str] = None,
        layer_key: Optional[str] = None,
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
        self.donor_sex_key = donor_sex_key
        self.experimental_batch_key = experimental_batch_key
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
            donor_sex_key=self.donor_sex_key,
            experimental_batch_key=self.experimental_batch_key,
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
        sampler_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "drop_last": self.drop_last,
        }

        indices = np.arange(len(dataset))
        sampler_kwargs["indices"] = indices
        sampler = BatchSampler(**sampler_kwargs)
        data_loader_kwargs = copy.copy(self.data_loader_kwargs)
        data_loader_kwargs.update({"sampler": sampler, "batch_size": None})
        return DataLoader(dataset, **data_loader_kwargs)

    def get_num_features(self) -> int:
        """Returns dimennsion of observed space."""
        return self.dataset.adata.shape[1]
