import copy
import warnings
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
        backed_mode: bool = True,
    ):
        # load anndata
        if isinstance(adata, str):
            if backed_mode:
                # Load in backed mode (READ-ONLY)
                print("Loading adata in backed mode (read-only)")
                self.adata = sc.read_h5ad(adata, backed="r")
                # Verify backed mode is read-only
                if hasattr(self.adata, "file") and hasattr(self.adata.file, "mode"):
                    if "w" in self.adata.file.mode or "+" in self.adata.file.mode:
                        raise ValueError(
                            f"adata file opened in write mode! Mode: {self.adata.file.mode}"
                        )
            else:
                self.adata = sc.read_h5ad(adata)
        else:
            self.adata = adata

        if known_cis_eqtls is not None:
            if isinstance(known_cis_eqtls, str):
                known_cis_eqtls = pd.read_csv(known_cis_eqtls, sep="\t", index_col=0)
            else:
                known_cis_eqtls = known_cis_eqtls
            tmp_cis_eqtls = pd.DataFrame(index=known_cis_eqtls.index, columns=self.adata.var.index)
            # Account for cases when not all genes have a known eQTL
            for c in self.adata.var.index:
                if c in known_cis_eqtls.columns:
                    tmp_cis_eqtls[c] = known_cis_eqtls[c]
                else:
                    tmp_cis_eqtls[c] = 0
            # Convert to tensor here once, instead of in __getitem__
            self.known_cis_eqtls = torch.from_numpy(tmp_cis_eqtls.to_numpy()).to(torch.long)
        else:
            self.known_cis_eqtls = None

        if not use_size_factor and size_factor_key is not None:
            raise ValueError("Set use_size_factor = True when passing size_factor_key")
        self.use_size_factor = use_size_factor
        self.size_factor_key = size_factor_key
        self.layer_key = layer_key

        self.y_key = y_key
        self.y, self.y_index = pd.factorize(
            self.adata.obs[self.y_key], sort=False, use_na_sentinel=False
        )
        if eqtl_genotypes is not None:
            assert (
                self.known_cis_eqtls is not None
            ), "eQTL genotypes provided, but not cis-eQTL gene associations."
            if isinstance(eqtl_genotypes, str):
                GT = pd.read_csv(eqtl_genotypes, sep="\t", index_col=0)
                GT = GT.loc[self.y_index]
            else:
                GT = eqtl_genotypes
                GT = GT.loc[self.y_index]
            if GT.shape[0] == 0:
                raise ValueError(
                    "Individual IDs in the cis-eQTL genotype matrix do not match individual IDs in adata.obs."
                )
            if GT.filter(tmp_cis_eqtls.index).shape[1] == 0:
                raise ValueError(
                    "SNP IDs in the cis-eQTL genotype matrix do not match SNP IDs in cis-eQTL associations."
                )
            # Consider only absence/presence of SNP, not dosage
            GT = GT.replace(2, 1)

            #####   Optimize genotype lookup   #####
            ## Problem: Using pd.DataFrame.loc in __getitem__ to look up genotypes for each cell is slow. It's faster to have
            ## the genotypes per cell already available.
            ## WARNING: Do not try to create a full cell-level genotype matrix (cells x SNPs), as memory will explode when using
            ## thousands of SNPs and millions of cells.
            ## Instead create a mapping from cell_idx to genotype_row_idx using numpy arrays once in __init__ .
            ## This is much more memory efficient and allows for direct and fast indexing in __getitem__.

            # Step 1: Convert GT to numpy contiguous array once here
            self.GT_array = np.ascontiguousarray(GT.to_numpy(), dtype=np.float32)  # donors × SNPs
            # Step 2: Map donor ID to row idx in GT_array
            donor_to_gt_idx = {donor: i for i, donor in enumerate(GT.index)}
            # Step 3: Create cell-level genotype index array
            self.cell_to_gt_idx = np.array(
                [donor_to_gt_idx[self.y_index[y_val]] for y_val in self.y], dtype=np.int32
            )
        else:
            self.GT_array = None
            self.cell_to_gt_idx = None

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
                if hasattr(self.adata, "isbacked") and self.adata.isbacked:
                    warnings.warn(
                        "`backed = True`, but `size_factor_key = None`. Computing size factors in backed mode will result in longer runtime. When reading adata in backed mode, we recommend pre-computing size factors, storing them in adata.obs and providing them via `size_factor_key`."
                    )
                    print("\nComputing size factors in backed mode...")
                    # Compute size factors in chunks
                    n_cells = self.adata.shape[0]
                    chunk_size = 10000
                    size_factor = np.zeros(n_cells, dtype=np.float32)
                    if self.layer_key is None:
                        X = self.adata.X
                    else:
                        X = self.adata.layers[self.layer_key]

                    for start in range(0, n_cells, chunk_size):
                        end = min(start + chunk_size, n_cells)
                        chunk_sums = X[start:end].sum(1)
                        if hasattr(chunk_sums, "A1"):  # sparse.sum can return matrix objects
                            chunk_sums = chunk_sums.A1
                        size_factor[start:end] = chunk_sums
                    print("Size factors computed.\n")

                else:
                    if self.layer_key is None:
                        X = self.adata.X
                    else:
                        X = self.adata.layers[self.layer_key]
                    size_factor = X.sum(1)
                    if hasattr(size_factor, "A1"):  # sparse.sum can return matrix objects
                        size_factor = size_factor.A1
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

        # contiguous arrays make operations faster: https://medium.com/@heyamit10/understanding-numpy-ascontiguousarray-with-practical-examples-a71d639fe65a
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.strict:
            if np.any(np.not_equal(np.mod(x, 1), 0)) and self.use_size_factor:
                raise TypeError(
                    "LIVI expects raw count data as input, but non-integers were found."
                )

        ## torch.from_numpy() is faster than torch.tensor():
        # https://stackoverflow.com/questions/68183227/read-data-from-numpy-array-into-a-pytorch-tensor-without-creating-a-new-tensor
        data["x"] = torch.from_numpy(x)

        data["y"] = torch.tensor(self.y[idx], dtype=torch.long)
        if self.use_size_factor:
            data["size_factor"] = torch.from_numpy(
                np.ascontiguousarray(self.size_factor[idx].reshape(-1, 1), dtype=np.float32)
            )

        if self.covariates is not None:
            covars = []
            for indices in self.covariates.values():
                covars.append(torch.tensor(indices[idx], dtype=torch.long))
            data["covariates"] = covars
        if self.known_cis_eqtls is not None:
            data["known_cis"] = self.known_cis_eqtls  # already a tensor
        # Use pre-computed index array from __init__ for fast genotype lookup
        if self.GT_array is not None:
            # gt_indices = self.cell_to_gt_idx[idx]
            data["GT_cells"] = torch.from_numpy(self.GT_array[self.cell_to_gt_idx[idx]])

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
        strict: bool = True,
        backed_mode: bool = True,
        data_split: List[float] = [1.0],
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        device: Union[torch.device, str] = "cuda",
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
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
        self.backed_mode = backed_mode

        self.data_split = data_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory and (str(device) != "cpu")  # only pin if using GPU
        self.persistent_workers = persistent_workers and (num_workers > 0)
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None

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
            backed_mode=self.backed_mode,
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

        data_loader_kwargs.update(
            {
                "sampler": sampler,
                "batch_size": None,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor,
            }
        )
        return DataLoader(dataset, **data_loader_kwargs)

    def get_num_features(self) -> int:
        """Returns dimension of observed space."""
        return self.dataset.adata.shape[1]
