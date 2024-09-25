### Run:
# python livi_inference.py --model_version --model_run_dir --checkpoint --adata -id --adata_layer --batch_column --sex_column --output_file_prefix -od
###


import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
import torch.nn as nn
from anndata import AnnData

pl.seed_everything(32)

sys.path.append("/data/danai/scripts/LIVI/")
from src.data_modules.livi_data import LIVIDataModule
from src.models.livi import LIVI, LIVIadv, LIVIadvBatchSex
from src.models.livi2 import LIVI2


def validate_passed_args(args: argparse.Namespace) -> Tuple[AnnData, str]:
    """Validate the passed arguments.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.

    Returns
    -------
    str: Output directory to save the LIVI embeddings.
    """

    assert os.path.isdir(args.model_run_dir), "Model checkpoint directory not found."
    assert os.path.isfile(args.adata), "AnnData file not found."

    adata = sc.read_h5ad(args.adata)

    assert (
        args.individual_column in adata.obs.columns
    ), f"`individual_column`: '{args.individual_column}' not in adata.obs columns."
    if args.batch_column:
        assert (
            args.batch_column in adata.obs.columns
        ), f"`batch_column`: '{args.batch_column}' not in adata.obs columns."
    if args.sex_column:
        assert (
            args.sex_column in adata.obs.columns
        ), f"`sex_column`: '{args.sex_column}' not in adata.obs columns."
    if args.adata_layer:
        assert (
            args.adata_layer in adata.layers
        ), f"`adata_layer`: '{args.adata_layer}' not in adata.layers."

    if args.output_dir:
        output_dir = args.output_dir
        if os.path.exists(output_dir):
            pass
        else:
            os.mkdir(output_dir)
    else:
        output_dir = args.model_run_dir

    return adata, output_dir


def setup_model_and_data(
    args: argparse.Namespace,
) -> Tuple[str, str, AnnData, LIVIDataModule, Union[LIVI, LIVIadv, LIVIadvBatchSex, LIVI2]]:
    """Loads trained LIVI model from `model_run_dir` and setups LIVIDataModule from the `adata`.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.

    Returns
    -------
    Tuple[str, str, AnnData, LIVIDataModule]:
        output_dir (str): Directory to save output files.
        of_prefix (str): Output file prefix.
        adata (AnnData): AnnData object containing scRNA-seq data and metadata.
        LIVI_data (LIVIDataModule): LIVI data loader.
        LIVI_model: Trained LIVI model.
    """

    print("\n ----- Validating passed args ----- \n")
    adata, output_dir = validate_passed_args(args)

    print("\n ----- Setting up adata ----- \n")

    adata.X = adata.X.astype(np.float32)

    if args.checkpoint == "last":
        checkpoint = "last.ckpt"
    else:
        checkpoint = [
            f for f in os.listdir(os.path.join(args.model_run_dir, "checkpoints")) if "epoch" in f
        ][0]

    if args.model_version == "wo-adversary":
        LIVI_model = LIVI.load_from_checkpoint(
            os.path.join(args.model_run_dir, "checkpoints", checkpoint)
        )
        z_dim = LIVI_model.z_dim
        n_genes = LIVI_model.x_dim

        LIVI_data = LIVIDataModule(
            adata=adata,
            y_key=args.individual_column,
            use_size_factor=True,
            layer_key=args.adata_layer if args.adata_layer else None,
            data_split=[1.0],
            batch_size=adata.shape[0],
            num_workers=1,
            seed=32,
            shuffle=False,
        )

    if args.model_version == "adversarial":
        LIVI_model = LIVIadv.load_from_checkpoint(
            os.path.join(args.model_run_dir, "checkpoints", checkpoint)
        )
        l1_weight = None

        z_dim = LIVI_model.z_dim
        n_genes = LIVI_model.x_dim
        w_dis = LIVI_model.adversary_weight

        LIVI_data = LIVIDataModule(
            adata=adata,
            y_key=args.individual_column,
            use_size_factor=True,
            layer_key=args.adata_layer if args.adata_layer else None,
            data_split=[1.0],
            batch_size=adata.shape[0],
            num_workers=1,
            seed=32,
            shuffle=False,
        )

    if args.model_version == "covariates":
        assert (
            args.sex_column is not None or args.batch_column is not None
        ), "To use LIVI with covariate correction you need to provide at least one covariate."

        LIVI_model = LIVIadvBatchSex.load_from_checkpoint(
            os.path.join(args.model_run_dir, "checkpoints", checkpoint)
        )

        z_dim = LIVI_model.z_dim
        n_genes = LIVI_model.x_dim
        w_dis = LIVI_model.adversary_weight

        LIVI_data = LIVIDataModule(
            adata=adata,
            y_key=args.individual_column,
            donor_sex_key=args.sex_column if args.sex_column else None,
            experimental_batch_key=args.batch_column if args.batch_column else None,
            use_size_factor=True,
            layer_key=args.adata_layer if args.adata_layer else None,
            data_split=[1.0],
            batch_size=adata.shape[0],
            num_workers=1,
            seed=32,
            shuffle=False,
        )

    if args.model_version in ["livi2", "LIVI2", "2"]:
        assert (
            args.sex_column is not None or args.batch_column is not None
        ), "To use LIVI with covariate correction you need to provide at least one covariate."

        LIVI_model = LIVI2.load_from_checkpoint(
            os.path.join(args.model_run_dir, "checkpoints", checkpoint)
        )

        z_dim = LIVI_model.z_dim
        n_genes = LIVI_model.x_dim
        zc = LIVI_model.n_gxc_factors
        za = LIVI_model.n_persistent_factors
        w_dis = LIVI_model.hparams.adversary_weight
        l1_weight = LIVI_model.hparams.l1_weight

        LIVI_data = LIVIDataModule(
            adata=adata,
            y_key=args.individual_column,
            donor_sex_key=args.sex_column if args.sex_column else None,
            experimental_batch_key=args.batch_column if args.batch_column else None,
            use_size_factor=True,
            layer_key=args.adata_layer if args.adata_layer else None,
            data_split=[1.0],
            batch_size=adata.shape[0],
            num_workers=1,
            seed=32,
            shuffle=False,
        )

    #     num_enc_layers = len([layer for layer in LIVI_model.encoder.net.modules() if isinstance(layer, nn.Linear)])
    #     num_adv_layers = len([layer for layer in LIVI_model.adversary.modules() if isinstance(layer, nn.Linear)]) - 1
    enc_layers = "-".join([str(dim) for dim in LIVI_model.hparams.encoder_hidden_dims])
    adv_layers = "-".join([str(dim) for dim in LIVI_model.hparams.adversary_hidden_dims])

    of_prefix = (
        f"run_{os.path.basename(args.model_run_dir)}__LIVI-{args.model_version}_zdim{z_dim}_{zc}-context-factors_{za}-persistent-factors_[{enc_layers}]-encoder-hidden-layers_adversary-weight-{w_dis}_[{adv_layers}]-adversary-hidden-layers_l1-weight-{l1_weight}_{n_genes}-genes_{args.checkpoint}-checkpoint"
        if args.output_file_prefix == "descriptive"
        and args.model_version in ["livi2", "LIVI2", "2"]
        else (
            f"run_{os.path.basename(args.model_run_dir)}__LIVI-{args.model_version}_zdim{z_dim}_{za}-persistent-factors_[{enc_layers}]-encoder-hidden-layers_adversary-weight-{w_dis}_[{adv_layers}]-adversary-hidden-layers_{n_genes}-genes_{args.checkpoint}-checkpoint"
            if args.output_file_prefix == "descriptive"
            and args.model_version in ["adversarial", "covariates"]
            else (
                f"run_{os.path.basename(args.model_run_dir)}__LIVI-{args.model_version}_zdim{z_dim}_{za}-persistent-factors_[{enc_layers}]-encoder-hidden-layers_{n_genes}-genes_{args.checkpoint}-checkpoint"
                if args.output_file_prefix == "descriptive"
                and args.model_version == "wo-adversary"
                else (
                    f"run_{os.path.basename(args.model_run_dir)}"
                    if args.output_file_prefix is None
                    else args.output_file_prefix
                )
            )
        )
    )

    LIVI_data.setup()

    return output_dir, of_prefix, adata, LIVI_data, LIVI_model


def get_livi_embeddings(
    args: argparse.Namespace,
    LIVI_model: Union[LIVI, LIVIadv, LIVIadvBatchSex, LIVI2],
    adata: AnnData,
    LIVI_data: LIVIDataModule,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Get LIVI embeddings from the provided LIVI model.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.
    LIVI_model (Union[LIVI, LIVIadv, LIVIadvBatchSex, LIVI2]): Trained LIVI model.
    adata (AnnData): AnnData object containing scRNA-seq data and metadata.
    LIVI_data (LIVIDataModule): LIVI data loader.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        U_context_df (pd.DataFrame): DataFrame containing cell-state-specific genetic embeddings.
        V_persistent_df (Optional[pd.DataFrame]): DataFrame containing persistent genetic
            embeddings if applicable, otherwise None.
    """

    print("\n ----- Getting genetic embeddings ----- \n")

    for i in LIVI_data._data_loader(LIVI_data.dataset):
        X, y = i[0], i[1]

    device = next(LIVI_model.U_context.parameters()).get_device()
    device = "cuda:" + str(device) if device != -1 else "cpu"

    U_context = LIVI_model.U_context(y.to(device)).detach().cpu().numpy()

    if args.variance_threshold:
        variable_factors = np.where(np.var(U_context, axis=0) >= args.variance_threshold)[0]
        U_context = U_context[:, variable_factors].astype(np.float32)
    else:
        U_context = U_context.astype(np.float32)

    colnames_context = (
        [f"Individual_Interaction_Factor{f}" for f in variable_factors + 1]
        if args.variance_threshold
        else (
            [
                f"Individual_Interaction_Factor{f}_{gf}"
                for f in range(1, int(LIVI_model.z_dim) + 1)
                for gf in range(1, LIVI_model.n_gxc_factors + 1)
            ]
            if args.model_version
            in [
                "livi2",
                "LIVI2",
                "2",
            ]  # TO-DO: select variable factors also under the LIVI2 hierarchical model
            else [f"Individual_Interaction_Factor{f}" for f in range(1, int(LIVI_model.z_dim) + 1)]
        )
    )
    U_context_df = pd.DataFrame(U_context, index=adata.obs.index, columns=colnames_context)
    U_context_df = (
        U_context_df.merge(
            adata.obs.filter([args.individual_column]), right_index=True, left_index=True
        )
        .drop_duplicates()
        .set_index(args.individual_column)
    )

    # In case we drop the persistent embedding
    try:
        V_persistent = LIVI_model.V_persistent(y.to(device)).detach().cpu().numpy()
        V_persistent = V_persistent.astype(np.float32)
        colnames_persistent = (
            [
                f"Individual_Persistent_Factor{f}"
                for f in range(1, LIVI_model.n_persistent_factors + 1)
            ]
            if args.model_version in ["livi2", "LIVI2", "2"]
            else [f"Individual_Persistent_Factor{f}" for f in range(1, V_persistent.shape[1] + 1)]
        )
        V_persistent_df = pd.DataFrame(
            V_persistent, index=adata.obs.index, columns=colnames_persistent
        )

        V_persistent_df = (
            V_persistent_df.merge(
                adata.obs.filter([args.individual_column]), right_index=True, left_index=True
            )
            .drop_duplicates()
            .set_index(args.individual_column)
        )
    except AttributeError:
        V_persistent_df = None

    return U_context_df, V_persistent_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_version",
        help="Choose between LIVI without adversarial component ('wo-adversary'), LIVI with adversarial component ('adversarial'), LIVI with adversarial component and gene-level correction for known covariates (e.g. donor sex, technical batch) ('covariates') or LIVI version 2.0 (LIVI2)",
        type=str,
        default="covariates",
        choices=["wo-adversary", "adversarial", "covariates", "LIVI2", "livi2", "2"],
        required=True,
    )
    parser.add_argument(
        "--model_run_dir",
        help="Absolute path of the directory containing the config files and checkpoints of the LIVI model.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        help="Which checkpoint to use; either 'last' or 'best'. ",
        type=str,
        default="last",
        choices=["best", "last"],
        required=True,
    )
    parser.add_argument(
        "--adata",
        help="Absolute path of the AnnData file containing the scRNA-seq data to be used for inference.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--individual_column",
        "-id",
        help="Column name in cell metadata (adata.obs) indicating the individual the sample (cell) comes from.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--adata_layer",
        help="key in adata layers containing the raw counts. If `None`, then adata.X is used .",
        type=str,
    )
    parser.add_argument(
        "--batch_column",
        help="Column name in cell metadata (adata.obs) indicating the experimental batch the sample (cell) comes from.",
        type=str,
    )
    parser.add_argument(
        "--sex_column",
        help="Column name in cell metadata (adata.obs) indicating the sex of the individual.",
        type=str,
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        help="Consider only factors whose variance across samples is above this threshold. Ignored if `variable_factors` are provided.",
    )
    parser.add_argument(
        "--output_file_prefix",
        "-ofp",
        help="Common prefix of the output files.",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        help="Absolute path of the directory to save the inference results.",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    output_dir, of_prefix, adata, LIVI_data, LIVI_model = setup_model_and_data(args)
    U_context_df, V_persistent_df = get_livi_embeddings(args, LIVI_model, adata, LIVI_data)

    print("\n ----- Writing genetic embeddings to file ----- \n")
    U_context_df.to_csv(
        os.path.join(output_dir, f"{of_prefix}_cell-state-specific_effects.tsv"),
        sep="\t",
        header=True,
        index=True,
    )
    if V_persistent_df is not None:
        V_persistent_df.to_csv(
            os.path.join(output_dir, f"{of_prefix}_persistent_effects.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
