### Run:
# python src/analysis/livi_analysis.py --model_run_dir --checkpoint --adata -id --adata_layer --batch_column --sex_column --age_column -ct -GT_matrix --plink -K --known_trans_eQTLs --known_cis_eQTLs --quantile_normalise --fdr -ofp -od
###

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData
from pandas_plink import read_plink
from scipy.stats import kruskal, ks_2samp, mannwhitneyu, pearsonr, zscore
from sklearn.preprocessing import StandardScaler

from src.analysis.livi_testing import (
    run_LIVI_genetic_association_testing,
    set_up_covariates,
)
from src.analysis.plotting import (
    cell_state_factors_heatmap,
    overlap_with_known_eQTLs,
    plot_ID_similarity,
    plot_U_factor_similarity,
    visualise_cell_state_latent,
)
from src.data_modules.livi_data import LIVIDataModule
from src.models.livi import LIVI
from src.models.livi_experimental import LIVI_cis, LIVI_cis_gen


def validate_and_read_passed_args(
    args: argparse.Namespace,
) -> Tuple[
    str,
    str,
    AnnData,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Validate the passed arguments and read the corresponding files.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.

    Returns
    -------
    Tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        output_dir (str): Output directory to save the testing results.
        of_prefix (str): Output file prefix.
        adata (AnnData): AnnData object containing gene counts and cell metadata.
        GT_matrix_standardised (pd.DataFrame): Standardized genotype matrix.
        bim (pd.DataFrame): SNP information contained in the .bim file, if PLINK genotype matrix is used, otherwise None.
        kinship (pd.DataFrame): Kinship matrix if provided, otherwise None.
    """

    assert os.path.isdir(args.model_run_dir), "Model directory not found."
    assert os.path.isfile(args.adata), "AnnData file not found."

    if args.output_dir:
        output_dir = args.output_dir
        if os.path.exists(output_dir):
            pass
        else:
            os.mkdir(output_dir)
    else:
        output_dir = args.model_run_dir

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
    if args.age_column:
        assert (
            args.age_column in adata.obs.columns
        ), f"`age_column`: '{args.age_column}' not in adata.obs columns."
    if args.celltype_column:
        assert (
            args.celltype_column in adata.obs.columns
        ), f"`celltype_column`: '{args.celltype_column}' not in adata.obs columns."

    if args.plink:
        bim, fam, bed = read_plink(args.genotype_matrix, verbose=False)
        GT_matrix = pd.DataFrame(bed.compute(), index=bim.snp, columns=fam.iid)
    else:
        assert os.path.isfile(args.genotype_matrix), "Genotype matrix not found"
        _, ext = os.path.splitext(args.genotype_matrix)
        if ext not in [".tsv", ".csv"]:
            raise TypeError(
                f"Genotype matrix must be either .tsv or .csv. File format provided: {ext}. To use a PLINK matrix please use the --plink flag"
            )
        GT_matrix = pd.read_csv(
            args.genotype_matrix, index_col=0, sep="\t" if ext == ".tsv" else ","
        )
        bim = None

    GT_matrix = GT_matrix.filter(adata.obs[args.individual_column].unique())
    if GT_matrix.shape[1] == 0:
        raise ValueError(
            "Individual IDs in adata.obs do not match individual IDs in the genotype matrix."
        )

    GT_matrix_standardised = pd.DataFrame(
        StandardScaler().fit_transform(GT_matrix.T.values),
        index=GT_matrix.columns,
        columns=GT_matrix.index,
    )
    del GT_matrix

    if args.kinship:
        assert os.path.isfile(args.kinship), "Kinship matrix file not found."
        _, ext = os.path.splitext(args.kinship)
        if ext not in [".tsv", ".csv"]:
            raise TypeError(
                f"Kinship matrix must be either .tsv or .csv. File format provided: {ext}."
            )
        kinship = pd.read_csv(args.kinship, index_col=0, sep="\t" if ext == ".tsv" else ",")
        kinship = kinship.loc[
            adata.obs[args.individual_column].unique(), adata.obs[args.individual_column].unique()
        ]
        if kinship.shape[0] == 0:
            raise ValueError(
                "Individual IDs in adata.obs do not match individual IDs in the kinship matrix."
            )
    else:
        kinship = None

    SNP_colname_trans = None
    if args.known_trans_eQTLs:
        assert os.path.isfile(args.known_trans_eQTLs), "Known trans eQTLs file not found."
        _, ext = os.path.splitext(args.known_trans_eQTLs)
        known_trans_eQTLs = pd.read_csv(args.known_trans_eQTLs, sep="\t" if ext == ".tsv" else ",")
        SNP_colnames = [
            c for c in known_trans_eQTLs.columns if "SNP" in c or "snp" in c or "variant" in c
        ]
        for c in SNP_colnames:
            if (
                known_trans_eQTLs.loc[
                    known_trans_eQTLs[c].isin(GT_matrix_standardised.columns)
                ].shape[0]
                != 0
            ):
                SNP_colname_trans = c
        if SNP_colname_trans is None and c == SNP_colnames[-1]:
            raise ValueError(
                "SNP IDs in known trans eQTLs do not match SNP IDs in the genotype matrix."
            )
    else:
        known_trans_eQTLs = None
    SNP_colname_cis = None
    if args.known_cis_eQTLs:
        assert os.path.isfile(args.known_cis_eQTLs), "Known cis eQTLs file not found."
        _, ext = os.path.splitext(args.known_cis_eQTLs)
        known_cis_eQTLs = pd.read_csv(args.known_cis_eQTLs, sep="\t" if ext == ".tsv" else ",")
        SNP_colnames = [
            c for c in known_cis_eQTLs.columns if "SNP" in c or "snp" in c or "variant" in c
        ]
        for c in SNP_colnames:
            if (
                known_cis_eQTLs.loc[known_cis_eQTLs[c].isin(GT_matrix_standardised.columns)].shape[
                    0
                ]
                != 0
            ):
                SNP_colname_cis = c
        if SNP_colname_cis is None and c == SNP_colnames[-1]:
            raise ValueError(
                "SNP IDs in known cis eQTLs do not match SNP IDs in the genotype matrix."
            )
    else:
        known_cis_eQTLs = None

    if args.checkpoint == "last":
        checkpoint = "last.ckpt"
    else:
        checkpoint = [
            f for f in os.listdir(os.path.join(args.model_run_dir, "checkpoints")) if "epoch" in f
        ][0]

    LIVI_model = LIVI.load_from_checkpoint(
        os.path.join(args.model_run_dir, "checkpoints", checkpoint),
        map_location=torch.device("cpu"),
    )

    if args.output_file_prefix:
        if args.output_file_prefix == "descriptive":
            zdim = LIVI_model.z_dim
            n_gxc = LIVI_model.n_gxc_factors
            n_persistent = LIVI_model.n_persistent_factors
            encoder_h = "_".join([str(d) for d in LIVI_model.hparams.encoder_hidden_dims])
            layer_norm = LIVI_model.hparams.layer_norm
            batch_norm_dec = LIVI_model.hparams.batch_norm_decoder
            lr = LIVI_model.hparams.learning_rate
            adversary_weight = LIVI_model.hparams.adversary_weight
            adversary_h = "_".join([str(d) for d in LIVI_model.hparams.adversary_hidden_dims])
            adversary_lr = LIVI_model.hparams.adversary_learning_rate
            adversary_steps = LIVI_model.hparams.adversary_steps
            l1_weight = LIVI_model.hparams.l1_weight
            l1_weight_A = LIVI_model.hparams.l1_weight_A
            warmup_epochs_vae = LIVI_model.hparams.warmup_epochs_vae
            warmup_epochs_G = LIVI_model.warmup_epochs_G

            of_prefix = f"zdim{zdim}_{n_gxc}-U-factors_{n_persistent}-V-factors_warmup-vae-{warmup_epochs_vae}_warmup-G-{warmup_epochs_G}_adversary-weight-{adversary_weight}_l1-weight-{l1_weight}_l1-weight-A-{l1_weight_A}"
        else:
            of_prefix = args.output_file_prefix
    else:
        of_prefix = os.path.basename(args.output_dir)

    return (
        output_dir,
        of_prefix,
        LIVI_model,
        adata,
        GT_matrix_standardised,
        bim,
        kinship,
        known_trans_eQTLs,
        SNP_colname_trans,
        known_cis_eQTLs,
        SNP_colname_cis,
    )


def LIVI_inference(LIVI_model, adata, of_prefix, output_dir, args):
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

    X = (
        torch.Tensor(adata.layers[args.adata_layer].todense())
        if args.adata_layer
        else torch.Tensor(adata.X.todense())
    )
    Y, _ = pd.factorize(adata.obs[args.individual_column])
    Y = torch.Tensor(Y).to(torch.long)
    batches, _ = pd.factorize(adata.obs[args.batch_column])
    batches = torch.Tensor(batches).to(torch.int)

    livi_results = LIVI_model.predict(x=X, y=Y, exp_batch_ids=batches)

    zbase = livi_results["cell-state_latent"].detach().numpy()
    zbase = pd.DataFrame(
        zbase,
        index=adata.obs.index,
        columns=[f"Cell-state_Factor{f}" for f in range(1, int(LIVI_model.z_dim) + 1)],
    )
    try:
        zbase.to_csv(
            os.path.join(output_dir, f"{of_prefix}_cell-state_latent.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
    except OSError:
        zbase.to_csv(
            os.path.join(output_dir, "_cell-state_latent.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
        warnings.warn(
            "Could not save cell-state latent dataframe under provided filename (filename too long).\nSaved as '_cell-state_latent.tsv' instead."
        )
    try:
        visualise_cell_state_latent(
            z=zbase,
            cell_metadata=adata.obs,
            output_dir=output_dir,
            of_prefix=of_prefix,
            format="png",
            args=args,
        )
        plt.close()
    except OSError:
        visualise_cell_state_latent(
            z=zbase,
            cell_metadata=adata.obs,
            output_dir=output_dir,
            of_prefix="",
            format="png",
            args=args,
        )
        plt.close()
        warnings.warn(
            "Could not save cell-state factor UMAP under provided filename (filename too long).\nSaved with default filename instead."
        )
    try:
        cell_state_factors_heatmap(
            cell_state_factors=zbase.to_numpy(),
            cell_idx=range(adata.obs.shape[0]),
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            row_cluster=True,
            column_cluster=False,
            metric="euclidean",
            factors=None,
            z_score=None,
            color_map="vlag",
            savefig=os.path.join(output_dir, f"{of_prefix}.png"),
            return_df=False,
        )
        plt.close()
        # Z-score across celltypes
        cell_state_factors_heatmap(
            cell_state_factors=zbase.to_numpy(),
            cell_idx=range(adata.obs.shape[0]),
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            color_map="vlag",
            row_cluster=True,
            column_cluster=False,
            z_score=1,
            savefig=os.path.join(output_dir, f"{of_prefix}_Zscored.png"),
            return_df=False,
        )
        plt.close()
    except OSError:
        cell_state_factors_heatmap(
            cell_state_factors=zbase.to_numpy(),
            cell_idx=range(adata.obs.shape[0]),
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            row_cluster=True,
            column_cluster=False,
            metric="euclidean",
            factors=None,
            z_score=None,
            color_map="vlag",
            savefig=os.path.join(output_dir, ""),
            format="png",
            return_df=False,
        )
        plt.close()
        # Z-score across celltypes
        cell_state_factors_heatmap(
            cell_state_factors=zbase.to_numpy(),
            cell_idx=range(adata.obs.shape[0]),
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            row_cluster=True,
            column_cluster=False,
            metric="euclidean",
            factors=None,
            z_score=1,
            color_map="vlag",
            savefig=os.path.join(output_dir, "Zscored.png"),
            return_df=False,
        )
        plt.close()
        warnings.warn(
            "Could not save cell-state factor heatmap under provided filename (filename too long).\nSaved with default filename instead."
        )

    cell_state_decoder = livi_results["cell-state_decoder"].detach().numpy()
    cell_state_decoder = pd.DataFrame(
        cell_state_decoder, index=adata.var.index, columns=zbase.columns
    )
    try:
        cell_state_decoder.to_csv(
            os.path.join(output_dir, f"{of_prefix}_cell-state_decoder.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
    except OSError:
        cell_state_decoder.to_csv(
            os.path.join(output_dir, "_cell-state_decoder.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
        warnings.warn(
            "Could not save cell-state decoder dataframe under provided filename (filename too long).\nSaved as '_cell-state_decoder.tsv' instead."
        )

    U_context = livi_results["U_embedding"].detach().numpy()
    if args.variance_threshold:
        variable_factors = np.where(np.var(U_context, axis=0) >= args.variance_threshold)[0]
        U_context = U_context[:, variable_factors].astype(np.float32)
    else:
        U_context = U_context.astype(np.float32)

    colnames_context = (
        [f"U_Factor{f}" for f in variable_factors + 1]
        if args.variance_threshold
        else [f"U_Factor{gf}" for gf in range(1, LIVI_model.n_gxc_factors + 1)]
    )
    U_context = pd.DataFrame(U_context, index=adata.obs.index, columns=colnames_context)
    U_context = (
        U_context.merge(
            adata.obs.filter([args.individual_column]), right_index=True, left_index=True
        )
        .drop_duplicates()
        .set_index(args.individual_column)
    )
    try:
        U_context.to_csv(
            os.path.join(output_dir, f"{of_prefix}_U_embedding.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
    except OSError:
        U_context.to_csv(
            os.path.join(output_dir, "_U_embedding.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
        warnings.warn(
            "Could not save U embedding dataframe under provided filename (filename too long).\nSaved as '_U_embedding.tsv' instead."
        )

    context_decoder = livi_results["GxC_decoder"].detach().numpy()
    context_decoder = pd.DataFrame(
        context_decoder,
        index=adata.var.index,
        columns=[f"GxC_Factor{f}" for f in range(1, LIVI_model.n_gxc_factors + 1)],
    )
    try:
        context_decoder.to_csv(
            os.path.join(output_dir, f"{of_prefix}_GxC_decoder.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
    except OSError:
        context_decoder.to_csv(
            os.path.join(output_dir, "_GxC_decoder.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
        warnings.warn(
            "Could not save CxG decoder dataframe under provided filename (filename too long).\nSaved as '_CxG_decoder.tsv' instead."
        )

    if livi_results["V_embedding"] is not None:
        V_persistent = livi_results["V_embedding"].detach().numpy()
        colnames_persistent = [
            f"V_Factor{f}" for f in range(1, LIVI_model.n_persistent_factors + 1)
        ]
        V_persistent = pd.DataFrame(
            V_persistent, index=adata.obs.index, columns=colnames_persistent
        )
        V_persistent = (
            V_persistent.merge(
                adata.obs.filter([args.individual_column]), right_index=True, left_index=True
            )
            .drop_duplicates()
            .set_index(args.individual_column)
        )
        try:
            V_persistent.to_csv(
                os.path.join(output_dir, f"{of_prefix}_V_embedding.tsv"),
                sep="\t",
                header=True,
                index=True,
            )
        except OSError:
            V_persistent.to_csv(
                os.path.join(output_dir, "_V_embedding.tsv"),
                sep="\t",
                header=True,
                index=True,
            )
            warnings.warn(
                "Could not save V embedding dataframe under provided filename (filename too long).\nSaved as '_V_embedding.tsv' instead."
            )

        persistent_decoder = livi_results["V_decoder"].detach().numpy()
        persistent_decoder = pd.DataFrame(
            persistent_decoder, index=adata.var.index, columns=colnames_persistent
        )
        try:
            persistent_decoder.to_csv(
                os.path.join(output_dir, f"{of_prefix}_V_decoder.tsv"),
                sep="\t",
                header=True,
                index=True,
            )
        except OSError:
            persistent_decoder.to_csv(
                os.path.join(output_dir, "_V_decoder.tsv"),
                sep="\t",
                header=True,
                index=True,
            )
            warnings.warn(
                "Could not save V decoder dataframe under provided filename (filename too long).\nSaved as '_V_decoder.tsv' instead"
            )

    else:
        V_persistent = None
        persistent_decoder = None

    assignment_matrix = torch.sigmoid(livi_results["assignment_matrix"]).detach().numpy()

    assignment_matrix = pd.DataFrame(
        assignment_matrix, index=zbase.columns, columns=U_context.columns
    )
    try:
        assignment_matrix.to_csv(
            os.path.join(output_dir, f"{of_prefix}_factor_assignment_matrix.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
    except OSError:
        assignment_matrix.to_csv(
            os.path.join(output_dir, "_factor_assignment_matrix.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
        warnings.warn(
            "Could not save A matrix dataframe under provided filename (filename too long).\nSaved as '_factor_assignment_matrix.tsv' instead."
        )

    return (
        zbase,
        cell_state_decoder,
        U_context,
        context_decoder,
        V_persistent,
        persistent_decoder,
        assignment_matrix,
    )


def main(args):
    (
        output_dir,
        of_prefix,
        LIVI_model,
        adata,
        GT_matrix_standardised,
        bim,
        kinship,
        known_trans_eQTLs,
        SNP_colname_trans,
        known_cis_eQTLs,
        SNP_colname_cis,
    ) = validate_and_read_passed_args(args)

    print("\n-------- Performing inference --------\n")

    (
        zbase,
        cell_state_decoder,
        U_context,
        context_decoder,
        V_persistent,
        persistent_decoder,
        A,
    ) = LIVI_inference(LIVI_model, adata, of_prefix, output_dir, args)

    print("\n-------- Running genetic association testing --------\n")

    covariates = set_up_covariates(args, adata.obs)

    associations = run_LIVI_genetic_association_testing(
        U_context=U_context,
        V_persistent=V_persistent,
        GT_matrix=GT_matrix_standardised,
        bim=bim,
        Kinship=kinship,
        output_dir=output_dir,
        output_file_prefix=of_prefix,
        covariates=covariates,
        quantile_norm=args.quantile_normalise,
        variance_threshold=args.variance_threshold,
        variable_factors=args.variable_factors,
        qval_threshold=(
            args.multiple_testing_threshold if args.multiple_testing_threshold else None
        ),
        return_associations=True,
    )

    associations_CxG = associations[0] if isinstance(associations, tuple) else associations
    associations_V = associations[1] if isinstance(associations, tuple) else None

    ## Exceptions for too-long filenames
    try:
        plot_U_factor_similarity(
            U=U_context,
            associated_factors=associations_CxG.Factor.unique(),
            A=A,
            assign_to_celltypes=True,
            cell_state_factors=zbase,
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            savefig=os.path.join(output_dir, of_prefix),
        )
    except OSError:
        plot_U_factor_similarity(
            U=U_context,
            associated_factors=associations_CxG.Factor.unique(),
            A=A,
            assign_to_celltypes=True,
            cell_state_factors=zbase,
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            savefig=os.path.join(output_dir, ""),
        )
        warnings.warn(
            "Could not save U factor similarity plot under provided filename (filename too long).\nSaved with default filename instead."
        )

    try:
        plot_ID_similarity(
            U=U_context,
            associated_factors=associations_CxG.Factor.unique(),
            savefig=os.path.join(output_dir, of_prefix),
        )
    except OSError:
        plot_ID_similarity(
            U=U_context,
            associated_factors=associations_CxG.Factor.unique(),
            savefig=os.path.join(output_dir, ""),
        )
        warnings.warn(
            "Could not save individual similarity plot under provided filename (filename too long).\nSaved with default filename instead."
        )

    try:
        overlap_with_known_eQTLs(
            known_trans_eQTLs=known_trans_eQTLs,
            SNP_colname_trans=SNP_colname_trans,
            CxG_effects_LIVI=associations_CxG,
            factor_assignment_matrix=A,
            known_cis_eQTLs=known_cis_eQTLs,
            SNP_colname_cis=SNP_colname_cis,
            persistent_effects_LIVI=associations_V,
            savefig=os.path.join(output_dir, of_prefix),
            format=None,
        )
    except OSError as err:
        overlap_with_known_eQTLs(
            known_trans_eQTLs=known_trans_eQTLs,
            SNP_colname_trans=SNP_colname_trans,
            CxG_effects_LIVI=associations_CxG,
            factor_assignment_matrix=A,
            known_cis_eQTLs=known_cis_eQTLs,
            SNP_colname_cis=SNP_colname_cis,
            persistent_effects_LIVI=associations_V,
            savefig=os.path.join(output_dir, ""),
            format=None,
        )
        warnings.warn(
            "Could not save overlap with known eQTLs plots under provided filename (filename too long).\nSaved with default filename instead."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_run_dir",
        type=str,
        required=True,
        help="Absolute path of the directory containing the config files and checkpoints of the LIVI model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        choices=["best", "last"],
        required=True,
        help="Which checkpoint to use; either 'last' or 'best'. ",
    )
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Absolute path of the AnnData file containing the scRNA-seq data to be used for inference.",
    )
    parser.add_argument(
        "--celltype_column",
        "-ct",
        type=str,
        required=True,
        help="Column name in cell metadata (adata.obs) indicating the celltype.",
    )
    parser.add_argument(
        "--individual_column",
        "-id",
        type=str,
        required=True,
        help="Column name in cell metadata (adata.obs) indicating the individual the sample (cell) comes from.",
    )
    parser.add_argument(
        "--genotype_matrix",
        "-GT_matrix",
        type=str,
        required=True,
        help="Absolute path of the .tsv file with the genotype matrix (the SNPs to test against LIVI's individual embeddings). For PLINK files please use in addition the --plink flag.",
    )
    parser.add_argument(
        "--plink",
        action="store_true",
        default=False,
        help="If PLINK genotype files (bim, bed, fam) are provided instead of a GT matrix in .tsv format.",
    )
    parser.add_argument(
        "--kinship",
        "-K",
        type=str,
        help="Absolute path of the .tsv file with the Kinship matrix (e.g. generated with PLINK) to be used for relatedness/population structure correction during variant testing.",
    )
    parser.add_argument(
        "--quantile_normalise",
        action="store_true",
        default=False,
        help="Whether to quantile normalise LIVI's individual embeddings prior to variant association testing.",
    )
    parser.add_argument(
        "--variable_factors",
        nargs="*",
        default=None,
        help="Test only those variable factors for interaction effects (zero-based index).",
    )
    parser.add_argument(
        "--variance_threshold",
        default=None,
        type=float,
        help="Test only factors whose variance across cells is above this threshold. Ignored if `variable_factors` are provided.",
    )
    parser.add_argument(
        "--adata_layer",
        default=None,
        type=str,
        help="key in adata layers containing the raw counts. If `None`, then adata.X is used .",
    )
    parser.add_argument(
        "--batch_column",
        default=None,
        type=str,
        help="Column name in cell metadata (adata.obs) indicating the experimental batch the sample (cell) comes from.",
    )
    parser.add_argument(
        "--sex_column",
        default=None,
        type=str,
        help="Column name in cell metadata (adata.obs) indicating the sex of the individual.",
    )
    parser.add_argument(
        "--age_column",
        default=None,
        type=str,
        help="Column name in cell metadata (adata.obs) indicating the age of the individual.",
    )
    parser.add_argument(
        "--other_covars",
        nargs="*",
        default=None,
        type=str,
        help="Column names in cell metadata (adata.obs) of other individual covariates to use in the null LMM.",
    )
    parser.add_argument(
        "--multiple_testing_threshold",
        "-fdr",
        type=float,
        help="Storey q-value threshold for multiple testing correction.",
    )
    parser.add_argument(
        "--known_trans_eQTLs",
        default=None,
        type=str,
        help="Absolute path of the .tsv file with known trans eQTLs to compare against LIVI associations (SNP IDs must be the same as in the genotype matrix!).",
    )
    parser.add_argument(
        "--known_cis_eQTLs",
        default=None,
        type=str,
        help="Absolute path of the .tsv file with known cis eQTLs to compare against LIVI associations (SNP IDs must be the same as in the genotype matrix!).",
    )
    parser.add_argument(
        "--output_file_prefix",
        "-ofp",
        default=None,
        type=str,
        help="Common prefix of the output files.",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        default=None,
        type=str,
        help="Absolute path of the directory to save the inference results.",
    )

    args = parser.parse_args()

    main(args)
