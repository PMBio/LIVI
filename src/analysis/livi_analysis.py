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
from datetime import datetime
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
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from tensorqtl import pgen

from src.analysis.livi_testing import (
    run_LIVI_genetic_association_testing,
    set_up_covariates,
)
from src.analysis.plotting import (
    cell_state_factors_heatmap,
    overlap_with_known_eQTLs,
    plot_D_factor_corr,
    plot_donor_similarity,
    plot_DxC_similarity,
    visualise_cell_state_latent,
)
from src.data_modules.livi_data import LIVIDataset
from src.models.livi import LIVI

# from src.models.livi_experimental import (
#     LIVI_cis_efficient,
#     LIVI_cis_Normal,
#     LIVI_cis_with_adversary,
# )


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
        GT_matrix (pd.DataFrame): Genotype matrix (donors x SNPs).
        variant_info (pd.DataFrame): SNP information contained in the .bim or .pvar file, if PLINK genotype matrix is used, otherwise None.
        kinship (pd.DataFrame): Kinship matrix if provided, otherwise None.
        GT_PCs (pd.DataFrame): Dataframe containing genotype principal components if provided, otherwise None.
        known_trans_eQTLs (pd.DataFrame): Dataframe containing known trans-eQTLs.
        SNP_colname_trans (str): SNP column in `known_trans_eQTLs`.
        known_cis_eQTLs (pd.DataFrame): Dataframe containing known trans-eQTLs.
        SNP_colname_cis (str):  SNP column in `known_cis_eQTLs`.
    """

    assert os.path.isdir(args.model_run_dir), "Model directory not found."
    assert os.path.isfile(args.adata), "AnnData file not found."
    assert os.path.isfile(args.covariates), "Covariates file not found."

    if args.method in ["tensorQTL", "TensorQTL", "tensorqtl"]:
        assert (
            args.genotype_pcs is not None
        ), "Genotype PCs are required when testing using TensorQTL."
        if not torch.cuda.is_available():
            warnings.warn(
                "Testing method is TensorQTL, but no GPU is available. This will slow down performance."
            )
    if args.method in ["limix", "LIMIX", "LMM"]:
        assert args.kinship is not None, "Kinship matrix required with testing using LIMIX-QTL."

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
    adata.obs[args.individual_column] = adata.obs[args.individual_column].astype(str)
    if args.batch_column:
        assert (
            args.batch_column in adata.obs.columns
        ), f"`batch_column`: '{args.batch_column}' not in adata.obs columns."
    if args.celltype_column:
        assert (
            args.celltype_column in adata.obs.columns
        ), f"`celltype_column`: '{args.celltype_column}' not in adata.obs columns."

    if args.plink and args.method in ["limix", "LIMIX", "LMM"]:
        variant_info, fam, bed = read_plink(args.genotype_matrix, verbose=False)
        GT_matrix = pd.DataFrame(
            bed.compute(), index=variant_info.snp, columns=fam.iid
        )  # SNPs x donors
    elif args.plink and args.method in ["tensorQTL", "TensorQTL", "tensorqtl"]:
        pgr = pgen.PgenReader(args.genotype_matrix)
        GT_matrix = pgr.load_genotypes()  # SNPs x donors
        variant_info = pgr.variant_df
        variant_info = variant_info.reset_index(names="variant_id")
    else:
        assert os.path.isfile(args.genotype_matrix), "Genotype matrix file not found"
        _, ext = os.path.splitext(args.genotype_matrix)
        if ext not in [".tsv", ".csv"]:
            raise TypeError(
                f"Genotype matrix must be either .tsv or .csv. File format provided: {ext}. To use a PLINK matrix please use the --plink flag"
            )
        GT_matrix = pd.read_csv(
            args.genotype_matrix, index_col=0, sep="\t" if ext == ".tsv" else ","
        )
        variant_info = None

    GT_matrix = GT_matrix.filter(adata.obs[args.individual_column].unique())
    if GT_matrix.shape[1] == 0:
        raise ValueError(
            "Individual IDs in adata.obs do not match individual IDs in the genotype matrix."
        )

    # GT_matrix_standardised = pd.DataFrame(
    #     StandardScaler().fit_transform(GT_matrix.T.values),  # donors x SNPs
    #     index=GT_matrix.columns,
    #     columns=GT_matrix.index,
    # )
    # del GT_matrix
    GT_matrix = GT_matrix.T  # donors x SNPs

    if args.kinship is not None:
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
                "Individual IDs in cell metadata do not match individual IDs in the kinship matrix."
            )
    else:
        kinship = None

    if args.genotype_pcs is not None:
        assert os.path.isfile(args.genotype_pcs), "Genotype PCs file not found."
        _, ext = os.path.splitext(args.genotype_pcs)
        if ext not in [".tsv", ".csv"]:
            raise TypeError(
                f"Genotype PCs file must be either .tsv or .csv. File format provided: {ext}."
            )
        n_gt_pcs = args.n_gt_pcs if args.n_gt_pcs else 10
        # load GT PCs; restrict to individuals in sample_df
        GT_PCs = pd.read_csv(
            args.genotype_pcs, sep="\t" if ext == ".tsv" else ",", index_col=0
        )  # donors x PCs
        GT_PCs.index = GT_PCs.index.astype(str)
        GT_PCs = GT_PCs.loc[GT_PCs.index.isin(adata.obs[args.individual_column].unique())]
        if GT_PCs.shape[0] == 0:
            raise ValueError(
                "Individual IDs in cell metadata do not match individual IDs in the genotype PCs."
            )
        # select leading components
        GT_PCs = GT_PCs.filter([f"PC{i}" for i in range(1, n_gt_pcs + 1)])
    else:
        GT_PCs = None

    SNP_colname_trans = None
    if args.known_trans_eQTLs:
        assert os.path.isfile(args.known_trans_eQTLs), "Known trans eQTLs file not found."
        _, ext = os.path.splitext(args.known_trans_eQTLs)
        known_trans_eQTLs = pd.read_csv(args.known_trans_eQTLs, sep="\t" if ext == ".tsv" else ",")
        SNP_colnames = [
            c for c in known_trans_eQTLs.columns if "SNP" in c or "snp" in c or "variant" in c
        ]
        for c in SNP_colnames:
            if known_trans_eQTLs.loc[known_trans_eQTLs[c].isin(GT_matrix.columns)].shape[0] != 0:
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
            if known_cis_eQTLs.loc[known_cis_eQTLs[c].isin(GT_matrix.columns)].shape[0] != 0:
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
            n_DxC = LIVI_model.n_DxC_factors
            n_persistent = LIVI_model.n_persistent_factors
            encoder_h = "_".join([str(d) for d in LIVI_model.hparams.encoder_hidden_dims])
            batch_norm_dec = LIVI_model.hparams.batch_norm_decoder
            lr = LIVI_model.hparams.learning_rate
            try:
                adversary_weight = LIVI_model.hparams.adversary_weight
                adversary_h = "_".join([str(d) for d in LIVI_model.hparams.adversary_hidden_dims])
                adversary_lr = LIVI_model.hparams.adversary_learning_rate
                adversary_steps = LIVI_model.hparams.adversary_steps
            except AttributeError:
                pass
            l1_weight = LIVI_model.hparams.l1_weight
            l1_weight_A = LIVI_model.hparams.A_weight
            warmup_epochs_vae = LIVI_model.hparams.warmup_epochs_vae
            warmup_epochs_G = LIVI_model.warmup_epochs_G

            of_prefix = f"zdim{zdim}_{n_DxC}-U-factors_{n_persistent}-V-factors_warmup-vae-{warmup_epochs_vae}_warmup-G-{warmup_epochs_G}_adversary-weight-{adversary_weight}_l1-weight-{l1_weight}_l1-weight-A-{l1_weight_A}"
        else:
            of_prefix = args.output_file_prefix
    else:
        of_prefix = os.path.basename(output_dir)

    return (
        output_dir,
        of_prefix,
        LIVI_model,
        adata,
        GT_matrix,
        variant_info,
        kinship,
        GT_PCs,
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
        'D_embedding' (torch.Tensor): Learned embedding of context-specific individual effects, if applicable.
        'DxC_decoder' (torch.Tensor): Gene loadings for the context-specific decoder, if applicable.
        'assignment_matrix' (torch.Tensor): Learned assignment matrix of U factors to cell-state factors.
        'V_embedding' (torch.Tensor): Learned embedding of persistent individual effects, if applicable.
        'V_decoder' (torch.Tensor): Gene loadings for the persistent decoder, if applicable.
    """

    ### Inference in batches to save memory

    dataset = LIVIDataset(
        adata=adata,
        y_key=args.individual_column,
        use_size_factor=True,
        size_factor_key=None,
        layer_key=None,
        covariates_keys=None,
        known_cis_eqtls=None,
        eqtl_genotypes=None,
        strict=False,
    )
    bs_inference = int(args.batch_size_inference)
    nbatches = adata.shape[0] // bs_inference
    batch_indices = [
        (int((current_batch - 1) * bs_inference), int(current_batch * bs_inference))
        for current_batch in range(1, nbatches + 1)
    ]
    # add last cells if any:
    if batch_indices[-1][1] < adata.shape[0]:
        batch_indices.append((batch_indices[-1][1], adata.shape[0]))

    cell_state_latent = pd.DataFrame(
        columns=[f"Cell-state_Factor{f}" for f in range(1, int(LIVI_model.z_dim) + 1)]
    )
    for i in range(len(batch_indices)):
        data = dataset.__getitem__(
            list(np.arange(batch_indices[i][0], batch_indices[i][1], 1, dtype=int))
        )
        livi_results_batch = LIVI_model.predict(x=data["x"], y=data["y"])
        c_latent_batch = livi_results_batch["cell-state_latent"].detach().numpy()
        c_latent_batch = pd.DataFrame(
            c_latent_batch,
            index=adata.obs.index[batch_indices[i][0] : batch_indices[i][1]],
            columns=[f"Cell-state_Factor{f}" for f in range(1, int(LIVI_model.z_dim) + 1)],
        )
        cell_state_latent = pd.concat([cell_state_latent, c_latent_batch], axis=0)

    Y, _ = pd.factorize(adata.obs[args.individual_column])
    Y = torch.Tensor(Y).to(torch.long)

    livi_results = LIVI_model.predict(x=data["x"], y=Y)

    try:
        cell_state_latent.to_csv(
            os.path.join(output_dir, f"{of_prefix}_cell-state_latent.tsv"),
            sep="\t",
            header=True,
            index=True,
        )
    except OSError:
        cell_state_latent.to_csv(
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
            z=cell_state_latent,
            cell_metadata=adata.obs,
            output_dir=output_dir,
            of_prefix=of_prefix,
            format="png",
            args=args,
        )
        plt.close()
    except OSError:
        visualise_cell_state_latent(
            z=cell_state_latent,
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
            cell_state_factors=cell_state_latent.to_numpy(),
            cell_idx=range(adata.obs.shape[0]),
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            row_cluster=True,
            column_cluster=False,
            metric="euclidean",
            factors=None,
            z_score=None,
            color_map="RdBu_r",
            savefig=os.path.join(output_dir, f"{of_prefix}.png"),
            return_df=False,
        )
        plt.close()
        # Z-score across celltypes
        cell_state_factors_heatmap(
            cell_state_factors=cell_state_latent.to_numpy(),
            cell_idx=range(adata.obs.shape[0]),
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            color_map="RdBu_r",
            row_cluster=True,
            column_cluster=False,
            z_score=1,
            savefig=os.path.join(output_dir, f"{of_prefix}_Zscored.png"),
            return_df=False,
        )
        plt.close()
    except OSError:
        cell_state_factors_heatmap(
            cell_state_factors=cell_state_latent.to_numpy(),
            cell_idx=range(adata.obs.shape[0]),
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            row_cluster=True,
            column_cluster=False,
            metric="euclidean",
            factors=None,
            z_score=None,
            color_map="RdBu_r",
            savefig=os.path.join(output_dir, ""),
            format="png",
            return_df=False,
        )
        plt.close()
        # Z-score across celltypes
        cell_state_factors_heatmap(
            cell_state_factors=cell_state_latent.to_numpy(),
            cell_idx=range(adata.obs.shape[0]),
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            row_cluster=True,
            column_cluster=False,
            metric="euclidean",
            factors=None,
            z_score=1,
            color_map="RdBu_r",
            savefig=os.path.join(output_dir, "Zscored.png"),
            return_df=False,
        )
        plt.close()
        warnings.warn(
            "Could not save cell-state factor heatmap under provided filename (filename too long).\nSaved with default filename instead."
        )

    cell_state_decoder = livi_results["cell-state_decoder"].detach().numpy()
    cell_state_decoder = pd.DataFrame(
        cell_state_decoder, index=adata.var.index, columns=cell_state_latent.columns
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

    if livi_results["D_embedding"] is not None:
        D_context = livi_results["D_embedding"].detach().numpy()
        if args.variance_threshold:
            variable_factors = np.where(np.var(D_context, axis=0) >= args.variance_threshold)[0]
            D_context = D_context[:, variable_factors].astype(np.float32)
        else:
            D_context = D_context.astype(np.float32)

        colnames_context = (
            [f"D_Factor{f}" for f in variable_factors + 1]
            if args.variance_threshold
            else [f"D_Factor{gf}" for gf in range(1, LIVI_model.n_DxC_factors + 1)]
        )
        D_context = pd.DataFrame(D_context, index=adata.obs.index, columns=colnames_context)
        D_context = (
            D_context.merge(
                adata.obs.filter([args.individual_column]), right_index=True, left_index=True
            )
            .drop_duplicates()
            .set_index(args.individual_column)
        )
        try:
            D_context.to_csv(
                os.path.join(output_dir, f"{of_prefix}_D_embedding.tsv"),
                sep="\t",
                header=True,
                index=True,
            )
        except OSError:
            D_context.to_csv(
                os.path.join(output_dir, "_D_embedding.tsv"),
                sep="\t",
                header=True,
                index=True,
            )
            warnings.warn(
                "Could not save D embedding dataframe under provided filename (filename too long).\nSaved as '_D_embedding.tsv' instead."
            )

        DxC_decoder = livi_results["DxC_decoder"].detach().numpy()
        DxC_decoder = pd.DataFrame(
            DxC_decoder,
            index=adata.var.index,
            columns=[f"DxC_Factor{f}" for f in range(1, LIVI_model.n_DxC_factors + 1)],
        )
        try:
            DxC_decoder.to_csv(
                os.path.join(output_dir, f"{of_prefix}_DxC_decoder.tsv"),
                sep="\t",
                header=True,
                index=True,
            )
        except OSError:
            DxC_decoder.to_csv(
                os.path.join(output_dir, "_DxC_decoder.tsv"),
                sep="\t",
                header=True,
                index=True,
            )
            warnings.warn(
                "Could not save DxC decoder dataframe under provided filename (filename too long).\nSaved as '_DxC_decoder.tsv' instead."
            )

    else:
        D_context = None
        DxC_decoder = None

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

    if livi_results["assignment_matrix"] is not None:

        assignment_matrix = torch.sigmoid(livi_results["assignment_matrix"]).detach().numpy()
        assignment_matrix = pd.DataFrame(
            assignment_matrix, index=cell_state_latent.columns, columns=D_context.columns
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
    else:
        assignment_matrix = None

    cell_state_decoder = None
    return (
        cell_state_latent,
        cell_state_decoder,
        D_context,
        DxC_decoder,
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
        GT_matrix,
        variant_info,
        kinship,
        GT_PCs,
        known_trans_eQTLs,
        SNP_colname_trans,
        known_cis_eQTLs,
        SNP_colname_cis,
    ) = validate_and_read_passed_args(args)

    print("\n-------- Performing inference --------\n")

    (
        cell_state_latent,
        cell_state_decoder,
        D_context,
        DxC_decoder,
        V_persistent,
        persistent_decoder,
        A,
    ) = LIVI_inference(LIVI_model, adata, of_prefix, output_dir, args)

    print("\n-------- Running genetic association testing --------\n")

    covariates = set_up_covariates(args, D_context)

    start = datetime.now()
    associations = run_LIVI_genetic_association_testing(
        D_context=D_context,
        V_persistent=V_persistent,
        GT_matrix=GT_matrix,
        variant_info=variant_info,
        Kinship=kinship,
        genotype_pcs=GT_PCs,
        method=args.method,
        fdr_method=args.fdr_method,
        output_dir=output_dir,
        output_file_prefix=of_prefix,
        covariates=covariates,
        quantile_norm=args.quantile_normalise,
        variance_threshold=args.variance_threshold,
        variable_factors=args.variable_factors,
        fdr_threshold=(args.fdr_threshold if args.fdr_threshold else None),
        return_associations=True,
    )

    end = datetime.now()
    duration = (end - start).seconds
    duration_minutes = duration / 60
    duration_hours = duration_minutes / 60

    with open(os.path.join(output_dir, "association_testing_execution_time.txt"), "w") as outfile:
        outfile.write(
            f"Execution time in seconds: {duration}\nExecution time in minutes: {duration_minutes}\nExecution time in hours: {duration_hours}\n"
        )

    associations_DxC = associations[0] if isinstance(associations, tuple) else associations
    associations_V = associations[1] if isinstance(associations, tuple) else None

    if D_context is not None and associations_DxC is not None and A is not None:
        ## Exceptions for too-long filenames
        try:
            plot_D_factor_corr(
                D=D_context,
                associated_factors=associations_DxC.Factor.unique(),
                A=A,
                savefig=os.path.join(output_dir, of_prefix),
                format="png",
            )
        except OSError:
            plot_D_factor_corr(
                D=D_context,
                associated_factors=associations_DxC.Factor.unique(),
                A=A,
                savefig=os.path.join(output_dir, ""),
                format="png",
            )
            warnings.warn(
                "Could not save D factor similarity plot under provided filename (filename too long).\nSaved with default filename instead."
            )

        try:
            plot_DxC_similarity(
                D=D_context,
                associated_factors=associations_DxC.Factor.unique(),
                A=A,
                cell_state_factors=cell_state_latent,
                cell_metadata=adata.obs,
                celltype_column=args.celltype_column,
                donor_column=args.individual_column,
                savefig=os.path.join(output_dir, of_prefix),
                format="png",
            )
        except OSError:
            plot_DxC_similarity(
                D=D_context,
                associated_factors=associations_DxC.Factor.unique(),
                A=A,
                cell_state_factors=cell_state_latent,
                cell_metadata=adata.obs,
                celltype_column=args.celltype_column,
                donor_column=args.individual_column,
                savefig=os.path.join(output_dir, ""),
                format="png",
            )
            warnings.warn(
                "Could not save DxC similarity plot under provided filename (filename too long).\nSaved with default filename instead."
            )
        try:
            plot_donor_similarity(
                D=D_context,
                associated_factors=associations_DxC.Factor.unique(),
                savefig=os.path.join(output_dir, of_prefix),
                format="png",
            )
        except OSError:
            plot_donor_similarity(
                D=D_context,
                associated_factors=associations_DxC.Factor.unique(),
                savefig=os.path.join(output_dir, ""),
                format="png",
            )
            warnings.warn(
                "Could not save individual similarity plot under provided filename (filename too long).\nSaved with default filename instead."
            )

        if known_trans_eQTLs is not None:
            try:
                overlap_with_known_eQTLs(
                    known_trans_eQTLs=known_trans_eQTLs,
                    SNP_colname_trans=SNP_colname_trans,
                    DxC_effects_LIVI=associations_DxC,
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
                    DxC_effects_LIVI=associations_DxC,
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
        default="best",
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
        "--batch_size_inference",
        "-bs",
        type=float,
        default=5e5,
        help="Number of samples (cells) to use per batch when performing inference. Larger numbers are recommended when memory resources are not limited.",
    )
    parser.add_argument(
        "--covariates",
        default=None,
        type=str,
        help="Absolute path of the .tsv file containing gene expression PCs aggregated at the donor level.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tensorQTL",
        choices=["tensorQTL", "LIMIX"],
        help="Whether to use LIMIX or TensorQTL for SNP association testing. LIMIX can account for repeated samples (e.g. when a donor is in multiple batches), while TensorQTL is fast.",
    )
    parser.add_argument(
        "--plink",
        action="store_true",
        default=False,
        help="If PLINK genotype files (bed, bim, fam if `method` is LIMIX, or pgen, pvar, psam if `method` is tensorQTL) are provided instead of a GT matrix in .tsv format.",
    )
    parser.add_argument(
        "--kinship",
        "-K",
        type=str,
        help="Absolute path of the .tsv file with the Kinship matrix (e.g. generated with PLINK) to be used for relatedness/population structure correction during variant testing.",
    )
    parser.add_argument(
        "--genotype_pcs",
        "-GT_pcs",
        default=None,
        type=str,
        help="Absolute path of the .tsv file with the genotype PCs (individuals x PCs) to be used for relatedness/population structure correction during variant testing.",
    )
    parser.add_argument(
        "--n_gt_pcs",
        default=None,
        type=int,
        help="Number of genotype principal components to use as covariates during SNP association testing. If omitted 10 PCs are used by default.",
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
        "--fdr_threshold",
        "-fdr",
        type=float,
        help="False discovery rate (FDR) threshold for multiple testing correction.",
    )
    parser.add_argument(
        "--fdr_method",
        default="Benjamini-Hochberg",
        type=str,
        choices=["Storey", "qvalue", "Benjamini-Hochberg", "BH", "Benjamini-Yekutieli", "BY"],
        help="False discovery rate (FDR) controlling method for multiple testing correction.",
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
