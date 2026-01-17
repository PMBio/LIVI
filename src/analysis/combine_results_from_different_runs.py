import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import os
import random
import re
import sys
import warnings
from itertools import combinations, islice, product
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib import cm, colors
from matplotlib_venn import venn3, venn3_circles
from multipy.fdr import qvalue
from pandas_plink import read_plink
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, zscore
from sklearn.preprocessing import StandardScaler
from statsmodels.stats import multitest
from tensorqtl import pgen, trans

from src.analysis._utils import assign_D_to_celltype, calculate_DxC_effect

# import src.analysis.livi_testing as testing
from src.analysis.livi_testing import (
    run_LIVI_genetic_association_testing,
    set_up_covariates,
)
from src.analysis.plotting import overlap_with_known_eQTLs, venny4py_custom_colors

np.random.seed(32)


def correlate_factors_across_runs(
    factors_across_different_runs: List[pd.DataFrame],
) -> Dict[str, np.ndarray]:
    """Compute pairwise correlations of U factors across different runs of LIVI model.

    This function normalizes the input factor matrices from each run, then computes
    the correlation between factors from different runs using the dot product method
    (equivalent to Pearson's correlation for normalized data). The results are
    returned as a dictionary where keys represent the pair of runs and values are
    the correlation matrices.

    Parameters
    ----------
        factors_across_different_runs (List[pd.DataFrame]): A list of pandas DataFrames,
            each containing factor matrices for a model run. Factors are expected to
            have columns that match a regex pattern "F|factor".

    Returns
    -------
        factor_corrs (Dict[str, np.ndarray]): A dictionary where keys are strings indicating the pair
            of runs (e.g., "corr_run0_run1"), and values are 2D NumPy arrays containing
            the correlation values between factors of the corresponding runs.
    """

    # By normalising each U embedding, calculating Pearson's corr becomes equivalent to taking the dot product between the 2 matrices, which is much more efficient computationally
    factors_across_different_runs_norm = [
        zscore(fs.filter(regex="F|factor").to_numpy().astype(float), axis=0)
        for fs in factors_across_different_runs
    ]

    factor_corrs = {}
    for i, j in combinations(range(len(factors_across_different_runs)), 2):
        # key = f"corr_run{model_runs[i]}_run{model_runs[j]}"
        key = f"corr_run{i}_run{j}"
        factor_corrs[key] = (
            np.dot(factors_across_different_runs_norm[i].T, factors_across_different_runs_norm[j])
            / factors_across_different_runs_norm[i].shape[0]
        )

    return factor_corrs


def aggregate_correlated_factors_across_runs(
    factors_across_different_runs: List[pd.DataFrame],
    corr_threshold: float,
    factor_correlations: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Aggregate correlated factors across multiple runs of LIVI into "robust factors".

    This function identifies mutually correlated factors across different runs of a
    model, aggregates their values, and returns a DataFrame of "robust factors".
    Correlations between factors are determined using a threshold and precomputed
    correlation matrices if provided; otherwise, they are computed within the function.

    Parameters
    ----------
        factors_across_different_runs (List[pd.DataFrame]): A list of pandas DataFrames,
            each containing factor matrices for different runs of a model. Factors
            are expected to have column names matching the pattern "U_Factor<n>".
        corr_threshold (float): The minimum correlation value required for factors
            to be considered correlated and eligible for aggregation.
        factor_correlations (Optional[Dict[str, np.ndarray]]): A dictionary containing
            precomputed correlation matrices for factor pairs across runs. If not provided,
            correlations will be computed using `correlate_factors_across_runs`.

    Returns
    -------
        robust_factors_df (pd.DataFrame): A DataFrame containing aggregated robust factors. Columns are
            named "Robust_U<n>" where <n> is the index of the aggregated robust factor.
    """

    if factor_correlations is None:
        factor_corrs = correlate_factors_across_runs(factors_across_different_runs)
    else:
        factor_corrs = factor_correlations

    # Create a dictionary to track correlations between factors and runs:
    # Each key is a run, and the value is another dictionary: the inner dictionary's keys are the indices fi of factors within run i,
    # and the values are dictionaries; in this 3rd level dictionary, the keys are other runs j and the values factors in that run j (fj)
    # that are correlated to factor fi in run i.
    corr_factors_all = {
        run_idx: {factor: dict() for factor in range(df.shape[1] - 1)}
        for run_idx, df in enumerate(factors_across_different_runs)
    }
    # Check and store all above-threshold correlations for each factor
    for i_j, corr_mat in factor_corrs.items():
        i, j = i_j.replace("_", "").split("run")[1:]
        i = int(i)
        j = int(j)
        for fi in range(corr_mat.shape[0]):
            for fj in range(corr_mat.shape[1]):
                if corr_mat[fi, fj] >= corr_threshold:
                    # Check if there is already another factor from run j correlated with the factor i from run i
                    if j in corr_factors_all[i][fi].keys():
                        previous_values = corr_factors_all[i][fi][j]
                        if isinstance(previous_values, list):
                            corr_factors_all[i][fi][j].append(fj)
                        else:
                            corr_factors_all[i][fi][j] = [previous_values, fj]
                    else:
                        corr_factors_all[i][fi][j] = fj

    aggregated_robust_factors = []
    # Identify mutually correlated factor groups across all runs
    # Start by run 0
    for factor_index, correlated_runs_factors in corr_factors_all[0].items():
        factor_runs = list(correlated_runs_factors.keys())
        # Only consider factors that are correlated with at least one factor in all other runs
        if not all([run in factor_runs for run in range(1, len(factors_across_different_runs))]):
            continue

        # Start by current factor
        factors2aggregate = {(0, factor_index)}
        for rj, fsj in correlated_runs_factors.items():
            key = f"corr_run0_run{rj}"
            correlation_matrix = factor_corrs[key]
            if isinstance(fsj, list):
                corr_factors_rj = list(
                    np.array(fsj)[
                        np.where(correlation_matrix[factor_index][fsj] >= corr_threshold)[0]
                    ]
                )
                for fj in corr_factors_rj:
                    factors2aggregate.add((rj, fj))
            else:
                if correlation_matrix[factor_index][fsj] >= corr_threshold:
                    factors2aggregate.add((rj, fsj))

        # Aggregate these factors
        aggregated_values = np.zeros(
            factors_across_different_runs[0].shape[0]
        )  # len N individuals
        n_factors = 0
        for r_index, f_index in factors2aggregate:
            factor_name = f"U_Factor{f_index+1}"
            aggregated_values += factors_across_different_runs[r_index][factor_name].to_numpy()
            n_factors += 1
        # Normalize by the number of aggregated factors
        factor_avg = pd.Series(
            aggregated_values / n_factors, index=factors_across_different_runs[0].index.unique()
        )
        aggregated_robust_factors.append(factor_avg)

    robust_factors_df = pd.concat(aggregated_robust_factors, axis=1)
    robust_factors_df.columns = [f"Robust_U{i+1}" for i in range(len(aggregated_robust_factors))]

    return robust_factors_df


def validate_and_read_passed_args(
    args: argparse.Namespace,
) -> Tuple[
    str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
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
        adata (anndata.AnnData): AnnData object containing gene counts and cell metadata.
        GT_matrix (pd.DataFrame): Genotype matrix (donors x SNPs).
        variant_info (pd.DataFrame): SNP information contained in the .bim or .pvar file, if PLINK genotype matrix is used, otherwise None.
        kinship (pd.DataFrame): Kinship matrix if provided, otherwise None.
        GT_PCs (pd.DataFrame): Dataframe containing genotype principal components if provided, otherwise None.
        known_trans_eQTLs (pd.DataFrame): Dataframe containing known trans-eQTLs.
        SNP_colname_trans (str): SNP column in `known_trans_eQTLs`.
        known_cis_eQTLs (pd.DataFrame): Dataframe containing known trans-eQTLs.
        SNP_colname_cis (str):  SNP column in `known_cis_eQTLs`.
    """

    assert os.path.isdir(args.results_dir), "Results directory not found."
    assert os.path.isfile(args.adata), "AnnData file not found."

    adata = sc.read_h5ad(args.adata)

    if args.output_dir:
        output_dir = args.output_dir
        if os.path.exists(output_dir):
            pass
        else:
            os.mkdir(output_dir)
    else:
        output_dir = args.results_dir

    assert (
        args.celltype_column in adata.obs.columns
    ), f"`celltype_column`: '{args.celltype_column}' not in adata.obs columns."
    if args.individual_column:
        assert (
            args.individual_column in adata.obs.columns
        ), f"`individual_column`: '{args.individual_column}' not in adata.obs columns."
        adata.obs[args.individual_column] = adata.obs[args.individual_column].astype(str)
    if args.batch_column:
        assert (
            args.batch_column in adata.obs.columns
        ), f"`batch_column`: '{args.batch_column}' not in adata.obs columns."

    if args.test_aggregated_factors:
        assert os.path.isfile(args.covariates), "Covariates file not found."

        if args.kinship is not None:
            assert os.path.isfile(args.kinship), "Kinship matrix file not found."
            _, ext = os.path.splitext(args.kinship)
            if ext not in [".tsv", ".csv"]:
                raise TypeError(
                    f"Kinship matrix must be either .tsv or .csv. File format provided: {ext}."
                )
            kinship = pd.read_csv(args.kinship, index_col=0, sep="\t" if ext == ".tsv" else ",")
            kinship.index = kinship.index.astype(str)
            kinship = kinship.loc[
                adata.obs[args.individual_column].unique(),
                adata.obs[args.individual_column].unique(),
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

        if args.plink and args.testing_method in ["limix", "LIMIX", "LMM"]:
            variant_info, fam, bed = read_plink(args.genotype_matrix, verbose=False)
            GT_matrix = pd.DataFrame(
                bed.compute(), index=variant_info.snp, columns=fam.iid
            )  # SNPs x donors
        elif args.plink and args.testing_method in ["tensorQTL", "TensorQTL", "tensorqtl"]:
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
                "Individual IDs in cell metadata do not match individual IDs in the genotype matrix."
            )

        GT_matrix = GT_matrix.T  # donors x SNPs

        if not args.prs:
            GT_matrix = pd.DataFrame(
                StandardScaler().fit_transform(GT_matrix.values),
                index=GT_matrix.index,
                columns=GT_matrix.columns,
            )

        SNP_colname_trans = None
        if args.known_trans_eQTLs:
            assert os.path.isfile(args.known_trans_eQTLs), "Known trans eQTLs file not found."
            _, ext = os.path.splitext(args.known_trans_eQTLs)
            known_trans_eQTLs = pd.read_csv(
                args.known_trans_eQTLs, sep="\t" if ext == ".tsv" else ","
            )
            SNP_colnames = [
                c for c in known_trans_eQTLs.columns if "SNP" in c or "snp" in c or "variant" in c
            ]
            for c in SNP_colnames:
                if (
                    known_trans_eQTLs.loc[known_trans_eQTLs[c].isin(GT_matrix.columns)].shape[0]
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
                if known_cis_eQTLs.loc[known_cis_eQTLs[c].isin(GT_matrix.columns)].shape[0] != 0:
                    SNP_colname_cis = c
            if SNP_colname_cis is None and c == SNP_colnames[-1]:
                raise ValueError(
                    "SNP IDs in known cis eQTLs do not match SNP IDs in the genotype matrix."
                )
        else:
            known_cis_eQTLs = None

    else:
        GT_matrix = None
        variant_info = None
        kinship = None
        GT_PCs = None
        known_trans_eQTLs = None
        SNP_colname_trans = None
        known_cis_eQTLs = None
        SNP_colname_cis = None

    return (
        output_dir,
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


def main(args):
    (
        output_dir,
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

    model_replicates = [
        d
        for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d)) and "Gseed" in d
    ]
    if args.model_prefix is not None:
        model_replicates = [d for d in model_replicates if args.model_prefix in d]

    gseed = [
        re.search("Gseed(-)?([0-9]{1,})", replicate).groups()[1]
        for replicate in model_replicates
        if re.search("Gseed(-)?([0-9]{1,})", replicate) is not None
    ]

    seeds_associations = []
    for i in range(len(model_replicates)):
        if args.prs:
            if os.path.isdir(os.path.join(args.results_dir, model_replicates[i], "PRS")):
                path2file = os.path.join(args.results_dir, model_replicates[i], "PRS")
                files_i = [
                    f for f in os.listdir(path2file) if os.path.isfile(os.path.join(path2file, f))
                ]
            else:
                path2file = os.path.join(args.results_dir, model_replicates[i])
                files_i = [
                    f for f in os.listdir(path2file) if os.path.isfile(os.path.join(path2file, f))
                ]
            D_associations = [
                re.match(
                    f"(.*PRS_{args.testing_method}_results_{args.fdr_method}*_D-embedding.tsv)", f
                )  # assumes same fdr method was used for individual run associations
                for f in files_i
                if re.match(
                    f"(.*PRS_{args.testing_method}_results_{args.fdr_method}*_D-embedding.tsv)", f
                )
                is not None
            ]
        else:
            path2file = os.path.join(args.results_dir, model_replicates[i])
            files_i = [
                f for f in os.listdir(path2file) if os.path.isfile(os.path.join(path2file, f))
            ]
            D_associations = [
                re.match(
                    f"(.*{args.testing_method}_results_{args.fdr_method}.*_D-embedding.tsv)", f
                )
                for f in files_i
                if re.match(
                    f"(.*{args.testing_method}_results_{args.fdr_method}.*_D-embedding.tsv)", f
                )
                is not None
                and "PRS" not in f
            ]
        if len(D_associations) > 0:
            if len(D_associations) > 1:
                warnings.warn(
                    f"Found more than one file with DxC associations for {model_replicates[i]}. Using file: {D_associations[0].groups()[0]}."
                )
            D_associations = D_associations[0].groups()[0]
            seed_associations = pd.read_csv(
                os.path.join(path2file, D_associations), index_col=False, sep="\t"
            )
            seed_associations = seed_associations.assign(
                random_seed=[gseed[i]] * seed_associations.shape[0]
            )
            seeds_associations.append(seed_associations)
        else:
            warnings.warn(f"No DxC associations found for {model_replicates[i]}.")

    associations_all_seeds = pd.concat(seeds_associations, ignore_index=True)

    # N of additional unique SNPs for each additional run
    n_uniq_snps_runs = {}
    for i in range(1, len(gseed)):
        n_uniq_snps_runs[f"{len(gseed)-i}_runs"] = associations_all_seeds.loc[
            associations_all_seeds.random_seed.isin(gseed[:-i])
        ].SNP_id.nunique()
    n_uniq_snps_runs[f"{len(gseed)}_runs"] = associations_all_seeds.loc[
        associations_all_seeds.random_seed.isin(gseed)
    ].SNP_id.nunique()

    n_uniq_snps_runs = pd.DataFrame.from_dict(
        n_uniq_snps_runs, orient="index", columns=["N_unique_fSNPs"]
    ).reset_index(names="N_runs")
    n_uniq_snps_runs.N_runs = pd.Categorical(n_uniq_snps_runs.N_runs).reorder_categories(
        sorted(n_uniq_snps_runs.N_runs)
    )

    sns.barplot(x=n_uniq_snps_runs.N_runs, y=n_uniq_snps_runs.N_unique_fSNPs, color="mediumblue")
    plt.savefig(
        os.path.join(output_dir, "N-unique-fSNPs_vs_N-runs.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.savefig(
        os.path.join(output_dir, "N-unique-fSNPs_vs_N-runs.pdf"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    seed_snps_dict = (
        associations_all_seeds.groupby("random_seed")
        .apply(lambda x: set(x.SNP_id.tolist()), include_groups=False)
        .to_dict()
    )
    # Find significant SNPs across all random initialisations
    intersect_snps = set(seed_snps_dict[gseed[0]])
    for gs in gseed[1:]:
        values = set(seed_snps_dict[gs])
        # Keep only elements that are also in the new set of values
        intersect_snps &= values

    print(f"Number of common fSNPs across all runs: {len(intersect_snps)}")

    # # Overlap between intersect SNPs and eQTLGen
    # plt.close()
    # v = venn3(subsets=[intersect_snps, set(known_trans_eQTLs.snp_id), set(known_cis_eQTLs.snp_id)],
    #       set_labels=("LIVI CxG intersection across runs", "$trans$-eQTLGen", "$cis$-eQTLGen"))
    # v.set_labels[0].set_text(f"LIVI CxG intersection across runs\n(n = {len(intersect_snps)})")
    # v.set_labels[1].set_text(f"$trans$-eQTLGen\n(n = {known_trans_eQTLs.snp_id.nunique()})")
    # v.set_labels[2].set_text(f"$cis$-eQTLGen\n(n= {known_cis_eQTLs.snp_id.nunique()})")
    # # Set font size for the labels
    # [l.set_fontsize(20) for l in v.set_labels]
    # for l in v.subset_labels:
    #     try:
    #         l.set_fontsize(16)
    #     except AttributeError:
    #         pass
    # for circle in v.patches:
    #     if circle is not None:  # Check if the patch exists to avoid errors
    #         circle.set_edgecolor('darkgrey')  # Set edge color if you also want to adjust the color
    #         circle.set_linestyle('dashed')
    # # plt.title("Overlap between significant SNPs\n", fontdict={"fontsize":18, "weight":"bold"})
    # plt.tight_layout()
    # filename = f"{args.model_prefix}_Intersection-fSNPs-different-seeds_vs_eQTLGen.png" if args.model_prefix is not None else "Intersection-fSNPs-different-seeds_vs_eQTLGen.png"
    # plt.savefig(os.path.join(output_dir, filename),
    #             transparent=True, dpi=300, bbox_inches="tight")

    seed_snps_dict_plot = seed_snps_dict.copy()
    if len(seed_snps_dict) > 4:
        seed_snps_dict_plot = {}
        k4 = list(islice(seed_snps_dict.keys(), 4))
        for k in k4:
            seed_snps_dict_plot[k] = seed_snps_dict[k]

    filename = (
        f"{args.model_prefix}_SNP-overlap-different-seeds"
        if args.model_prefix is not None
        else "SNP-overlap-different-seeds"
    )
    if args.prs:
        filename = filename.replace("SNP", "PRS")
    venny4py_custom_colors(
        sets=seed_snps_dict_plot,
        plot_title="SNP overlap between different random initialisations",
        out=os.path.join(output_dir, filename),
        ext=".png",
    )
    plt.close()

    intersect_snps_one = intersect_snps.copy()
    for gs in gseed:
        livi_associations = associations_all_seeds.loc[associations_all_seeds.random_seed == gs]
        median_N_DxC_snp = (
            livi_associations.loc[livi_associations.SNP_id.isin(intersect_snps)]
            .groupby("SNP_id", observed=True)
            .apply(lambda x: x.Factor.nunique(), include_groups=False)
            .median()
        )
        # For simplicity consider only SNPs that are associated with 1 U factor
        snps_one = (
            livi_associations.groupby(by="SNP_id", observed=True).apply(
                lambda x: x.Factor.nunique(), include_groups=False
            )
            == 1
        )
        snps_one = snps_one.loc[snps_one].index
        intersect_snps_one &= set(snps_one)

    print(
        f"Number of common fSNPs across all runs associated only with one D factor: {len(intersect_snps_one)}"
    )  # used to judge concordance at the single-cell level

    print(
        "------------  Assessing celltype assignment concordance between different random seeds  ------------"
    )

    seeds_D_ct = {}
    seeds_DxC_effects = []
    for i in range(len(model_replicates)):
        livi_associations = associations_all_seeds.loc[
            associations_all_seeds.random_seed == gseed[i]
        ]

        zbase = pd.read_csv(
            os.path.join(
                args.results_dir,
                model_replicates[i],
                f"{model_replicates[i]}_cell-state_latent.tsv",
            ),
            sep="\t",
            index_col=0,
        )
        seed_A = pd.read_csv(
            os.path.join(
                args.results_dir,
                model_replicates[i],
                f"{model_replicates[i]}_factor_assignment_matrix.tsv",
            ),
            index_col=0,
            sep="\t",
        )

        D_ct_dict = assign_D_to_celltype(
            cell_state_latent=zbase,
            A=seed_A,
            cell_metadata=adata.obs,
            celltype_column=args.celltype_column,
            top_one=True,
            assignment_threshold=0.8,
        )

        seeds_D_ct[gseed[i]] = D_ct_dict

        livi_associations = livi_associations.loc[
            livi_associations.SNP_id.isin(intersect_snps_one)
        ]
        DxC_effects_all_snps = []
        for snp in intersect_snps_one:
            DxC_effect_snp = calculate_DxC_effect(
                DxC_associations=livi_associations,
                SNP_id=snp,
                cell_state_latent=zbase,
                A=seed_A,
            )
            DxC_effects_all_snps.append(DxC_effect_snp)
        DxC_effects_all_snps = pd.concat(DxC_effects_all_snps, axis=1, ignore_index=False)
        seeds_DxC_effects.append(DxC_effects_all_snps)

    seeds_D_ct_df = pd.DataFrame.from_dict(seeds_D_ct, orient="index")

    associations_all_seeds = associations_all_seeds.assign(
        Celltypes=associations_all_seeds.apply(
            lambda x: seeds_D_ct_df.loc[x.random_seed, x.Factor], axis=1
        )
    )
    ## N_celltypes should be either 1 or None, after changing the U to celltype assignment approach
    associations_all_seeds = associations_all_seeds.loc[associations_all_seeds.Celltypes.notna()]
    associations_all_seeds = associations_all_seeds.assign(
        N_celltypes=associations_all_seeds.apply(lambda x: len(set(x.Celltypes)), axis=1)
    )

    if args.save_results_summary:
        associations_all_seeds.to_csv(
            os.path.join(
                output_dir,
                (
                    "DxC_associations_all_seeds.tsv"
                    if not args.prs
                    else "DxC_associations_all_seeds_PRS.tsv"
                ),
            ),
            sep="\t",
            index=False,
            header=True,
        )

    plot_data = associations_all_seeds.loc[associations_all_seeds.N_celltypes < 5].filter(
        ["SNP_id", "Celltypes", "random_seed"]
    )
    snps_all_runs = plot_data.groupby("SNP_id", observed=True).apply(
        lambda x: x.random_seed.nunique() == len(model_replicates), include_groups=False
    )
    snps_all_runs = snps_all_runs[snps_all_runs].index
    # Global stats
    plot_data = plot_data.loc[plot_data.SNP_id.isin(snps_all_runs)]
    plot_data = plot_data.explode("Celltypes")
    plot_data = plot_data.drop_duplicates()
    plot_data = plot_data.groupby(["SNP_id", "Celltypes"]).size().reset_index(name="N_runs")
    sns.histplot(plot_data.N_runs, discrete=True, binwidth=0.8, color="navy")
    plt.xticks(range(1, len(model_replicates) + 1))
    plt.xlabel("$N$ random model initialisations", fontsize=15)
    variable = "PRS" if args.prs else "SNP"
    plt.ylabel(f"Counts\n({variable}-celltype pair)", fontsize=15)
    plt.title(
        f"Number of different random model initialisations (max {len(model_replicates)})\nthat a {variable} was assigned to a specific celltype"
    )
    filename = (
        f"{args.model_prefix}_Histogram_celltype-concordance-different-seeds.png"
        if args.model_prefix is not None
        else "Histogram_celltype-concordance-different-seeds.png"
    )
    if args.prs:
        filename = filename.replace(".png", "_PRS.png")
    plt.savefig(os.path.join(output_dir, filename), dpi=400, bbox_inches="tight", transparent=True)
    plt.savefig(
        os.path.join(output_dir, filename.replace("png", "pdf")),
        dpi=400,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    # Sample SNPs to plot
    if args.snp_sample:
        sample_plot_data = plot_data.loc[plot_data.SNP_id.isin(args.snp_sample)]
        if sample_plot_data.shape[0] == 0:
            warnings.warn(
                f"SNP sample {args.snp_sample} not included in tested SNPs. Selecting a random sample instead..."
            )
            sample_plot_data = plot_data.loc[plot_data.SNP_id.isin(snps_all_runs[:6])]
    else:
        sample_plot_data = plot_data.loc[plot_data.SNP_id.isin(snps_all_runs[:6])]

    # NaNs can occur when a U factor is not assign to any cell-state-factors with prob > 0.8
    sample_plot_data = sample_plot_data.dropna(subset="Celltypes")

    if f"{args.celltype_column}_colors" in [k for k in adata.uns.keys()]:
        adata.obs[args.celltype_column] = pd.Categorical(adata.obs[args.celltype_column])
        ct_colors = dict(
            zip(
                adata.obs[args.celltype_column].cat.categories.tolist(),
                adata.uns[f"{args.celltype_column}_colors"].tolist(),
            )
        )
        ct_colors = {
            key: value
            for key, value in ct_colors.items()
            if key in sample_plot_data.Celltypes.tolist()
        }
    else:
        ct_colors = {}
        for ct in sample_plot_data.Celltypes.unique():
            ct_colors[ct] = "#" + "".join(
                [random.choice("0123456789ABCDEF") for j in range(6)]  # nosec
            )

    uniq_snps = sample_plot_data["SNP_id"].unique().tolist()
    uniq_cts = sample_plot_data["Celltypes"].unique().tolist()
    node_labels = uniq_snps + uniq_cts
    snps_colors = []
    while len(snps_colors) < len(uniq_snps):
        col = "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])  # nosec
        if col not in list(ct_colors.values()):
            snps_colors.append(col)
    node_colors = snps_colors + list(ct_colors.values())

    source = []
    target = []
    value = []
    link_colors = []
    link_labels = []

    for index, row in sample_plot_data.iterrows():
        source.append(uniq_snps.index(row["SNP_id"]))
        target.append(uniq_cts.index(row["Celltypes"]) + len(uniq_snps))
        value.append(row["N_runs"])
        link_colors.append(ct_colors[row["Celltypes"]])
        link_labels.append(f'{row["N_runs"]} runs')

    # Create the Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=30,
                    thickness=20,
                    line={"color": "black", "width": 0.5},
                    label=node_labels,
                    color=node_colors,
                ),
                textfont={"color": "black", "size": 14},
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    label=link_labels,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Celltype assignment concordance between different random initialisations",
        font_size=18,
    )
    # fig.show()
    filename = (
        f"{args.model_prefix}_Sankey-celltype-concordance-different-seeds_example.png"
        if args.model_prefix is not None
        else "Sankey-celltype-concordance-different-seeds_example.png"
    )
    if args.prs:
        filename = filename.replace(".png", "_PRS.png")
    fig.write_image(os.path.join(output_dir, filename), width=1200, height=1000)
    fig.write_html(os.path.join(output_dir, filename.replace("png", "html")))
    plt.close()

    # Also illustrate the same data with a Histogram for non-interactive visualisation
    n_cts_snp = (
        sample_plot_data.groupby("SNP_id")
        .apply(lambda x: x.Celltypes.nunique(), include_groups=False)
        .to_dict()
    )
    uniq_snps = sorted(uniq_snps)  # Sorting is necessary to have compatible order with `n_cts_snp`
    bar_width = 0.8
    spacing = 3
    xticks_pos = []
    legend_set = []
    offset = 0
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for idx, snp in enumerate(uniq_snps):
        snp_data = sample_plot_data[sample_plot_data["SNP_id"] == snp]
        if idx == 0:
            offset = 0
        else:
            n_cts_prev_snp = n_cts_snp[uniq_snps[idx - 1]]
            offset += n_cts_prev_snp * bar_width + spacing
        for idx2, ct in enumerate(snp_data.Celltypes.unique()):
            position = idx2 * bar_width + offset
            if idx2 == 0:
                start_pos = position
            if idx2 == (snp_data.Celltypes.nunique() - 1):
                end_pos = position
            count = snp_data[snp_data.Celltypes == ct]["N_runs"].values[0]
            if ct not in legend_set:
                ax.bar(position, count, width=bar_width, label=ct, color=ct_colors[ct])
                legend_set.append(ct)
            else:
                ax.bar(position, count, width=bar_width, color=ct_colors[ct])
        xticks_pos.append((end_pos + start_pos) / 2)

    ax.set_xlabel(f"{variable}", fontsize=14)
    ax.set_ylabel("Number of random seeds", fontsize=15)
    ax.set_yticks(np.arange(1, len(gseed) + 1, 1))
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(uniq_snps, rotation=0, fontsize=13)
    ax.legend(title="Cell type", loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False)
    plt.savefig(
        os.path.join(output_dir, filename.replace("Sankey", "histogram")),
        dpi=400,
        bbox_inches="tight",
        transparent=True,
    )
    plt.savefig(
        os.path.join(output_dir, filename.replace("Sankey", "histogram").replace("png", "pdf")),
        dpi=400,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    print(
        "------------  Assessing concordance between different random seeds at the single-cell level ------------"
    )

    corrs_across_runs_pearson = []
    corrs_across_runs_pearson_pvals = []
    for i, j in combinations(range(len(seeds_DxC_effects)), 2):
        corrs = pearsonr(
            x=seeds_DxC_effects[i].abs(),
            y=seeds_DxC_effects[j].abs(),
            axis=0,
            alternative="greater",
        )
        corrs_across_runs_pearson.append(corrs.statistic)
        corrs_across_runs_pearson_pvals.append(corrs.pvalue)

    corrs_across_runs_pearson = np.vstack(corrs_across_runs_pearson)
    corrs_across_runs_pearson_pvals = np.vstack(corrs_across_runs_pearson_pvals)
    mean_corrs_across_runs_pearson = np.mean(corrs_across_runs_pearson, axis=0)
    mean_pval_across_runs_pearson = np.mean(corrs_across_runs_pearson_pvals, axis=0)

    ## Histogram
    sns.histplot(mean_corrs_across_runs_pearson, bins="fd", color="navy")
    plt.xlabel("Pearson's $ρ$", fontsize=15)
    plt.ylabel("Counts", fontsize=15)
    plt.title(
        f"Pearson's $ρ$ between {variable} effect on cells across {len(gseed)} different random model initialisations"
    )
    filename = (
        f"{args.model_prefix}_Histogram_{variable}-effect-concordance-different-seeds-single-cell-level_Pearson.png"
        if args.model_prefix is not None
        else f"Histogram_{variable}-effect-concordance-different-seeds-single-cell-level_Pearson.png"
    )
    if args.prs:
        filename = filename.replace(".png", "_PRS.png")
    plt.savefig(
        os.path.join(output_dir, filename), dpi=400, bbox_inches="tight", transparent=False
    )
    plt.savefig(
        os.path.join(output_dir, filename.replace("png", "svg")),
        dpi=400,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    ## Barplot
    sorted_indices = np.argsort(mean_corrs_across_runs_pearson)[::-1]
    correlations_sorted = mean_corrs_across_runs_pearson[sorted_indices]
    pvalues_sorted = mean_pval_across_runs_pearson[sorted_indices]
    snp_names_sorted = np.array(list(intersect_snps_one))[sorted_indices]
    pval_colors = ["cornflowerblue" if p < 0.05 else "grey" for p in pvalues_sorted]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(correlations_sorted)), correlations_sorted, color=pval_colors)
    # Labels and title
    plt.axhline(y=0, color="black", linewidth=1)
    plt.xticks(range(len(correlations_sorted)), snp_names_sorted, rotation=90)
    plt.xlabel(variable)
    plt.ylabel("Pearson's $ρ$", fontsize=15)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="cornflowerblue", lw=4, label="Significant (p<0.05)"),
            plt.Line2D([0], [0], color="grey", lw=4, label="Not Significant"),
        ],
        loc="upper right",
    )
    plt.title(
        f"Pearson's $ρ$ between {variable} effect on cells across {len(gseed)} different random model initialisations"
    )
    filename = filename.replace("Histogram", "Barplot")
    plt.savefig(
        os.path.join(output_dir, filename), dpi=400, bbox_inches="tight", transparent=False
    )
    plt.savefig(
        os.path.join(output_dir, filename.replace("png", "pdf")),
        dpi=400,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    if args.test_aggregated_factors:
        print(
            "------------ Calculate correlations between factors from different runs  ------------"
        )
        ### Calculate correlations between factors, then combine highly correalted factors (average value) to "robust factors" and test those
        seeds_D = []
        for i in range(len(model_replicates)):
            seed_D = pd.read_csv(
                os.path.join(
                    args.results_dir, model_replicates[i], f"{model_replicates[i]}_D_embedding.tsv"
                ),
                index_col=0,
                sep="\t",
            )
            seed_D = seed_D.assign(random_seed=[gseed[i]] * seed_D.shape[0])
            seeds_D.append(seed_D)

        # robust_factors = aggregate_correlated_factors_across_runs(seeds_D, args.factor_correlation_theshold, factor_correlations=None)

        #### Calculate correlations based on DxC decoder
        seeds_DxC_decoder = []
        for i in range(len(model_replicates)):
            seed_DxC_dec = pd.read_csv(
                os.path.join(
                    args.results_dir, model_replicates[i], f"{model_replicates[i]}_DxC_decoder.tsv"
                ),
                index_col=0,
                sep="\t",
            )
            seed_DxC_dec = seed_DxC_dec.assign(random_seed=[gseed[i]] * seed_DxC_dec.shape[0])
            seeds_DxC_decoder.append(seed_DxC_dec)

        factor_correlations = correlate_factors_across_runs(seeds_DxC_decoder)

        robust_factors = aggregate_correlated_factors_across_runs(
            seeds_D, args.factor_correlation_theshold, factor_correlations=factor_correlations
        )
        robust_factors.to_csv(
            os.path.join(output_dir, "Robust_aggregated_D_factors.tsv"),
            sep="\t",
            header=True,
            index=True,
        )

        robust_loadings = aggregate_correlated_factors_across_runs(
            [
                loadings.rename(
                    columns=dict(
                        zip(loadings.columns, [f.replace("DxC", "D") for f in loadings.columns])
                    )
                )
                for loadings in seeds_DxC_decoder
            ],
            args.factor_correlation_theshold,
            factor_correlations=factor_correlations,
        )
        robust_loadings.to_csv(
            os.path.join(output_dir, "Robust_aggregated_DxC_decoder_loadings.tsv"),
            sep="\t",
            header=True,
            index=True,
        )

        print(f"Found {robust_factors.shape[1]} robust factors across runs.")

        print("------------  Testing aggregated robust factors  ------------")
        covariates = set_up_covariates(args, robust_factors)

        associations_ofp = (
            f"{args.model_prefix}_Robust-factors-corr{args.factor_correlation_theshold}-across-random-seeds"
            if args.model_prefix
            else f"Robust-factors-corr{args.factor_correlation_theshold}-across-random-seeds"
        )
        if args.prs:
            associations_ofp = associations_ofp + "_PRS"

        associations = run_LIVI_genetic_association_testing(
            D_context=robust_factors,
            V_persistent=None,
            GT_matrix=GT_matrix,
            variant_info=variant_info,
            Kinship=kinship,
            genotype_pcs=GT_PCs,
            method=args.testing_method,
            output_dir=output_dir,
            output_file_prefix=associations_ofp,
            covariates=covariates,
            quantile_norm=args.quantile_normalise,
            variance_threshold=None,
            variable_factors=None,
            fdr_method=args.fdr_method,
            fdr_threshold=(args.fdr_threshold if args.fdr_threshold else None),
            return_associations=True,
        )

        if not args.prs and known_trans_eQTLs:
            corr = str(args.factor_correlation_theshold)
            overlap_with_known_eQTLs(
                known_trans_eQTLs=known_trans_eQTLs,
                SNP_colname_trans=SNP_colname_trans,
                DxC_effects_LIVI=associations,
                factor_assignment_matrix=None,
                known_cis_eQTLs=known_cis_eQTLs,
                SNP_colname_cis=SNP_colname_cis,
                persistent_effects_LIVI=None,
                savefig=os.path.join(output_dir, f"Robust-factors-corr{corr.replace('.', '')}"),
                format="png",
            )
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Absolute path of the directory containing the LIVI model(s) inference and testing results.",
    )
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Absolute path of the AnnData file containing the scRNA-seq data used for inference.",
    )
    parser.add_argument(
        "--celltype_column",
        "-ct",
        type=str,
        required=True,
        help="Column name in cell metadata (adata.obs) indicating the celltype.",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        help="Used to select a specific model, if multiple models are saved in `resutls_dir`",
    )
    parser.add_argument(
        "--snp_sample",
        type=str,
        nargs="*",
        help="Plot celltype assignments of different models runs for those SNPs.",
    )
    parser.add_argument(
        "--individual_column",
        "-id",
        type=str,
        help="Column name in cell metadata (adata.obs) indicating the individual the sample (cell) comes from.",
    )
    parser.add_argument(
        "--factor_correlation_theshold",
        type=float,
        help="Aggregate factors that have a correlation greater or equal to this threshold.",
    )
    parser.add_argument(
        "--test_aggregated_factors",
        action="store_true",
        default=False,
        help="If provided, then robust factors resulting from the aggregation of the highly correlated factors between runs, will be tested for SNP effects.",
    )
    parser.add_argument(
        "--testing_method",
        type=str,
        default=False,
        choices=["TensorQTL", "LIMIX"],
        help="Whether to use LIMIX or TensorQTL for SNP association testing. LIMIX can account for repeated samples (e.g. when a donor is in multiple batches), while TensorQTL is fast.",
    )
    parser.add_argument(
        "--genotype_matrix",
        "-GT_matrix",
        type=str,
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
        help="Absolute path of the .tsv file with the Kinship matrix (e.g. generated with PLINK) to be used for relatedness/population structure correction during variant testing. Required when testing_method == 'LIMIX'.",
    )
    parser.add_argument(
        "--genotype_pcs",
        "-GT_pcs",
        default=None,
        type=str,
        help="Absolute path of the .tsv file with the genotype PCs (individuals x PCs) to be used for relatedness/population structure correction during variant testing. Required when testing_method == 'tensorQTL'.",
    )
    parser.add_argument(
        "--n_gt_pcs",
        default=10,
        type=int,
        help="Number of genotype principal components to use as covariates during SNP association testing. If omitted 10 PCs are used by default.",
    )
    parser.add_argument(
        "--covariates",
        default=None,
        type=str,
        help="Absolute path of the .tsv file containing gene expression PCs aggregated at the donor level.",
    )
    parser.add_argument(
        "--prs", action="store_true", default=False, help="If testing PRS instead of SNPs."
    )
    parser.add_argument(
        "--quantile_normalise",
        action="store_true",
        default=False,
        help="Whether to quantile normalise LIVI's individual embeddings prior to variant association testing.",
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
        "--save_results_summary",
        action="store_true",
        default=False,
        help="If provided, write the dataframe summarizing the results of the different runs in a .tsv file.",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        default=None,
        type=str,
        help="Absolute path of the directory to save the output figures and testing results.",
    )

    args = parser.parse_args()

    main(args)
