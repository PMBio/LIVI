import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

### Run:
# python livi_testing.py --model_output_dir --cell_metadata_file -id -GT_matrix --plink -K --batch_column --age_column --sex_column --quantile_normalise --multiple_testing_threshold --output_dir --output_file_prefix
###

import argparse
import os
import re
import warnings
from datetime import datetime
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from glimix_core.lmm import LMM
from multipy.fdr import qvalue
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from pandas_plink import read_plink
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, quantile_transform
from statsmodels.stats import multitest
from tensorqtl import pgen, trans

from src.analysis.plotting import QQplot


# Kinship matrix normalisation taken from LIMIX: https://limix-tempdoc.readthedocs.io/en/latest/_modules/limix/qc/_covariance.html#normalise_covariance
def normalise_covariance(K):
    """Variance rescaling of covariance matrix 𝙺.

    Let n be the number of rows (or columns) of 𝙺 and let
    mᵢ be the average of the values in the i-th column.
    Gower rescaling is defined as

    .. math::

        𝙺(n - 1)/(𝚝𝚛𝚊𝚌𝚎(𝙺) - ∑mᵢ).

    Parameters
    ----------
    K : array_like
        Covariance matrix to be normalised.

    Returns
    -------
    Normalised covariance matrix
    """

    if isinstance(K, pd.DataFrame):
        K = K.astype(float)
        trace = K.values.trace()
    else:
        K = np.asarray(K, float)
        trace = K.trace()

    c = np.asarray((K.shape[0] - 1) / (trace - K.mean(axis=0).sum()), float)

    return K * c


def lrt_pvalues(null_lml: float, alt_lmls: Union[float, np.ndarray], dof: int = 1) -> np.ndarray:
    """Compute p-values from likelihood ratios.

    Parameters
    ----------
    null_lml (float): Log-likelihood of the null model.
    alt_lmls (Union[float, np.ndarray]): Log-likelihoods of the alternative models.
    dof (Optional[int]): Degrees of freedom for the chi-squared distribution. Default is 1.

    Returns
    -------
    np.ndarray: Likelihood ratio test p-values.
    """

    super_tiny = np.finfo(float).tiny
    tiny = np.finfo(float).eps

    lrs = np.clip(-2 * null_lml + 2 * np.asarray(alt_lmls, float), super_tiny, np.inf)
    pv = chi2(df=dof).sf(lrs)

    return np.clip(pv, super_tiny, 1 - tiny)


# Combine tensorQTL trans results in single dataframe
def flatten_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Flattens a multi-index DataFrame by stacking it and resetting the index.

    Args:
        df (DataFrame): The input DataFrame with a multi-index to be flattened.
        source_name (str): The name for the column to replace the stacked level's name.

    Returns:
        DataFrame: A flattened DataFrame with columns ["variant_id", "phenotype_id", source_name].
    """
    df_flattened = df.stack(future_stack=True).reset_index()
    df_flattened.columns = ["variant_id", "phenotype_id", source_name]

    return df_flattened


def run_tensorQTL(
    phenotype_df: pd.DataFrame,
    genotype_df: pd.DataFrame,
    variant_info: pd.DataFrame,
    covariates_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Runs TensorQTL's trans-association mapping on phenotype (here LIVI individual embeddings)
    and genotype data, returning a DataFrame with p-values, effect sizes, and allele information
    for each tested variant-factor pair.

    Args:
        phenotype_df (DataFrame): DataFrame containing LIVI individual embeddings with gene IDs as columns.
        genotype_df (DataFrame): DataFrame containing genotype data with variant IDs as columns.
        variant_info (DataFrame): DataFrame containing variant information, including reference and alternate alleles.
        covariates_df (Optional[DataFrame]): DataFrame of individual level covariates to include in the linear model (default is None).

    Returns:
        DataFrame: A DataFrame containing the merged trans-association results with columns:
                  - SNP_id (variant_id)
                  - Factor (phenotype_id)
                  - p_value
                  - effect_size
                  - effect_size_se
                  - ref_allele (from variant_info)
                  - alt_allele (from variant_info)
    """
    results_trans = trans.map_trans(
        genotype_df=genotype_df,
        phenotype_df=phenotype_df,
        covariates_df=covariates_df,
        batch_size=10000,
        return_sparse=False,  # report all test results
        maf_threshold=0.0,  # no filtering
    )

    # When `return_sparse=False`, TensorQTL returns a list of 4 dfs with indices the variants and columns the gene IDs.
    # The 1st df contains the pvalues, the 2nd df the effect sizes, the 3rd df the effect_size_se and the 4th Series the MAF
    dfs = [flatten_df(df, label) for (df, label) in zip(results_trans[:-1], ["pval", "b", "b_se"])]
    results_trans = dfs[0]
    for df in dfs[1:]:
        results_trans = pd.merge(results_trans, df, on=["variant_id", "phenotype_id"])

    # Add REF/ALT allele info
    if variant_info is not None:
        results_trans = results_trans.merge(
            variant_info.filter(["variant_id", "ref_allele", "alt_allele"]),
            on="variant_id",
            how="left",
        )

    results_trans.rename(
        columns={
            "phenotype_id": "Factor",
            "variant_id": "SNP_id",
            "pval": "p_value",
            "b": "effect_size",
            "b_se": "effect_size_se",
        },
        inplace=True,
    )

    return results_trans


def LMM_test_feature(
    feature_id: Union[int, str],
    phenotype_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    G: pd.DataFrame,
    QS: Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
    quantile_norm: bool,
) -> pd.DataFrame:
    """Perform a linear mixed model (LMM) test for the effect of a genetic variable (e.g. SNP or
    PRS) on a specified phenotype feature (e.g. a gene or factor).

    Parameters
    ----------
    feature_id (Union[int, str]): Identifier for the specific phenotype feature.
    phenotype_df (pd.DataFrame): DataFrame containing the values of all phenotype features.
    covariates_df (pd.DataFrame): DataFrame containing sample covariates to be included
        as fixed effects in the LMM.
    G (pd.DataFrame): DataFrame containing the genetic data.
    QS (tuple): Economic eigendecomposition in the form of ((Q0, Q1), S0) of a kinship
        matrix K (used as covariance of the random genetic effect).
    quantile_norm (bool): Flag indicating whether quantile normalization should be
        applied to the phenotype.

    Returns
    -------
    pd.DataFrame: DataFrame containing results of the LMM test, including phenotype feature ID,
        genetic variable ID, effect size, effect size standard error, and p-value.
    """
    feature_phenotype = phenotype_df.loc[feature_id]
    # Remove samples where the feature is NaN
    feature_phenotype = feature_phenotype.dropna()

    if covariates_df.empty:
        covariates_matrix = np.ones((feature_phenotype.shape[0], 1))
    else:
        covariates_matrix = covariates_df.loc[
            covariates_df.index.isin(feature_phenotype.index)
        ].values.astype(np.float32)
        # covariates_matrix = np.append(np.ones((feature_phenotype.shape[0], 1)), covariates_matrix, axis=1)

    # Subset Genetics matrix
    G_matrix = G.loc[G.index.isin(feature_phenotype.index)].values  # individuals x G_variable

    if quantile_norm:
        phenotype = quantile_transform(
            feature_phenotype.values.reshape(-1, 1), output_distribution="normal"
        )
    else:
        phenotype = feature_phenotype

    null_lmm = LMM(phenotype, covariates_matrix, QS, restricted=False)
    null_lmm.fit(verbose=False)
    # Log of the marginal likelihood of null model:
    null_loglk = null_lmm.lml()

    # Test the effect of G_variable
    flmm = null_lmm.get_fast_scanner()

    scannerOut = flmm.fast_scan(G_matrix, verbose=False)
    alt_loglks = scannerOut["lml"]
    effsizes = np.asarray(scannerOut["effsizes1"])
    effsizes_se = np.asarray(scannerOut["effsizes1_se"])
    # Compute p-values from likelihood ratios: null model vs. full model with genetics
    pvalues = np.asarray(lrt_pvalues(null_loglk, alt_loglks))

    feature_results = pd.DataFrame(
        {
            "feature_id": [feature_id] * G.shape[1],
            "variable": G.columns,
            "effect_size": effsizes,
            "effect_size_se": effsizes_se,
            "p_value": pvalues,
        }
    )

    return feature_results


def set_up_covariates(args: argparse.Namespace, D_context: pd.DataFrame) -> pd.DataFrame:
    """Set up sample covariates based on command-line arguments and metadata.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.
    D_context (pd.DataFrame): DxC LIVI factors.

    Returns
    -------
    pd.DataFrame: DataFrame containing sample (donor) covariates to be included
        as fixed effects in the L(M)M.
    """

    covariates = pd.DataFrame(index=D_context.index)

    # File with individual level covariates such as aggregated GEX PCs
    covars = pd.read_csv(
        args.covariates, sep="," if ".csv" in args.covariates else "\t", index_col=0
    )
    covars.index = covars.index.astype(str)
    covariates = covariates.merge(covars, how="left", left_index=True, right_index=True)

    return covariates


def run_LIVI_genetic_association_testing(
    D_context: pd.DataFrame,
    V_persistent: pd.DataFrame,
    GT_matrix: pd.DataFrame,
    output_dir: str,
    output_file_prefix: str,
    method: str = "limix",
    quantile_norm: bool = True,
    fdr_method: str = "BH",
    return_associations: bool = False,
    Kinship: Optional[pd.DataFrame] = None,
    genotype_pcs: Optional[pd.DataFrame] = None,
    variant_info: Optional[pd.DataFrame] = None,
    covariates: Optional[pd.DataFrame] = None,
    variance_threshold: Optional[float] = None,
    variable_factors: Optional[List[int]] = None,
    fdr_threshold: Optional[float] = None,
) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """Test genetic variables (e.g. SNPs or PRS) for effects on LIVI's individual embeddings and
    save the results to file. Optionally select also significant associations based on
    `fdr_threshold`.

    Parameters
    ----------
    D_context (pd.DataFrame): LIVI cell-state-specific genetic embedding.
    V_persistent (pd.DataFrame): LIVI persistent genetic embedding.
    GT_matrix (pd.DataFrame): Genotype matrix (donors x SNPs).
    output_dir (str): Output directory to save the testing results.
    output_file_prefix(str): Output file prefix.
    method (str): Whether to use LIMIX or TensorQTL for association testing. LIMIX can account for
        repeated samples (e.g. when a donor is in multiple batches), while TensorQTL is fast.
    fdr_method (str): False discovery rate (FDR) controlling method for multiple testing correction.
    Kinship (Optional[pd.DataFrame]): Precomputed Kinship matrix.
    genotype_pcs (Optional[pd.DataFrame]): Precomputed genotype principal components.
    variant_info (Optional[pd.DataFrame]): SNP information contained in the .bim file,
        if PLINK genotype matrix is used.
    covariates_df (Optional[pd.DataFrame]): DataFrame containing sample covariates
        to be included as fixed effects in the LMM.
    quantile_norm (bool): Flag indicating whether quantile normalization should be
        applied to the phenotype.
    fdr_threshold (Optional[float]): False discovery rate (FDR) threshold to call an association significant.

    Returns
    -------
    results_sign_context (pd.DataFrame): Significant SNP associations with the cell-state-specific genetic embedding.
    results_sign_persistent (pd.DataFrame): Significant SNP associations with the persistent genetic embedding,
        if there is one.
    """

    if return_associations and fdr_threshold is None:
        fdr_threshold = 0.05

    GT_matrix = GT_matrix.loc[D_context.index]

    if method in ["LIMIX", "LMM"]:
        # add intercept
        covariates["intercept"] = 1.0

        if Kinship is not None:
            kinship = Kinship
            kinship_mat = (
                kinship.loc[covariates.index, covariates.index].to_numpy()
                if covariates is not None
                else kinship.to_numpy()
            )
        else:
            kinship_mat = np.dot(GT_matrix.to_numpy(), GT_matrix.T.to_numpy())
            kinship_mat = normalise_covariance(kinship_mat)
        QS = economic_qs(kinship_mat)
    elif method == "TensorQTL":
        covariates = covariates.merge(genotype_pcs, how="left", right_index=True, left_index=True)
    else:
        raise ValueError(f"Supported methods are LIMIX and TensorQTL. Unknown method: {method}.")

    if D_context is not None:
        if variable_factors is not None:
            D_context = pd.DataFrame(
                D_context.to_numpy()[:, variable_factors],
                index=D_context.index,
                columns=[f"DxC_Factor{f+1}" for f in variable_factors],
            )
        elif variance_threshold is not None:
            variable_factors = np.where(
                np.var(D_context.to_numpy(), axis=0) >= variance_threshold
            )[0]
            D_context = pd.DataFrame(
                D_context.to_numpy()[:, variable_factors],
                index=D_context.index,
                columns=[f"DxC_Factor{f+1}" for f in variable_factors],
            )
        else:
            D_context = D_context

        print("\n ----- Running genetic association testing for D embedding ----- \n")

        if method in ["LIMIX", "LMM"]:
            method_prefix = "LMM"
            results = pd.DataFrame(
                columns=["Factor", "SNP_id", "effect_size", "effect_size_se", "p_value"]
            )
            for f in D_context.columns:
                print(f"Testing: {f}")
                results_factor = LMM_test_feature(
                    feature_id=f,
                    phenotype_df=D_context.T,
                    covariates_df=covariates,
                    G=GT_matrix,
                    QS=QS,
                    quantile_norm=quantile_norm,
                )
                results_factor.rename(
                    columns={"feature_id": "Factor", "variable": "SNP_id"}, inplace=True
                )

                if results.empty:
                    results = results_factor
                else:
                    results = pd.concat([results, results_factor], axis=0)

            if variant_info is not None:
                results = results.merge(
                    variant_info.filter(["snp", "a1"]).rename(
                        columns={"snp": "SNP_id", "a1": "assessed_allele"}
                    ),
                    on="SNP_id",
                    how="left",
                )
        elif method == "TensorQTL":
            method_prefix = "TensorQTL"
            results = run_tensorQTL(
                phenotype_df=D_context.T,
                genotype_df=GT_matrix.T,
                variant_info=variant_info,
                covariates_df=covariates,
            )
        else:
            raise ValueError(
                f"Supported methods are LIMIX and TensorQTL. Unknown method: {method}."
            )

        try:
            filename = (
                f"{output_file_prefix}_{method_prefix}_results_D-embedding_variable-factors.tsv"
                if variable_factors or variance_threshold
                else f"{output_file_prefix}_{method_prefix}_results_D-embedding.tsv"
            )
            results.to_csv(os.path.join(output_dir, filename), sep="\t", header=True, index=False)
        except OSError:
            filename = (
                f"_{method_prefix}_results_D-embedding_variable-factors.tsv"
                if variable_factors or variance_threshold
                else f"_{method_prefix}_results_D-embedding.tsv"
            )
            results.to_csv(os.path.join(output_dir, filename), sep="\t", header=True, index=False)
            warnings.warn(
                f"Could not save testing results for D under provided filename (filename too long).\nSaved as '{filename}' instead."
            )
        try:
            qqplot_filename = (
                f"{output_file_prefix}_{method_prefix}_QQplot_context-specific-effects.png"
            )
            QQplot(
                results.p_value,
                savefig=os.path.join(output_dir, qqplot_filename),
            )
            plt.close()
        except OSError:
            qqplot_filename = f"_{method_prefix}_QQplot_context-specific-effects.png"
            QQplot(
                results.p_value,
                savefig=os.path.join(output_dir, qqplot_filename),
            )
            plt.close()
            warnings.warn(
                "Could not save QQplot for D under provided filename (filename too long).\nSaved as '_QQplot_<method>_context-specific-effects.png' instead."
            )
        try:
            histplot_filename = f"{output_file_prefix}_{method_prefix}_Pvalue-Histogram_context-specific-effects.png"
            sns.histplot(results.p_value, bins=500, color="royalblue")
            plt.savefig(
                os.path.join(output_dir, histplot_filename),
                transparent=True,
                bbox_inches="tight",
                dpi=200,
            )
            plt.close()
        except OSError:
            histplot_filename = f"_{method_prefix}_Pvalue-Histogram_context-specific-effects.png"
            sns.histplot(results.p_value, bins=500, color="royalblue")
            plt.savefig(
                os.path.join(output_dir, histplot_filename),
                transparent=True,
                bbox_inches="tight",
                dpi=200,
            )
            plt.close()
        if fdr_threshold is not None:
            results_sign_context = FDR_correction(
                results, cut_off=fdr_threshold, method=fdr_method
            )
            try:
                filename_sign = (
                    f"{output_file_prefix}_{method_prefix}_results_{fdr_method}-{fdr_threshold}_D-embedding_variable-factors.tsv"
                    if variable_factors or variance_threshold
                    else f"{output_file_prefix}_{method_prefix}_results_{fdr_method}-{fdr_threshold}_D-embedding.tsv"
                )
                results_sign_context.to_csv(
                    os.path.join(output_dir, filename_sign), sep="\t", header=True, index=False
                )
            except OSError:
                filename_sign = (
                    f"_{method_prefix}_results_{fdr_method}-{fdr_threshold}_D-embedding_variable-factors.tsv"
                    if variable_factors or variance_threshold
                    else f"_{method_prefix}_results_{fdr_method}-{fdr_threshold}_D-embedding.tsv"
                )
                results_sign_context.to_csv(
                    os.path.join(output_dir, filename_sign), sep="\t", header=True, index=False
                )
                warnings.warn(
                    f"Could not save significant results for U under provided filename (filename too long).\nSaved as '{filename_sign}' instead."
                )

        print("----- Done ----- \n")

    if V_persistent is not None:
        print("\n\n ----- Running genetic association testing for V embedding ----- \n")

        if method in ["limix", "LIMIX", "LMM"]:
            results = pd.DataFrame(
                columns=["Factor", "SNP_id", "effect_size", "effect_size_se", "p_value"]
            )
            for f in V_persistent.columns:
                print(f"Testing: {f}")
                results_factor = LMM_test_feature(
                    feature_id=f,
                    phenotype_df=V_persistent.T,
                    covariates_df=covariates,
                    G=GT_matrix,
                    QS=QS,
                    quantile_norm=quantile_norm,
                )
                results_factor.rename(
                    columns={"feature_id": "Factor", "variable": "SNP_id"}, inplace=True
                )

                if results.empty:
                    results = results_factor
                else:
                    results = pd.concat([results, results_factor], axis=0)

            if variant_info is not None:
                results = results.merge(
                    variant_info.filter(["snp", "a1"]).rename(
                        columns={"snp": "SNP_id", "a1": "assessed_allele"}
                    ),
                    on="SNP_id",
                    how="left",
                )
        elif args.method == "TensorQTL":
            results = run_tensorQTL(
                phenotype_df=V_persistent.T,
                genotype_df=GT_matrix.T,
                variant_info=variant_info,
                covariates_df=covariates,
            )
        else:
            raise ValueError(
                f"Supported methods are LIMIX and TensorQTL. Unknown method: {method}."
            )

        try:
            filename = f"{output_file_prefix}_{method_prefix}_results_V-embedding.tsv"
            results.to_csv(os.path.join(output_dir, filename), sep="\t", header=True, index=False)
        except OSError:
            filename = f"_{method_prefix}_results_V-embedding.tsv"
            results.to_csv(os.path.join(output_dir, filename), sep="\t", header=True, index=False)
            warnings.warn(
                f"Could not save testing results for V under provided filename (filename too long).\nSaved as '{filename}' instead."
            )
        try:
            qqplot_filename = f"{output_file_prefix}_QQplot_{method_prefix}_persistent-effects.png"
            QQplot(
                results.p_value,
                savefig=os.path.join(output_dir, qqplot_filename),
            )
            plt.close()
        except OSError:
            qqplot_filename = f"_QQplot_{method_prefix}_persistent-effects.png"
            QQplot(
                results.p_value,
                savefig=os.path.join(output_dir, qqplot_filename),
            )
            plt.close()
            warnings.warn(
                "Could not save QQplot for V under provided filename (filename too long).\nSaved as '_QQplot_persistent-effects.png' instead."
            )
        try:
            histplot_filename = (
                f"{output_file_prefix}_{method_prefix}_Pvalue-Histogram_persistent-effects.png"
            )
            sns.histplot(results.p_value, bins=500, color="royalblue")
            plt.savefig(
                os.path.join(output_dir, histplot_filename),
                transparent=True,
                bbox_inches="tight",
                dpi=200,
            )
            plt.close()
        except OSError:
            histplot_filename = f"_{method_prefix}_Pvalue-Histogram_persistent-effects.png"
            sns.histplot(results.p_value, bins=500, color="royalblue")
            plt.savefig(
                os.path.join(output_dir, histplot_filename),
                transparent=True,
                bbox_inches="tight",
                dpi=200,
            )
            plt.close()

        if fdr_threshold is not None:
            results_sign_persistent = FDR_correction(
                results, cut_off=fdr_threshold, method=fdr_method
            )
            try:
                filename_sign = f"{output_file_prefix}_{method_prefix}_results_{fdr_method}-{fdr_threshold}_V-embedding.tsv"
                results_sign_persistent.to_csv(
                    os.path.join(output_dir, filename_sign), sep="\t", header=True, index=False
                )
            except OSError:
                filename_sign = (
                    f"_{method_prefix}_results_{fdr_method}-{fdr_threshold}_V-embedding.tsv"
                )
                results_sign_persistent.to_csv(
                    os.path.join(output_dir, filename_sign), sep="\t", header=True, index=False
                )
                warnings.warn(
                    f"Could not save significant results for V under provided filename (filename too long).\nSaved as '{filename_sign}' instead."
                )

        print("----- Done ----- \n")

    if return_associations:
        if D_context is not None:
            if V_persistent is not None:
                return results_sign_context, results_sign_persistent
            else:
                return results_sign_context
        else:
            if V_persistent is not None:
                return None, results_sign_persistent
            else:
                return None, None


def FDR_correction(
    testing_results: pd.DataFrame, cut_off: float = 0.05, method: str = "Storey"
) -> pd.DataFrame:
    """Perform False Discovery Rate (FDR) correction on testing results.

    Parameters
    ----------
    testing_results (pd.DataFrame): DataFrame containing testing results.
    cut_off (float, optional): False discovery rate (FDR) threshold for significance.
        Default is 0.05.
    method (str, optional): Method used for adjustment of pvalues. Supported are FDR controlling methods,
        specifically Benjamini-Hochberg, Benjamini-Yekutieli and Storey's Q-value method.

    Returns
    -------
    pd.DataFrame: DataFrame containing testing results after FDR correction.
    """

    ## Multiple testing correction across everything
    if method in ["Storey", "storey", "qvalue", "q_value"]:
        testing_results = testing_results.assign(
            corrected_pvalue=qvalue(testing_results["p_value"].to_numpy(), threshold=cut_off)[1]
        )
    elif method in ["Benjamini-Hochberg", "benjamini_hochberg", "BH", "bh", "fdr_bh"]:
        testing_results = testing_results.assign(
            corrected_pvalue=multitest.multipletests(
                testing_results["p_value"].values,
                method="fdr_bh",
                is_sorted=False,
                returnsorted=False,
            )[1]
        )
    elif method in ["Benjamini-Yekutieli", "benjamini_yekutieli", "BY", "by", "fdr_by"]:
        testing_results = testing_results.assign(
            corrected_pvalue=multitest.multipletests(
                testing_results["p_value"].values,
                method="fdr_by",
                is_sorted=False,
                returnsorted=False,
            )[1]
        )
    else:
        raise ValueError(
            f"Unsupported method {method}. Valid options are 'Storey', 'Benjamini-Hochberg' and 'Benjamini-Yekutieli'."
        )

    testing_results_sign = testing_results.loc[testing_results.corrected_pvalue < cut_off]
    print(f"number of fQTLs: {testing_results_sign.shape[0]}")
    print(f"number of unique fSNPs: {testing_results_sign.SNP_id.nunique()}")
    print(f"number of unique factors: {testing_results_sign.Factor.nunique()}")

    return testing_results_sign


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
        metadata (pd.DataFrame): DataFrame containing cell metadata.
        GT_matrix (pd.DataFrame): Genotype matrix (donors x SNPs).
        variant_info (pd.DataFrame): SNP information contained in the .bim or .pvar file, if PLINK genotype matrix is used, otherwise None.
        kinship (pd.DataFrame): Kinship matrix if provided, otherwise None.
        GT_PCs (pd.DataFrame): Dataframe containing genotype principal components if provided, otherwise None.
        D_context (pd.DataFrame): LIVI cell-state-specific genetic embedding.
        V_persistent (pd.DataFrame): LIVI persistent genetic embedding if applicable, otherwise None.
    """

    assert os.path.isdir(args.model_output_dir), "Model directory not found."
    assert os.path.isfile(args.cell_metadata_file), "Cell metadata file not found."
    assert os.path.isfile(args.covariates), "Covariates file not found."

    if args.method == "TensorQTL":
        assert (
            args.genotype_pcs is not None
        ), "Genotype PCs are required when testing using TensorQTL."
        if not torch.cuda.is_available():
            warnings.warn(
                "Testing method is TensorQTL, but no GPU is available. This will slow down performance."
            )
    if args.method in ["LIMIX", "LMM"]:
        assert args.kinship is not None, "Kinship matrix required with testing using LIMIX-QTL."

    if args.output_dir:
        output_dir = args.output_dir
        if os.path.exists(output_dir):
            pass
        else:
            os.mkdir(output_dir)
    else:
        output_dir = args.model_output_dir

    of_prefix = (
        args.output_file_prefix
        if args.output_file_prefix
        else os.path.basename(args.model_output_dir)
    )

    _, ext = os.path.splitext(args.cell_metadata_file)
    if ext not in [".tsv", ".csv"]:
        raise TypeError(
            f"Cell metadata file must be either .tsv or .csv. File format provided: {ext}"
        )

    metadata = pd.read_csv(
        args.cell_metadata_file, index_col=0, sep="\t" if ext == ".tsv" else ","
    )

    assert (
        args.individual_column in metadata.columns
    ), f"`individual_column`: '{args.individual_column}' not in cell metadata columns."
    metadata[args.individual_column] = metadata[args.individual_column].astype(str)

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
            metadata[args.individual_column].unique(), metadata[args.individual_column].unique()
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
        GT_PCs = GT_PCs.loc[GT_PCs.index.isin(metadata[args.individual_column].unique())]
        if GT_PCs.shape[0] == 0:
            raise ValueError(
                "Individual IDs in cell metadata do not match individual IDs in the genotype PCs."
            )
        # select leading components
        GT_PCs = GT_PCs.filter([f"PC{i}" for i in range(1, n_gt_pcs + 1)])
    else:
        GT_PCs = None

    if args.plink and args.method in ["LIMIX", "LMM"]:
        variant_info, fam, bed = read_plink(args.genotype_matrix, verbose=False)
        GT_matrix = pd.DataFrame(
            bed.compute(), index=variant_info.snp, columns=fam.iid
        )  # SNPs x donors
    elif args.plink and args.method == "TensorQTL":
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

    GT_matrix = GT_matrix.filter(metadata[args.individual_column].unique())
    if GT_matrix.shape[1] == 0:
        raise ValueError(
            "Individual IDs in cell metadata do not match individual IDs in the genotype matrix."
        )

    # GT_matrix_standardised = pd.DataFrame(
    #     StandardScaler().fit_transform(GT_matrix.T.values), # donors x SNPs
    #     index=GT_matrix.columns,
    #     columns=GT_matrix.index,
    # )
    # del GT_matrix

    GT_matrix = GT_matrix.T  # donors x SNPs

    files = [
        f
        for f in os.listdir(args.model_output_dir)
        if os.path.isfile(os.path.join(args.model_output_dir, f))
    ]
    D_context = [
        re.match("(.*D_embedding.tsv)", f)
        for f in files
        if re.match(".*D_embedding.tsv", f) is not None
    ]
    if len(D_context) > 0:
        D_context = D_context[0].groups()[0]
        D_context = pd.read_csv(
            os.path.join(args.model_output_dir, D_context), index_col=0, sep="\t"
        )
        D_context.index = D_context.index.astype(str)
        if D_context.loc[D_context.index.isin(GT_matrix.index)].shape[0] == 0:
            raise ValueError(
                "Individual IDs in D embedding do not match individual IDs in the genotype matrix."
            )
    else:
        D_context = None

    V_persistent = [
        re.match("(.*V_embedding.tsv)", f)
        for f in files
        if re.match(".*V_embedding.tsv", f) is not None
    ]
    if len(V_persistent) > 0:
        V_persistent = V_persistent[0].groups()[0]
        V_persistent = pd.read_csv(
            os.path.join(args.model_output_dir, V_persistent), index_col=0, sep="\t"
        )
        V_persistent.index = V_persistent.index.astype(str)
        if V_persistent.loc[V_persistent.index.isin(GT_matrix.index)].shape[0] == 0:
            raise ValueError(
                "Individual IDs in V embedding do not match individual IDs in the genotype matrix."
            )
    else:
        V_persistent = None

    return (
        output_dir,
        of_prefix,
        metadata,
        GT_matrix,
        variant_info,
        kinship,
        GT_PCs,
        D_context,
        V_persistent,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_output_dir",
        type=str,
        required=True,
        help="Absolute path of the directory containing the inference results (embeddings) files of the LIVI model.",
    )
    parser.add_argument(
        "--cell_metadata_file",
        type=str,
        required=True,
        help="Absolute path of the file containing metadata info (individual ID, batch etc.) for each cell. Similar to adata.obs. ATTENTION: Cell IDs must be the first column!",
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
        "--covariates",
        default=None,
        type=str,
        required=True,
        help="Absolute path of the .tsv file containing gene expression PCs aggregated at the donor level.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="TensorQTL",
        choices=["TensorQTL", "LIMIX", "LMM"],
        help="Whether to use LIMIX/LMM or TensorQTL for SNP association testing. LIMIX can account for repeated samples (e.g. when a donor is in multiple batches), while TensorQTL is fast.",
    )
    parser.add_argument(
        "--plink",
        action="store_true",
        default=False,
        help="If PLINK genotype files (bim, bed, fam if `method` is LIMIX/LMM, or pgen, pvar, psam if `method` is TensorQTL) are provided instead of a GT matrix in .tsv format.",
    )
    parser.add_argument(
        "--kinship",
        "-K",
        type=str,
        help="Absolute path of the .tsv file with the Kinship matrix (e.g. generated with PLINK) to be used for relatedness/population structure correction during variant testing. Required when testing_method == 'LMM'",
    )
    parser.add_argument(
        "--genotype_pcs",
        "-GT_pcs",
        default=None,
        type=str,
        help="Absolute path of the .tsv file with the genotype PCs (individuals x PCs) to be used for relatedness/population structure correction during variant testing. Required when testing_method == 'TensorQTL'",
    )
    parser.add_argument(
        "--n_gt_pcs",
        default=10,
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
        help="Test only those factors whose variance across cells is above this threshold. Ignored if `variable_factors` are provided.",
    )
    parser.add_argument(
        "--fdr_threshold",
        "-fdr",
        default=None,
        type=float,
        help="False discovery rate (FDR) threshold for multiple testing correction.",
    )
    parser.add_argument(
        "--fdr_method",
        default=None,
        type=str,
        choices=["Storey", "qvalue", "Benjamini-Hochberg", "BH", "Benjamini-Yekutieli", "BY"],
        help="False discovery rate (FDR) controlling method for multiple testing correction.",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        default=None,
        type=str,
        help="Absolute path of the directory to save the testing results.",
    )
    parser.add_argument(
        "--output_file_prefix",
        "-ofp",
        default=None,
        type=str,
        help="Common prefix of the output results files.",
    )

    args = parser.parse_args()

    (
        od,
        of_prefix,
        metadata,
        GT_matrix,
        variant_info,
        kinship,
        gt_pcs,
        D,
        V,
    ) = validate_and_read_passed_args(args)
    covariates = set_up_covariates(args, D)

    start = datetime.now()

    run_LIVI_genetic_association_testing(
        D_context=D,
        V_persistent=V,
        GT_matrix=GT_matrix,
        variant_info=variant_info,
        Kinship=kinship,
        genotype_pcs=gt_pcs,
        method=args.method,
        fdr_method=args.fdr_method,
        output_dir=od,
        output_file_prefix=of_prefix,
        covariates=covariates,
        quantile_norm=args.quantile_normalise,
        variance_threshold=args.variance_threshold,
        variable_factors=args.variable_factors,
        fdr_threshold=(args.fdr_threshold if args.fdr_threshold else None),
    )

    end = datetime.now()

    duration = (end - start).seconds
    duration_minutes = duration / 60
    duration_hours = duration_minutes / 60

    with open(os.path.join(od, "association_testing_execution_time.txt"), "w") as outfile:
        outfile.write(
            f"Execution time in seconds: {duration}\nExecution time in minutes: {duration_minutes}\nExecution time in hours: {duration_hours}\n"
        )
