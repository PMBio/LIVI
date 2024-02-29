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
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glimix_core.lmm import LMM
from multipy.fdr import qvalue
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from pandas_plink import read_plink
from scipy.stats import chi2, norm
from sklearn.preprocessing import StandardScaler, quantile_transform

from src.analysis.plotting import QQplot


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
    import numpy as np

    super_tiny = np.finfo(float).tiny
    tiny = np.finfo(float).eps

    lrs = np.clip(-2 * null_lml + 2 * np.asarray(alt_lmls, float), super_tiny, np.inf)
    pv = chi2(df=dof).sf(lrs)

    return np.clip(pv, super_tiny, 1 - tiny)


def LMM_test_feature(
    feature_id: Union[int, str],
    phenotype_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    G_scaled: pd.DataFrame,
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
    G_scaled (pd.DataFrame): DataFrame containing the standardised genetic data.
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
        covariates_matrix = np.ones(feature_phenotype.shape[0], 1)
    else:
        covariates_matrix = covariates_df.loc[
            covariates_df.index.isin(feature_phenotype.index)
        ].values.astype(np.float32)

    # Subset Genetics matrix
    G_matrix = G_scaled.loc[
        G_scaled.index.isin(feature_phenotype.index)
    ].values  # individuals x G_variable

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
            "feature_id": [feature_id] * G_scaled.shape[1],
            "variable": G_scaled.columns,
            "effect_size": effsizes,
            "effect_size_se": effsizes_se,
            "p_value": pvalues,
        }
    )

    return feature_results


def set_up_covariates(args: argparse.Namespace, metadata: pd.DataFrame) -> pd.DataFrame:
    """Set up sample covariates based on command-line arguments and metadata.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.
    metadata (pd.DataFrame): DataFrame containing sample metadata.

    Returns
    -------
    pd.DataFrame: DataFrame containing sample covariates to be included
        as fixed effects in the LMM.
    """

    covariates = pd.DataFrame(index=metadata[args.individual_column].unique())
    if args.batch_column is not None:
        metadata[args.batch_column] = metadata[args.batch_column].astype(np.int32)
        # adata.obs[args.batch_column] = pd.Categorical(adata.obs[args.batch_column])
        covariates = covariates.merge(
            metadata.filter([args.individual_column, args.batch_column])
            .drop_duplicates()
            .set_index(args.individual_column),
            right_index=True,
            left_index=True,
        )

    if args.sex_column is not None:
        metadata[args.sex_column] = pd.Categorical(metadata[args.sex_column])
        metadata[args.sex_column].replace(
            {
                metadata[args.sex_column].cat.categories[0]: 1,
                metadata[args.sex_column].cat.categories[1]: 0,
            },
            inplace=True,
        )
        covariates = covariates.merge(
            metadata.filter([args.individual_column, args.sex_column])
            .drop_duplicates()
            .set_index(args.individual_column),
            right_index=True,
            left_index=True,
        )

    if args.age_column is not None:
        if "age_scaled" not in metadata.columns:
            age_scaled = StandardScaler().fit_transform(
                metadata[args.age_column].to_numpy().reshape(-1, 1)
            )
            age_scaled = pd.DataFrame(age_scaled, index=metadata.index, columns=["age_scaled"])
            metadata = metadata.merge(age_scaled, left_index=True, right_index=True)
        covariates = covariates.merge(
            metadata.filter([args.individual_column, "age_scaled"])
            .drop_duplicates()
            .set_index(args.individual_column),
            right_index=True,
            left_index=True,
        )

    return covariates


def run_LIVI_genetic_association_testing(
    U_context,
    V_persistent,
    GT_matrix,
    output_dir,
    output_file_prefix,
    Kinship=None,
    bim=None,
    covariates=None,
    quantile_norm=True,
    variance_threshold=None,
    variable_factors=None,
    qval_threshold=None,
    return_associations=False,
) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """Test genetic variables (e.g. SNPs or PRS) for effects on LIVI's individual embeddings and
    save the results to file. Optionally select also significant associations based on
    `fdr_threshold`.

    Parameters
    ----------
    U_context (pd.DataFrame): LIVI cell-state-specific genetic embedding.
    V_persistent (pd.DataFrame): LIVI persistent genetic embedding.
    GT_matrix (pd.DataFrame): Standardised genetic variable matrix.
    output_dir (str): Output directory to save the testing results.
    output_file_prefix(str): Output file prefix.
    Kinship (Optional[pd.DataFrame]): Precomputed Kinship matrix.
    bim (Optional[pd.DataFrame]): SNP information contained in the .bim file,
        if PLINK genotype matrix is used.
    covariates_df (Optional[pd.DataFrame]): DataFrame containing sample covariates
        to be included as fixed effects in the LMM.
    quantile_norm (bool): Flag indicating whether quantile normalization should be
        applied to the phenotype.
    qval_threshold (Optional[float]): Storey q-value threshold to call an association significant.

    Returns
    -------
    results_sign_context (pd.DataFrame): Significant SNP associations with the cell-state-specific genetic embedding.
    results_sign_persistent (pd.DataFrame): Significant SNP associations with the persistent genetic embedding, if there is one.
    """

    if return_associations and qval_threshold is None:
        qval_threshold = 0.05

    if covariates is not None:
        GT_matrix = GT_matrix.loc[covariates.index]
    #   U_context = U_context.loc[covariates.index]

    if Kinship is not None:
        kinship = Kinship
        kinship_mat = (
            kinship.loc[covariates.index, covariates.index].to_numpy()
            if covariates is not None
            else kinship.to_numpy()
        )
        QS = economic_qs(kinship_mat)
    else:
        QS = economic_qs_linear(GT_matrix)

    if U_context is not None:
        if covariates is not None:
            U_context = U_context.loc[covariates.index]
        print("\n ----- Running genetic association testing for U_context ----- \n")

        results = pd.DataFrame(
            columns=["Factor", "SNP_id", "effect_size", "effect_size_se", "p_value"]
        )

        if variable_factors is not None:
            U_context = pd.DataFrame(
                U_context.to_numpy()[:, variable_factors],
                index=U_context.index,
                columns=[f"CxG_Factor{f+1}" for f in variable_factors],
            )
        elif variance_threshold is not None:
            variable_factors = np.where(
                np.var(U_context.to_numpy(), axis=0) >= variance_threshold
            )[0]
            U_context = pd.DataFrame(
                U_context.to_numpy()[:, variable_factors],
                index=U_context.index,
                columns=[f"CxG_Factor{f+1}" for f in variable_factors],
            )
        else:
            U_context = U_context

        for f in U_context.columns:
            print(f"Testing: {f}")
            results_factor = LMM_test_feature(
                feature_id=f,
                phenotype_df=U_context.T,
                covariates_df=covariates,
                G_scaled=GT_matrix,
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

        if bim is not None:
            results = results.merge(
                bim.filter(["snp", "a1"]).rename(
                    columns={"snp": "SNP_id", "a1": "assessed_allele"}
                ),
                on="SNP_id",
                how="left",
            )
        filename = (
            f"{output_file_prefix}_LMM_results_Ucontext_variable-factors.tsv"
            if variable_factors or variance_threshold
            else f"{output_file_prefix}_LMM_results_Ucontext.tsv"
        )

        results.to_csv(os.path.join(output_dir, filename), sep="\t", header=True, index=False)
        QQplot(
            results.p_value,
            savefig=os.path.join(
                output_dir, f"{output_file_prefix}_QQplot_context-specific-effects.png"
            ),
        )
        plt.close()
        if qval_threshold is not None:
            results_sign_context = FDR_correction(results, cut_off=qval_threshold)
            filename_sign = (
                f"{output_file_prefix}_LMM_results_StoreyQ{qval_threshold}_Ucontext_variable-factors.tsv"
                if variable_factors or variance_threshold
                else f"{output_file_prefix}_LMM_results_StoreyQ{qval_threshold}_Ucontext.tsv"
            )
            results_sign_context.to_csv(
                os.path.join(output_dir, filename_sign), sep="\t", header=True, index=False
            )

        print("----- Done ----- \n")

    if V_persistent is not None:
        if covariates is not None:
            V_persistent = V_persistent.loc[covariates.index]
        print("\n\n ----- Running genetic association testing for V_persistent ----- \n")

        results = pd.DataFrame(
            columns=["Factor", "SNP_id", "effect_size", "effect_size_se", "p_value"]
        )

        for f in V_persistent.columns:
            print(f"Testing: {f}")
            results_factor = LMM_test_feature(
                feature_id=f,
                phenotype_df=V_persistent.T,
                covariates_df=covariates,
                G_scaled=GT_matrix,
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

        if bim is not None:
            results = results.merge(
                bim.filter(["snp", "a1"]).rename(
                    columns={"snp": "SNP_id", "a1": "assessed_allele"}
                ),
                on="SNP_id",
                how="left",
            )
        filename = f"{output_file_prefix}_LMM_results_Vpersistent.tsv"

        results.to_csv(os.path.join(output_dir, filename), sep="\t", header=True, index=False)
        QQplot(
            results.p_value,
            savefig=os.path.join(
                output_dir, f"{output_file_prefix}_QQplot_persistent-effects.png"
            ),
        )
        plt.close()
        if qval_threshold is not None:
            results_sign_persistent = FDR_correction(results, cut_off=qval_threshold)
            filename_sign = (
                f"{output_file_prefix}_LMM_results_StoreyQ{qval_threshold}_Vpersistent.tsv"
            )
            results_sign_persistent.to_csv(
                os.path.join(output_dir, filename_sign), sep="\t", header=True, index=False
            )

        print("----- Done ----- \n")

    if return_associations:
        if V_persistent is not None:
            return results_sign_context, results_sign_persistent
        else:
            return results_sign_context


def FDR_correction(testing_results: pd.DataFrame, cut_off: float = 0.05) -> pd.DataFrame:
    """Perform False Discovery Rate (FDR) correction on testing results.

    Parameters
    ----------
    testing_results (pd.DataFrame): DataFrame containing testing results.
    cut_off (float, optional): Storey q-value threshold for significance.
        Default is 0.05.

    Returns
    -------
    pd.DataFrame: DataFrame containing testing results after FDR correction.
    """

    ## Multiple testing correction across everything
    testing_results = testing_results.assign(
        Storey_qvals=qvalue(testing_results["p_value"].to_numpy(), threshold=cut_off)[1]
    )

    testing_results_sign = testing_results.loc[testing_results.Storey_qvals < cut_off]
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
        GT_matrix_standardised (pd.DataFrame): Standardized genotype matrix.
        bim (pd.DataFrame): SNP information contained in the .bim file, if PLINK genotype matrix is used, otherwise None.
        kinship (pd.DataFrame): Kinship matrix if provided, otherwise None.
        U_context (pd.DataFrame): LIVI cell-state-specific genetic embedding.
        V_persistent (pd.DataFrame): LIVI persistent genetic embedding if applicable, otherwise None.
    """

    assert os.path.isdir(args.model_output_dir), "Model directory not found"
    assert os.path.isfile(args.cell_metadata_file), "Cell metadata file not found"

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
    if args.batch_column:
        assert (
            args.batch_column in metadata.columns
        ), f"`batch_column`: '{args.batch_column}' not in cell metadata columns."
    if args.sex_column:
        assert (
            args.sex_column in metadata.columns
        ), f"`sex_column`: '{args.sex_column}' not in cell metadata columns."
    if args.age_column:
        assert (
            args.age_column in metadata.columns
        ), f"`age_column`: '{args.age_column}' not in cell metadata columns."

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

    GT_matrix = GT_matrix.filter(metadata[args.individual_column].unique())
    if GT_matrix.shape[1] == 0:
        raise ValueError(
            "Individual IDs in cell metadata do not match individual IDs in the genotype matrix."
        )

    GT_matrix_standardised = pd.DataFrame(
        StandardScaler().fit_transform(GT_matrix.T.values),
        index=GT_matrix.columns,
        columns=GT_matrix.index,
    )
    del GT_matrix

    if args.kinship is not None:
        assert os.path.isfile(args.kinship), "Kinship matrix file not found."
        _, ext = os.path.splitext(args.kinship)
        if ext not in [".tsv", ".csv"]:
            raise TypeError(
                f"Kinship matrix must be either .tsv or .csv. File format provided: {ext}."
            )
        kinship = pd.read_csv(args.kinship, index_col=0, sep="\t" if ext == ".tsv" else ",")
        kinship = kinship.loc[
            metadata[args.individual_column].unique(), metadata[args.individual_column].unique()
        ]
        if kinship.shape[0] == 0:
            raise ValueError(
                "Individual IDs in cell metadata do not match individual IDs in the kinship matrix."
            )
    else:
        kinship = None

    files = [
        f
        for f in os.listdir(args.model_output_dir)
        if os.path.isfile(os.path.join(args.model_output_dir, f))
    ]
    U_context = [
        re.match("(.*CxG_embedding.tsv)", f)
        for f in files
        if re.match(".*CxG_embedding.tsv", f) is not None
    ]
    if len(U_context) > 0:
        U_context = U_context[0].groups()[0]
        U_context = pd.read_csv(
            os.path.join(args.model_output_dir, U_context), index_col=0, sep="\t"
        )
        if U_context.loc[U_context.index.isin(GT_matrix_standardised.index)].shape[0] == 0:
            raise ValueError(
                "Individual IDs in U context do not match individual IDs in the genotype matrix."
            )
    else:
        U_context = None

    V_persistent = [
        re.match("(.*persistent_embedding.tsv)", f)
        for f in files
        if re.match(".*persistent_embedding.tsv", f) is not None
    ]
    if len(V_persistent) > 0:
        V_persistent = V_persistent[0].groups()[0]
        V_persistent = pd.read_csv(
            os.path.join(args.model_output_dir, V_persistent), index_col=0, sep="\t"
        )
        if V_persistent.loc[V_persistent.index.isin(GT_matrix_standardised.index)].shape[0] == 0:
            raise ValueError(
                "Individual IDs in V_persistent do not match individual IDs in the genotype matrix."
            )
    else:
        V_persistent = None

    return (
        output_dir,
        of_prefix,
        metadata,
        GT_matrix_standardised,
        bim,
        kinship,
        U_context,
        V_persistent,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_output_dir",
        help="Absolute path of the directory containing the inference results (embeddings) files of the LIVI model.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cell_metadata_file",
        help="Absolute path of the file containing metadata info (individual ID, batch etc.) for each cell. Similar to adata.obs. ATTENTION: Cell IDs must be the first column!",
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
        "--genotype_matrix",
        "-GT_matrix",
        help="Absolute path of the .tsv file with the genotype matrix (the SNPs to test against LIVI's individual embeddings). For PLINK files please use in addition the --plink flag.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--plink",
        action="store_true",
        help="If PLINK genotype files (bim, bed, fam) are provided instead of a GT matrix in .tsv format.",
        default=False,
    )
    parser.add_argument(
        "--kinship",
        "-K",
        help="Absolute path of the .tsv file with the Kinship matrix (e.g. generated with PLINK) to be used for relatedness/population structure correction during variant testing.",
        type=str,
    )
    parser.add_argument(
        "--quantile_normalise",
        action="store_true",
        help="Whether to quantile normalise LIVI's individual embeddings prior to variant association testing.",
        default=False,
    )
    parser.add_argument(
        "--variable_factors",
        nargs="*",
        help="Test only those variable factors for interaction effects (zero-based index).",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        help="Test only those factors whose variance across cells is above this threshold. Ignored if `variable_factors` are provided.",
    )
    parser.add_argument(
        "--batch_column",
        help="Column name in cell metadata (adata.obs) indicating the batch the sample (cell) comes from.",
        type=str,
    )
    parser.add_argument(
        "--age_column",
        help="Column name in cell metadata (adata.obs) indicating the age of the individual.",
        type=str,
    )
    parser.add_argument(
        "--sex_column",
        help="Column name in cell metadata (adata.obs) indicating the sex of the individual.",
        type=str,
    )
    parser.add_argument(
        "--multiple_testing_threshold",
        "-fdr",
        type=float,
        help="Storey q-value threshold for multiple testing correction.",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        help="Absolute path of the directory to save the testing results.",
        type=str,
    )
    parser.add_argument(
        "--output_file_prefix",
        "-ofp",
        help="Common prefix of the output results files.",
        type=str,
    )

    args = parser.parse_args()

    (
        od,
        of_prefix,
        metadata,
        GT_matrix_standardised,
        bim,
        kinship,
        Uint,
        Vadd,
    ) = validate_and_read_passed_args(args)
    covariates = set_up_covariates(args, metadata)

    run_LIVI_genetic_association_testing(
        U_context=Uint,
        V_persistent=Vadd,
        GT_matrix=GT_matrix_standardised,
        bim=bim,
        Kinship=kinship,
        output_dir=od,
        output_file_prefix=of_prefix,
        covariates=covariates,
        quantile_norm=args.quantile_normalise,
        variance_threshold=args.variance_threshold,
        variable_factors=args.variable_factors,
        qval_threshold=args.multiple_testing_threshold
        if args.multiple_testing_threshold
        else None,
    )
