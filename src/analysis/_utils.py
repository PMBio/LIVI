from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import seaborn as sns
import torch
import umap
from anndata import AnnData
from scipy.stats import iqr

pl.seed_everything(32)


def compute_umap(
    latent_space_df: pd.DataFrame,
    colnames: List[str] = ["UMAP1", "UMAP2"],
    add_latent: bool = True,
) -> pd.DataFrame:
    """Compute UMAP (Uniform Manifold Approximation and Projection) from a given latent space.

    Parameters
    ----------
        latent_space_df (pd.DataFrame): DataFrame containing the latent space values.
        colnames (List[str], optional): Column names for the UMAP dimensions. Default is ["UMAP1", "UMAP2"].
        add_latent (bool, optional): Whether to merge the UMAP dimensions with the original latent space DataFrame. Default is True.

    Returns
    -------
        pd.DataFrame: DataFrame containing UMAP dimensions and, optionally, the original latent space values.
    """
    reducer = umap.UMAP(metric="euclidean", unique=True)
    umap_embedding = reducer.fit_transform(latent_space_df.values)
    umap_df = pd.DataFrame(umap_embedding, columns=colnames, index=latent_space_df.index)
    if add_latent:
        umap_df = umap_df.merge(latent_space_df, right_index=True, left_index=True)

    return umap_df


def add_livi_umaps_to_cell_metadata(
    livi_results: Dict[str, Union[np.ndarray, torch.Tensor]],
    latent_space: Union[List[str], str],
    cell_metadata: pd.DataFrame,
    metric: str = "euclidean",
    factors: Optional[Union[List[int], np.ndarray]] = None,
    return_factors_umap_df: bool = False,
) -> Union[pd.DataFrame, None]:
    """Computes UMAP on LIVI's embeddings and stores it in cell metadata to facilitate
    visualisation.

    Parameters
    ----------
        livi_results (dict): Dict with LIVI inference results.
        latent_space (list): Which LIVI latent space to reduce (can also be more than one).
            Must be included in livi_results.keys().
        adata (pd.DataFrame): DataFrame containing cell metadata to which the results are added.
        metric (str): The distance metric to use to compute the UMAP embedding (default is Euclidean distance).
        factors (list of integers or numpy 1D-array): Factors indices to use to compute the UMAP embedding.
            Defaults to None, i.e. use all the factors.
        return_factors_umap_df (bool): Whether to return the dataframe containing the UMAP embeddings,
            besides adding it to adata.obs.

    Returns
    -------
    umap_df (Optional[pd.DataFrame]): If `return_factors_umap_df = True`, returns a separate pd.DataFrame
        containing the UMAP 2D embeddings of LIVI's latent factors.
    """

    if not isinstance(latent_space, list):
        if isinstance(latent_space, str):
            latent_space = [latent_space]
        else:
            raise ValueError("Parameter 'latent_space' must be either a list of str or a str")

    if not all([ls in [key for key in livi_results.keys()] for ls in latent_space]):
        raise KeyError("`latent space` not in `livi_results` ")

    umap_df = pd.DataFrame()
    for latent in latent_space:
        livi_latent = livi_results[latent]
        if livi_latent is None:
            continue
        if isinstance(livi_latent, torch.Tensor):
            livi_latent = livi_latent.detach().cpu().numpy()
        livi_latent = livi_latent[:, factors] if factors is not None else livi_latent
        colnames = (
            [f"Factor{f+1}" for f in factors]
            if factors is not None
            else [f"Factor{f+1}" for f in range(livi_latent.shape[1])]
        )
        livi_latent = pd.DataFrame(livi_latent, index=cell_metadata.index, columns=colnames)
        cols = [f"UMAP1_{latent}", f"UMAP2_{latent}"]
        latent_umap_df = compute_umap(livi_latent, colnames=cols, add_latent=False)

        if umap_df.empty:
            umap_df = latent_umap_df
        else:
            umap_df = pd.concat([umap_df, latent_umap_df], axis=1)

    cell_metadata = cell_metadata.merge(umap_df, right_index=True, left_index=True)

    if return_factors_umap_df:
        return umap_df


def select_important_genes_for_factor_IQR(
    genes_factors_loadings: np.ndarray,
    factor_idx: int,
    gene_names: List[str],
    direction: str = "both",
    threshold: float = 1.5,
    plot: bool = True,
) -> List[str]:
    """Select characteristic genes for a factor based on the interquartile range (IQR) of their
    loadings.

    Parameters
    ----------
        genes_factors_loadings (np.ndarray): 2D array containing gene loadings for factors (genes x factors).
        factor_idx (int): Index of the factor for which important genes should be selected.
        gene_names (List[str]): List of candidate genes; the order must correspond to the order of the rows of `genes_factors_loadings`.
        direction (str, optional): Direction of selection. Valid choices are 'high', 'low', or 'both'. Default is 'both'.
        threshold (float, optional): Threshold multiplier for defining the upper and lower cutoffs. Default is 1.5.
        plot (bool): Whether to visualise the distribution of the gene loadings and the selected cut-offs.
    Returns
    -------
        List[str]: List of factor characteristic gene names based on the selected criteria.
    """

    factor_loadings = genes_factors_loadings[:, factor_idx]
    p_75 = np.percentile(factor_loadings, 75)
    p_25 = np.percentile(factor_loadings, 25)
    IQR = iqr(factor_loadings)
    mean_f = np.mean(factor_loadings)
    upper_cutoff = p_75 + threshold * IQR
    lower_cutoff = p_25 - threshold * IQR

    if direction == "both":
        important_genes_factor_up = np.where(factor_loadings > upper_cutoff)[0]
        important_genes_factor_down = np.where(factor_loadings < lower_cutoff)[0]
        important_genes_factor = np.concatenate(
            [important_genes_factor_up, important_genes_factor_down]
        )
    elif direction == "high":
        important_genes_factor_up = np.where(factor_loadings > upper_cutoff)[0]
        important_genes_factor = important_genes_factor_up
    elif direction == "low":
        important_genes_factor_down = np.where(factor_loadings < lower_cutoff)[0]
        important_genes_factor = important_genes_factor_down
    else:
        raise ValueError("Valid choices for direction are 'high', 'low' or 'both'.")

    if plot:
        sns.histplot(factor_loadings, kde=True, color="royalblue")
        plt.xlabel(f"Gene Loadings - Factor {factor_idx+1}")
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        plt.vlines(x=mean_f, ymin=0, ymax=y_max, color="darkslategrey", linestyles="--")
        plt.vlines(x=upper_cutoff, ymin=0, ymax=y_max, color="darkslategrey", linestyles=":")
        plt.vlines(x=lower_cutoff, ymin=0, ymax=y_max, color="darkslategrey", linestyles=":")

    return gene_names[important_genes_factor].tolist()


def calculate_GxC_effect(
    LIVI_associations: pd.DataFrame,
    SNP_id: str,
    cell_state_latent: pd.DataFrame,
    A: pd.DataFrame,
):
    """Compute effect of a given SNP on cell-states.

    Parameters
    ----------
        LIVI_associations (pd.DataFrame): Dataframe containing LIVI CxG effects.
        SNP_id (str): ID of the SNP, whose effect should be calculated.
        cell_state_latent (pd.DataFrame): DataFrame containing the cell-state latent space.
        A (pd.DataFrame): Dataframe containing LIVI factor assignment matrix.

    Returns
    -------
        GxC_effect (pd.DataFrame): Dataframe containing the effect of the given SNP on each associated cell-state factor.
    """

    snp_associations = LIVI_associations.loc[LIVI_associations.SNP_id == SNP_id]
    factor_beta = snp_associations.filter(["Factor", "effect_size"]).set_index("Factor").T
    ## Visualise the effect as if all individuals had the assessed allele
    # genotypes = GT_matrix.loc[SNP_id, IIDs].astype(int)
    # genotypes = genotypes.replace({2:1})
    # genotypes = genotypes.to_numpy().reshape(-1,1) # genotypes is now n_cells x 1  SNP

    ## G_effect = genotypes*factor_beta.values

    A = A.filter(snp_associations.Factor)

    GxC_effect = (
        cell_state_latent.to_numpy() @ A.to_numpy() * factor_beta.to_numpy()  # G_effect.to_numpy()
    )
    GxC_effect = pd.DataFrame(
        GxC_effect,
        index=cell_state_latent.index,
        columns=["GxC_" + f[1] for f in snp_associations.Factor.str.split("_")],
    )

    return GxC_effect


def calculate_GxC_gene_effect(
    LIVI_associations: pd.DataFrame,
    SNP_id: str,
    cell_state_latent: pd.DataFrame,
    A: pd.DataFrame,
    GxC_decoder: pd.DataFrame,
):
    """Compute effect of a given SNP on cell-states.

    Parameters
    ----------
        LIVI_associations (pd.DataFrame): Dataframe containing LIVI CxG effects.
        SNP_id (str): ID of the SNP, whose effect should be calculated.
        cell_state_latent (pd.DataFrame): DataFrame containing the cell-state latent space.
        A (pd.DataFrame): Dataframe containing LIVI factor assignment matrix.

    Returns
    -------
        GxC_effect_gene (pd.DataFrame): Dataframe containing the effect of the given SNP on each gene.
    """

    snp_associations = LIVI_associations.loc[LIVI_associations.SNP_id == SNP_id]
    factor_beta = snp_associations.filter(["Factor", "effect_size"]).set_index("Factor").T

    A = A.filter(snp_associations.Factor)
    GxC_decoder = GxC_decoder.filter(
        snp_associations.Factor.str.replace("U", "GxC")
    ).T  # U x genes

    GxC_effect = cell_state_latent.to_numpy() @ A.to_numpy() * factor_beta.to_numpy()  # cells x U
    GxC_effect_gene = GxC_effect @ GxC_decoder.to_numpy()  # cells x genes
    GxC_effect_gene = pd.DataFrame(
        GxC_effect_gene,
        index=cell_state_latent.index,
        columns=GxC_decoder.columns,
    )

    return GxC_effect_gene
