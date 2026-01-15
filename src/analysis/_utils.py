from typing import Dict, List, Optional, Tuple, Union

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
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
        cell_metadata (pd.DataFrame): DataFrame containing cell metadata to which the results are added.
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


def assign_D_to_celltype(
    cell_state_latent: pd.DataFrame,
    A: pd.DataFrame,
    cell_metadata: pd.DataFrame,
    celltype_column: str,
    top_one: bool = False,
    assignment_threshold: float = 0.8,
    strict: bool = True,
):
    """Assigns D factors to known celltypes. Only D factors assigned to at least one cell-state
    based on `assignment_threshold` are assigned to a celltype.

    Parameters
    ----------
        cell_state_latent (pd.DataFrame): DataFrame containing the cell-state latent space (cells x factors).
        A (pd.DataFrame): Dataframe containing LIVI factor assignment matrix.
        cell_metadata (pd.DataFrame): DataFrame containing cell information, such as celltype, donor ID etc.
            Cell IDs must be the index, and they must be the same as in `cell_state_latent`.
        celltype_column (str): Column in `cell_metadata` containing celltype information.
        top_one (bool): If True, a celltypes x D factors matrix is calculated, and each D factor is assigned
            to the celltype (row) that has the max value for that D factor (column). Else, D factors are assigned
            to all the cell-state factors that have values >= 0.9 (or values >= `assignment_threshold` if strict
            is False). Cell-state factor values are aggregated across cells within each cell type, and cell-
            state factors are mapped to the cell type that has the highest average value (across cells) for that
            factor. Default is False.

    Returns
    -------
    D_celltype (Dict): Dictionary with D factors as keys and assigned celltype as values (or None if the D
        factor was not assigned to any specific cell-state).
    """

    cc_softmax = nn.Softmax(dim=1)(torch.from_numpy(cell_state_latent.to_numpy())).numpy()
    cc_softmax = pd.DataFrame(
        cc_softmax, index=cell_state_latent.index, columns=cell_state_latent.columns
    )
    celltypes_factors = cc_softmax.merge(
        cell_metadata[celltype_column], right_index=True, left_index=True, how="left"
    )
    # Average of each factor across cells belonging to the same celltype
    celltypes_factors = celltypes_factors.groupby(
        by=celltype_column, observed=True
    ).mean()  # celltypes x C factors

    if not top_one and strict:
        D_not_assigned = A.columns[A[A >= 0.9].isna().sum(axis=0) == cell_state_latent.shape[1]]
    else:
        D_not_assigned = A.columns[
            A[A >= assignment_threshold].isna().sum(axis=0) == cell_state_latent.shape[1]
        ]

    if top_one:
        celltypes_D = pd.DataFrame(
            celltypes_factors.to_numpy() @ A.to_numpy(),
            index=celltypes_factors.index,
            columns=A.columns,
        )  # celltypes x D factors
        # Assign each D factor to the celltype with the highest value
        D_celltype = celltypes_D.idxmax(axis=0).to_dict()
    else:
        threshold = 0.9 if strict else assignment_threshold
        # Assign each cell-state factor to the celltype with the highest value
        temp_dict = celltypes_factors.idxmax(axis=0).to_dict()
        # Substitute cell-state factor names with corresponding celltype names
        A_ct = A.rename(index=temp_dict).T  # D x celltypes
        # For each D factor pick the celltypes (cell-state factors with A values >= threshold)
        D_celltype = (
            A_ct.apply(lambda x: x >= threshold, axis=1)
            .apply(lambda x: x.index[x].unique().tolist(), axis=1)
            .to_dict()
        )

    for d in D_not_assigned:
        D_celltype[d] = None

    return D_celltype


def calculate_DxC_effect(
    DxC_associations: pd.DataFrame,
    SNP_id: str,
    cell_state_latent: pd.DataFrame,
    A: pd.DataFrame,
):
    """Compute effect of a given SNP on cell-states.

    Parameters
    ----------
        DxC_associations (pd.DataFrame): Dataframe containing LIVI DxC effects.
        SNP_id (str): ID of the SNP, whose effect should be calculated.
        cell_state_latent (pd.DataFrame): DataFrame containing the cell-state latent space.
        A (pd.DataFrame): Dataframe containing LIVI factor assignment matrix.

    Returns
    -------
        DxC_effect (pd.DataFrame): Dataframe containing the effect of the given SNP at the single-cell
            level for cells belonging to different cell-states.
    """

    snp_associations = DxC_associations.loc[DxC_associations.SNP_id == SNP_id]
    factor_beta = snp_associations.filter(["Factor", "effect_size"]).set_index("Factor").T

    A = A.filter(snp_associations.Factor)
    cell_state_softmax = nn.Softmax(dim=1)(torch.from_numpy(cell_state_latent.to_numpy())).numpy()

    ## Visualise the effect as if all individuals had the assessed allele
    DxC_effect = cell_state_softmax @ A.to_numpy() * factor_beta.to_numpy()  # cells x DxC effect
    DxC_effect = pd.DataFrame(
        DxC_effect,
        index=cell_state_latent.index,
        columns=["DxC_" + f[1] for f in snp_associations.Factor.str.split("_")],
    )

    return DxC_effect


def calculate_DxC_gene_effect(
    DxC_associations: pd.DataFrame,
    SNP_id: str,
    cell_state_latent: pd.DataFrame,
    A: pd.DataFrame,
    DxC_decoder: pd.DataFrame,
    factor_id: Optional[str] = None,
):
    """Compute effect of a given SNP on cell-states.

    Parameters
    ----------
        LIVI_associations (pd.DataFrame): Dataframe containing LIVI DxC effects.
        SNP_id (str): ID of the SNP, whose effect should be calculated.
        cell_state_latent (pd.DataFrame): DataFrame containing the cell-state latent space.
        A (pd.DataFrame): Dataframe containing LIVI factor assignment matrix.
        factor_id (Optional[str]): ID of the factor to use for reconstruction. For SNPs associated with more than one factors, it
            is recommended to specify which one to use when calculating the effect of the SNP on genes, in order to obtain
            meaningful results.

    Returns
    -------
        DxC_effect_gene (pd.DataFrame): Dataframe containing the effect of the given SNP on each gene.
    """

    snp_associations = DxC_associations.loc[DxC_associations.SNP_id == SNP_id]
    if factor_id is not None:
        snp_associations = snp_associations.loc[snp_associations.Factor == factor_id]
    factor_beta = snp_associations.filter(["Factor", "effect_size"]).set_index("Factor").T

    A = A.filter(snp_associations.Factor)
    DxC_decoder = DxC_decoder.filter(
        snp_associations.Factor.str.replace("D", "DxC")
    ).T  # D x genes

    cell_state_softmax = nn.Softmax(dim=1)(torch.from_numpy(cell_state_latent.to_numpy())).numpy()
    DxC_effect = cell_state_softmax @ A.to_numpy() * factor_beta.to_numpy()  # cells x DxC effect
    DxC_effect_gene = DxC_effect @ DxC_decoder.to_numpy()  # cells x genes
    DxC_effect_gene = pd.DataFrame(
        DxC_effect_gene,
        index=cell_state_latent.index,
        columns=DxC_decoder.columns,
    )

    return DxC_effect_gene


def find_trans_fSNPs(
    DxC_effects: pd.DataFrame,
    DxC_decoder: pd.DataFrame,
    gene_metadata: Optional[pd.DataFrame] = None,
    adata: Optional["AnnData"] = None,
) -> pd.DataFrame:
    """Identify SNP–factor pairs whose top gene loadings map to genes **in trans** relative to the
    SNP (i.e., on a different chromosome or >5 Mb away on the same chromosome).

    Parameters
    ----------
    DxC_effects (pandas.DataFrame): Table with at least columns `SNP_id` (formatted like "chr:pos"),
        `Factor`, and optionally `SNP_chrom`, `SNP_pos`.
    DxC_decoder (pandas.DataFrame): Factor loading matrix indexed by gene IDs; columns are factors IDs.
        The function selects the top absolute-loading gene per factor.
    gene_metadata (pandas.DataFrame, optional): Gene annotations indexed by gene ID, containing
        chromosome and genomic coordinates. If not provided, `adata.var` is used.
    adata (AnnData, optional): AnnData object whose `.var` holds gene annotations if `gene_metadata`
        is not provided.

    Returns
    -------
    pandas.DataFrame: A dataframe with columns:
        - `SNP_id`: The SNP identifier ("chr:pos").
        - `Factor`: The factor associated with the SNP whose top gene is in trans.

    Raises
    ------
    AssertionError: If neither `gene_metadata` nor `adata` is provided, or if required columns are
        missing from the gene metadata.
    """
    assert (
        gene_metadata is not None or adata is not None
    ), "Please provide gene genomic locations either as a pandas DataFrame or in adata.var"

    if gene_metadata is None and adata is not None:
        gene_metadata = adata.var

    gene_metadata.columns = gene_metadata.columns.str.replace(" ", "_", regex=False)
    gene_metadata.columns = gene_metadata.columns.str.lower()
    chrom_col = "gene_chromosome"
    try:
        gene_metadata[chrom_col]
    except KeyError:
        chrom_col = "gene_chrom"
        try:
            gene_metadata[chrom_col]
        except KeyError:
            chrom_col = "chromosome"
            try:
                gene_metadata[chrom_col]
            except KeyError:
                chrom_col = "chrom"
                try:
                    gene_metadata[chrom_col]
                except KeyError:
                    chrom_col = "genechrom"
                    assert (
                        chrom_col in gene_metadata.columns
                    ), "`gene_chromosome` column missing from gene metadata"

    start_col = "gene_start"
    try:
        gene_metadata[start_col]
    except KeyError:
        start_col = "start"
        try:
            gene_metadata[start_col]
        except KeyError:
            start_col = "genestart"
            assert (
                start_col in gene_metadata.columns
            ), "`gene_start` column missing from gene metadata"
    gene_metadata[start_col] = gene_metadata[start_col].astype(np.float64)

    end_col = "gene_end"
    try:
        gene_metadata[end_col]
    except KeyError:
        end_col = "end"
        try:
            gene_metadata[end_col]
        except KeyError:
            end_col = "geneend"
            assert end_col in gene_metadata.columns, "`gene_end` column missing from gene metadata"
    gene_metadata[end_col] = gene_metadata[end_col].astype(np.float64)

    if any([c not in DxC_effects.columns for c in ["SNP_chrom", "SNP_pos"]]):
        DxC_effects[["SNP_chrom", "SNP_pos"]] = DxC_effects.SNP_id.str.split(":", expand=True)
        DxC_effects.SNP_pos = DxC_effects.SNP_pos.astype(np.float64)
    DxC_effects = DxC_effects.assign(
        upstream_5MB=DxC_effects.apply(lambda x: max(0, x.SNP_pos - 5000000), axis=1),
        downstream_5MB=DxC_effects.apply(lambda x: x.SNP_pos + 5000000, axis=1),
    )

    trans_fsnps = []
    factors = []
    for _, row in DxC_effects.iterrows():
        top_gene_factor = (
            DxC_decoder[row["Factor"].replace("D", "DxC")].abs().nlargest(1, keep="all").index
        )
        gene_sub = gene_metadata.loc[top_gene_factor]
        # Only keep genes located on different chromosome or on the same chromosome, but more than 5 Mbp away from the SNP
        gene_sub = gene_sub.loc[
            (
                (
                    (gene_sub[chrom_col] == str(row["SNP_chrom"]))
                    & (gene_sub[start_col] > row["downstream_5MB"])
                )
                | (
                    (gene_sub[chrom_col] == str(row["SNP_chrom"]))
                    & (gene_sub[end_col] < row["upstream_5MB"])
                )
            )
            | (gene_sub[chrom_col] != str(row["SNP_chrom"]))
        ]
        if gene_sub.shape[0] > 0:
            trans_fsnps.append(row["SNP_id"])
            factors.append(row["Factor"])

    trans_fsnps = pd.DataFrame({"SNP_id": trans_fsnps, "Factor": factors})

    return trans_fsnps


def aggregate_cell_counts(
    adata: AnnData,
    aggregate_cols: List[str],
    sum_gene: bool = False,
    layer: Optional[str] = None,
) -> AnnData:
    """Aggregates single-cell data into pseudobulk profiles based on specified cell metadata
    columns.

    This function groups cells in an AnnData object (`adata`) by the specified `aggregate_cols` and averages
    the expression data across those cells to create pseudobulk profiles. It can also aggregate values from a
    specified layer in `adata.layers` instead of the primary data matrix in `adata.X`. Optionally, it can also
    sum the expression over all genes.

    Parameters:
    ----------
        adata (anndata.AnnData): AnnData object containing gene counts in `adata.X`, cell metadata in `adata.obs`
            and gene metadata in `adata.var`.
        aggregate_cols (List[str]): Column names from `adata.obs` used to group cells for aggregation (e.g., ['donor']).
        sum_gene (bool): If True, sums expression across genes (rows). If False, retains gene-level resolution in the
            aggregated expression. Default is False.
        layer (Optional[str]): If specified, aggregates data from the given layer instead of `adata.X`. Default is None,
            i.e. aggregate data in `adata.X`

    Returns:
    -------
        adata_aggr(anndata.AnnData): An `AnnData` object with pseudobulk expression (mean across each group of the specified `aggregate_cols`).
            The `obs` contains metadata about the aggregated groups, including the number of cells aggregated and the original cell IDs.
            The `var` contains original gene metadata, or a summary row if `sum_gene=True`.
    """

    grouped = adata.obs.groupby(by=aggregate_cols, observed=True)
    grouped_obs = adata.obs.filter(aggregate_cols).drop_duplicates().reset_index(drop=True)

    init = True
    for k, v in grouped.groups.items():
        if isinstance(k, list):
            pseudocell = "__".join([str(key) for key in k])
        else:
            pseudocell = k
        if init:
            if layer:
                pseudoexpression_matrix = adata[v].layers[layer].mean(axis=0)  # mean across donors
            else:
                pseudoexpression_matrix = adata[v].X.mean(axis=0)
            if sum_gene:
                pseudoexpression_matrix = pseudoexpression_matrix.sum(axis=1)  # sum across genes

            metadata = (
                adata.obs.loc[v]
                .filter(aggregate_cols)
                .drop_duplicates()
                .reset_index(drop=True)
                .rename(index={0: pseudocell})
            )
            # Save also the number of cells that were aggregated as well as their IDs
            metadata = pd.concat(
                [
                    metadata,
                    pd.DataFrame(
                        {"ncells_aggregated": len(v), "cell_ids": ", ".join(v.tolist())},
                        index=metadata.index,
                    ),
                ],
                axis=1,
            )
            init = False
        else:
            if layer:
                pseudoexpression = adata[v].layers[layer].mean(axis=0)
            else:
                pseudoexpression = adata[v].X.mean(axis=0)
            if sum_gene:
                pseudoexpression = pseudoexpression.sum(axis=1)  # sum across genes

            pseudoexpression_matrix = np.vstack([pseudoexpression_matrix, pseudoexpression])
            tmp = (
                adata.obs.loc[v]
                .filter(aggregate_cols)
                .drop_duplicates()
                .reset_index(drop=True)
                .rename(index={0: pseudocell})
            )
            metadata = pd.concat(
                [
                    metadata,
                    pd.concat(
                        [
                            tmp,
                            pd.DataFrame(
                                {"ncells_aggregated": len(v), "cell_ids": ", ".join(v.tolist())},
                                index=tmp.index,
                            ),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
    # Check that the aggregation was done correctly
    assert metadata.shape[0] == grouped_obs.shape[0]

    ## If summing up the gene counts:
    if sum_gene:
        if "GeneSymbol" in adata.var.columns:
            gene_meta = pd.DataFrame(
                data=[" and ".join(adata.var.GeneSymbol)],
                index=["__".join(adata.var.index)],
                columns=["GeneSymbol"],
            )
        elif "features" in adata.var.columns:
            gene_meta = pd.DataFrame(
                data=[" and ".join(adata.var.features)],
                index=["__".join(adata.var.index)],
                columns=["GeneSymbol"],
            )
        else:
            gene_meta = pd.DataFrame(index=["__".join(adata.var.index)])
    else:
        gene_meta = adata.var

    adata_aggr = AnnData(X=np.asarray(pseudoexpression_matrix), obs=metadata, var=gene_meta)

    return adata_aggr


def find_cells_with_high_loadings_for_factor(
    factor_loadings: Union[np.ndarray, pd.Series],
    iqr_threshold: float = 0.5,
    value: Optional[float] = None,
    plot: bool = True,
) -> Union[np.ndarray, pd.Series]:
    """Identifies cells with high loadings for a given factor, using either an IQR-based cutoff
    (`iqr_threshold`) or a user-defined threshold (`value`).

    Parameters:
    ----------
        factor_loadings (Union[np.ndarray, pd.Series]): Array or Series of factor loadings per cell.
        iqr_threshold (float): Multiplier for the interquartile range (IQR) to define outlier threshold. Default is 0.5.
        value (Optional[float]): If specified, this value overrides the IQR-based upper cutoff. Default is None.
        plot (bool): If True, plots a histogram of loadings with thresholds. Default is True.

    Returns:
    -------
        high_loading_cells (Union[np.ndarray, pd.Series]): Cells with high factor loadings. Returns a `pd.Series` if
            `factor_loadings` is a `pd.Series`, otherwise returns a NumPy array of selected values.
    """
    if isinstance(factor_loadings, pd.Series):
        name = factor_loadings.name
        cell_idx = factor_loadings.index
        factor_loadings = factor_loadings.to_numpy()
    else:
        name = "Factor loadings"
        cell_idx = None

    p_75 = np.percentile(factor_loadings, 75)
    p_25 = np.percentile(factor_loadings, 25)
    IQR = iqr(factor_loadings)
    mean_f = np.mean(factor_loadings)

    lower_cutoff = p_25 - iqr_threshold * IQR
    upper_cutoff = value if value is not None else p_75 + iqr_threshold * IQR

    cell_subset = np.where(factor_loadings > upper_cutoff)[0]

    if plot:
        sns.histplot(factor_loadings, kde=True, color="royalblue")
        plt.xlabel(f"{name}")
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        plt.vlines(x=mean_f, ymin=0, ymax=y_max, color="darkslategrey", linestyles="--")
        plt.vlines(x=upper_cutoff, ymin=0, ymax=y_max, color="darkslategrey", linestyles=":")
        plt.vlines(x=lower_cutoff, ymin=0, ymax=y_max, color="darkslategrey", linestyles=":")

    return (
        pd.Series(factor_loadings[cell_subset], index=cell_idx[cell_subset], name=name)
        if cell_idx is not None
        else factor_loadings[cell_subset]
    )


def annotate_factor_GSEA(
    factor_loadings: pd.Series,
    adata_var: pd.DataFrame,
    hgnc_column: str,
    n_top_genes: int = 30,
    background_genes: Optional[List[str]] = None,
    databases: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Annotates factors by gene set enrichment analysis (GSEA) on top-loading genes for the given
    factor and optionally plots enriched terms.

    Parameters:
    ----------
        factor_loadings (pd.Series): Gene loadings for a factor, indexed by gene IDs matching `adata_var`.
        adata_var (pd.DataFrame): DataFrame corresponding to `adata.var` containing gene metadata.
        hgnc_column (str): Name of the column in `adata_var` that contains HGNC gene symbols.
        n_top_genes (int): Number of top absolute-loading genes to use for the enrichment analysis. Default is 30.
        background_genes (Optional[List[str]]): List of background genes for the enrichment analysis.
            If None, all genes in `adata_var[hgnc_column]` are used. Default is None.
        databases (Optional[List[str]]): List of gene set databases to query (e.g., "GO_Biological_Process_2023").
            If None, a default list is used. Default is None.
        plot_results (bool): Whether to generate a dotplot of the top enriched terms. Default is False.

    Returns:
    -------
        top_pathway (List[str]): Name of the top enriched pathway with the smallest adjusted p-value.
            If more than one pathways have the same p-value, all of them are included.
        pathway_genes (List[str]): Genes involved in the top enriched pathway(s).
    """

    if databases is None:
        databases_bio = [
            "GO_Biological_Process_2023",
            "GO_Molecular_Function_2023",
            "KEGG_2021_Human",
            "Reactome_2022",
        ]
    else:
        databases_bio = databases

    if background_genes is None:
        background_genes = adata_var[hgnc_column].tolist()

    enr_bio = gp.enrichr(
        gene_list=adata_var.loc[factor_loadings.abs().nlargest(n_top_genes).index][
            hgnc_column
        ].tolist(),
        gene_sets=databases_bio,
        background=background_genes,
        outdir=None,
    )
    enr_bio_sign = enr_bio.results.loc[enr_bio.results["Adjusted P-value"] <= 0.05]
    if enr_bio_sign.shape[0] > 0:
        top_pathways = (
            enr_bio_sign.groupby("Gene_set")
            .apply(
                lambda x: x.nsmallest(n=1, columns=["Adjusted P-value"], keep="all").Term.tolist(),
                include_groups=False,
            )
            .rename("Term")
            .to_frame()
        )
        top_pathways = top_pathways.explode("Term")
        top_genes = enr_bio_sign.loc[enr_bio_sign.Term.isin(top_pathways.Term)].Genes.to_frame()
        top_genes.index = top_pathways.index
        top_pathways_genes = pd.concat([top_pathways, top_genes], axis=1, ignore_index=False)
    else:
        top_pathways_genes = None
    return top_pathways_genes


def assign_factor_to_known_pathways(
    known_pathways: pd.DataFrame,
    gene_loadings: pd.Series,
    n_top_genes: int = 20,
    return_pathways_genes: bool = False,
) -> Union[str, Tuple[str, pd.Series, pd.Series]]:
    """Ranks top genes for a given factor, and then assigns the factor to known pathway(s) based on
    pathway occurrences among the top-loading genes.

    Parameters:
    ----------
        known_pathways (pd.DataFrame): DataFrame containing known pathway annotations for genes, with columns
            like 'Geneid' (e.g. ENSEMBL gene ID), 'Gene' (e.g. HGNC gene name), and 'Pathway_Name'.
        gene_loadings (pd.Series): Series of gene loadings for a given factor. The row indices must correspond to
            the gene IDs in the `Geneid` column of `known_pathways`.
        n_top_genes (int): Number of top absolute-loading genes to consider for pathway assignment. Default is 20.
        return_pathways_genes (bool): If True, also returns gene counts per pathway and contributing genes. Default is False.

    Returns:
    -------
        top_pathway (str): Name of the top enriched pathway supported by at least two genes.
        pathway_counts (pd.Series): (Returned if `return_pathways_genes=True`) Number of top genes supporting each pathway.
        top_pathways_genes (pd.Series): (Returned if `return_pathways_genes=True`) Top genes supporting each enriched pathway.
    """
    info_loadings = pd.DataFrame(gene_loadings.sort_values(ascending=False))
    info_loadings["rank"] = np.arange(1, info_loadings.shape[0] + 1)
    info_loadings = info_loadings.merge(
        pd.DataFrame(gene_loadings.abs().sort_values(ascending=False)).assign(
            absolute_rank=np.arange(1, gene_loadings.shape[0] + 1)
        )["absolute_rank"],
        right_index=True,
        left_index=True,
    )
    info_loadings = info_loadings.merge(
        known_pathways.filter(regex="ene|Pathway_Name"), on="Geneid", how="left"
    )

    top_genes = (
        info_loadings.loc[info_loadings.absolute_rank.isin(np.arange(1, int(n_top_genes) + 1))]
        .dropna(subset=["Gene"])
        .Gene.unique()
    )

    pathway_counts = (
        info_loadings.loc[info_loadings.Gene.isin(top_genes)]
        .groupby("Pathway_Name", observed=True)
        .apply(lambda x: x.Gene.nunique(), include_groups=False)
        .sort_values(ascending=False)
    )

    pathway_counts = pathway_counts[pathway_counts >= 2]
    top_pathway = pathway_counts.nlargest(1, keep="all").index.tolist()

    if len(top_pathway) > 1:
        top_pathway = ", ".join(top_pathway)
    elif len(top_pathway) == 1:
        top_pathway = top_pathway[0]
    else:
        top_pathway = None

    if return_pathways_genes:
        cut_off = pathway_counts.nlargest(1, keep="all").iloc[0] - 4
        top_pathways_genes = (
            info_loadings.loc[
                (info_loadings.Gene.isin(top_genes))
                & (
                    info_loadings.Pathway_Name.isin(
                        pathway_counts[pathway_counts >= cut_off].index
                    )
                )
            ]
            .groupby("Pathway_Name", observed=True)
            .apply(lambda x: x.Gene.unique(), include_groups=False)
        )
        return top_pathway, pathway_counts, top_pathways_genes
    else:
        return top_pathway
