import argparse
import os
import re
import sys
from textwrap import wrap
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
import scanpy as sc
import seaborn as sns
import torch
import umap
import webcolors
from anndata import AnnData
from matplotlib import cm, colormaps, colors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from scipy.stats import probplot, zscore
from tqdm import tqdm

from src.analysis._utils import add_livi_umaps_to_cell_metadata, compute_umap

sc.settings.set_figure_params(
    dpi=100, dpi_save=400, facecolor="white", transparent=True, figsize=(7, 6)
)
plt.rcParams["axes.grid"] = False


def visualise_cell_state_latent(
    z: pd.DataFrame,
    cell_metadata: pd.DataFrame,
    output_dir: str,
    of_prefix: str,
    format: str,
    args: argparse.Namespace,
) -> None:
    """Visualize LIVI's cell-state latent space using UMAP (Uniform Manifold Approximation and
    Projection) .

    Parameters
    ----------
        z (pd.DataFrame): DataFrame containing cell-state factor values (cells x factors).
        cell_metadata (pd.DataFrame): DataFrame containing cell metadata.
        output_dir (str): Directory to save the figure.
        of_prefix (str): Prefix for the output figure.
        format (str): The file format, e.g. 'png', 'pdf', 'svg', ..., to save the figure to.
        args (argparse.Namespace): Parsed command-line arguments; must include the name
        of cell-type and batch columns in cell_metadata.

    Returns
    -------
        None
    """

    umap_df = compute_umap(z, colnames=["UMAP1", "UMAP2"], add_latent=False)
    umap_df = umap_df.merge(
        cell_metadata.filter([args.celltype_column, args.batch_column]),
        how="left",
        right_index=True,
        left_index=True,
    )

    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15, 5), dpi=100)
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue=args.celltype_column,
        data=umap_df,
        ax=axs[0],
        s=3,
        palette="tab20",
        rasterized=format == "svg",
    )
    axs[0].legend(
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        ncol=2,
        markerscale=3,
    )
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue=args.batch_column,
        data=umap_df,
        ax=axs[1],
        s=3,
        palette="tab20",
        rasterized=format == "svg",
    )
    axs[1].legend(
        title="Batch",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        ncol=1,
        markerscale=3,
    )
    plt.suptitle(f"{os.path.basename(args.model_run_dir)}")
    plt.savefig(
        os.path.join(output_dir, f"{of_prefix}_cell-state_latent_UMAP.{format}"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )


def visualise_livi_embeddings(
    cell_metadata: pd.DataFrame,
    output_dir: str,
    of_prefix: str,
    format: str,
    plot_title: str,
    celltype_column: str,
    batch_column: Optional[str] = None,
    sex_column: Optional[str] = None,
    livi_embeddings: Optional[dict] = None,
) -> None:
    """Visualize LIVI embeddings using UMAP (Uniform Manifold Approximation and Projection).

    Parameters
    ----------
        cell_metadata (pd.DataFrame): DataFrame containing cell information.
        output_dir (str): Directory to save the figures.
        of_prefix (str): Prefix for the output figures.
        format (str): The file format, e.g. 'png', 'pdf', 'svg', ..., to save the figure to.
        plot_title (str): Title for the plots.
        celltype_column (str): Column containing cell type information.
        batch_column (str, optional): Column containing batch information.
        sex_column (str, optional): Column containing sex information.
        livi_embeddings (dict, optional): Dictionary containing LIVI embeddings.
            Must be provided is the UMAPs are not precomputed.

    Returns
    -------
        None
    """

    if cell_metadata.filter(like="UMAP").shape[1] == 0 and livi_embeddings is None:
        raise ValueError(
            "No precomputed UMAP in `cell_metadata` and no `livi_embeddings` to compute it provided."
        )

    ## Cell-state latent space
    if (
        "UMAP1_cell-state_latent" not in cell_metadata.columns
        and "UMAP2_cell-state_latent" not in cell_metadata.columns
    ):
        try:
            zbase = livi_embeddings["zbase"]
        except KeyError:
            zbase = livi_embeddings["cell-state_latent"]
        if isinstance(zbase, torch.Tensor):
            zbase = zbase.detach().cpu().numpy()
        zbase = pd.DataFrame(
            zbase,
            index=cell_metadata.index,
            columns=[f"Cell-state_Factor{f+1}" for f in range(zbase.shape[1])],
        )
        umap_df = compute_umap(zbase, colnames=["UMAP1", "UMAP2"], add_latent=False)
        cell_metadata = cell_metadata.merge(
            umap_df,
            how="left",
            right_index=True,
            left_index=True,
        )

    r = 2 if batch_column else 1

    fig = plt.figure(layout="constrained", figsize=(7 * r, 5), dpi=100)
    ax1 = fig.add_subplot(1, 1 * r, 1)  # UMAP cell type
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue=celltype_column,
        data=cell_metadata,
        ax=ax1,
        s=3,
        palette="tab20",
        rasterized=format == "svg",
    )
    ax1.legend(
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        ncol=2,
    )
    if batch_column:
        ax2 = fig.add_subplot(1, 2, 2)
        sns.scatterplot(
            x="UMAP1",
            y="UMAP2",
            hue=batch_column,
            data=cell_metadata,
            ax=ax2,
            s=3,
            palette="tab20",
            rasterized=format == "svg",
        )
        ax2.legend(
            title="Batch",
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            frameon=False,
            fontsize=13,
            title_fontsize=13,
            ncol=1,
        )
    plt.suptitle(plot_title)
    plt.savefig(
        os.path.join(output_dir, f"{of_prefix}_cell-state_latent_UMAP.{format}"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.close()

    ## CxG latent
    if (
        "UMAP1_CxG_effects" not in cell_metadata.columns
        and "UMAP2_CxG_effects" not in cell_metadata.columns
    ):
        U = livi_embeddings["CxG_latent"]
        if isinstance(U, torch.Tensor):
            U = U.detach().cpu().numpy()
        U = pd.DataFrame(
            U,
            index=cell_metadata.index,
            columns=[f"CxG_Factor{f+1}" for f in range(U.shape[1])],
        )
        umap_df = compute_umap(U, colnames=["UMAP1_CxG", "UMAP2_CxG"], add_latent=False)
        cell_metadata = cell_metadata.merge(
            umap_df,
            how="left",
            right_index=True,
            left_index=True,
        )

    r = 3 if batch_column and sex_column else 2 if batch_column else 1

    fig = plt.figure(layout="constrained", figsize=(7 * r, 5), dpi=100)
    ax1 = fig.add_subplot(1, 1 * r, 1)  # UMAP cell type
    sns.scatterplot(
        x="UMAP1_CxG",
        y="UMAP2_CxG",
        hue=celltype_column,
        data=cell_metadata,
        ax=ax1,
        s=3,
        palette="tab20",
        rasterized=format == "svg",
    )
    ax1.legend(
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        ncol=2,
    )
    if batch_column:
        if sex_column:
            ax3 = fig.add_subplot(1, 3, 3)
            sns.scatterplot(
                x="UMAP1_CxG",
                y="UMAP2_CxG",
                hue=sex_column,
                data=cell_metadata,
                ax=ax3,
                s=6,
                palette="tab20",
                rasterized=format == "svg",
            )
            ax3.legend(
                title="Sex",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=13,
                title_fontsize=13,
                ncol=1,
            )
            ax2 = fig.add_subplot(1, 3, 2)
        else:
            ax2 = fig.add_subplot(1, 2, 2)
        sns.scatterplot(
            x="UMAP1_CxG",
            y="UMAP1_CxG",
            hue=batch_column,
            data=cell_metadata,
            ax=ax2,
            s=3,
            palette="tab20",
            rasterized=format == "svg",
        )
        ax2.legend(
            title="Batch",
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            frameon=False,
            fontsize=13,
            title_fontsize=13,
            ncol=1,
        )
    plt.suptitle(plot_title)
    plt.savefig(
        os.path.join(output_dir, f"{of_prefix}_CxG_latent_UMAP.{format}"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.close()

    ## Persistent latent
    if (
        "UMAP1_persistent_effects" not in cell_metadata.columns
        and "UMAP2_persistent_effects" not in cell_metadata.columns
    ):
        V = livi_embeddings["persistent_latent"]
        if V is None:
            sys.exit(0)
        if batch_column is None and sex_column is None:
            sys.exit(0)
        if isinstance(V, torch.Tensor):
            V = V.detach().cpu().numpy()
        V = pd.DataFrame(
            V,
            index=cell_metadata.index,
            columns=[f"CxG_Factor{f+1}" for f in range(U.shape[1])],
        )
        umap_df = compute_umap(
            U, colnames=["UMAP1_persistent", "UMAP2_persistent"], add_latent=False
        )
        cell_metadata = cell_metadata.merge(
            umap_df,
            how="left",
            right_index=True,
            left_index=True,
        )

    r = 2 if batch_column and sex_column else 1
    fig = plt.figure(layout="constrained", figsize=(7 * r, 5), dpi=100)
    if batch_column:
        if sex_column:
            ax2 = fig.add_subplot(1, 1, 2)
            sns.scatterplot(
                x="UMAP1_persistent",
                y="UMAP2_persistent",
                hue=sex_column,
                data=cell_metadata,
                ax=ax2,
                s=6,
                palette="tab20",
                rasterized=format == "svg",
            )
            ax2.legend(
                title="Sex",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=13,
                title_fontsize=13,
                ncol=1,
            )
            ax1 = fig.add_subplot(1, 1, 1)
        else:
            ax1 = fig.add_subplot(1, 1, 1)
        sns.scatterplot(
            x="UMAP1_persistent",
            y="UMAP2_persistent",
            hue=batch_column,
            data=cell_metadata,
            ax=ax1,
            s=6,
            palette="tab20",
            rasterized=format == "svg",
        )
        ax1.legend(
            title="Batch",
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            frameon=False,
            fontsize=13,
            title_fontsize=13,
            ncol=1,
        )
    plt.suptitle(plot_title)
    plt.savefig(
        os.path.join(output_dir, f"{of_prefix}_persistent_latent_UMAP.{format}"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.close()


def cell_state_factors_heatmap(
    cell_state_factors: np.ndarray,
    cell_idx: Union[List[int], np.ndarray],
    cell_metadata: pd.DataFrame,
    celltype_column: str,
    row_cluster: bool,
    column_cluster: bool,
    metric: str = "euclidean",
    factors: Optional[Union[List[int], np.ndarray]] = None,
    zscore: Optional[int] = None,
    color_map: Optional[str] = None,
    savefig: Optional[str] = None,
    format: Optional[str] = None,
    return_df: bool = False,
) -> Union[pd.DataFrame, None]:
    """Make a heatmap of the (variational) autoencoder (AE) latent space (factors).

    Parameters
    ----------
        cell_state_factors (numpy 2D-array): LIVI's cell-state latent space (factors) (cells x factors).
        cell_idx (list or numpy 1D-array): Integer-location based indices of the cells in the `cell_metadata`.
        cell_metadata (pandas.DataFrame): Dataframe containing metadata information about the cells.
        celltype_column (str): Column in cell_metadata containing the cell type information.
        factors (numpy 1D-array): Selected factors to plot. If None all factors (latent space dims) will be used.
        metric (str): Distance metric to use for the hierarchical clustering
            (see: https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist)
        zscore (int, optional): Whether to z-score the data row- (0) or column-wise (1). Default is None.
        color_map (str, optional): Name of `matplotlib` colormap to use.
        savefig (str, optional): Absolute filepath to save the figure. If None, the figure is not saved. Default is None.
        format (str, optional): The file format, e.g. 'png', 'pdf', 'svg', ..., to save the figure to. If None, then the file format is inferred from the
            extension of savefig, if savefig is not None.
    """

    if savefig:
        prefix, ext = os.path.splitext(savefig)
        ext = "." + format if format else ".png" if ext == "" else ext
    else:
        ext = "." + format if format else ".png"

    if factors is not None:
        df_plot = pd.DataFrame(cell_state_factors[:, factors], columns=[str(t) for t in factors])
    else:
        df_plot = pd.DataFrame(
            cell_state_factors,
            columns=["Factor" + str(t) for t in range(1, cell_state_factors.shape[1] + 1)],
        )

    df_plot["Cell type"] = cell_metadata.iloc[cell_idx][celltype_column].values
    # Average of selected factors across cells belonging to the same day
    df_celltype = df_plot.groupby("Cell type").mean().reset_index().set_index("Cell type")

    sns.set_style("white")

    if color_map is None:
        color_map = "vlag" if zscore is not None else None
    if color_map == "vlag":
        sns.clustermap(
            df_celltype,
            row_cluster=row_cluster,
            col_cluster=column_cluster,
            metric=metric,
            z_score=zscore,
            cmap=color_map,
            center=0.0,
            figsize=(10, 10),
            rasterized=ext == ".svg",
        )
    else:
        sns.clustermap(
            df_celltype,
            row_cluster=row_cluster,
            col_cluster=column_cluster,
            metric=metric,
            z_score=zscore,
            cmap=color_map,
            figsize=(10, 10),
            rasterized=ext == ".svg",
        )
    if savefig:
        plt.savefig(
            f"{prefix}_Heatmap_cell-state-Factors_Celltype{ext}",
            bbox_inches="tight",
            dpi=200,
            transparent=True,
        )
    
    if return_df:
        return df_celltype


def plot_celltype_factors(
    cell_type_factor_df: pd.DataFrame,
    celltype_column: str,
    marker_size: int = 2,
    d: int = 10,
    legend_fontsize: int = 20,
    axis_title_fontsize: int = 26,
    heatmap_color_map: str = "vlag",
    zscore: bool = False,
    return_celltype_factors: bool = False,
    savefig: Optional[str] = None,
    format: Optional[str] = None,
) -> Union[None, Tuple[dict, dict]]:
    """Plot factors that characterize each cell type on a latent space UMAP. Characteristic factor
    for each celltype is the one with the largest (zscored) value (factor values are z-scored
    within each celltype).

    Parameters
    ----------
        cell_type_factor_df (pd.DataFrame): DataFrame containing cell type label and factor values for each cell.
        celltype_column (str): Column in cell_metadata containing the cell type information.
        marker_size (int, optional): Size of markers in scatterplots. Default is 2.
        d (int, optional): Controls figure size. Resulting figure will have a size of ~ 2d x (d/2)n_unique_celltype_factors. Default is 10.
        legend_fontsize (int, optional): Font size for plot legends. Default is 20.
        axis_title_fontsize (int, optional): Font size for axis titles. Default is 26.
        heatmap_color_map (str, optional): Color map for the heatmap. Default is "vlag".
        zscore (bool, optional): Whether to z-score the factor values across factors for a given celltype. Default is False.
        return_celltype_factors (bool, optional): Whether to return the characteristic celltype factors. Default is False.
        savefig (str, optional): Absolute filepath to save the figure. If None, the figure is not saved. Default is None.
        format (str, optional): The file format, e.g. 'png', 'pdf', 'svg', ..., to save the figure to. If None, then the file format is inferred from the
            extension of savefig, if savefig is not None.

    Returns
    -------
        None or Tuple[dict, dict]: If return_celltype_factors is True, returns a tuple containing dictionaries
      of the factors with highest and lowest value for each cell type.
    """

    if savefig:
        prefix, ext = os.path.splitext(savefig)
        ext = "." + format if format else ".png" if ext == "" else ext
    else:
        ext = "." + format if format else ".png"

    df_celltype = (
        cell_type_factor_df.groupby(celltype_column).mean().reset_index().set_index(celltype_column)
    )

    if zscore:
        df_celltype_zscored = pd.DataFrame(
            zscore(df_celltype.to_numpy(), axis=1),
            index=df_celltype.index,
            columns=df_celltype.columns,
        )
        celltype_factor_high = df_celltype_zscored.idxmax(axis=1).to_dict()
        celltype_factor_low = df_celltype_zscored.idxmin(axis=1).to_dict()
    else:
        celltype_factor_high = df_celltype.idxmax(axis=1).to_dict()
        celltype_factor_low = df_celltype.idxmin(axis=1).to_dict()
        
    if cell_type_factor_df.filter(like="UMAP").shape[1] == 0:
        latent = cell_type_factor_df.filter(regex="F|factor")
        umap_df = compute_umap(latent, colnames=["UMAP1", "UMAP2"], add_latent=False)
        cell_type_factor_df = cell_type_factor_df.merge(
            umap_df,
            how="left",
            right_index=True,
            left_index=True,
        )

    unique_factors = list(
        set(list(celltype_factor_high.values()) + list(celltype_factor_low.values()))
    )

    if len(unique_factors) % 2 == 0:
        n_rows = len(unique_factors) // 2 + 1
    else:
        n_rows = int(len(unique_factors) // 2) + 2

    d = d
    figure_size = (2 * d, n_rows * d)
    fig, axs = plt.subplots(ncols=2, nrows=n_rows, figsize=(20, 40), constrained_layout=False)
    axs = axs.flatten()

    for i, ax in enumerate(tqdm(axs)):
        if i == 0:
            sns.scatterplot(
                x="UMAP1",
                y="UMAP2",
                hue=celltype_column,
                data=cell_type_factor_df,
                ax=axs[0],
                s=3,
                palette="tab20",
                rasterized=ext == ".svg",
            )
            axs[0].legend(
                title="Cell type",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize + 2,
                ncol=2,
            )
            axs[0].set_title(
                label="Cell type", fontdict={"fontsize": axis_title_fontsize}, loc="center"
            )
        elif i == 1:
            if zscore:
                sns.heatmap(df_celltype_zscored, cmap=heatmap_color_map, center=0.0, ax=axs[1])
            else:
                sns.heatmap(df_celltype, cmap=heatmap_color_map, center=0.0, ax=axs[1])
        elif 1 < i <= len(unique_factors) + 2:
            sns.scatterplot(
                x="UMAP1",
                y="UMAP2",
                hue=unique_factors[i - 2],
                data=cell_type_factor_df,
                ax=ax,
                s=marker_size,
                palette="vlag",
                legend=False,
                rasterized=ext == ".svg",
            )
            sm = cm.ScalarMappable(
                cmap="vlag",
                norm=colors.Normalize(
                    vmin=cell_type_factor_df[unique_factors[i - 2]].min(),
                    vmax=cell_type_factor_df[unique_factors[i - 2]].max(),
                ),
            )
            cb = plt.colorbar(sm, ax=ax)
            ax.set_title(
                label=unique_factors[i - 2],
                fontdict={"fontsize": axis_title_fontsize},
               loc="center",
            )
            cb.ax.tick_params(labelsize=12)
            ax.xaxis.label.set_fontsize(axis_title_fontsize - 2)
            ax.yaxis.label.set_fontsize(axis_title_fontsize - 2)
            ax.tick_params(labelsize=legend_fontsize)
        else:
            fig.delaxes(ax)

    if savefig:
        plt.savefig(
            f"{prefix}_celltype-factors_UMAP{ext}", bbox_inches="tight", dpi=400, transparent=True
        )
    plt.close()
    
    if return_celltype_factors:
        return (celltype_factor_high, celltype_factor_low)


def QQplot(pvalues: Union[list, np.ndarray], savefig: Optional[str] = None) -> None:
    """Generate a QQ plot to assess the deviation of observed p-values from expected uniform
    distribution.

    Parameters
    ----------
        pvalues (array-like): The observed p-values.
        savefig (str or None): If provided, the path to save the generated QQ plot. Default is None.

    Returns
    -------
        None
    """
    (osm, osr), _ = probplot(pvalues, dist="uniform")
    b, a = np.polyfit(osm, osr, deg=1)
    ax = plt.gca()
    df = pd.DataFrame({"osm": -np.log10(osm), "osr": -np.log10(osr)})
    sns.scatterplot(x="osm", y="osr", data=df, ax=ax, edgecolor=None)
    x = np.linspace(0, ax.get_xlim()[1])
    ax.plot(x, a + b * x, c="lightgrey", linestyle=":", scaley=True)
    ax.set(xlabel=r"Expected $-\log_{10} P$", ylabel=r"Observed $-\log_{10} P$")

    if savefig is not None:
        plt.savefig(savefig, transparent=True, dpi=200, bbox_inches="tight")


def plot_A_sparsity(
    A: pd.DataFrame,
    associated_factors: list,
    plot_title: str,
    savefig: Optional[str] = None,
    format: Optional[str] = None,
) -> None:
    """Plot the sparsity of LIVI's design matrix, A and clusters formed between CxG factors based
    on their assignments to cell-state factors.

    Parameters
    ----------
        A (pd.DataFrame): LIVI's design matrix, A.
        associated_factors (list): List of CxG factors associated with genetic variables.
        plot_title (str): Title for the plot.
        savefig (str or None): If provided, the absolute path to save the generated plots. Default is None.
        format (str or None): The file format, e.g. 'png', 'pdf', 'svg', ..., to save the figure to. If None, then the file format is inferred from the
            extension of savefig, if savefig is not None.

    Returns
    -------
        None
    """

    if savefig:
        prefix, ext = os.path.splitext(savefig)
        ext = "." + format if format else ".png" if ext == "" else ext
    else:
        ext = "." + format if format else ".png"

    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    sns.histplot(
        A.apply(lambda x: x.sum(), axis=0),
        kde=True,
        color="cornflowerblue",
        ax=axs[0],
        rasterized=ext == ".svg",
    )
    axs[0].set_title(
        "\n".join(wrap("Number of assigned cell-state factors for each GxC factor", 60)),
        fontsize=13,
    )
    axs[0].tick_params(axis="both", labelsize=12)

    sns.histplot(
        A.filter(associated_factors).apply(lambda x: x.sum(), axis=0),
        kde=True,
        color="navy",
        bins=20,
        ax=axs[1],
        rasterized=ext == ".svg",
    )
    axs[1].set_title(
        "\n".join(
            wrap("Number of assigned cell-state factors for each SIGNIFICANT GxC factor", 60)
        ),
        fontsize=13,
    )
    axs[1].tick_params(axis="both", labelsize=12)

    fig.suptitle(plot_title)
    fig.tight_layout()
    if savefig:
        plt.savefig(
            f"{prefix}_A-design-matrix_sparsity{ext}",
            dpi=200,
            transparent=True,
            bbox_inches="tight",
        )
    plt.close()

    sns.clustermap(
        A.filter(associated_factors).astype(int),
        col_cluster=True,
        row_cluster=False,
        metric="hamming",
        cmap="Reds",
        rasterized=ext == ".svg",
    )
    if savefig:
        plt.savefig(
            f"{prefix}_A-design-matrix_CxG_factor_clustering{ext}",
            dpi=200,
            transparent=True,
            bbox_inches="tight",
        )
    plt.close()


def plot_CxG_factor_cor(
    U: pd.DataFrame,
    associated_factors: list,
    savefig: Optional[str] = None,
    format: Optional[str] = None,
) -> None:
    """Plot the pairwise correlations of SNP-associated CxG factors and clustering of the
    individuals based on those.

    Parameters
    ----------
        U (pd.DataFrame): Dataframe containing CxG factors (individuals x factors).
        associated_factors (list): List of associated factors to filter and plot.
        savefig (str or None): If provided, the path to save the generated plots. Default is None.
        format (str or None): The file format, e.g. 'png', 'pdf', 'svg', ..., to save the figure to. If None, then the file format is inferred from the
            extension of savefig, if savefig is not None.

    Returns
    -------
        None
    """

    U = U.filter(associated_factors)
    pairwise_correlations = U.corr(method="pearson")

    if savefig:
        prefix, ext = os.path.splitext(savefig)
        ext = "." + format if format else ".png" if ext == "" else ext
    else:
        ext = "." + format if format else ".png"

    # Visualize pearson cor between significant factors
    plt.figure(figsize=(12, 10))
    sns.heatmap(pairwise_correlations, cmap="vlag", center=0, rasterized=ext == ".svg")
    plt.title("Pearson's $\\rho$ between significant CxG factors", fontsize=15, pad=20)
    if savefig:
        plt.savefig(
            f"{prefix}_CxG_factor_correlations{ext}",
            dpi=200,
            transparent=True,
            bbox_inches="tight",
        )
    plt.close()
    # Cluster individuals based on significant factor values
    clm = sns.clustermap(
        U.filter(associated_factors),
        col_cluster=False,
        row_cluster=True,
        cmap="vlag",
        center=0,
        cbar_pos=(0.99, 0.14, 0.022, 0.2),
        rasterized=ext == ".svg",
    )
    # ax = clm.ax_heatmap
    # ax.set_title("Clustering of individuals based on the cosine distance between CxG factors", fontsize=15, pad=20)
    if savefig:
        clm.savefig(
            f"{prefix}_IID_clustering_based_on_CxG_factors{ext}",
            dpi=200,
            transparent=True,
            bbox_inches="tight",
        )
    plt.close()
