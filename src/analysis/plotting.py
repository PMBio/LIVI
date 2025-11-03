import argparse
import os
import re
import sys
import textwrap
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
from anndata import AnnData
from matplotlib import cm, colormaps, colors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, probplot, zscore
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.analysis._utils import (
    add_livi_umaps_to_cell_metadata,
    aggregate_cell_counts,
    assign_U_to_celltype,
    calculate_GxC_effect,
    calculate_GxC_gene_effect,
    compute_umap,
)

sc.settings.set_figure_params(
    dpi=100, dpi_save=400, facecolor="white", transparent=True, figsize=(7, 6)
)
plt.rcParams["axes.grid"] = False

rng = np.random.default_rng()


## Customize gseapy plot (https://github.com/zqfang/GSEApy/blob/master/gseapy/plot.py)
from gseapy.plot import DotPlot


class gp_DotPlot(DotPlot):

    def __init__(
        self,
        df: pd.core.frame.DataFrame,
        x: Optional[str] = None,
        y: str = "Term",
        hue: str = "Adjusted P-value",
        dot_scale: float = 5.0,
        x_order: Optional[List[str]] = None,
        y_order: Optional[List[str]] = None,
        thresh: float = 0.05,
        n_terms: int = 10,
        title: str = "",
        figsize: Tuple[float, float] = (6, 5.5),
        cmap: str = "viridis_r",
        ofname: Optional[str] = None,
        ax: Optional[str] = None,
        fig: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            df=df,
            x=x,
            y=y,
            hue=hue,
            dot_scale=dot_scale,
            x_order=x_order,
            y_order=y_order,
            thresh=thresh,
            n_terms=n_terms,
            title=title,
            figsize=figsize,
            cmap=cmap,
            ofname=ofname,
            **kwargs,
        )

        self.axis = ax
        self.fig = fig

    def get_ax(self):
        """Setup figure axes."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        # create fig
        if self.axis is not None and self.fig is not None:
            ax = self.axis
        else:
            if hasattr(sys, "ps1") and (self.ofname is None):
                # working inside python console, show figure
                fig = plt.figure(figsize=self.figsize)
            else:
                # If working on commandline, don't show figure
                fig = Figure(figsize=self.figsize)
                _canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            self.fig = fig
        return ax

    def scatter(
        self,
        outer_ring: bool = False,
        rasterize: bool = False,
    ):
        """Build scatter."""
        from matplotlib.category import UnitData

        # scatter colormap range
        # df = df.assign(colmap=self.data[self.colname].round().astype("int"))
        # make area bigger to better visualization
        # area = df["Hits_ratio"] * plt.rcParams["lines.linewidth"] * 100
        df = self.data.assign(
            area=(self.data["Hits_ratio"] * self.scale * plt.rcParams["lines.markersize"]).pow(2)
        )
        colmap = df[self.colname].astype(int)
        vmin = np.percentile(colmap.min(), 2)
        vmax = np.percentile(colmap.max(), 98)
        # vmin = np.percentile(df.colmap.min(), 2)
        # vmax = np.percentile(df.colmap.max(), 98)
        ax = self.get_ax()
        # if self.x is None:
        x, xlabel = self.set_x()
        y = self.y
        # set x, y order
        xunits = UnitData(self.get_x_order()) if self.x_order else None
        yunits = UnitData(self.get_y_order()) if self.y_order else None

        # outer ring
        if outer_ring:
            smax = df["area"].max()
            # TODO:
            # Matplotlib BUG: when setting edge colors,
            # there's the center of scatter could not aligned.
            # Must set backend to TKcario... to fix it
            # Instead, I just add more dots in the plot to get the ring
            blk_sc = ax.scatter(
                x=x,
                y=y,
                s=smax * 1.6,
                edgecolors="none",
                c="black",
                data=df,
                marker=self.marker,
                xunits=xunits,  # set x categorical order
                yunits=yunits,  # set y categorical order
                zorder=0,
                rasterized=rasterize,
            )
            wht_sc = ax.scatter(
                x=x,
                y=y,
                s=smax * 1.3,
                edgecolors="none",
                c="white",
                data=df,
                marker=self.marker,
                xunits=xunits,  # set x categorical order
                yunits=yunits,  # set y categorical order
                zorder=1,
                rasterized=rasterize,
            )
            # data = np.array(rg.get_offsets()) # get data coordinates
        # inner circle
        sc = ax.scatter(
            x=x,
            y=y,
            data=df,
            s="area",
            edgecolors="none",
            c=self.colname,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            marker=self.marker,
            xunits=xunits,  # set x categorical order
            yunits=yunits,  # set y categorical order
            zorder=2,
            rasterized=rasterize,
        )
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=27)
        ax.set_axisbelow(True)  # set grid blew other element
        ax.grid(axis="y", zorder=-1)  # zorder=-1.0
        ax.margins(x=0.20)

        ax.set_title(self.title, fontsize=30, fontweight="bold")
        self.add_colorbar(sc)

        return ax


def make_gp_dotplot(
    df: pd.DataFrame,
    column: str = "Adjusted P-value",
    x: Optional[str] = None,
    y: str = "Term",
    x_order: Union[List[str], bool] = False,
    y_order: Union[List[str], bool] = False,
    title: str = "",
    cutoff: float = 0.05,
    top_term: int = 10,
    size: float = 5,
    figsize: Tuple[float, float] = (4, 6),
    cmap: str = "viridis_r",
    ofname: Optional[str] = None,
    xticklabels_rot: Optional[float] = None,
    yticklabels_rot: Optional[float] = None,
    marker: str = "o",
    show_ring: bool = False,
    ax: Optional[str] = None,
    fig: Optional[str] = None,
    rasterize: bool = False,
    **kwargs: Any,
):

    dot = gp_DotPlot(
        df=df,
        x=x,
        y=y,
        x_order=x_order,
        y_order=y_order,
        hue=column,
        title=title,
        thresh=cutoff,
        n_terms=int(top_term),
        dot_scale=size,
        figsize=figsize,
        cmap=cmap,
        ofname=ofname,
        marker=marker,
        ax=ax,
        fig=fig,
    )
    ax = dot.scatter(outer_ring=show_ring, rasterize=rasterize)

    if xticklabels_rot:
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(xticklabels_rot)

    if yticklabels_rot:
        for label in ax.get_yticklabels():
            label.set_ha("right")
            label.set_rotation(yticklabels_rot)

    if ofname is None:
        return ax
    dot.fig.savefig(ofname, bbox_inches="tight", dpi=300)


def visualise_cell_state_latent(
    z: pd.DataFrame,
    cell_metadata: pd.DataFrame,
    output_dir: str,
    of_prefix: str,
    format: str,
    args: argparse.Namespace,
    save_umap=True,
) -> None:
    """Visualize LIVI's cell-state latent space using UMAP (Uniform Manifold Approximation and
    Projection) .

    Parameters
    ----------
        z (pd.DataFrame): DataFrame containing cell-state factor values (cells x factors).
        cell_metadata (pd.DataFrame): DataFrame containing cell metadata.
        output_dir (str): Directory to save the figure.
        of_prefix (str): Prefix for the output figure.
        format (str): The file format, e.g. 'png', 'pdf', 'eps', ..., to save the figure to.
        args (argparse.Namespace): Parsed command-line arguments; must include the name
        of cell-type and batch columns in cell_metadata.

    Returns
    -------
        None
    """

    umap_df = compute_umap(z, colnames=["UMAP1", "UMAP2"], add_latent=False)
    if save_umap:
        umap_df.to_csv(
            os.path.join(output_dir, f"{of_prefix}_cell-state_latent_UMAP.tsv"),
            sep="\t",
            index=True,
            header=True,
        )
    umap_df = umap_df.merge(
        cell_metadata.filter([args.celltype_column, args.batch_column]),
        how="left",
        right_index=True,
        left_index=True,
    )

    ncol_ct = np.ceil(cell_metadata[args.celltype_column].nunique() / 6)
    ncol_batch = np.ceil(cell_metadata[args.batch_column].nunique() / 6)
    legend_width = (ncol_ct + ncol_batch) * 2
    fig, axs = plt.subplots(
        nrows=1, ncols=2, constrained_layout=True, figsize=(10 + legend_width, 5), dpi=100
    )
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue=args.celltype_column,
        data=umap_df,
        ax=axs[0],
        s=3,
        palette="tab20",
        rasterized=format == "pdf",
    )
    axs[0].legend(
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        ncol=ncol_ct,
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
        rasterized=format == "pdf",
    )
    axs[1].legend(
        title="Batch",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        ncol=ncol_batch,
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
    format: str,
    plot_title: str,
    celltype_column: str,
    individual_column: str,
    batch_column: Optional[str] = None,
    sex_column: Optional[str] = None,
    livi_embeddings: Optional[dict] = None,
    of_prefix: Optional[str] = None,
) -> None:
    """Visualize LIVI embeddings using UMAP (Uniform Manifold Approximation and Projection).

    Parameters
    ----------
        cell_metadata (pd.DataFrame): DataFrame containing cell information.
        output_dir (str): Directory to save the figures.
        of_prefix (str): Prefix for the output figures.
        format (str): The file format, e.g. 'png', 'pdf', 'eps', ..., to save the figure to.
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
        umap_df = umap_df.merge(
            cell_metadata.filter([celltype_column, batch_column]),
            how="left",
            right_index=True,
            left_index=True,
        )

    r = 2 if batch_column else 1

    fig = plt.figure(layout="constrained", figsize=(7 * r, 5), dpi=100)
    ax1 = fig.add_subplot(1, r, 1)  # UMAP cell type
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue=celltype_column,
        data=umap_df,
        ax=ax1,
        s=3,
        palette="tab20",
        rasterized=format == "pdf",
    )
    ax1.legend(
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        ncol=2,
        markerscale=4,
    )
    if batch_column:
        ax2 = fig.add_subplot(1, r, 2)
        sns.scatterplot(
            x="UMAP1",
            y="UMAP2",
            hue=batch_column,
            data=umap_df,
            ax=ax2,
            s=3,
            palette="tab10",
            rasterized=format == "pdf",
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

    ## U embedding
    if "UMAP1_U" not in cell_metadata.columns and "UMAP2_U" not in cell_metadata.columns:
        U = livi_embeddings["U_embedding"]
        if U is None:
            sys.exit(0)
        if batch_column is None and sex_column is None:
            sys.exit(0)
        if isinstance(U, torch.Tensor):
            U = U.detach().cpu().numpy()
        U = pd.DataFrame(
            U,
            index=cell_metadata[individual_column].unique(),
            columns=[f"U_Factor{f+1}" for f in range(U.shape[1])],
        )
        umap_df = compute_umap(U, colnames=["UMAP1_U", "UMAP2_U"], add_latent=False)
        umap_df = umap_df.merge(
            cell_metadata.filter([individual_column, batch_column, sex_column])
            .drop_duplicates()
            .set_index(individual_column),
            how="left",
            right_index=True,
            left_index=True,
        )

    r = 2 if batch_column and sex_column else 1
    fig = plt.figure(layout="constrained", figsize=(7 * r, 5), dpi=100)
    if batch_column:
        ax1 = fig.add_subplot(1, r, 1)
        sns.scatterplot(
            x="UMAP1_U",
            y="UMAP2_U",
            hue=batch_column,
            data=umap_df,
            ax=ax1,
            s=6,
            palette="tab10",
            rasterized=format == "pdf",
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
        if sex_column:
            ax2 = fig.add_subplot(1, r, 2)
            sns.scatterplot(
                x="UMAP1_U",
                y="UMAP2_U",
                hue=sex_column,
                data=umap_df,
                ax=ax2,
                s=6,
                palette=["lightpink", "slategrey"],
                rasterized=format == "pdf",
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

    plt.suptitle("\n".join(textwrap.wrap(plot_title, 60)))
    plt.savefig(
        os.path.join(output_dir, f"{of_prefix}_U-embedding_UMAP.{format}"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.close()

    if batch_column or sex_column:
        pca_U = PCA(n_components=U.shape[1] - 1, random_state=32).fit_transform(U.to_numpy())
        pca_U = pd.DataFrame(
            pca_U, index=U.index, columns=[f"U_PC{i+1}" for i in range(pca_U.shape[1])]
        )
        pca_U = pca_U.merge(
            cell_metadata.filter([individual_column, batch_column, sex_column])
            .drop_duplicates()
            .set_index(individual_column),
            right_index=True,
            left_index=True,
        )

        if batch_column:
            fig, axs = plt.subplots(
                ncols=3, nrows=5, figsize=(12, 13), constrained_layout=True
            )  # ncols=2, nrows=n_rows, figsize=figure_size, constrained_layout=False)
            axs = axs.flatten()
            for p1 in range(2, 11):
                sns.scatterplot(
                    x="U_PC1",
                    y=f"U_PC{p1}",
                    hue=batch_column,
                    data=pca_U,
                    ax=axs[p1 - 2],
                    s=6,
                    rasterized=format == "pdf",
                )
                axs[p1 - 2].legend(
                    title="Batch",
                    loc="center left",
                    bbox_to_anchor=(1.03, 0.5),
                    frameon=False,
                    fontsize=14,
                    title_fontsize=16,
                    ncol=1,
                    markerscale=4,
                )
            axs_idx = p1 - 1
            for p2 in range(10, 16):
                sns.scatterplot(
                    x="U_PC2",
                    y=f"U_PC{p2}",
                    hue=batch_column,
                    data=pca_U,
                    ax=axs[axs_idx],
                    s=6,
                    rasterized=format == "pdf",
                )
                axs[axs_idx].legend(
                    title="Batch",
                    loc="center left",
                    bbox_to_anchor=(1.03, 0.5),
                    frameon=False,
                    fontsize=14,
                    title_fontsize=16,
                    ncol=1,
                    markerscale=4,
                )
                axs_idx += 1
            fig.suptitle("\n".join(textwrap.wrap(plot_title, 60)))
            plt.savefig(
                os.path.join(output_dir, f"{of_prefix}_U-embedding-PCA_Batch.{format}"),
                dpi=600,
                transparent=True,
                bbox_inches="tight",
            )
            plt.close()

        if sex_column:
            fig, axs = plt.subplots(
                ncols=3, nrows=5, figsize=(12, 13), constrained_layout=True
            )  # ncols=2, nrows=n_rows, figsize=figure_size, constrained_layout=False)
            axs = axs.flatten()
            for p1 in range(2, 11):
                sns.scatterplot(
                    x="U_PC1",
                    y=f"U_PC{p1}",
                    hue=sex_column,
                    data=pca_U,
                    ax=axs[p1 - 2],
                    palette=["lightpink", "slategrey"],
                    s=6,
                    rasterized=format == "pdf",
                )
                axs[p1 - 2].legend(
                    title="Sex",
                    loc="center left",
                    bbox_to_anchor=(1.03, 0.5),
                    frameon=False,
                    fontsize=14,
                    title_fontsize=16,
                    ncol=1,
                    markerscale=4,
                )

            axs_idx = p1 - 1
            for p2 in range(10, 16):
                sns.scatterplot(
                    x="U_PC2",
                    y=f"U_PC{p2}",
                    hue=sex_column,
                    data=pca_U,
                    ax=axs[axs_idx],
                    palette=["lightpink", "slategrey"],
                    s=6,
                    rasterized=format == "pdf",
                )
                axs[axs_idx].legend(
                    title="Sex",
                    loc="center left",
                    bbox_to_anchor=(1.03, 0.5),
                    frameon=False,
                    fontsize=14,
                    title_fontsize=16,
                    ncol=1,
                    markerscale=4,
                )
                axs_idx += 1
            fig.suptitle("\n".join(textwrap.wrap(plot_title, 60)))
            plt.savefig(
                os.path.join(output_dir, f"{of_prefix}_U-embedding-PCA_sex.{format}"),
                dpi=600,
                transparent=True,
                bbox_inches="tight",
            )
            plt.close()

    ## GxC latent
    if "UMAP1_GxC" not in cell_metadata.columns and "UMAP2_GxC" not in cell_metadata.columns:
        GxC = livi_embeddings["GxC_latent"]
        if isinstance(GxC, torch.Tensor):
            GxC = GxC.detach().cpu().numpy()
        GxC = pd.DataFrame(
            GxC,
            index=cell_metadata.index,
            columns=[f"GxC_Factor{f+1}" for f in range(GxC.shape[1])],
        )
        umap_df = compute_umap(GxC, colnames=["UMAP1_GxC", "UMAP2_GxC"], add_latent=False)
        umap_df = umap_df.merge(
            cell_metadata.filter([celltype_column, batch_column, sex_column]),
            how="left",
            right_index=True,
            left_index=True,
        )

    r = 3 if batch_column and sex_column else 2 if batch_column else 1

    fig = plt.figure(layout="constrained", figsize=(7 * r, 5), dpi=100)
    ax1 = fig.add_subplot(1, r, 1)  # UMAP cell type
    sns.scatterplot(
        x="UMAP1_GxC",
        y="UMAP2_GxC",
        hue=celltype_column,
        data=umap_df,
        ax=ax1,
        s=3,
        palette="tab20",
        rasterized=format == "pdf",
    )
    ax1.legend(
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        ncol=2,
        markerscale=4,
    )
    if batch_column:
        ax2 = fig.add_subplot(1, r, 2)
        sns.scatterplot(
            x="UMAP1_GxC",
            y="UMAP2_GxC",
            hue=batch_column,
            data=umap_df,
            ax=ax2,
            s=3,
            palette="tab10",
            rasterized=format == "pdf",
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
        if sex_column:
            ax3 = fig.add_subplot(1, r, 3)
            sns.scatterplot(
                x="UMAP1_GxC",
                y="UMAP2_GxC",
                hue=sex_column,
                data=umap_df,
                ax=ax3,
                s=6,
                palette=["lightpink", "slategrey"],
                rasterized=format == "pdf",
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
        else:
            ax2 = fig.add_subplot(1, 2, 2)

    plt.suptitle("\n".join(textwrap.wrap(plot_title, 60)))
    plt.savefig(
        os.path.join(output_dir, f"{of_prefix}_GxC-latent_UMAP.{format}"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.close()

    pca_GxC = PCA(n_components=GxC.shape[1] - 1, random_state=32).fit_transform(GxC.to_numpy())
    pca_GxC = pd.DataFrame(
        pca_GxC, index=GxC.index, columns=[f"GxC_PC{i+1}" for i in range(pca_GxC.shape[1])]
    )
    pca_GxC = pca_GxC.merge(
        cell_metadata.filter([individual_column, batch_column, celltype_column, sex_column]),
        right_index=True,
        left_index=True,
    )

    fig, axs = plt.subplots(
        ncols=3, nrows=5, figsize=(20, 13), constrained_layout=True
    )  # ncols=2, nrows=n_rows, figsize=figure_size, constrained_layout=False)
    axs = axs.flatten()
    for p1 in range(2, 11):
        sns.scatterplot(
            x="GxC_PC1",
            y=f"GxC_PC{p1}",
            hue=celltype_column,
            data=pca_GxC,
            ax=axs[p1 - 2],
            s=4,
            palette="tab20",
            rasterized=format == "pdf",
        )
        axs[p1 - 2].legend(
            title="Cell type",
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            frameon=False,
            fontsize=14,
            title_fontsize=16,
            ncol=2,
            markerscale=4,
        )
    axs_idx = p1 - 1
    for p2 in range(10, 16):
        sns.scatterplot(
            x="GxC_PC2",
            y=f"GxC_PC{p2}",
            hue=celltype_column,
            data=pca_GxC,
            ax=axs[axs_idx],
            s=4,
            palette="tab20",
            rasterized=format == "pdf",
        )
        axs[axs_idx].legend(
            title="Cell type",
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            frameon=False,
            fontsize=14,
            title_fontsize=16,
            ncol=2,
            markerscale=4,
        )
        axs_idx += 1

    fig.suptitle("\n".join(textwrap.wrap(plot_title, 60)))
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{of_prefix}_GxC-latent-PCA_celltype.{format}"),
        dpi=600,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    if batch_column:
        fig, axs = plt.subplots(
            ncols=3, nrows=5, figsize=(12, 13), constrained_layout=True
        )  # ncols=2, nrows=n_rows, figsize=figure_size, constrained_layout=False)
        axs = axs.flatten()
        for p1 in range(2, 11):
            sns.scatterplot(
                x="GxC_PC1",
                y=f"GxC_PC{p1}",
                hue=batch_column,
                data=pca_GxC,
                ax=axs[p1 - 2],
                s=4,
                rasterized=format == "pdf",
            )
            axs[p1 - 2].legend(
                title="Batch",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=14,
                title_fontsize=16,
                ncol=1,
                markerscale=4,
            )
        axs_idx = p1 - 1
        for p2 in range(10, 16):
            sns.scatterplot(
                x="GxC_PC2",
                y=f"GxC_PC{p2}",
                hue=batch_column,
                data=pca_GxC,
                ax=axs[axs_idx],
                s=4,
                rasterized=format == "pdf",
            )
            axs[axs_idx].legend(
                title="Batch",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=14,
                title_fontsize=16,
                ncol=1,
                markerscale=4,
            )
            axs_idx += 1
        fig.suptitle("\n".join(textwrap.wrap(plot_title, 60)))
        plt.savefig(
            os.path.join(output_dir, f"{of_prefix}_GxC-latent-PCA_Batch.{format}"),
            dpi=600,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()

    if sex_column:
        fig, axs = plt.subplots(
            ncols=3, nrows=5, figsize=(12, 13), constrained_layout=True
        )  # ncols=2, nrows=n_rows, figsize=figure_size, constrained_layout=False)
        axs = axs.flatten()
        for p1 in range(2, 11):
            sns.scatterplot(
                x="GxC_PC1",
                y=f"GxC_PC{p1}",
                hue=sex_column,
                data=pca_GxC,
                ax=axs[p1 - 2],
                s=4,
                palette=["lightpink", "slategrey"],
                rasterized=format == "pdf",
            )
            axs[p1 - 2].legend(
                title="Sex",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=14,
                title_fontsize=16,
                ncol=1,
                markerscale=4,
            )
        axs_idx = p1 - 1
        for p2 in range(10, 16):
            sns.scatterplot(
                x="GxC_PC2",
                y=f"GxC_PC{p2}",
                hue=sex_column,
                data=pca_GxC,
                ax=axs[axs_idx],
                s=4,
                palette=["lightpink", "slategrey"],
                rasterized=format == "pdf",
            )
            axs[axs_idx].legend(
                title="Sex",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=14,
                title_fontsize=16,
                ncol=1,
                markerscale=4,
            )
            axs_idx += 1
        fig.suptitle("\n".join(textwrap.wrap(plot_title, 60)))
        plt.savefig(
            os.path.join(output_dir, f"{of_prefix}_GxC-latent-PCA_sex.{format}"),
            dpi=600,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()

    ## V embedding
    if "UMAP1_V" not in cell_metadata.columns and "UMAP2_V" not in cell_metadata.columns:
        V = livi_embeddings["V_embedding"]
        if V is None:
            sys.exit(0)
        if batch_column is None and sex_column is None:
            sys.exit(0)
        if isinstance(V, torch.Tensor):
            V = V.detach().cpu().numpy()
        V = pd.DataFrame(
            V,
            index=cell_metadata[individual_column].unique(),
            columns=[f"V_Factor{f+1}" for f in range(V.shape[1])],
        )
        umap_df = compute_umap(U, colnames=["UMAP1_V", "UMAP2_V"], add_latent=False)
        umap_df = umap_df.merge(
            cell_metadata.filter([individual_column, batch_column, sex_column])
            .drop_duplicates()
            .set_index(individual_column),
            how="left",
            right_index=True,
            left_index=True,
        )

    r = 2 if batch_column and sex_column else 1
    fig = plt.figure(layout="constrained", figsize=(7 * r, 5), dpi=100)
    if batch_column:
        ax1 = fig.add_subplot(1, r, 1)
        sns.scatterplot(
            x="UMAP1_V",
            y="UMAP2_V",
            hue=batch_column,
            data=umap_df,
            ax=ax1,
            s=6,
            palette="tab10",
            rasterized=format == "pdf",
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
        if sex_column:
            ax2 = fig.add_subplot(1, r, 2)
            sns.scatterplot(
                x="UMAP1_V",
                y="UMAP2_V",
                hue=sex_column,
                data=umap_df,
                ax=ax2,
                s=6,
                palette=["lightpink", "slategrey"],
                rasterized=format == "pdf",
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

    plt.suptitle("\n".join(textwrap.wrap(plot_title, 60)))
    plt.savefig(
        os.path.join(output_dir, f"{of_prefix}_V-embedding_UMAP.{format}"),
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
    z_score: Optional[int] = None,
    color_map: Optional[str] = None,
    label_fontsize: Optional[int] = None,
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
        format (str, optional): The file format, e.g. 'png', 'pdf', 'eps', ..., to save the figure to. If None, then the file format is inferred from the
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
    # Average of selected factors across cells belonging to the same celltype
    df_celltype = (
        df_plot.groupby(by="Cell type", observed=True).mean().reset_index().set_index("Cell type")
    )

    sns.set_style("white")

    if color_map is None:
        color_map = "vlag" if z_score is not None else None
    if color_map in ["vlag", "RdBu_r", "seismic"]:
        sns.clustermap(
            df_celltype,
            row_cluster=row_cluster,
            col_cluster=column_cluster,
            metric=metric,
            z_score=z_score,
            cmap=color_map,
            center=0.0,
            figsize=(10, 10),
            annot_kws={"size": label_fontsize} if label_fontsize is not None else label_fontsize,
            rasterized=ext == ".pdf",
        )
    else:
        sns.clustermap(
            df_celltype,
            row_cluster=row_cluster,
            col_cluster=column_cluster,
            metric=metric,
            z_score=z_score,
            cmap=color_map,
            figsize=(10, 10),
            annot_kws={"size": label_fontsize} if label_fontsize is not None else label_fontsize,
            rasterized=ext == ".pdf",
        )
    if savefig:
        plt.savefig(
            f"{prefix}_Heatmap_cell-state-Factors_Celltype{ext}",
            bbox_inches="tight",
            dpi=400,
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
    z_score: bool = False,
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
        format (str, optional): The file format, e.g. 'png', 'pdf', 'eps', ..., to save the figure to. If None, then the file format is inferred from the
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
        cell_type_factor_df.groupby(by=celltype_column, observed=True)
        .mean()
        .reset_index()
        .set_index(celltype_column)
    )

    if z_score:
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

    figure_size = (2 * d, n_rows * d)
    fig, axs = plt.subplots(ncols=2, nrows=n_rows, figsize=figure_size, constrained_layout=True)
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
                rasterized=ext == ".pdf",
            )
            axs[0].legend(
                title="Cell type",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize + 2,
                ncol=2,
                markerscale=4,
            )
            axs[0].set_title(
                label="Cell type", fontdict={"fontsize": axis_title_fontsize}, loc="center"
            )
        elif i == 1:
            if zscore:
                sns.heatmap(df_celltype_zscored, cmap=heatmap_color_map, center=0.0, ax=axs[1])
            else:
                sns.heatmap(df_celltype, cmap=heatmap_color_map, center=0.0, ax=axs[1])
        elif 1 < i <= len(unique_factors) + 1:
            sns.scatterplot(
                x="UMAP1",
                y="UMAP2",
                hue=unique_factors[i - 2],
                data=cell_type_factor_df,
                ax=ax,
                s=marker_size,
                palette="vlag",
                legend=False,
                rasterized=ext == ".pdf",
            )
            sm = cm.ScalarMappable(
                cmap="vlag",
                # norm=colors.Normalize(
                #     vmin=cell_type_factor_df[unique_factors[i - 2]].min(),
                #     vmax=cell_type_factor_df[unique_factors[i - 2]].max(),
                # ),
                norm=colors.TwoSlopeNorm(
                    vcenter=0.0,
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
    #   plt.close()

    if return_celltype_factors:
        return (celltype_factor_high, celltype_factor_low)


def QQplot(
    pvalues: Union[list, np.ndarray],
    linecolor: Optional[str] = None,
    truncate: float = 0.001,
    savefig: Optional[str] = None,
) -> None:
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
    (osm, osr), _ = probplot(pvalues, dist="uniform")  # expected, observed
    b, a = np.polyfit(osm, osr, deg=1)
    if truncate > 0:
        cutoff = int(len(osr) * truncate)
        # Remove very small p-values
        osm = osm[cutoff:]
        osr = osr[cutoff:]

    color = linecolor if linecolor is not None else "navy"
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    df = pd.DataFrame({"osm": -np.log10(osm), "osr": -np.log10(osr)})
    sns.scatterplot(x="osm", y="osr", data=df, ax=ax, edgecolor=None, color=color)
    x = np.linspace(0, ax.get_xlim()[1] + 1)
    # Draw diagonal line
    ax.plot(x, a + b * x, c="lightgrey", linestyle="--", scaley=True)
    if truncate > 0:
        # Add a horizontal line to indicate truncation
        last_point_osm = -np.log10(osm[0])
        last_point_osr = -np.log10(osr[0])
        ax.hlines(
            y=last_point_osr,
            xmin=last_point_osm,
            xmax=ax.get_xlim()[1] - 0.3,
            linestyles=":",
            linewidth=4,
            color=color,
            label="Truncated",
        )
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    ax.set(xlabel=r"Expected $-\log_{10} P$", ylabel=r"Observed $-\log_{10} P$")
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    if savefig is not None:
        plt.savefig(savefig, transparent=True, dpi=200, bbox_inches="tight")


def plot_U_factor_corr(
    U: pd.DataFrame,
    associated_factors: list,
    A: pd.DataFrame,
    savefig: Optional[str] = None,
    format: Optional[str] = None,
) -> None:
    """Creates a heatmap of the pairwise Pearson correlations of SNP-associated U factors.

    Parameters
    ----------
        U (pd.DataFrame): Dataframe containing U factors (individuals x factors).
        associated_factors (list): List of associated factors to filter and plot.
        A (pd.DataFrame): Dataframe containing the factor assignment matrix, A.
        savefig (str or None): If provided, the path to save the generated plots. Default is None.
        format (str or None): The file format, e.g. 'png', 'pdf', 'eps', ..., to save the figure to. If None, then the file format is inferred from the
            extension of savefig, if savefig is not None.

    Returns
    -------
        None
    """

    A = A.filter(associated_factors)
    U = U.filter(associated_factors)
    pairwise_correlations = U.corr(method="pearson")

    if savefig:
        prefix, ext = os.path.splitext(savefig)
        ext = "." + format if format else ".png" if ext == "" else ext
    else:
        ext = "." + format if format else ".png"

    # Visualize pearson cor between significant U factors
    plt.figure(figsize=(12, 10))
    sns.heatmap(pairwise_correlations, cmap="RdBu_r", center=0, rasterized=ext == ".pdf")
    plt.title("Pearson's $\\rho$ between significant $U$ factors", fontsize=15, pad=20)
    if savefig:
        plt.savefig(
            f"{prefix}_U_factor_correlations{ext}",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )
    plt.close()


def plot_GxC_similarity(
    U: pd.DataFrame,
    associated_factors: list,
    A: pd.DataFrame,
    cell_state_factors: pd.DataFrame,
    cell_metadata: pd.DataFrame,
    celltype_column: str,
    donor_column: str,
    savefig: str,
    format: str,
) -> None:
    """Creates a clustermap of the cosine similarity between GxC factors and their assigned cell
    types.

    Parameters
    ----------
        U (pd.DataFrame): Dataframe containing U factors (individuals x factors).
        associated_factors (list): List of associated factors to filter and plot.
        A (pd.DataFrame): Dataframe containing the factor assignment matrix, A.
        cell_state_factors (pd.DataFrame or None): Dataframe containing LIVI cell-state latent.
        cell_metadata (pd.DataFrame or None): Dataframe containing cell metadata.
        celltype_column (str or None): Column name in cell metadata indicating the cell type.
        donor_column (str or None): Column name in cell metadata indicating the donor.
        savefig (str or None): If provided, the path to save the generated plots. Default is None.
        format (str or None): The file format, e.g. 'png', 'pdf', 'eps', ..., to save the figure to. If None, then the file format is inferred from the
            extension of savefig, if savefig is not None.

    Returns
    -------
        None
    """

    A = A.filter(associated_factors)
    U = U.filter(associated_factors)

    if savefig:
        prefix, ext = os.path.splitext(savefig)
        ext = "." + format if format else ".png" if ext == "" else ext
    else:
        ext = "." + format if format else ".png"

    cc_softmax = nn.Softmax(dim=1)(torch.from_numpy(cell_state_factors.to_numpy())).numpy()
    GxC = pd.DataFrame(
        (cc_softmax @ A.to_numpy()) * U.loc[cell_metadata[donor_column]].to_numpy(),
        index=cell_state_factors.index,
        columns=A.columns,
    )
    GxC.columns = [c.replace("U_", "GxC ") for c in GxC.columns]

    GxC = GxC.merge(cell_metadata[celltype_column], right_index=True, left_index=True)
    GxC = GxC.groupby(celltype_column, observed=True).apply(lambda x: x.mean())

    clm = sns.clustermap(
        GxC,
        col_cluster=True,
        row_cluster=True,
        metric="cosine",
        cmap="Reds",
        cbar_pos=(0.99, 0.14, 0.022, 0.2),
        rasterized=ext == ".pdf",
    )

    if savefig:
        clm.savefig(
            f"{prefix}_GxC-factor_celltype_clustering{ext}",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )
    plt.close()


def plot_donor_similarity(
    U: pd.DataFrame,
    associated_factors: list,
    donor_metadata: Optional[pd.DataFrame] = None,
    donor_covariate: Optional[str] = None,
    covariate_colors: Optional[List[str]] = None,
    savefig: Optional[str] = None,
    format: Optional[str] = None,
):
    """Cluster donors and draw a heatmap of donor similarity based on LIVI's U embedding.
    Optionally color the donor IDs according to a donor covariate of interest.

        Parameters
        ----------
        U (pd.DataFrame): Dataframe containing U factors (individuals x factors).
        associated_factors (list): List of factors with significant genetic associations (used to calculate the donor similarities).
        donor_metadata (pd.DataFrame or None): Dataframe containing donor metadata. Default is None.
        donor_covariate (str or None): Column name in donor metadata indicating the donor covariate to use to color the donor IDs. Default is None.
        covariate_colors (list or None): If provided, each donor category will be colored with the corresponding color from the list (as indicated by the color order), otherwise colors will be chosen randomly. Default is None, i.e. use random colors.
        savefig (str or None): If provided, the path to save the generated plots. Default is None.
        format (str or None): The file format, e.g. 'png', 'pdf', 'eps', ..., to save the figure to. If None, then the file format is inferred from the
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

    if donor_covariate is not None:
        assert (
            donor_metadata is not None
        ), "Donor covariate provided, but no donor metadata dataframe."
        if covariate_colors is not None:
            assert (
                len(covariate_colors) == donor_metadata[donor_covariate].nunique()
            ), "Number of colors provided doesn't match the number of covariate categories."
        color_dict = dict(
            zip(
                donor_metadata[donor_covariate].unique(),
                (
                    covariate_colors
                    if covariate_colors is not None
                    else [
                        "#" + "".join([rng.choice(list("0123456789ABCDEF")) for j in range(6)])
                        for i in range(donor_metadata[donor_covariate].nunique())
                    ]
                ),
            )
        )
        row_colors = donor_metadata[donor_covariate].map(color_dict)
    else:
        row_colors = None

    # Cluster individuals based on significant U factor values
    U = U.filter(associated_factors)

    clm = sns.clustermap(
        U,
        col_cluster=False,
        row_cluster=True,
        metric="cosine",
        cmap="RdBu_r",
        center=0,
        cbar_pos=(0.99, 0.14, 0.022, 0.2),
        rasterized=ext == ".pdf",
    )
    if savefig:
        clm.savefig(
            f"{prefix}_IID_clustering_based_on_U_factors{ext}",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )
    plt.close()

    iid_distances = squareform(
        pdist(U, metric="cosine")
    )  # 0: identical, 1: unrelated, 2: opposite
    iid_distances = pd.DataFrame(iid_distances, index=U.index, columns=U.index)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        iid_distances, cmap="RdBu", center=1, rasterized=True, xticklabels=5, yticklabels=5
    )
    # Color donor ids according to the covariate
    if donor_covariate is not None:
        for label, color in color_dict.items():
            plt.plot([], [], label=label, color=color, linewidth=5)
        plt.legend(loc="center left", bbox_to_anchor=(-0.2, 1.2), title=donor_covariate)
        # Set the tick labels color
        for tick_label in plt.gca().get_yticklabels():
            tick_label.set_color(row_colors[tick_label.get_text()])
            tick_label.set_fontsize(3)
        for tick_label in plt.gca().get_xticklabels():
            tick_label.set_color(row_colors[tick_label.get_text()])
            tick_label.set_fontsize(3)
    plt.title(
        "Cosine distance between individuals based on significant $U$ factors", fontsize=15, pad=20
    )
    if savefig:
        plt.savefig(
            f"{prefix}_Heatmap_IID_cosine-distance_based_on_U_factors{ext}",
            bbox_inches="tight",
            dpi=400,
            transparent=True,
        )
    plt.close()


### Edited from venny4py package ###
def venny4py_custom_colors(
    sets,
    plot_title,
    custom_colors=None,
    out="./",
    asax=False,
    ext="png",
    dpi=300,
    size=3.5,
    alpha=0.4,
    fontsize=None,
):
    from itertools import combinations

    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse
    from venny4py.venny4py import get_shared as vp_get_shared
    from venny4py.venny4py import get_unique as vp_get_unique

    shared = vp_get_shared(sets)
    unique = vp_get_unique(shared)
    if custom_colors is None:
        custom_colors = ["green", "darkorange", "maroon", "royalblue"]  # colors
    lw = size * 0.12  # line width
    fs = fontsize if fontsize is not None else size * 2  # font size
    nc = 2  # legend cols
    cs = 4  # columnspacing

    if asax is False:
        plt.rcParams["figure.dpi"] = 200  # dpi in notebook
        plt.rcParams["savefig.dpi"] = dpi  # dpi in saved figure
        fig, ax = plt.subplots(1, 1, figsize=(size, size))
    else:
        ax = asax

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # 4 sets
    if len(sets) == 4:
        # draw ellipses
        ew = 45  # width
        eh = 75  # height
        xe = [35, 48, 52, 65]  # x coordinates
        ye = [35, 45, 45, 35]  # y coordinates
        ae = [225, 225, 315, 315]  # angles

        for i, s in enumerate(sets):
            ax.add_artist(
                Ellipse(
                    xy=(xe[i], ye[i]),
                    width=ew,
                    height=eh,
                    fc=custom_colors[i],
                    angle=ae[i],
                    alpha=alpha,
                )
            )
            ax.add_artist(
                Ellipse(
                    xy=(xe[i], ye[i]),
                    width=ew,
                    height=eh,
                    fc="None",
                    angle=ae[i],
                    ec="black",
                    lw=lw,
                )
            )

        # annotate
        xt = [12, 32, 68, 88, 14, 34, 66, 86, 26, 28, 50, 50, 72, 74, 37, 60, 40, 63, 50]  # x
        yt = [67, 79, 79, 67, 41, 70, 70, 41, 59, 26, 11, 60, 26, 59, 51, 17, 17, 51, 35]  # y

        for j, s in enumerate(sets):
            ax.text(
                xt[j],
                yt[j],
                len(sets[s]),
                ha="center",
                va="center",
                fontsize=fs,
                transform=ax.transData,
            )

        for k in unique:
            j += 1
            ax.text(
                xt[j],
                yt[j],
                len(unique[k]),
                ha="center",
                va="center",
                fontsize=fs,
                transform=ax.transData,
            )

        # legend
        handles = [
            mpatches.Patch(color=custom_colors[i], label=l, alpha=alpha)
            for i, l in enumerate(sets)
        ]
        ax.legend(
            labels=sets,
            handles=handles,
            fontsize=fs * 1.1,
            frameon=False,
            bbox_to_anchor=(0.5, 1.05),
            bbox_transform=ax.transAxes,
            loc=9,
            handlelength=1.5,
            ncol=nc,
            columnspacing=cs,
            handletextpad=0.5,
        )
        plt.title(
            label=plot_title,
            fontdict={"weight": "bold", "fontsize": fs},
        )
        if asax is False:
            fig.savefig(
                f"{out}.{ext}",
                dpi=dpi,
                bbox_inches="tight",
                transparent=True,
            )


def overlap_with_known_eQTLs(
    known_trans_eQTLs: pd.DataFrame,
    SNP_colname_trans: str,
    GxC_effects_LIVI: pd.DataFrame,
    factor_assignment_matrix: Optional[pd.DataFrame] = None,
    known_cis_eQTLs: Optional[pd.DataFrame] = None,
    SNP_colname_cis: Optional[str] = None,
    persistent_effects_LIVI: Optional[pd.DataFrame] = None,
    savefig: Optional[str] = None,
    format: Optional[str] = None,
):
    if savefig:
        prefix, ext = os.path.splitext(savefig)
        ext = "." + format if format else ".png" if ext == "" else ext
    else:
        ext = "." + format if format else ".png"

    if known_cis_eQTLs is not None:
        only_trans_eQTLs = (
            known_trans_eQTLs.loc[
                ~known_trans_eQTLs[SNP_colname_trans].isin(known_cis_eQTLs[SNP_colname_cis])
            ][SNP_colname_trans]
            .unique()
            .tolist()
        )
        only_cis_eQTLs = (
            known_cis_eQTLs.loc[
                ~known_cis_eQTLs[SNP_colname_cis].isin(known_trans_eQTLs[SNP_colname_trans])
            ][SNP_colname_cis]
            .unique()
            .tolist()
        )
        both_eQTLs = (
            known_trans_eQTLs.loc[
                known_trans_eQTLs[SNP_colname_trans].isin(known_cis_eQTLs[SNP_colname_cis])
            ][SNP_colname_cis]
            .unique()
            .tolist()
        )

        GxC_effects_LIVI = GxC_effects_LIVI.assign(
            is_known_eQTL=GxC_effects_LIVI.apply(
                lambda x: (
                    "only trans-eQTLs"
                    if x.SNP_id in only_trans_eQTLs
                    else (
                        "only cis-eQTLs"
                        if x.SNP_id in only_cis_eQTLs
                        else "cis and trans eQTLs" if x.SNP_id in both_eQTLs else "only LIVI"
                    )
                ),
                axis=1,
            )
        )
        # Overlap between SNPs
        v = venn3(
            subsets=[
                set(GxC_effects_LIVI.SNP_id),
                set(known_trans_eQTLs[SNP_colname_trans]),
                set(known_cis_eQTLs[SNP_colname_cis]),
            ],
            set_labels=("LIVI GxC", "known $trans$-eQTLs", "known $cis$-eQTLs"),
        )
        [lbl.set_fontsize(15) for lbl in v.set_labels]
        for lbl in v.subset_labels:
            try:
                lbl.set_fontsize(12)
            except AttributeError:
                pass
        plt.title(
            "Overlap between significant SNPs\n", fontdict={"fontsize": 18, "weight": "bold"}
        )
        plt.tight_layout()
        plt.savefig(
            f"{prefix}_Venn_LIVI-GxC-vs-known-eQTLs{ext}",
            transparent=True,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        # Overlap between associated factors
        ssets = {
            "only cis-eQTLs": set(
                GxC_effects_LIVI.loc[
                    GxC_effects_LIVI.is_known_eQTL == "only cis-eQTLs"
                ].Factor.unique()
            ),
            "only trans-eQTLs": set(
                GxC_effects_LIVI.loc[
                    GxC_effects_LIVI.is_known_eQTL == "only trans-eQTLs"
                ].Factor.unique()
            ),
            "cis and trans eQTLs": set(
                GxC_effects_LIVI.loc[
                    GxC_effects_LIVI.is_known_eQTL == "cis and trans eQTLs"
                ].Factor.unique()
            ),
            "only LIVI GxC": set(
                GxC_effects_LIVI.loc[GxC_effects_LIVI.is_known_eQTL == "only LIVI"].Factor.unique()
            ),
        }

        venny4py_custom_colors(
            ssets,
            plot_title="Overlap between $U$ Factors associated with SNPs that are (not) known eQTLs",
            out=f"{prefix}_Venn_U-factor-overlap-LIVI-vs-known-eQTLs",
            asax=False,
            ext=ext,
            dpi=300,
            size=3.5,
        )
        plt.close()

        cis_factors = list(
            set(
                GxC_effects_LIVI.loc[
                    GxC_effects_LIVI.is_known_eQTL == "only cis-eQTLs"
                ].Factor.unique()
            )
            .difference(
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.is_known_eQTL == "cis and trans eQTLs"
                    ].Factor.unique()
                )
            )
            .difference(
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.is_known_eQTL == "only LIVI"
                    ].Factor.unique()
                )
            )
        )

        trans_factors = list(
            set(
                GxC_effects_LIVI.loc[
                    GxC_effects_LIVI.is_known_eQTL == "only trans-eQTLs"
                ].Factor.unique()
            )
            .difference(
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.is_known_eQTL == "cis and trans eQTLs"
                    ].Factor.unique()
                )
            )
            .difference(
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.is_known_eQTL == "only LIVI"
                    ].Factor.unique()
                )
            )
        )

        LIVIonly_factors = list(
            set(
                GxC_effects_LIVI.loc[GxC_effects_LIVI.is_known_eQTL == "only LIVI"].Factor.unique()
            )
            .difference(
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.is_known_eQTL == "only cis-eQTLs"
                    ].Factor.unique()
                )
            )
            .difference(
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.is_known_eQTL == "only trans-eQTLs"
                    ].Factor.unique()
                )
            )
            .difference(
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.is_known_eQTL == "cis and trans eQTLs"
                    ].Factor.unique()
                )
            )
        )

        if factor_assignment_matrix is not None:
            plot_df = factor_assignment_matrix.filter(GxC_effects_LIVI.Factor.unique()).T

            plot_df = (
                plot_df.reset_index()
                .assign(
                    Group=plot_df.reset_index().apply(
                        lambda x: (
                            "only LIVI GxC"
                            if x["index"] in LIVIonly_factors
                            else (
                                "LIVI GxC and $trans$-eQTL"
                                if x["index"] in trans_factors
                                else (
                                    "LIVI GxC and $cis$-eQTL"
                                    if x["index"] in cis_factors
                                    else "LIVI GxC and both $cis$ and $trans$ eQTL"
                                )
                            )
                        ),
                        axis=1,
                    )
                )
                .set_index("index")
            )
            plot_df = plot_df.groupby(by="Group", observed=True).mean()

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                plot_df,
                annot=False,
                cmap="inferno",
                cbar_kws={"label": "mean $A$ across $U$ factors"},
                rasterized=True,
            )
            plt.title("Cell-state factor prevalence among significant $U$ factors")
            plt.ylabel("")
            plt.xlabel("")
            plt.tight_layout()
            plt.savefig(
                f"{prefix}_Cell-state-factor-prevalence-among-significant-U-factors{ext}",
                dpi=500,
                transparent=True,
                bbox_inches="tight",
            )
            plt.close()

        if persistent_effects_LIVI is not None:
            # Overlap between SNPs
            ssets = {
                "known cis-eQTLs": set(known_cis_eQTLs[SNP_colname_cis].tolist()),
                "known trans-eQTLs": set(known_trans_eQTLs[SNP_colname_trans].tolist()),
                "LIVI GxC": set(GxC_effects_LIVI.SNP_id.tolist()),
                "LIVI persistent": set(persistent_effects_LIVI.SNP_id.tolist()),
            }

            venny4py_custom_colors(
                ssets,
                plot_title="Overlap between significant SNPs",
                out=f"{prefix}_Venn_LIVI-persistent-vs-known-eQTLs",
                asax=False,
                ext=ext,
                dpi=300,
                size=3.5,
            )
            plt.close()

    else:
        # Overlap between SNPs
        v = venn2(
            subsets=[set(GxC_effects_LIVI.SNP_id), set(known_trans_eQTLs[SNP_colname_trans])],
            set_labels=("LIVI GxC", "known $trans$-eQTLs"),
        )
        [lbl.set_fontsize(15) for lbl in v.set_labels]
        for lbl in v.subset_labels:
            try:
                lbl.set_fontsize(12)
            except AttributeError:
                pass
        plt.title(
            "Overlap between significant SNPs\n", fontdict={"fontsize": 18, "weight": "bold"}
        )
        plt.tight_layout()
        plt.savefig(
            f"{prefix}_Venn_LIVI_vs_known-eQTLs{ext}",
            transparent=True,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        # Overlap between associated factors
        v = venn2(
            subsets=[
                set(
                    GxC_effects_LIVI.loc[
                        ~GxC_effects_LIVI.SNP_id.isin(known_trans_eQTLs[SNP_colname_trans])
                    ].Factor.tolist()
                ),
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.SNP_id.isin(known_trans_eQTLs[SNP_colname_trans])
                    ].Factor.tolist()
                ),
            ],
            set_labels=("only LIVI GxC", "known $trans$-eQTLs"),
        )
        [lbl.set_fontsize(15) for lbl in v.set_labels]
        for lbl in v.subset_labels:
            try:
                lbl.set_fontsize(12)
            except AttributeError:
                pass
        plt.title(
            "Overlap between $U$ Factors associated with SNPs that are (not) known $trans$ eQTLs\n",
            fontdict={"fontsize": 18, "weight": "bold"},
        )
        plt.tight_layout()
        plt.savefig(
            f"{prefix}_Venn_U-factor-overlap-LIVI-vs-known-eQTLs{ext}",
            transparent=True,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        LIVIonly_factors = list(
            set(
                GxC_effects_LIVI.loc[
                    ~GxC_effects_LIVI.SNP_id.isin(known_trans_eQTLs[SNP_colname_trans])
                ].Factor.tolist()
            ).difference(
                set(
                    GxC_effects_LIVI.loc[
                        GxC_effects_LIVI.SNP_id.isin(known_trans_eQTLs[SNP_colname_trans])
                    ].Factor.tolist()
                )
            )
        )

        if factor_assignment_matrix is not None:
            plot_df = factor_assignment_matrix.filter(GxC_effects_LIVI.Factor.unique()).T
            plot_df = (
                plot_df.reset_index()
                .assign(
                    Group=plot_df.reset_index().apply(
                        lambda x: (
                            "only LIVI GxC"
                            if x["index"] in LIVIonly_factors
                            else "LIVI and $trans$-eQTL"
                        ),
                        axis=1,
                    )
                )
                .set_index("index")
            )
            plot_df = plot_df.groupby(by="Group", observed=True).mean()

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                plot_df,
                annot=False,
                cmap="inferno",
                cbar_kws={"label": "mean $A$ across $U$ factors"},
                rasterized=True,
            )
            plt.title("Cell-state factor prevalence among significant $U$ factors")
            plt.ylabel("")
            plt.xlabel("")
            plt.tight_layout()
            plt.savefig(
                f"{prefix}_Cell-state-factor-prevalence-among-significant-U-factors{ext}",
                dpi=500,
                transparent=True,
                bbox_inches="tight",
            )
            plt.close()

        if persistent_effects_LIVI is not None:
            # Overlap between SNPs
            v = venn3(
                subsets=[
                    set(GxC_effects_LIVI.SNP_id),
                    set(known_trans_eQTLs[SNP_colname_trans]),
                    set(persistent_effects_LIVI.SNP_id),
                ],
                set_labels=("LIVI GxC", "known $trans$-eQTLs", "LIVI persistent"),
            )
            [lbl.set_fontsize(15) for lbl in v.set_labels]
            for lbl in v.subset_labels:
                try:
                    lbl.set_fontsize(12)
                except AttributeError:
                    pass
            plt.title(
                "Overlap between significant SNPs\n", fontdict={"fontsize": 18, "weight": "bold"}
            )
            plt.tight_layout()
            plt.savefig(
                f"{prefix}_Venn_LIVI-persistent-vs-known-eQTLs{ext}",
                transparent=True,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def plot_gene_loadings_for_factor(
    GxC_decoder: pd.DataFrame,
    factor: str,
    adata_var: pd.DataFrame,
    gene_name_column: str,
    color: Optional[str] = None,
    spines_invisible: bool = False,
    n_top_genes: int = 5,
    genes_to_annotate: Optional[List[str]] = None,
    annotation_fontsize: int = 14,
    offset: float = 0.07,
    x_distance: float = 5e-2,
    savefig: Optional[str] = None,
) -> None:
    """Plots the distribution of gene loadings for a given factor and annotates the genes with the
    highest loadings. Alternatively, it can highlight a list of user-specified genes (e.g. genes
    belonging to a specific regulatory pathway).

    Parameters:
    ----------
        GxC_decoder (pd.DataFrame): DataFrame containing gene loadings for each factor (columns).
        factor (str): Name of the factor whose gene loadings will be visualized.
        adata_var (pd.DataFrame): DataFrame equivalent to `adata.var` containing gene metadata with same index as `GxC_decoder`.
        gene_name_column (str): Column in `adata_var` containing the gene names (those will be annotated on the plot).
        color (Optional[str]): Outline color for the boxplot. Default is "darkblue".
        spines_invisible (bool): If True, hides the top, right, and left spines of the plot. Default is False.
        n_top_genes (int): Number of top-loading genes to annotate if `genes_to_annotate` is not provided. Default is 5.
            Warning that large values (>10-15) can lead to cluttered plots.
        genes_to_annotate (Optional[List[str]]): List of user-specified gene IDs to annotate. Overrides top-gene selection.
            Default is None.
        annotation_fontsize (int): Font size for gene name annotations. Default is 14.
        x_distance (float): If the x-distance between adjacent annotations is smaller than `x_distance`, the positions of the
            annotation texts are adjusted to reduce overlap. Default is 5e-2.
        offset (float): Vertical offset to adjust annotation text to avoid overlap. Default is 0.07.
        savefig (Optional[str]): Absolute path to save the resulting figure (if provided). Default is None.

    Returns:
    -------
        None
    """

    if GxC_decoder.index.name is None:
        idx_name = "GeneID"
    else:
        idx_name = GxC_decoder.index.name
    gene_loadings = GxC_decoder[factor].reset_index()

    if genes_to_annotate is not None:
        gene_loadings = gene_loadings.assign(
            annotate_gene=gene_loadings.apply(lambda x: x[idx_name] in genes_to_annotate, axis=1)
        )
    else:
        top_genes = GxC_decoder[factor].abs().nlargest(n_top_genes).index.tolist()
        gene_loadings = gene_loadings.assign(
            annotate_gene=gene_loadings.apply(lambda x: x[idx_name] in top_genes, axis=1)
        )

    if adata_var.index.name != idx_name:
        adata_var.rename_axis(idx_name, axis=0, inplace=True)
    gene_loadings = gene_loadings.merge(
        adata_var[gene_name_column].reset_index(), on=idx_name, how="left"
    )

    fig, axs = plt.subplots(figsize=(7.5, 5), nrows=1, ncols=1, constrained_layout=True)

    sns.boxplot(
        data=gene_loadings,
        x=factor,
        fill=False,
        color="darkblue" if color is None else color,
        ax=axs,
    )
    axs.legend().remove()
    axs.set_title(factor.replace("_", " ").replace("or", "or "), fontsize=annotation_fontsize + 2)
    axs.set_yticks([])
    axs.set_xlabel("Loadings", fontsize=annotation_fontsize + 2)
    axs.set_xticklabels(axs.get_xticklabels(), fontsize=15)
    if spines_invisible:
        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)
        axs.spines["left"].set_visible(False)
    # axis.spines["bottom"].set_visible(False)

    f_loadings_anno = gene_loadings.loc[gene_loadings["annotate_gene"]]
    f_loadings_anno = f_loadings_anno.sort_values(
        by=factor, ascending=gene_loadings[factor].max() != gene_loadings[factor].abs().max()
    ).reset_index(drop=True)
    x_previous = 0
    y_previous = []
    offset = offset
    for idx, row in f_loadings_anno.iterrows():
        y_loc = 0.05 if idx % 2 == 0 else -0.05
        if np.abs(np.abs(row[factor]) - x_previous) < x_distance:
            if y_loc > 0:
                y_loc += offset
                while y_loc in y_previous:
                    y_loc += offset
            else:
                y_loc -= offset
                while y_loc in y_previous:
                    y_loc -= offset

        axs.annotate(
            row[gene_name_column],
            xy=(row[factor], 0),
            xytext=(row[factor], y_loc),
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc3",
                color="darkgrey",
                linewidth=0.8,
            ),
            ha="center",
            va="center",
            size=annotation_fontsize,
            color="black",
            weight="regular",
            alpha=0.9,
        )
        x_previous = np.abs(row[factor])
        y_previous.append(y_loc)

    if savefig:
        plt.savefig(savefig, transparent=True, bbox_inches="tight", dpi=500)


def plot_gene_loadings_for_associated_variable(
    GxC_associations: pd.DataFrame,
    variable: str,
    GxC_decoder: pd.DataFrame,
    adata_var: pd.DataFrame,
    gene_name_column: str,
    color: Optional[str] = None,
    spines_invisible: bool = False,
    n_top_genes: int = 5,
    genes_to_annotate: Optional[List[str]] = None,
    annotation_fontsize: int = 14,
    offset: float = 0.07,
    x_distance: float = 5e-2,
    d: int = 5,
    savefig: Optional[str] = None,
) -> None:
    """Plots distributions of loadings for all factors associated with a given donor variable, and
    annotates the top genes for each factor. Alternatively, it can highlight a list of user-
    specified genes (e.g. genes belonging to a specific regulatory pathway).

    Parameters:
    ----------
        GxC_associations (pd.DataFrame): DataFrame containing variable–factor associations, obtained using LIVI's testing pipeline.
        variable (str): Name of the variable (e.g., SNP ID) for which factor loadings should be visualized.
        GxC_decoder (pd.DataFrame): DataFrame with gene loadings (row) for each GxC factor (columns).
        adata_var (pd.DataFrame): DataFrame equivalent to `adata.var` containing gene metadata with same index as `GxC_decoder`.
        gene_name_column (str): Column in `adata_var` containing the gene names (those will be annotated on the plot).
        color (Optional[str]): Color for boxplot outlines. If None, defaults to "darkblue".
        spines_invisible (bool): Whether to hide the top, right, and left plot spines. Default is False.
        n_top_genes (int): Number of top-loading genes to annotate if `genes_to_annotate` is not provided. Default is 5.
             Warning that large values (>10-15) can lead to cluttered plots.
        genes_to_annotate (Optional[List[str]]): List of user-specified gene IDs to annotate. Overrides `n_top_genes`. Default is None.
        annotation_fontsize (int): Font size for gene name annotations. Default is 14.
        x_distance (float): If the x-distance between adjacent annotations is smaller than `x_distance`, the positions of the
            annotation texts are adjusted to reduce overlap. Default is 5e-2.
        offset (float): Vertical offset to adjust annotation text to avoid overlap. Default is 0.07.
        d (int): Scaling factor for figure size (width and height per subplot). Default is 5.
        savefig (Optional[str]): Absolute path to save the resulting figure (if provided). Default is None.

    Returns:
    -------
        None
    """
    GxC_associations_variable = GxC_associations.loc[GxC_associations.SNP_id == variable]
    if GxC_decoder.index.name is None:
        idx_name = "GeneID"
    else:
        idx_name = GxC_decoder.index.name
    gene_loadings = GxC_decoder.filter(
        [f.replace("U", "GxC") for f in GxC_associations_variable.Factor.unique().tolist()]
    )
    gene_loadings = pd.melt(
        gene_loadings, var_name="Factor", value_name="loadings", ignore_index=False
    )
    gene_loadings = (
        gene_loadings.assign(gene_factor=gene_loadings.index + "__" + gene_loadings.Factor)
        .reset_index()
        .set_index("gene_factor")
    )

    if genes_to_annotate is not None:
        gene_loadings = gene_loadings.assign(
            annotate_gene=gene_loadings.apply(lambda x: x[idx_name] in genes_to_annotate, axis=1)
        )
    else:
        top_genes = {}
        for f_gxc in GxC_associations_variable.Factor.unique():
            f_gxc = f_gxc.replace("U", "GxC")
            top_genes[f_gxc] = GxC_decoder[f_gxc].abs().nlargest(n_top_genes).index.tolist()
        gene_loadings = gene_loadings.assign(
            annotate_gene=gene_loadings.apply(lambda x: x[idx_name] in top_genes[x.Factor], axis=1)
        )

    if adata_var.index.name != idx_name:
        adata_var.rename_axis(idx_name, axis=0, inplace=True)
    gene_loadings = gene_loadings.merge(
        adata_var[gene_name_column].reset_index(), on=idx_name, how="left"
    )

    if GxC_associations_variable.Factor.nunique() % 3 == 0:
        n_rows = GxC_associations_variable.Factor.nunique() // 3
    else:
        n_rows = (GxC_associations_variable.Factor.nunique() // 3) + 1
    n_cols = (
        3
        if GxC_associations_variable.Factor.nunique() > 2
        else GxC_associations_variable.Factor.nunique()
    )

    figure_size = ((n_cols + 0.5) * d, n_rows * d)
    fig, axs = plt.subplots(
        figsize=figure_size, nrows=n_rows, ncols=n_cols, constrained_layout=True
    )
    if n_rows > 1:
        axs = axs.flatten()

    for f_idx, f_gxc in enumerate(gene_loadings.Factor.unique()):
        f_gxc_loadings = gene_loadings.loc[gene_loadings.Factor == f_gxc]
        if n_rows > 1 or n_cols > 1:
            axis = axs[f_idx]
        else:
            axis = axs

        sns.boxplot(
            data=f_gxc_loadings,
            x="loadings",
            fill=False,
            color="darkblue" if color is None else color,
            ax=axis,
        )
        axis.legend().remove()
        axis.set_title(f_gxc.replace("_", " ").replace("or", "or "), fontsize=annotation_fontsize)
        axis.set_yticks([])
        axis.set_xlabel("Loadings", fontsize=annotation_fontsize + 2)
        axis.tick_params(axis="x", which="major", labelsize=annotation_fontsize - 3)
        if spines_invisible:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_visible(False)
        # axis.spines["bottom"].set_visible(False)

        f_gxc_loadings_anno = f_gxc_loadings.loc[f_gxc_loadings["annotate_gene"]]
        f_gxc_loadings_anno = f_gxc_loadings_anno.sort_values(
            by="loadings",
            ascending=f_gxc_loadings.loadings.max() != f_gxc_loadings.loadings.abs().max(),
        ).reset_index(drop=True)
        x_previous = 0
        y_previous = []
        offset = offset
        for idx, row in f_gxc_loadings_anno.iterrows():
            y_loc = 0.05 if idx % 2 == 0 else -0.05
            if np.abs(np.abs(row["loadings"]) - x_previous) < x_distance:
                if y_loc > 0:
                    y_loc += offset
                    while y_loc in y_previous:
                        y_loc += offset
                else:
                    y_loc -= offset
                    while y_loc in y_previous:
                        y_loc -= offset

            axis.annotate(
                row["GeneSymbol"],
                xy=(row["loadings"], 0),
                xytext=(row["loadings"], y_loc),
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="arc3",
                    color="darkgrey",
                    linewidth=0.8,
                ),
                ha="center",
                va="center",
                size=annotation_fontsize,
                color="black",
                weight="regular",
                alpha=0.9,
            )
            x_previous = np.abs(row["loadings"])
            y_previous.append(y_loc)

    if GxC_associations_variable.Factor.nunique() > 1:
        fig.suptitle(variable.replace("_", " "), fontsize=annotation_fontsize + 4, ha="left")
    else:
        axis.set_title(variable.replace("_", " "), fontsize=annotation_fontsize + 4, ha="center")

    if n_rows > 1 and len(axs) > gene_loadings.Factor.nunique():
        for i in range(len(axs)):
            if i >= gene_loadings.Factor.nunique():
                fig.delaxes(axs[i])

    if savefig:
        plt.savefig(savefig, transparent=True, bbox_inches="tight", dpi=500)


def plot_ct_gex_vs_gt(
    adata,
    gene: Union[str, List[str]],
    iid_column: str,
    GT_matrix: pd.DataFrame,
    SNP_id: str,
    GxC_associations: pd.DataFrame,
    filter_maf: bool = True,
    test_gt_groups: bool = True,
    annotation_fontsize: int = 20,
    celltype: Optional[Union[str, List[str]]] = None,
    celltype_column: Optional[str] = None,
    cell_indices: Optional[List[int]] = None,
    hgnc_column: Optional[str] = None,
    violin_colors: Optional[List[str]] = None,
    savefig: bool = True,
    output_dir: Optional[str] = None,
    output_prefix: Optional[str] = None,
    format: Optional[str] = None,
    return_df: bool = False,
) -> Optional[pd.DataFrame]:
    """Plots pseudobulk gene expression (at the donor level) versus genotype for a given SNP and
    gene, across one or more specified celltype(s) or set of cells. Optionally genotypes not
    present in at least 5% of the individuals are ignored.

    Parameters:
    ----------
        adata (anndata.AnnData): AnnData object with single-cell expression values and cell metadata (incl. donor IDs).
        gene (Union[str, List[str]]): Name or list of gene IDs to plot expression for. If more than one gene is passed,
            expression values are summed over all genes.
        iid_column (str): Column in `adata.obs` indicating individual (donor) IDs. Individual IDs must be the same as
            in the genotype matrix.
        GT_matrix (pd.DataFrame): Genotype matrix with SNPs as rows and individual IDs as columns.
        SNP_id (str): ID of the SNP to use.
        GxC_associations (pd.DataFrame): DataFrame containing LIVI testing results, including the assessed allele for the SNP.
            SNP IDs must be the same as in the genotype matrix.
        filter_maf (bool): Whether to omit genotypes present in less than 5% of the donors. Default is True.
        annotation_fontsize (int): Fontsize for axis (tick)labels. Ticks of the x axis (i.e. the SNP genotypes) are annotated
            with the given fontsize, while axis/plot title fontsizes are adjusted to be slightly larger than that.
        celltype (Optional[Union[str, List[str]]]): Aggregate expression values only across cells of the given celltype(s).
            Default is None.
        celltype_column (Optional[str]): Column in `adata.obs` that contains cell type annotations. Required if `celltype` is used.
        cell_indices (Optional[List[int]]): Aggregate expression values only across these specific cells.
            `cell_indices` must be present in `adata.obs`. Default is None.
        hgnc_column (Optional[str]): Column in `adata.var` with gene symbols. Used for labeling. Default is None.
        violin_colors (Optional[List[str]]): List of colors to use for the distributions of the different genotypes.
            Must be of the same length as the number of the different possible genotype groups.
        savefig (bool): Whether to save the figure to a file. Default is True.
        output_dir (Optional[str]): Absolute path of the directory to save the plot if `savefig=True`. If not specified,
            the current working directory is used.
        output_prefix (Optional[str]): Optional prefix for the output filename. Default is None.
        format (Optional[str]): File format to save (e.g., 'pdf', 'png'). Default is 'png'.
        return_df (bool): If True, returns the DataFrame used for plotting, containing donor-level gene expression and genotypes.
            Default is False.

    Returns:
    -------
        plot_df (Optional[pd.DataFrame]): DataFrame containing donor-level gene expression and genotypes, only returned if `return_df=True`.
    """

    if isinstance(gene, list) and len(gene) > 1:
        warnings.warn(
            "More than one gene was passed. Expression values will be summed over genes."
        )

    if cell_indices is not None:
        adata_aggr = aggregate_cell_counts(
            adata=adata[cell_indices, gene], aggregate_cols=[iid_column], layer=None, sum_gene=True
        )
    elif celltype is not None:
        assert (
            celltype_column is not None
        ), "Please specify `celltype_column` when `celltype` is passed."
        if isinstance(celltype, str):
            celltype = [celltype]
        adata_aggr = aggregate_cell_counts(
            adata=adata[adata.obs.loc[adata.obs[celltype_column].isin(celltype)].index, gene],
            aggregate_cols=[iid_column],
            layer=None,
            sum_gene=True,
        )
    else:
        warnings.warn(
            "Neither `cell_indices` nor `celltype` were provided.\nAggregating gene expression over all cells."
        )
        adata_aggr = aggregate_cell_counts(
            adata=adata[:, gene], aggregate_cols=[iid_column], layer=None, sum_gene=True
        )

    plot_df = pd.DataFrame(
        adata_aggr.X.astype(np.float32),
        index=adata_aggr.obs[iid_column],
        columns=adata_aggr.var.index,
    )
    plot_df = (
        plot_df.reset_index(drop=False)
        .merge(
            GT_matrix.loc[SNP_id].T.rename_axis(iid_column).reset_index(drop=False),
            on=iid_column,
            how="left",
        )
        .set_index(iid_column)
    )

    alleles = re.search(
        re.compile(".*[0-9]+.{1}([A-Z]+).{1}([A-Z]+)"), GT_matrix.loc[SNP_id].T.name
    ).groups()

    if (
        alleles[0] == GxC_associations.loc[GxC_associations.SNP_id == SNP_id].assessed_allele
    ).all():
        allele_dict = {
            0: "/".join([alleles[1], alleles[1]]),
            1: "/".join([alleles[0], alleles[1]]),
            2: "/".join([alleles[0], alleles[0]]),
        }
        plot_df = plot_df.replace({SNP_id: allele_dict})
        plot_df[SNP_id] = pd.Categorical(
            plot_df[SNP_id],
            categories=[
                "/".join([alleles[1], alleles[1]]),
                "/".join([alleles[0], alleles[1]]),
                "/".join([alleles[0], alleles[0]]),
            ],
        )
    else:
        allele_dict = {
            0: "/".join([alleles[0], alleles[0]]),
            1: "/".join([alleles[0], alleles[1]]),
            2: "/".join([alleles[1], alleles[1]]),
        }
        plot_df = plot_df.replace({SNP_id: allele_dict})
        plot_df[SNP_id] = pd.Categorical(
            plot_df[SNP_id],
            categories=[
                "/".join([alleles[0], alleles[0]]),
                "/".join([alleles[0], alleles[1]]),
                "/".join([alleles[1], alleles[1]]),
            ],
        )

    if filter_maf:
        # Remove genotypes not present in at least 5% of the donors
        snp_counts = plot_df[SNP_id].value_counts()
        total_donors = sum(snp_counts.tolist())
        min_donors = int(np.round(total_donors * 0.05))
        plot_df = plot_df.loc[plot_df[SNP_id].isin(snp_counts.loc[snp_counts >= min_donors].index)]
        plot_df[SNP_id] = plot_df[SNP_id].cat.remove_unused_categories()

    gene_hgnc = adata.var.loc[gene][hgnc_column] if hgnc_column is not None else gene
    if isinstance(gene, list):
        gene_hgnc = " and ".join(gene_hgnc)
        gene_hgnc = textwrap.fill(gene_hgnc, width=30)
        gene = "__".join(gene)

    gt_group_median = (
        plot_df.filter([gene, SNP_id]).groupby(by=SNP_id, observed=True)[gene].median()
    )
    if violin_colors is None:
        if len(plot_df[SNP_id].cat.categories) == 3:
            violin_colors = (
                ["lightsteelblue", "cornflowerblue", "blue"]
                if gt_group_median.iloc[0] > gt_group_median.iloc[-1]
                else ["mistyrose", "indianred", "brown"]
            )
        else:
            violin_colors = (
                ["lightsteelblue", "blue"]
                if gt_group_median.iloc[0] > gt_group_median.iloc[-1]
                else ["mistyrose", "brown"]
            )
    else:
        violin_colors = violin_colors

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 7), constrained_layout=False)
    sns.violinplot(
        plot_df,
        y=gene,
        x=SNP_id,
        hue=SNP_id,
        palette=violin_colors,
        ax=axs,
        legend=False,
        rasterized=True,
    )
    axs.set_ylabel(
        gene_hgnc, fontdict={"fontsize": annotation_fontsize + 2, "fontstyle": "italic"}
    )
    axs.xaxis.get_ticklabels()[0].set_color(violin_colors[0])
    axs.xaxis.get_label().set_fontsize(22)
    [t.set_fontsize(annotation_fontsize) for t in axs.xaxis.get_ticklabels()]
    [t.set_fontsize(annotation_fontsize - 3) for t in axs.yaxis.get_ticklabels()]
    gt_groups = plot_df[[gene, SNP_id]].groupby(by=SNP_id, observed=True).groups
    gt_groups = [plot_df.loc[gt_groups[k], gene].to_numpy() for k in gt_groups.keys()]
    pval1 = mannwhitneyu(gt_groups[0], gt_groups[1], alternative="two-sided")[1]
    if test_gt_groups:
        y_max = axs.get_ylim()[1]
        axs.text(
            axs.get_xticks()[0] - 0.15,
            y_max - (0.15 * y_max),
            f"Mann-Whitney \n$p$-value = {pval1:.2e}",
            fontdict={"color": "black", "fontsize": "large", "ma": "right"},
        )
    if len(gt_groups) > 2:
        if test_gt_groups:
            pval2 = mannwhitneyu(gt_groups[0], gt_groups[2], alternative="two-sided")[1]
            axs.text(
                axs.get_xticks()[1] + 0.1,
                y_max - (0.15 * y_max),
                f"Mann-Whitney \n$p$-value = {pval2:.2e}",
                fontdict={"color": "black", "fontsize": "large", "ma": "right"},
            )
        axs.xaxis.get_ticklabels()[1].set_color(violin_colors[1])
        axs.xaxis.get_ticklabels()[2].set_color(violin_colors[2])
    else:
        axs.xaxis.get_ticklabels()[1].set_color(violin_colors[-1])

    axs.plot(
        gt_group_median.index,
        gt_group_median.values,
        marker="o",
        linestyle="-",
        color="midnightblue" if "blue" in violin_colors else "maroon",
    )
    if celltype is not None:
        CT = " and ".join(celltype) if len(celltype) > 1 else celltype[0]
    else:
        CT = ""
    axs.set_title(CT, fontdict={"fontweight": "bold", "fontsize": annotation_fontsize + 2})

    if savefig:
        od = output_dir if output_dir else os.getcwd()
        op = output_prefix if output_prefix else ""
        ext = "." + format if format else ".png"
        genes_filename = gene_hgnc.replace(" ", "-").replace("\n", "-")

        plt.savefig(
            os.path.join(
                od,
                (
                    f"{op}_{SNP_id}_vs_{genes_filename}_{CT.replace(' ', '-')}{ext}"
                    if CT != ""
                    else f"{op}_{SNP_id}_vs_{genes_filename}{ext}"
                ),
            ),
            bbox_inches="tight",
            dpi=400,
            transparent=True,
        )

    #  plt.close()
    if return_df:
        return plot_df


def visualize_GxC_effect(
    SNP_id: str,
    adata: AnnData,
    celltype_column: str,
    return_GxC_effect: bool = False,
    marker_size: int = 2,
    axis_title_fontsize: int = 26,
    d: int = 10,
    model_results_dir: Optional[str] = None,
    cell_state_latent: Optional[pd.DataFrame] = None,
    GxC_associations: Optional[pd.DataFrame] = None,
    assignment_matrix: Optional[pd.DataFrame] = None,
    GxC_decoder: Optional[pd.DataFrame] = None,
    gene: Optional[str] = None,
    factor_id: Optional[str] = None,
    hgnc_column: Optional[str] = None,
    umap_cell_state: Optional[Union[str, pd.DataFrame]] = None,
    savefig: Optional[str] = None,
    format: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Visualise SNP effect on cells.

    Parameters
    ----------
        SNP_id (str): ID of the SNP, whose effect should be calculated.
        adata (AnnData): AnnData object containing the cell metadata in adata.obs.
        celltype_column (str): Column name in adata.obs indicating the celltype.
        return_GxC_effect (bool): If True, returns the quantified SNP effect at the single-cell level.
        marker_size (int): Scatterplot marker size. Default is 2.
        axis_title_fontsize (int): Fontsize of the title for each axis/plot.
        d (int): Controls figure size. Resulting figure will have a width of 2.5 x d and a height d x number of axis/plots.
            Default is 10.
        model_results_dir (str or None): Absolute path of the directory containing the inference and testing
            results files of the LIVI model. Must be provided if `cell_state_latent`, `GxC_associations`,
            `assignment_matrix` are None. Default is None.
        cell_state_latent (pd.DataFrame or None): DataFrame containing the cell-state latent space. Can be used instead
            of `model_results_dir`. Default is None.
        GxC_associations (pd.DataFrame or None): Dataframe containing LIVI GxC associations. Can be used instead of
            `model_results_dir`. Default is None.
        assignment_matrix (pd.DataFrame or None): Dataframe containing LIVI factor assignment matrix. Can be used instead
            of `model_results_dir`. Default is None.
        GxC_decoder (Optional[pd.DataFrame]): DataFrame with gene loadings (row) for each GxC factor (columns).
        gene (Optional[str]): Plot the effect of the SNPs on the specific genes(s). Gene IDs must be the same as in `GxC_decoder`
            and `adata.var` index.
        factor_id (Optional[str]): ID of the factor to use for reconstruction. For SNPs associated with more than one factors, it
            is recommended to specify which one to use when calculating the effect of the SNP on genes, in order to obtain
            meaningful results.
        hgnc_column (Optional[str]): Column in `adata.var` containing the HGNC gene names. If provided the genes will be annotated
            using HGNC names instead of the gene IDs in `gene`.
        umap_cell_state (str, pd.DataFrame or None): Dataframe or name of the file containing a precomputed 2D UMAP of LIVI
            cell-state latent. If None, the UMAP is computed. Default is None.
        savefig (Optional[str]): If provided, save the generated plots in this path (and prefix). Default is None.
        format (Optional[str]): The file format, e.g. 'png', 'pdf', 'eps', ..., to save the figure to. If None, then the file
            format is inferred from the extension of savefig, if savefig is not None.


    Returns:
    -------
        GxC_effect (Optional[pd.DataFrame]): Dataframe containing the effect of the given SNP at the single-cell level for cells
            belonging to different cell-states.
    """
    if all(
        [
            arg is None
            for arg in [model_results_dir, cell_state_latent, GxC_associations, assignment_matrix]
        ]
    ):
        raise ValueError(
            "Either provide pd.DataFrames with LIVI inference and testing results or an absolute path to the directory that contains them."
        )

    if model_results_dir is not None:
        assert os.path.isdir(model_results_dir), "Model results directory doesn't exist."

        files = [
            f
            for f in os.listdir(model_results_dir)
            if os.path.isfile(os.path.join(model_results_dir, f))
        ]

        if cell_state_latent is None:
            cell_state_latent = [
                re.match("(.*cell-state_latent.tsv)", f)
                for f in files
                if re.match("(.*cell-state_latent.tsv)", f) is not None
            ]
            if len(cell_state_latent) > 0:
                cell_state_latent = cell_state_latent[0].groups()[0]
                cell_state_latent = pd.read_csv(
                    os.path.join(model_results_dir, cell_state_latent), sep="\t", index_col=0
                )
            else:
                raise FileNotFoundError(
                    "No cell-state latent found in `model_results_dir`. Make sure the filename ends in 'cell-state_latent.tsv'."
                )

        if GxC_associations is None:
            GxC_associations = [
                re.match("(.*_LMM_results_StoreyQ[0-9].[0-9]{1,3}_Ucontext.tsv)", f)
                for f in files
                if re.match("(.*_LMM_results_StoreyQ[0-9].[0-9]{1,3}_Ucontext.tsv)", f) is not None
            ]
            if len(GxC_associations) > 0:
                GxC_associations = GxC_associations[0].groups()[0]
                GxC_associations = pd.read_csv(
                    os.path.join(model_results_dir, GxC_associations), sep="\t", index_col=False
                )
            else:
                raise FileNotFoundError(
                    "No genetic associations found in `model_results_dir`. Make sure the filename ends in 'LMM_results_StoreyQ<alpha-threshold>_Ucontext.tsv'."
                )

        if assignment_matrix is None:
            assignment_matrix = [
                re.match("(.*factor_assignment_matrix.tsv)", f)
                for f in files
                if re.match("(.*factor_assignment_matrix.tsv)", f) is not None
            ]
            if len(assignment_matrix) > 0:
                assignment_matrix = assignment_matrix[0].groups()[0]
                assignment_matrix = pd.read_csv(
                    os.path.join(model_results_dir, assignment_matrix), sep="\t", index_col=0
                )
            else:
                raise FileNotFoundError(
                    "No factor assignment matrix found in `model_results_dir`. Make sure the filename ends in 'factor_assignment_matrix.tsv'."
                )
        if GxC_decoder is None and gene is not None:
            GxC_decoder = [
                re.match("(.*GxC_decoder.tsv)", f)
                for f in files
                if re.match("(.*GxC_decoder.tsv)", f) is not None
            ]
            if len(GxC_decoder) > 0:
                GxC_decoder = GxC_decoder[0].groups()[0]
                GxC_decoder = pd.read_csv(
                    os.path.join(model_results_dir, GxC_decoder), sep="\t", index_col=0
                )
            else:
                raise FileNotFoundError(
                    "No GxC decoder found in `model_results_dir`. Make sure the filename ends in 'GxC_decoder.tsv'."
                )
    if gene is not None:
        assert (
            GxC_decoder is not None
        ), "To visualize effect on individual genes `GxC_decoder` cannot be `None`."
        GxC_effect = calculate_GxC_gene_effect(
            GxC_associations=GxC_associations,
            SNP_id=SNP_id,
            cell_state_latent=cell_state_latent,
            A=assignment_matrix,
            GxC_decoder=GxC_decoder,
            factor_id=factor_id,
        )
        GxC_effect = GxC_effect.filter(gene)
        if hgnc_column is not None:
            hgnc_name = adata.var.loc[gene, hgnc_column].to_dict()
            # GxC_effect = GxC_effect.rename(columns=hgnc_name)
        else:
            hgnc_name = None

    else:
        GxC_effect = calculate_GxC_effect(
            GxC_associations=GxC_associations,
            SNP_id=SNP_id,
            cell_state_latent=cell_state_latent,
            A=assignment_matrix,
        )
        hgnc_name = None

    if umap_cell_state is not None:
        if isinstance(umap_cell_state, pd.DataFrame):
            umap_base = umap_cell_state
        else:
            if not os.path.isfile(umap_cell_state) and not os.path.isfile(
                os.path.join(model_results_dir, umap_cell_state)
            ):
                raise FileNotFoundError("UMAP cell state file not found.")
            else:
                umap_base = (
                    umap_cell_state
                    if os.path.isfile(umap_cell_state)
                    else os.path.join(model_results_dir, umap_cell_state)
                )
                umap_base = pd.read_csv(umap_base, sep="\t", index_col=0)
    else:
        umap_base = compute_umap(cell_state_latent, colnames=["UMAP1", "UMAP2"], add_latent=False)

    umap_base = umap_base.drop(columns=[c for c in adata.obs.columns if c in umap_base.columns])
    umap_base = umap_base.merge(
        adata.obs.filter([celltype_column]), how="left", right_index=True, left_index=True
    )
    umap_base = umap_base.merge(GxC_effect, right_index=True, left_index=True)

    legend_fontsize = axis_title_fontsize - 6
    # Plot the effect for each factor/gene (column in `GxC_effect`) associated with the given SNP
    if len(GxC_effect.columns) % 2 == 0:
        n_rows = len(GxC_effect.columns) // 2 + 1
    else:
        n_rows = int(len(GxC_effect.columns) // 2) + 1

    figure_size = (2.5 * d, n_rows * d)
    fig, axs = plt.subplots(ncols=2, nrows=n_rows, figsize=figure_size, constrained_layout=False)
    axs = axs.flatten()

    ncol_ct = np.ceil(adata.obs[celltype_column].nunique() / 6)
    for i, ax in enumerate(tqdm(axs)):
        if i == 0:
            sns.scatterplot(
                x="UMAP1",
                y="UMAP2",
                hue=celltype_column,
                data=umap_base,
                ax=axs[0],
                s=3,
                palette=list(adata.uns[f"{celltype_column}_colors"]),
                rasterized=True,
            )
            axs[0].legend(
                title="Cell type",
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                frameon=False,
                fontsize=legend_fontsize + 2,
                title_fontsize=legend_fontsize + 3,
                ncol=ncol_ct,
                markerscale=6,
            )
            axs[0].set_title(
                label="Cell type", fontdict={"fontsize": axis_title_fontsize}, loc="center"
            )
            axs[0].xaxis.label.set_fontsize(axis_title_fontsize - 1)
            axs[0].yaxis.label.set_fontsize(axis_title_fontsize - 1)
            axs[0].tick_params(labelsize=legend_fontsize)

        elif 0 < i < len(GxC_effect.columns) + 1:
            GE = GxC_effect.columns[i - 1]
            if GxC_effect[GE].min() < 0 and GxC_effect[GE].max() < 0:
                comap = "Blues_r"
            elif GxC_effect[GE].min() <= 0 and GxC_effect[GE].max() > 0:
                comap = "RdBu_r"
            else:
                comap = "Reds"
            sns.scatterplot(
                x="UMAP1",
                y="UMAP2",
                hue=GE,
                data=umap_base,
                ax=ax,
                s=marker_size,
                palette=comap,
                legend=False,
                rasterized=True,
            )
            norM = (
                colors.TwoSlopeNorm(
                    vcenter=0.0, vmin=GxC_effect[GE].min(), vmax=GxC_effect[GE].max()
                )
                if comap == "RdBu_r"
                else colors.Normalize(vmin=GxC_effect[GE].min(), vmax=GxC_effect[GE].max())
            )
            sm = cm.ScalarMappable(cmap=comap, norm=norM)
            cb = plt.colorbar(sm, ax=ax)
            cb.ax.tick_params(labelsize=axis_title_fontsize - 10)

            ax.set_title(
                label=(
                    f"{SNP_id} effect on ${hgnc_name[GE]}$"
                    if hgnc_name is not None
                    else f"{SNP_id} effect on {GE}"
                ),
                fontdict={"fontsize": axis_title_fontsize},
                loc="center",
            )
            ax.xaxis.label.set_fontsize(axis_title_fontsize - 1)
            ax.yaxis.label.set_fontsize(axis_title_fontsize - 1)
            ax.tick_params(labelsize=legend_fontsize)
        else:
            fig.delaxes(ax)

    plt.tight_layout()

    if savefig is not None:
        prefix, ext = os.path.splitext(savefig)
        ext = "." + format if format is not None else ".png" if ext == "" else ext
        effect_on = "genes" if gene is not None else "cells"
        plt.savefig(
            f"{prefix}_SNP-{SNP_id}-effect-on-{effect_on}{ext}",
            bbox_inches="tight",
            dpi=500,
            transparent=True,
        )

    if return_GxC_effect:
        return GxC_effect
