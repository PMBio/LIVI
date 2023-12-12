"""Simulate synthetic population scRNA-seq data."""

from typing import List, Optional, Union

import anndata
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp


def sample_mafs(
    n_snps: int, maf_min: float, maf_max: float, rng: np.random.Generator
) -> Union[float, np.ndarray]:
    """Samples minor allele frequencies (MAFs).

    Args:
        n_snps: Number of SNPs to simulate.
        maf_min: Minimum MAF.
        maf_max: Maximum MAF.
        rng: Random generator.

    Returns:
        MAFs for each SNP.
    """
    assert maf_min <= maf_max and maf_min >= 0 and maf_max <= 1
    return rng.random(n_snps) * (maf_max - maf_min) + maf_min


def sample_genotypes(
    n_individuals: int, mafs: Union[np.ndarray, float], rng: np.random.Generator
) -> np.ndarray:
    """Samples genotypes using minor allele frequencies.

    Args:
        n_individuals: Number of individuals to simulate.
        mafs: MAFs, e.g. from sample_mafs.
        rng: Random generator.

    Returns:
        (n_individuals x n_snps) genotypes, encoded as
        - 0 (homozygous reference)
        - 1 (heterozygous)
        - 2 (homozygous alternative)
    """
    G = []
    mafs = np.asarray(mafs, float)
    for maf in mafs:
        probs = [(1 - maf) ** 2, 1 - ((1 - maf) ** 2 + maf**2), maf**2]
        g = rng.choice([0, 1, 2], p=probs, size=n_individuals)
        G.append(np.asarray(g, float))

    return np.stack(G, axis=1)


def column_standardize(X: np.ndarray, center: bool = True) -> np.ndarray:
    """Scale and (optionally) center columns of X."""
    if center:
        X = X - X.mean(0)
    with np.errstate(divide="raise", invalid="raise"):
        return X / X.std(0, ddof=1)


def compute_sample_var(U: np.ndarray) -> float:
    """Computes the sum of column variances of U.

    Equivalent to the expected sample variance of UU^T. Slightly faster than np.var(U, 0,
    ddof=1).sum().
    """
    n = U.shape[0]
    g = (U**2).sum() - 1 / n * (U.sum(0) ** 2).sum()
    return g / (n - 1)


def sample_effect_sizes(
    W: np.ndarray,
    n_snps: int,
    causals: List[int],
    var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generates effect sizes for effects on single factors.

    Effects are scaled such that the sum of gene-level variances for effects
    matches var (assuming that genotypes and latent contexts have been
    standardized!).

    Args:
        W: Loading matrix.
        n_snps: Number of SNPs
        causals: List with indices of SNPs with persistent effect.
        var: Total gene-level variance.
        rng: Random generator.

    Returns:
        (n_snps x n_factors) Effect size matrix.
    """
    n_factors = W.shape[1]
    n_causals = len(causals)
    A = np.zeros((n_snps, n_factors))
    W_norm = np.sqrt((W**2).sum(0))  # n_factors
    for s in causals:
        factor = rng.choice(n_factors)
        alpha = rng.choice([-1, 1])
        alpha = alpha * np.sqrt(var / n_causals)
        alpha = alpha / W_norm[factor]
        A[s, factor] = alpha
    return A


def compute_genetic_effects(
    G: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Computes the cell-by-gene genetic effect.

    Args:
        G: (n_cells x n_snps) genotype matrix.
        C: Latent contexts.
        W: Loading matrix.
        A: Effect size matrix.

    Returns:
        Genetic effect matrix.
    """
    return (C * (G @ A)) @ W.T


def simulate(
    n_cells: int,
    n_factors: int,
    n_individuals: int,
    n_genes: int,
    n_snps: int,
    maf_min: float,
    maf_max: float,
    causals_g: List[int],
    causals_gxc: List[int],
    loading_sparsity: float,
    frac_var_genetics: float,
    frac_gxc: float,
    size_factors: Optional[np.ndarray] = None,
    rng: np.random.Generator = np.random.default_rng(123),
) -> anndata.AnnData:
    """Simulates from a Poisson factor model with latent genetic effects.

    Args:
        n_cells: Number of cells to simulate.
        n_factors: Number of factors to simulate.
        n_individuals: Number of individuals to simulate.
        n_genes: Number of genes to simulate.
        n_snps: Number of SNPs to simulate.
        maf_min: Minimum minor allele frequency.
        maf_max: Maximum minor allele frequency.
        causals_g: Indices for SNPs with persistent effect.
        causals_gxc: Indices for SNPs with dynamic effect.
        loading_sparsity: Prior for sparsity mask on factor loadings.
        frac_var_genetics: Fraction of total variance explained by genetics.
        frac_gxc: Fraction of genetic variance explained by GxC.
        size_factors: Optional cell size factors. If not given, samples size
            factors randomly from [10**4, 10**5).
        rng: Random generator.

    Returns:
        anndata.AnnData object with simulated data.
    """
    # check values
    if loading_sparsity < 0 or loading_sparsity > 1.0:
        raise ValueError("Loading sparsity has to be from [0, 1]")
    all_causals = causals_g + causals_gxc
    c_min, cmax = min(all_causals), max(all_causals)
    if c_min < 0 or cmax > n_snps:
        raise ValueError("Causal indices have to be positive and smaller than n_snps")

    # sample individuals & genotypes
    individuals = rng.choice(n_individuals, size=n_cells)
    mafs = sample_mafs(n_snps=n_snps, maf_min=maf_min, maf_max=maf_max, rng=rng)
    G = sample_genotypes(n_individuals=n_individuals, mafs=mafs, rng=rng)
    G = G[individuals, :]  # expand to cell level
    G = column_standardize(G)  # important to match total variance

    # sample latent contexts
    C = rng.normal(size=(n_cells, n_factors))
    C = column_standardize(C)  # ensure standardization

    # sample loadings
    W = rng.normal(size=(n_genes, n_factors))
    B = rng.choice([0, 1], p=[1 - loading_sparsity, loading_sparsity], size=W.shape)
    W = B * W

    X_base = C @ W.T
    total_var_base = compute_sample_var(X_base)
    total_var_genetics = total_var_base * frac_var_genetics / (1 - frac_var_genetics)

    # compute variances
    var_gxc = total_var_genetics * frac_gxc
    var_g = total_var_genetics * (1 - frac_gxc)

    # compute effect sizes
    A_g = sample_effect_sizes(W=W, n_snps=n_snps, causals=causals_g, var=var_g, rng=rng)
    A_gxc = sample_effect_sizes(W=W, n_snps=n_snps, causals=causals_gxc, var=var_gxc, rng=rng)

    # compute gene-level effects
    X_g = compute_genetic_effects(
        G=G,
        C=np.ones_like(C),
        W=W,
        A=A_g,
    )
    X_gxc = compute_genetic_effects(G=G, C=C, W=W, A=A_gxc)
    X = X_base + X_gxc + X_g

    # compute total variance for each component
    total_var = compute_sample_var(X)
    total_var_g = compute_sample_var(X_g)
    total_var_gxc = compute_sample_var(X_gxc)

    # compute expression mean and sample
    if size_factors is None:
        size_factors = rng.integers(10**4, 10**5, size=n_cells).reshape(-1, 1)

    X = scipy.special.softmax(X, axis=1) * size_factors
    X = sp.csr_matrix(rng.poisson(X))

    obs_names = [f"cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(
        {
            "individual": individuals,
            "size_factor": size_factors.ravel(),
        },
        index=obs_names,
    )
    obsm = {
        "genotype": G,
        "contexts": C,
    }
    varm = {"W": W}
    uns = {
        "simulation": {
            "var_g": var_g,
            "var_gxc": var_gxc,
            "total_var": total_var,
            "total_var_g": total_var_g,
            "total_var_gxc": total_var_gxc,
            "A_g": A_g,
            "A_gxc": A_gxc,
        }
    }
    return anndata.AnnData(
        X=X,
        dtype=X.dtype,
        obs=obs,
        obsm=obsm,
        varm=varm,
        uns=uns,
    )


def main():
    adata = simulate(
        n_cells=10000,
        n_factors=10,
        n_individuals=100,
        n_genes=1000,
        n_snps=5,
        maf_min=0.3,
        maf_max=0.5,
        causals_g=[0, 1],
        causals_gxc=[1, 2, 3],
        loading_sparsity=0.5,
        frac_var_genetics=0.05,
        frac_gxc=0.2,
    )
    print(adata)
    for k, v in adata.uns["simulation"].items():
        if "var" in k:
            print(f"{k:14}: {v:6.2f}")
    adata.write("data/test.h5ad")


if __name__ == "__main__":
    main()
