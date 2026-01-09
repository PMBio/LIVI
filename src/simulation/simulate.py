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


def compute_sample_var(D: np.ndarray) -> float:
    """Computes the sum of column variances of U.

    Equivalent to the expected sample variance of UU^T. Slightly faster than np.var(U, 0,
    ddof=1).sum().
    """
    n = D.shape[0]
    g = (D**2).sum() - 1 / n * (D.sum(0) ** 2).sum()
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
    matches var (assuming that genotypes and latent contexts are independent
    and have each been standardized).

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
    B = np.zeros((n_snps, n_factors))
    W_norm = np.sqrt((W**2).sum(0))  # n_factors
    for s in causals:
        factor = rng.choice(n_factors)
        beta = rng.choice([-1, 1])
        beta = beta * np.sqrt(var / n_causals)
        beta = beta / W_norm[factor]
        B[s, factor] = beta
    return B


def compute_genetic_effects(
    G: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Computes the cell-by-gene genetic effect.

    Args:
        G: (n_cells x n_snps) genotype matrix.
        C: Latent contexts.
        W: Loading matrix.
        B: Effect size matrix.

    Returns:
        Genetic effect matrix.
    """
    return (C * (G @ B)) @ W.T


def sample_loadings(
    n_genes: int,
    n_factors: int,
    loading_sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Samples sparse loading matrix.

    Args:
        n_genes: Number of genes
        n_factors: Number of factors
        loading_sparsity: Number between 0 and 1 indicating the expected
            fraction of non-zero elements in the loading matrix.

    Returns:
        Loading matrix.
    """
    W = rng.normal(size=(n_genes, n_factors))
    B = rng.choice([0, 1], p=[1 - loading_sparsity, loading_sparsity], size=W.shape)
    return B * W


def simulate(
    n_cells: int,
    n_factors: int,
    n_factors_g: int,
    n_factors_DxC: int,
    n_individuals: int,
    n_genes: int,
    n_snps: int,
    maf_min: float,
    maf_max: float,
    causals_g: List[int],
    causals_DxC: List[int],
    loading_sparsity: float,
    frac_var_genetics: float,
    frac_DxC: float,
    size_factors: Optional[np.ndarray] = None,
    n_celltypes: int = 1,
    frac_var_celltypes: float = 0.0,
    seed: int = 123,
) -> anndata.AnnData:
    """Simulates from a Poisson factor model with latent genetic effects.

    Base latent contexts are simulated from a Gaussian distribution:

        c ~ N(0, aK + (1-a)I)

    where K is a block-diagonal matrix (celltype kernel) and a is the fraction
    of latent variance driven by cell-type differences. Let W, W_g, W_DxC be
    loading matrices for context effects, persistent and dynamic genetic effects,
    respectively. The expected expression is modeled as

        E[X] = Softmax(X_base + X_g + X_DxC),

    where

        X_base = C @ W
        X_g = (G @ B_g) @ W_g
        X_DxC = ((C @ A) o (G @ B_DxC)) @ W_DxC

    Here, o denotes the element-wise (Hadamard) product, B_g, B_DxC are the
    genetic effect size matrices and A is a design matrix assigning exactly one
    cell state factor to each genetic effect factor. Observed counts are sampled
    from a Poisson model at the gene level.

    Args:
        n_cells: Number of cells to simulate.
        n_factors: Number of base factors to simulate.
        n_factors_g: Number of persistent genetic effect factors to simulate.
        n_factors_DxC: Number of DxC effect factors to simulate.
        n_individuals: Number of individuals to simulate.
        n_genes: Number of genes to simulate.
        n_snps: Number of SNPs to simulate.
        maf_min: Minimum minor allele frequency.
        maf_max: Maximum minor allele frequency.
        causals_g: Indices for SNPs with persistent effect.
        causals_DxC: Indices for SNPs with dynamic effect.
        loading_sparsity: Prior for sparsity mask on factor loadings.
        frac_var_genetics: Fraction of total variance explained by genetics.
        frac_DxC: Fraction of genetic variance explained by DxC.
        size_factors: Optional cell size factors. If not given, samples size
            factors randomly from [10**4, 10**5).
        n_celltypes: Number of cell types (clusters in latent space) to
            simulate. Defaults to one.
        frac_var_celltypes: Fraction of base latent variance driven by
            differences between celltypes. Should be in [0, 1). Defaults to
            zero.
        seed: Random seed.

    Returns:
        anndata.AnnData object with simulated data.
    """
    rng = np.random.default_rng(seed)

    # check values
    if loading_sparsity < 0 or loading_sparsity > 1.0:
        raise ValueError("Loading sparsity has to be from [0, 1]")
    all_causals = causals_g + causals_DxC
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
    celltypes = rng.choice(n_celltypes, size=n_cells)
    C = rng.normal(size=(n_cells, n_factors))
    if n_celltypes > 1:
        E = pd.get_dummies(celltypes).to_numpy()
        E = E / np.sqrt(compute_sample_var(E))
        C = np.sqrt(1 - frac_var_celltypes) * C
        C = C + np.sqrt(frac_var_celltypes) * E @ rng.normal(size=(n_celltypes, n_factors))
    C = column_standardize(C)  # ensure standardization

    # sample loadings
    loadings = dict()
    loadings_kwargs = dict(n_genes=n_genes, loading_sparsity=loading_sparsity, rng=rng)
    loadings["W"] = sample_loadings(**loadings_kwargs, n_factors=n_factors)
    loadings["W_g"] = sample_loadings(**loadings_kwargs, n_factors=n_factors_g)
    loadings["W_DxC"] = sample_loadings(**loadings_kwargs, n_factors=n_factors_DxC)

    # sample design matrix A of shape (n_factors x n_factors_DxC)
    A = np.zeros((n_factors, n_factors_DxC))
    # choose one context factor for each genetic effect factor
    A[rng.choice(n_factors, size=n_factors_DxC), np.arange(n_factors_DxC)] = 1

    X_base = C @ loadings["W"].T
    total_var_base = compute_sample_var(X_base)
    total_var_genetics = total_var_base * frac_var_genetics / (1 - frac_var_genetics)

    # compute target sample variances
    var_DxC = total_var_genetics * frac_DxC
    var_g = total_var_genetics * (1 - frac_DxC)

    # compute effect sizes
    B_g = sample_effect_sizes(
        W=loadings["W_g"], n_snps=n_snps, causals=causals_g, var=var_g, rng=rng
    )
    B_DxC = sample_effect_sizes(
        W=loadings["W_DxC"], n_snps=n_snps, causals=causals_DxC, var=var_DxC, rng=rng
    )

    # compute gene-level effects
    X_g = compute_genetic_effects(
        G=G,
        C=np.ones((n_cells, n_factors_g)),
        W=loadings["W_g"],
        B=B_g,
    )
    X_DxC = compute_genetic_effects(G=G, C=C @ A, W=loadings["W_DxC"], B=B_DxC)

    X = X_base + X_DxC + X_g

    # compute actual sample variances for each component
    total_var = compute_sample_var(X)
    total_var_g = compute_sample_var(X_g)
    total_var_DxC = compute_sample_var(X_DxC)

    # compute expression mean and sample gene counts
    if size_factors is None:
        size_factors = rng.integers(10**4, 10**5, size=n_cells).reshape(-1, 1)

    X = scipy.special.softmax(X, axis=1)
    X = sp.csr_matrix(rng.poisson(X * size_factors))

    obs_names = [f"cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(
        {
            "individual": individuals,
            "size_factor": size_factors.ravel(),
            "celltype": celltypes,
        },
        index=obs_names,
    )
    obsm = {
        "genotype": G,
        "contexts": C,
    }
    varm = loadings
    uns = {
        "simulation": {
            "target_var_g": var_g,
            "target_var_DxC": var_DxC,
            "total_var": total_var,
            "total_var_g": total_var_g,
            "total_var_DxC": total_var_DxC,
            "B_g": B_g,
            "B_DxC": B_DxC,
            "A": A,
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
        n_cells=1000,
        n_factors=10,
        n_factors_g=20,
        n_factors_DxC=20,
        n_individuals=25,
        n_genes=100,
        n_snps=5,
        maf_min=0.3,
        maf_max=0.5,
        causals_g=[0, 1],
        causals_DxC=[1, 2, 3],
        loading_sparsity=0.5,
        frac_var_genetics=0.05,
        frac_DxC=0.2,
    )
    print(adata)
    for k, v in adata.uns["simulation"].items():
        if "var" in k:
            print(f"{k:14}: {v:6.2f}")


if __name__ == "__main__":
    main()
