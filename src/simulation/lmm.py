"""Utilities for running LMM tests."""

from typing import List, Optional

import numpy as np
from glimix_core.lmm import LMM
from numpy_sugar import epsilon
from numpy_sugar.linalg import economic_qs_linear
from scipy.stats import chi2
from tqdm import trange


def row_aggregate(
    matrices: List[np.ndarray], groups: np.ndarray, fun: str = "mean"
) -> List[np.ndarray]:
    """Aggregates rows by group for multiple matrices.

    Args:
        matrices: 2D Matrices to aggregate.
        groups: Group indicator.
        fun: 'mean' or 'sum'.

    Returns:
        Aggregated matrices.
    """
    if fun not in ["mean", "sum"]:
        raise ValueError("fun has to be one of 'mean' or 'sum'")
    M = np.hstack(matrices)
    f = np.mean if fun == "mean" else np.sum

    ranges = list()
    d = 0
    for m in matrices:
        ranges.append(range(d, d + m.shape[1]))
        d += m.shape[1]
    unique_groups = np.unique(groups)
    out = np.zeros((len(unique_groups), M.shape[1]))
    for i, g in enumerate(np.unique(groups)):
        out[i, :] = f(M[groups == g, :], axis=0)
    return [out[:, ranges[i]] for i in range(len(ranges))]


def lrt_pvalues(null_lml: float, alt_lmls: np.ndarray, dof: int = 1) -> np.ndarray:
    """Computes p-values from likelihood ratios.

    Args:
        null_lml: Log of the marginal likelihood under the null hypothesis.
        alt_lmls: Log of the marginal likelihoods under the alternative hypotheses.
        dof:  Degrees of freedom.

    Returns:
        P-values.
    """
    lrs = np.clip(-2 * null_lml + 2 * np.asarray(alt_lmls, float), epsilon.super_tiny, np.inf)
    pv = chi2(df=dof).sf(lrs)
    return np.clip(pv, epsilon.super_tiny, 1 - epsilon.tiny)


def compute_min_pvals(X: np.ndarray, axis: int = 0):
    """Returns minimum P-values across axis, Bonferroni adjusted."""
    return np.minimum(X.min(axis) * X.shape[axis], 1.0)


def test_association(
    Y: np.ndarray,
    G: np.ndarray,
    X: np.ndarray,
    E: Optional[np.ndarray] = None,
    fast: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """Runs LMM association test.

    Args:
        Y: Responses.
        G: Covariates to test.
        X: Other covariates / fixed effects.
        E: Group covariates / random effects.
        fast: Use fast (``True``) or regular testing. Defaults to ``False``.
            Re-uses parameter estimates from null model when fitting alternative
            models.
        show_progress Show progress bar.

    Returns:
        P-values.
    """

    QS = economic_qs_linear(E) if E is not None else None
    pvals = np.ones((Y.shape[1], G.shape[1]))
    for i in trange(Y.shape[1], disable=not show_progress):
        # fit null model without genotypes
        null_lmm = LMM(y=Y[:, [i]], X=X, QS=QS, restricted=False)
        null_lmm.fit(verbose=False)

        if fast:
            alt_lmls = null_lmm.get_fast_scanner().fast_scan(G, verbose=False)["lml"]
        else:
            alt_lmls = []
            for j in range(G.shape[1]):
                g = G[:, [j]]
                W = np.concatenate([X, g], axis=1)

                # alternative model
                alt_lmm = LMM(Y[:, [i]], W, QS, restricted=False)
                alt_lmm.fit(verbose=False)
                alt_lmls.append(alt_lmm.lml())

        # compute LRT pvalues
        pvals[i, :] = lrt_pvalues(null_lmm.lml(), np.asarray(alt_lmls), dof=1)
    return pvals


def main():
    from limix.qc import quantile_gaussianize
    from simulate import simulate

    adata = simulate(
        n_cells=30000,
        n_factors=40,
        n_individuals=100,
        n_genes=1000,
        n_snps=60,
        maf_min=0.3,
        maf_max=0.5,
        causals_g=list(range(20)),
        causals_gxc=[],
        loading_sparsity=0.5,
        frac_var_genetics=0.0025,
        frac_gxc=0.0,
    )
    var_g = adata.uns["simulation"]["total_var_g"]
    var_gxc = adata.uns["simulation"]["total_var_gxc"]
    var_total = adata.uns["simulation"]["total_var"]
    print(f"FTV persistent effects: {var_g/var_total:.4f}")
    print(f"FTV dynamic effects: {var_gxc/var_total:.4f}")

    # aggregate cells for each individual to test for persistent effects:
    #   (1) adjust library size
    #   (2) log transform
    #   (3) quantile normalize
    #   (4) mean aggregate

    size_factors = adata.X.sum(1).A
    Y = np.log1p(np.asarray(adata.X.todense()) / size_factors)
    Y = quantile_gaussianize(Y)
    G = np.asarray(adata.obsm["genotype"])
    individuals = adata.obs["individual"].to_numpy()

    # get individual-level exprs and genotypes
    [Y, G] = row_aggregate([Y, G], groups=individuals, fun="mean")
    pvals = test_association(Y=Y, G=G, X=np.ones((Y.shape[0], 1)), fast=True)
    #  adjust p-values
    pvals = compute_min_pvals(pvals) * pvals.shape[1]

    print("LMM test results (alpha = 0.05)")
    print(f" True positive rate:  {(pvals[:10] < 0.05).mean()}")
    print(f" False positive rate: {(pvals[10:] < 0.05).mean()}")


if __name__ == "__main__":
    main()
