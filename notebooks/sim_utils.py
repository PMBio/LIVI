import os
import sys

sys.path.append("..")

import re
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from glimix_core.lmm import LMM
from numpy_sugar import epsilon
from numpy_sugar.linalg import economic_qs_linear
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2, false_discovery_control
from tqdm import trange

from src.models.livi import LIVI


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


def test_embedding(U, adata, show_progress):
    G = (
        pd.DataFrame(adata.obsm["genotype"], index=adata.obs["individual"])
        .drop_duplicates()
        .loc[U.index]
    )
    X = np.ones((U.shape[0], 1))
    return test_association(Y=U.to_numpy(), G=G.to_numpy(), X=X, show_progress=show_progress)


def eval_checkpoint(checkpoint, show_progress=False):
    results = dict()

    # load config
    config_file = os.path.dirname(checkpoint).replace(
        "checkpoints", "tensorboard/version_0/hparams.yaml"
    )
    with open(config_file) as f:
        config = yaml.safe_load(f)

    adata_path = config["datamodule"]["adata"]
    pattern = r"adata_(\w+)-([\d.]+)\.h5ad"
    results["adata"] = adata_path
    match = re.match(pattern, os.path.basename(adata_path))
    results["experiment"] = match.group(1)
    results["param"] = match.group(2)

    # load data
    adata = sc.read_h5ad(config["datamodule"]["adata"])
    y, y_index = pd.factorize(adata.obs["individual"])

    # identify simulated effects
    causal_snps = adata.uns["simulation"]["B_gxc"].any(1)
    causal_factors = adata.uns["simulation"]["B_gxc"].any(0)

    # load model parameters
    model = LIVI.load_from_checkpoint(checkpoint)
    U = model.U_context.weight.cpu().detach().numpy()
    U = pd.DataFrame(U, index=y_index)

    # factor correspondence
    base_loadings = model.decoder.mean[0].weight.cpu().detach().numpy()

    idx_I = [f"I{i}" for i in range(base_loadings.shape[1])]
    idx_G = [f"G{i}" for i in range(adata.varm["W"].shape[1])]
    idx = idx_I + idx_G
    df = pd.DataFrame(np.corrcoef(x=base_loadings.T, y=adata.varm["W"].T), index=idx, columns=idx)
    # find best pairing
    row_ids, col_ids = linear_sum_assignment(df.loc[idx_I, idx_G].abs().to_numpy(), maximize=True)
    results["factor_corr"] = df.loc[idx_I, idx_G].iloc[row_ids, col_ids]
    results["MCC"] = np.abs(np.diag(df.loc[idx_I, idx_G].iloc[row_ids, col_ids])).mean()

    # run tests
    pvals = test_embedding(U, adata, show_progress)
    qvals = false_discovery_control(pvals, axis=None, method="bh").reshape(pvals.shape)
    snp_qvals = qvals.min(0)
    factor_qvals = qvals.min(1)

    results["qvals"] = snp_qvals
    results["power"] = (snp_qvals[causal_snps] < 0.05).mean()

    # check if correct factors were identified
    if results["power"] == 0:
        results["correspondence"] = 0
    else:
        A_I = torch.sigmoid(model.A).cpu().detach().numpy()[col_ids, :]
        A_G = adata.uns["simulation"]["A"]

        # restrict to true positive factors & compute correlation with groundtruth factor

        a_G = A_G[:, (factor_qvals < 0.05) & causal_factors].argmax(0)
        a_I = A_I[:, (factor_qvals < 0.05) & causal_factors].argmax(0)
        results["correspondence"] = (a_G == a_I).mean()

    results["model"] = "LIVI (adv: %.2f)" % model.hparams.adversary_weight
    results["seed"] = config["seed"]
    return results


def run_pca(adata_path, show_progress=False):
    adata = sc.read_h5ad(adata_path)
    causals = adata.uns["simulation"]["B_gxc"].any(1)

    # params
    pattern = r"adata_(\w+)-([\d.]+)\.h5ad"
    match = re.match(pattern, os.path.basename(adata_path))

    # run pca
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=100)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)

    # global test
    U = pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names)
    U["individual"] = adata.obs["individual"]
    U = U.groupby("individual").mean()
    pvals = test_embedding(U, adata, show_progress)

    results = list()

    snp_qvals = false_discovery_control(pvals, axis=None, method="bh").reshape(pvals.shape).min(0)
    power_global = (snp_qvals[causals] < 0.05).mean()
    results.append(
        {
            "experiment": match[1],
            "param": match[2],
            "power": power_global,
            "model": "PCA (global)",
        }
    )

    # by celltype test
    snp_qvals = np.ones_like(snp_qvals)
    for c in adata.obs["leiden"].unique():
        adata_sub = adata[adata.obs["leiden"] == c, :]
        U = pd.DataFrame(adata_sub.obsm["X_pca"], index=adata_sub.obs_names)
        U["individual"] = adata_sub.obs["individual"]
        U = U.groupby("individual").mean()
        pvals = test_embedding(U, adata_sub, show_progress)

        snp_qvals = np.minimum(
            snp_qvals,
            false_discovery_control(pvals, axis=None, method="bh").reshape(pvals.shape).min(0),
        )
    snp_qvals = np.minimum(snp_qvals * adata.obs["leiden"].nunique(), 1.0)
    power_bycelltype = (snp_qvals[causals] < 0.05).mean()
    results.append(
        {
            "experiment": match[1],
            "param": match[2],
            "power": power_bycelltype,
            "model": "PCA (celltype)",
        }
    )
    return results
