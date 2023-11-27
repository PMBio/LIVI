from typing import Callable, Dict

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import glob
import importlib
import os

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from joblib import Parallel, delayed

from src.data.dcrm_data import DCRMDataModule
from src.lmm import row_aggregate, test_association

SIGNIFICANCE_LEVEL = 0.05


def get_geno_ids(adata):
    geno_ids = {
        "static": adata.uns["simulation"]["A_g"].any(1),
        "dynamic": adata.uns["simulation"]["A_gxc"].any(1),
    }
    geno_ids["control"] = ~(geno_ids["static"] | geno_ids["dynamic"])
    return geno_ids


def eval_tests(pvals, geno_ids):
    results = dict()
    # evaluate all tests independently
    for test, pv in pvals.items():
        pv = np.minimum(pv * pv.size, 1.0)
        # compute positive rate for each type of simulated SNP
        for k, idx in geno_ids.items():
            results[f"pr_{k}_{test}"] = (pv.min(0)[idx] < SIGNIFICANCE_LEVEL).mean()
    return results


def monkey_log(logs: Dict) -> Callable:
    """Monkey-patch for model.log to write to logs.

    Args:
        logs: Dictionary to write to.

    Returns:
        Function to monkey-patch model.log
    """

    def log(name, value, **kwargs):
        logs[name] = value

    return log


def get_step_logs(model, data):
    """Returns all values logged during model.step()."""
    logs = dict()
    model.log = monkey_log(logs)
    _ = model.step(data, None)
    return logs


def eval_vae(model, dm):
    """Collect logs and evaluate VAE."""
    adata = dm.adata
    data = dm.dataset[:]
    logs = get_step_logs(model, data)

    individual_ids = data[1].numpy()

    Z = model(data[0]).mean.detach().numpy()
    adata.obsm["Z"] = Z

    # cluster latent states
    sc.pp.neighbors(adata, use_rep="Z")
    sc.tl.leiden(adata)
    clusters = adata.obs["leiden"]

    [Y, G] = row_aggregate(
        [Z, adata.obsm["genotype"]],
        groups=individual_ids,
        fun="mean",
    )

    pvals = dict()
    pvals["tglobal"] = test_association(
        Y=Y,
        G=G,
        X=np.ones((Y.shape[0], 1)),
    )

    # test within each cluster
    pv = np.ones_like(pvals["tglobal"])
    for c in clusters.unique():
        cluster_ids = clusters == c
        adata_sub = adata[cluster_ids, :]
        # get individuals with at least one cell in cluster
        indiv = np.unique(individual_ids[cluster_ids])
        # aggregate in cluster
        [Y] = row_aggregate(
            [adata_sub.obsm["Z"]],
            groups=individual_ids[cluster_ids],
            fun="mean",
        )
        pv = np.minimum(
            test_association(
                Y=Y,
                G=G[indiv, :],
                X=np.ones((Y.shape[0], 1)),
            ),
            pv,
        )
    pvals["tcluster"] = np.minimum(pv * clusters.nunique(), 1.0)

    geno_ids = get_geno_ids(adata)
    results = eval_tests(pvals, geno_ids)
    results = {**results, **logs}
    return results


def eval_dcrm(model, dm):
    """Collect logs and evaluate DCRM."""
    adata = dm.adata
    data = dm.dataset[:]
    logs = get_step_logs(model, data)

    individual_ids = data[1].numpy()

    [G] = row_aggregate([adata.obsm["genotype"]], groups=individual_ids, fun="mean")
    pvals = dict()
    pvals["tstatic"] = test_association(
        Y=model.U_static.weight.detach().numpy(),
        G=G,
        X=np.ones((G.shape[0], 1)),
    )
    pvals["tdynamic"] = test_association(
        Y=model.U_dynamic.weight.detach().numpy(),
        G=G,
        X=np.ones((G.shape[0], 1)),
    )

    geno_ids = get_geno_ids(adata)
    results = eval_tests(pvals, geno_ids)
    results = {**results, **logs}
    return results


EVAL_FUN = {
    "VAE": eval_vae,
    "DCRM": eval_dcrm,
    "DCRMadv": eval_dcrm,
}


def evaluate_run(checkpoint_dir: str, config_file: str):
    """Evaluates all checkpoints in a given directory.

    Aggregates results and model parameters in a DataFrame.
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    # load config
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    module_str_list = config["model"]["_target_"].split(".")
    module_str = ".".join(module_str_list[:-1])
    model_str = module_str_list[-1]

    # get model class
    module = importlib.import_module(module_str)
    model_cls = getattr(module, model_str)
    model_kwargs = config["model"]

    # load data
    datamodule_kwargs = config["datamodule"]
    del datamodule_kwargs["_target_"]
    dm = DCRMDataModule(**datamodule_kwargs)
    dm.setup()

    results = list()
    for checkpoint in checkpoints:
        print(checkpoint)
        # load model
        model = model_cls.load_from_checkpoint(checkpoint)
        model.eval()

        # evaluate model
        ckpt_results = EVAL_FUN[model_str](model, dm)

        ckpt_results["model"] = model_str
        # extract epoch from checkpoint name
        if "last" in checkpoint:
            epoch = "last"
        else:
            epoch = int(checkpoint.split("_")[-1].split(".")[0])
        ckpt_results["epoch"] = epoch
        results.append({**model_kwargs, **datamodule_kwargs, **ckpt_results})
    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to checkpoint directory (may contain wildcards to match multiple directories)",
    )
    parser.add_argument(
        "--config_path",
        default="../tensorboard/version_0/hparams.yaml",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--config_path_relative",
        default=True,
        type=bool,
        help="Whether the config path is relative to the checkpoint directory",
    )
    parser.add_argument("--out_file", default="results.csv", type=str, help="Path to output file")
    parser.add_argument("--n_jobs", default=1, type=int, help="Number of parallel jobs to run")
    return parser.parse_args()


def main():
    args = parse_args()
    print("-" * 80)
    print("\n".join([f"{k}: {v}" for k, v in vars(args).items()]))
    print("-" * 80)

    checkpoints_dirs = glob.glob(args.checkpoint_dir)
    print("[INFO] Found %d directories" % len(checkpoints_dirs))
    if args.config_path_relative:
        config_paths = [os.path.join(d, args.config_path) for d in checkpoints_dirs]
    else:
        config_paths = [args.config_path] * len(checkpoints_dirs)

    df = pd.concat(
        Parallel(n_jobs=args.n_jobs)(
            delayed(evaluate_run)(d, c) for d, c in zip(checkpoints_dirs, config_paths)
        ),
        axis=0,
    )
    df.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    main()