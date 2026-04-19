<div align="center">

# Latent Interaction Variational Inference (LIVI)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

LIVI is a probabilistic model for single-cell RNA-seq data collected from a large population of individuals. At its core, LIVI builds on variational autoencoders (VAEs), employing structured linear decoders to decompose observed variation in single-cell expression to cell-state variation, donor-driven variation and their interaction. The resulting model has properties that resemble classical factor analysis, where the decoder is a factor loadings matrix instead of a neural network with non-linear activations. \
Once trained, LIVI enables efficient donor-level association testing, while retaining single-cell resolution and interpretation. Because donor latent factors are inferred without information on specific donor-level characteristics, such as SNP genotypes, they can be used as quantitative phenotypes to test for genetic effects without the risk of circularity. Following association testing at the donor level, the discovered effects can be projected back onto single cells via LIVI's latent donor-cell-state interaction model ($D \\times C$), and the decoder weights can be inspected to identify the affected sets of genes.

Check out our preprint for more details on the model and analyses: [Vagiaki *et al.*, 2026](https://doi.org/10.64898/2026.02.04.703363)

## Quick start

### How to install

Install dependencies

```bash
# clone project
git clone https://github.com/PMBio/LIVI
cd LIVI

# [OPTIONAL] create conda environment
conda create -n LIVIenv python=3.11
conda activate LIVIenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### How to train LIVI

LIVI builds on the [Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template) structure with configuration-based experiments. This means that you need to specify your own config files based on your dataset and file system. Assuming that you have cloned this repo into a directory `/your/homedir/LIVI`. You need to specify a model, a datamodule, a paths and an experiment config under `/your/homedir/LIVI/configs/model`, `/your/homedir/LIVI/configs/datamodule`, `/your/homedir/LIVI/configs/paths` and `/your/homedir/LIVI/configs/experiment` , respectively.

- Example of model config: [configs/model/LIVIcis_onek1k_10K-HVG-HEX.yaml](configs/model/LIVIcis_onek1k_10K-HVG-HEX.yaml) . You can use the model config to specify the numbers of latent factors, learning rate, warm-up epochs etc. `x_dim` should correspond to the number of genes and `y_dim` to the number of individuals in your dataset.
- Example of datamodule config: [configs/datamodule/onek1k_10K-HVG-HEX_LIVIcis.yaml](configs/datamodule/onek1k_10K-HVG-HEX_LIVIcis.yaml). You can use the datamodule config to specify the anndata object you wish to train LIVI on (including which are the donor IDs and covariates IDs keys in `adata.obs`), as well as training specific parameters like batch size.

After you have the model, datamodule and path configs, you can specify them in your experiment config. Example of experiment config using the model and datamodule config from above: [configs/experiment/LIVIcis-cell-state_onek1k_10K-HVG-HEX_train-end-to-end.yaml](configs/experiment/LIVIcis-cell-state_onek1k_10K-HVG-HEX_train-end-to-end.yaml)

You can create you custom path config as below:

```
# path to root directory --> DO NOT CHANGE
root_dir: ${oc.env:PROJECT_ROOT}

# path to data directory --> ADAPT
data_dir:/your/data/directory

# path to logging directory --> ADAPT
log_dir: /your/logging/directory

# path to output directory, created dynamically by hydra --> DO NOT CHANGE
# use it to store all files generated during the run, like ckpts and metrics
output_dir: ${hydra:runtime.output_dir}

# path to working directory --> DO NOT CHANGE
work_dir: ${hydra:runtime.cwd}
```

**Finally, you can train the model with chosen experiment config like this:**

```bash
python src/train.py experiment=experiment_name.yaml
```

Train model on CPU/GPU

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=100 datamodule.batch_size=528
```

### Inference and downstream association testing

The following performs inference on the gene expression data stored in `--adata`, using the "best" model checkpoint stored under `--model_run_dir`. Subsequently, it runs association testing between inferred donor factors and the SNP genotypes in `--genotype_matrix` (prefix of .bed, .bim, .fam PLINK files), while accounting for covariates (e.g. expression PCs) specified under `--covariates` and population structure specified under `--kinship` using a LMM. Output files are saved under `-od`. \
**For a full list of options please run `python src/analysis/livi_analysis.py --h`.**

```bash
python src/analysis/livi_analysis.py \
    --model_run_dir /path/to/model/checkpoints/ \
    --adata /path/to/adata.h5ad \
    --celltype_column CELLTYPE_COLUMN \
    --individual_column INDIVIDUAL_COLUMN \
    --covariates /path/to/association/testing/covariate_file.tsv \
    --fdr_threshold FDR \
    --genotype_matrix /path/to/PLINK/genotype/matrix --plink \
    --method LMM \
    --kinship /path/to/Kinship_matrix.tsv \
    -od /path/to/output/directory
```

### Interpretation of association testing results

Examples of downstream intrepretation and plotting of association testing results can be found in [https://github.com/danaivagiaki/LIVI_analyses](https://github.com/danaivagiaki/LIVI_analyses)

## Citation

If you use LIVI in your research, please cite:

```bibtex
@article{vagiaki2026livi,
  title = {Mapping trans-eQTLs at single-cell resolution using Latent Interaction Variational Inference},
  author = {Vagiaki, Danai and Heinen, Tobias and Saraswat, Manu and Clarke, Brian and Stegle, Oliver},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.02.04.703363},
  URL = {https://www.biorxiv.org/content/early/2026/02/06/2026.02.04.703363},
}
```

## Acknowledgments

This project builds on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).
