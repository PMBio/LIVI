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

Check out our preprint for more details on the model and analyses: [Vagiaki et al., 2026](https://doi.org/10.64898/2026.02.04.703363)

## Quick start

We are working on more comprehensive documentation. In the meantime, if you need assistance using our tool beyond this *Quick start* guide, feel free to reach out at [danai.vagiaki@embl.de](danai.vagiaki@embl.de)

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

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

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
