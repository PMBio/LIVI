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

## How to run

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

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
