#!/bin/bash

snakemake --snakefile /data/danai/scripts/LIVI/src/Snakefile_train_LIVI_LSF.smk --software-deployment-method conda --conda-frontend conda --jobs 10 --latency-wait 60 --executor lsf --keep-going --rerun-incomplete
