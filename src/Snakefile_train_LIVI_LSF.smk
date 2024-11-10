import os
import yaml
from datetime import datetime

def read_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


LIVI_ENV="LIVIHydra_new"
LIVI_DIR="/data/danai/scripts/LIVI"
LIVI_CKPT_DIR="/data/danai/Data/hydra_logs/LIVI2/trainable_A/runs"
WARMUP_MODEL_YAML="LIVIcis_wo_adv_onek1k.yaml"
WARMUP_YAML="LIVI-VAE-warmup_onek1k.yaml"
MODEL_YAML="LIVIcis_wo_adv_onek1k_resume-training.yaml"
EXPERIMENT_YAML="LIVIcis-cell-state_onek1k.yaml"
RANDOM_SEEDS=[32, 50, 100, 200, 500]


warmup_yaml = read_yaml(os.path.join(LIVI_DIR, "configs", "experiment", WARMUP_YAML))
vae = warmup_yaml["task_name"]
exp_yaml = read_yaml(os.path.join(LIVI_DIR, "configs", "experiment", EXPERIMENT_YAML))
experiment = exp_yaml["task_name"]
experiment_dir = os.path.join(LIVI_CKPT_DIR, experiment)
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)


rule all:
    input:
        os.path.join(experiment_dir, f"{vae}.done"),
        expand(os.path.join(experiment_dir, f"{experiment}_Gseed"+"{seed}"+".done"), seed=RANDOM_SEEDS)


rule pretrain_vae:
    input:
        os.path.join(LIVI_DIR, "configs", "model", WARMUP_MODEL_YAML)
    output:
        touch(os.path.join(experiment_dir, f"{vae}.done")),
        output_file = os.path.join(LIVI_CKPT_DIR, f'{datetime.now().strftime("%Y-%m-%d_%H-%M")}_{vae}', "checkpoints", "last.ckpt")
    params:
        LIVI_dir=LIVI_DIR,
        model_yml=WARMUP_YAML,
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M"),
        experiment_yml=os.path.join(LIVI_DIR, "configs", "experiment", EXPERIMENT_YAML)
    conda:
        LIVI_ENV
    resources:
        mem="150GB", # for Cardinal
        lsf_queue="gpu",
        lsf_extra="-r -gpu num=1:j_exclusive=yes:gmem=10G -o VAE-warmup.out -e VAE-warmup.err"
    shell:
        """
        python {params.LIVI_dir}/src/utils/update_task-name.py --experiment_yaml {params.experiment_yml} --warmup_ckpt_timestamp {params.timestamp}
        python {params.LIVI_dir}/src/train.py experiment={params.model_yml}
        """


rule train_UV:
    input:
        rules.pretrain_vae.output.output_file
    output:
        directory(os.path.join(LIVI_CKPT_DIR, f'{datetime.now().strftime("%Y-%m-%d_%H-%M")}_{experiment}_Gseed'+"{seed}")),
        touch(os.path.join(experiment_dir, f"{experiment}_Gseed"+"{seed}"+".done"))
    params:
        LIVI_dir=LIVI_DIR,
        LIVI_ckpt_dir=LIVI_CKPT_DIR,
        model_yml=os.path.join(LIVI_DIR, "configs", "model", MODEL_YAML),
        experiment_dir=experiment_dir,
        experiment_yml=os.path.join(LIVI_DIR, "configs", "experiment", EXPERIMENT_YAML),
        experiment_seed_yml=os.path.splitext(EXPERIMENT_YAML)[0]+"_Gseed{seed}.yaml"
    conda:
        LIVI_ENV
    resources:
        mem="150GB", # for Cardinal
        lsf_queue="gpu",
        lsf_extra="-r -gpu num=1:j_exclusive=yes:gmem=10G -o LIVI_GxC.out -e LIVI_GxC.err"
    shell:
        """
        python {params.LIVI_dir}/src/utils/update_ckpt-train-yaml.py --warmup_file {input} --LIVI_dir {params.LIVI_dir}
        python {params.LIVI_dir}/src/utils/update_model-yaml.py --model_yaml {params.model_yml} --experiment_yaml {params.experiment_yml} --random_seed {wildcards.seed}
        python {params.LIVI_dir}/src/train_from_ckpt.py experiment={params.experiment_seed_yml}
        """
