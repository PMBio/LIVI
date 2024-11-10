import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import os
import re
from collections import OrderedDict

import yaml


# Custom Loader to preserve order
class OrderedLoader(yaml.SafeLoader):
    pass


def construct_ordered_mapping(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


OrderedLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_ordered_mapping
)


# Custom Dumper to preserve order when writing
class OrderedDumper(yaml.SafeDumper):
    pass


def represent_ordered_mapping(dumper, data):
    return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())


OrderedDumper.add_representer(OrderedDict, represent_ordered_mapping)


def read_yaml(file_path):
    with open(file_path) as file:
        data = yaml.load(file, Loader=OrderedLoader)
    return data


# Function to write YAML file with preserved order
def write_yaml(file_path, data):
    with open(file_path, "w") as file:
        yaml.dump(data, file, Dumper=OrderedDumper, default_flow_style=False)


def append_yaml(file_path, data):
    with open(file_path, "a") as file:
        yaml.dump(data, file, Dumper=OrderedDumper, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_yaml",
        type=str,
        required=True,
        help="Absolute path of the .yaml file specifying the experiment to run.",
    )
    parser.add_argument(
        "--warmup_ckpt_timestamp",
        type=str,
        required=True,
        help="Absolute path of the directory to write LIVI model checkpoints",
    )

    args = parser.parse_args()

    assert os.path.isfile(args.experiment_yaml)

    # Update the task name in the experiment .yaml to reflect the VAE checkpoint that was used to train LIVI
    timestamp = args.warmup_ckpt_timestamp
    experiment_yaml = read_yaml(args.experiment_yaml)
    taskname = experiment_yaml["task_name"]
    taskname = re.sub("_train-from-ckpt-[0-9]+-[0-9]+-[0-9]+_[0-9]+-[0-9]+", "", taskname)
    experiment_yaml["task_name"] = f"{taskname}_train-from-ckpt-{timestamp}"
    experiment_yml_out = args.experiment_yaml
    with open(experiment_yml_out, "w") as outfile:
        outfile.write("# @package _global_")
        outfile.write("\n\n")
    append_yaml(experiment_yml_out, experiment_yaml)
