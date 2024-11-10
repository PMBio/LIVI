import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import os
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
        "--model_yaml",
        type=str,
        required=True,
        help="Absolute path of the .yaml file specifying the LIVI model.",
    )
    parser.add_argument(
        "--experiment_yaml",
        type=str,
        required=True,
        help="Absolute path of the .yaml file specifying the experiment to run.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Absolute path of the directory to write LIVI model checkpoints",
    )

    args = parser.parse_args()

    assert os.path.isfile(args.model_yaml)
    assert os.path.isfile(args.experiment_yaml)

    # Update the random seed in the model .yaml
    model_dir = os.path.dirname(args.model_yaml)
    model_name = os.path.splitext(os.path.basename(args.model_yaml))[0]
    model_yaml = read_yaml(args.model_yaml)
    model_yaml["genetics_seed"] = args.random_seed
    write_yaml(os.path.join(model_dir, f"{model_name}_Gseed{args.random_seed}.yaml"), model_yaml)
    # Update the name of the model .ymal in the experiment .yaml
    experiment_dir = os.path.dirname(args.experiment_yaml)
    experiment_name = os.path.splitext(os.path.basename(args.experiment_yaml))[0]
    experiment_yaml = read_yaml(args.experiment_yaml)
    experiment_yaml["defaults"][0][
        "override /model"
    ] = f"{model_name}_Gseed{args.random_seed}.yaml"
    taskname = experiment_yaml["task_name"]
    experiment_yaml["task_name"] = f"{taskname}_Gseed{args.random_seed}"
    experiment_seed_yml = os.path.join(
        experiment_dir, f"{experiment_name}_Gseed{args.random_seed}.yaml"
    )
    with open(experiment_seed_yml, "w") as outfile:
        outfile.write("# @package _global_")
        outfile.write("\n\n")
    append_yaml(
        os.path.join(experiment_dir, f"{experiment_name}_Gseed{args.random_seed}.yaml"),
        experiment_yaml,
    )
