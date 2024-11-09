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
        "--warmup_file",
        type=str,
        required=True,
        help="Absolute path of the .yaml file used for VAE warm-up. Alternatively, directly the filename of the VAE checkpoint file can be provided.",
    )
    parser.add_argument(
        "--LIVI_dir",
        type=str,
        required=True,
        help="Absolute path of the cloned LIVI repo directory.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Absolute path of the directory to write LIVI model checkpoints",
    )

    args = parser.parse_args()

    assert os.path.isfile(args.warmup_file)
    assert os.path.isdir(args.LIVI_dir)

    # Get the checkpoint path
    warm_up_file = os.path.split(args.warmup_file)[1]
    if ".yaml" in warm_up_file or ".yml" in warm_up_file:
        yaml_data = read_yaml(args.warmup_yaml_file)
        task_name = yaml_data["task_name"]  # checkpoints are saved in this dir
        ckpt_path = os.path.join(args.ckpt_dir, task_name, "checkpoints", "last.ckpt")
    elif ".ckpt" in warm_up_file:
        ckpt_path = args.warmup_file
    else:
        raise ValueError("--warmup_file: Expected .yaml or .ckpt file.")

    # Update the ckpt_path in train_from_ckpt.yaml accordingly
    train_ckpt_yaml = read_yaml(os.path.join(args.LIVI_dir, "configs", "train_from_ckpt.yaml"))
    train_ckpt_yaml["ckpt_path"] = ckpt_path
    output_yml = os.path.join(args.LIVI_dir, "configs", "train_from_ckpt.yaml")
    with open(output_yml, "w") as outfile:
        outfile.write("# @package _global_")
        outfile.write("\n\n")
    append_yaml(output_yml, train_ckpt_yaml)
