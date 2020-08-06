import yaml


def load_config(config_file_path):
    """Load config file as a dictionary."""
    with open(config_file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config