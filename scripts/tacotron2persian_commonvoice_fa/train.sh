#! /bin/bash

# get path to config file
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo $CONFIG_PATH

python -m tac2persian.train --config_path="$CONFIG_PATH"