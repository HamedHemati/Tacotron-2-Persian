#! /bin/bash

# Get full path to the config file automatically
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo $CONFIG_PATH


DATASET_PATH="PATH TO DATASET"
OUTPUT_PATH="PATH TO THE OUTPUT DIRECTORY"
NUM_WORKERS=5

python -m tac2persian.data_preprocessing.preprocess_commonvoice_fa --dataset_path="$DATASET_PATH" \
                                                                   --output_path="$OUTPUT_PATH" \
                                                                   --config_path="$CONFIG_PATH" \
                                                                   --num_workers="$NUM_WORKERS"