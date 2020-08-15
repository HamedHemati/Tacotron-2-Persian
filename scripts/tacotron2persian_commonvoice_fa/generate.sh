#! /bin/bash

# get path to config file
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo $CONFIG_PATH


CHECKPOINT_ID="100"

SPEAKER=""
LANGUAGE="fa"

INP_TEXT="دلا نزدِ کسی بنشین که او از دل خبر دارد."


python -m TTS.acoustic_model.tacotron2.generate --config_path="$CONFIG_PATH" \
                                                --checkpoint_id="$CHECKPOINT_ID" \
                                                --speaker="$SPEAKER" \
                                                --language="$LANGUAGE" \
                                                --text="$TEXT"
