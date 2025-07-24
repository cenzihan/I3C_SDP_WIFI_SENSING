#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo ">>> Starting Office Scenario (Glass Wall) Model Training..."

# --- Activate Conda Environment ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdp-wifi-sensing
echo ">>> Conda environment 'sdp-wifi-sensing' activated."

# --- Find Python Executable ---
PYTHON_EXEC=$(which python)
echo "--- Using Python from: $PYTHON_EXEC ---"

# --- Run the Training Script ---
# We explicitly point to the office training script and its config.
# The --config_path argument ensures it loads the correct config,
# overriding any default behavior in the python script.
$PYTHON_EXEC -u -m src.train_office_glasswall --config_path config_office_glasswall.yml "$@"

echo ">>> Office Scenario training finished." 