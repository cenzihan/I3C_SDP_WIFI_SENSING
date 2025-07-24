#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo ">>> Starting Office Scenario Data Preprocessing..."

# --- Activate Conda Environment ---
# Find the conda environment path robustly
CONDA_ENV_PATH=$(conda info --envs | grep 'sdp-wifi-sensing' | awk '{print $NF}')
if [ -z "$CONDA_ENV_PATH" ] || [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Error: Could not find the path for conda environment 'sdp-wifi-sensing'."
    exit 1
fi
# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdp-wifi-sensing
echo ">>> Conda environment 'sdp-wifi-sensing' activated."

# --- Run the Preprocessing Script ---
# Find the python executable in the activated environment
PYTHON_EXEC=$(which python)
echo "--- Using Python from: $PYTHON_EXEC ---"

# Run the script as a module from the project root
$PYTHON_EXEC -u -m src.preprocess_office_glasswall "$@"

echo ">>> Office Scenario preprocessing finished." 