#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo ">>> Starting WiFi Sensing Transformer Model Training..."

# Get the path to the conda environment robustly, taking the last column
CONDA_ENV_PATH=$(conda info --envs | grep 'sdp-wifi-sensing' | awk '{print $NF}')

if [ -z "$CONDA_ENV_PATH" ] || [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Error: Could not find the path for conda environment 'sdp-wifi-sensing'."
    echo "Please make sure the environment exists by running 'conda env list'."
    exit 1
fi

# Define the python executable from the conda environment
PYTHON_EXEC="$CONDA_ENV_PATH/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at '$PYTHON_EXEC'"
    exit 1
fi

echo "--- Using Python from: $PYTHON_EXEC ---"

# Activate Conda environment
# The path to conda.sh can vary depending on the installation.
# This path is more robust and standard for environments created with --prefix.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdp-wifi-sensing

# Run the training script
# Find the python executable in the conda environment
PYTHON_EXEC=$(conda run -n sdp-wifi-sensing which python)
echo "--- Using Python from: $PYTHON_EXEC ---"
$PYTHON_EXEC -u -m src.train "$@"

echo ">>> Training finished." 