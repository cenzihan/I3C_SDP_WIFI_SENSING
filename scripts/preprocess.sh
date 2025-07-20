#!/bin/bash

# This script activates the Conda environment and runs the parallel preprocessing script.
# It executes the preprocessing script as a module to ensure correct package imports.

# Exit immediately if a command exits with a non-zero status.
set -e

CONDA_ENV_NAME="sdp-wifi-sensing"
echo ">>> Activating Conda environment: $CONDA_ENV_NAME"

# Find the Conda activation script
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Conda activation script not found."
    exit 1
fi

conda activate "$CONDA_ENV_NAME"

# Verify that the correct python is being used
PYTHON_PATH=$(which python)
echo "--- Using Python from: $PYTHON_PATH ---"

echo ">>> Starting Parallel Data Preprocessing..."

# Run the preprocessing script as a module from the project root
# The `python -m src.preprocess` command tells Python to treat the `src` directory as a package
# and run the `preprocess.py` file as a module. This correctly resolves internal imports.
python -m src.preprocess

echo ">>> Preprocessing finished." 