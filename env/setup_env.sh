#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the name of the conda environment
ENV_NAME="sdp-wifi-sensing"

echo ">>> Checking if conda environment '$ENV_NAME' already exists..."

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo ">>> Environment '$ENV_NAME' already exists. Skipping creation."
else
    echo ">>> Environment '$ENV_NAME' does not exist. Creating..."
    # Create the conda environment from the .yml file
    conda env create -f environment.yml
    echo ">>> Environment '$ENV_NAME' created successfully."
fi

echo ""
echo ">>> To activate the environment, run:"
echo ">>> conda activate $ENV_NAME"
echo "" 