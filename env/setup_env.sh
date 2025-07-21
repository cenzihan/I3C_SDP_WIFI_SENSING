#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the name of the conda environment
ENV_NAME="sdp-wifi-sensing"

echo ">>> Checking if conda environment '$ENV_NAME' already exists..."

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo ">>> Environment '$ENV_NAME' already exists. Updating it from environment.yml..."
    # Update the existing environment. --prune removes packages not in the yml.
    conda env update --name "$ENV_NAME" --file env/environment.yml --prune
    echo ">>> Environment '$ENV_NAME' updated successfully."
else
    echo ">>> Environment '$ENV_NAME' does not exist. Creating..."
    # Create the conda environment from the .yml file
    conda env create -f env/environment.yml
    echo ">>> Environment '$ENV_NAME' created successfully."
fi

echo ""
echo ">>> To activate the environment, run:"
echo ">>> conda activate $ENV_NAME"
echo "" 