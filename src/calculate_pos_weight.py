import os
import torch
import glob
from tqdm import tqdm
import argparse
import sys

# Add project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import get_config

def calculate_weights(data_dirs: list):
    """
    Calculates the pos_weight for BCEWithLogitsLoss by iterating through
    preprocessed training data from one or more directories.

    Args:
        data_dirs (list): A list of directories containing the training .pt files.
    """
    print(f"--- Scanning directories: {data_dirs} ---")
    
    sample_files = []
    for data_dir in data_dirs:
        files = glob.glob(os.path.join(data_dir, "sample_*.pt"))
        if not files:
            print(f"Warning: No preprocessed files found in '{data_dir}'.")
        sample_files.extend(files)

    if not sample_files:
        print(f"Error: No preprocessed files found in any of the specified directories.")
        print("Please run the preprocessing script first.")
        return

    # Assuming 3 classes based on the project description
    num_classes = 3 
    pos_counts = torch.zeros(num_classes)
    neg_counts = torch.zeros(num_classes)

    print(f"Found {len(sample_files)} training samples. Analyzing...")

    for sample_file in tqdm(sample_files, desc="Calculating Class Weights"):
        try:
            _, label = torch.load(sample_file, map_location='cpu')
            if label.shape[0] != num_classes:
                print(f"Warning: Skipping file {sample_file} with unexpected label shape: {label.shape}")
                continue
            pos_counts += label
            neg_counts += (1 - label)
        except Exception as e:
            print(f"Warning: Could not load or process file {sample_file}. Error: {e}")

    total_samples = len(sample_files)
    print("\n--- Calculation Complete ---")
    print(f"Total samples processed: {total_samples}")
    print(f"Positive counts per class: {pos_counts.numpy()}")
    print(f"Negative counts per class: {neg_counts.numpy()}")

    # To avoid division by zero, if a class has zero positive samples,
    # its weight is set to 1.0, assuming no imbalance.
    pos_counts[pos_counts == 0] = 1

    pos_weight = neg_counts / pos_counts

    print("\n--- Results ---")
    print("Calculated 'pos_weight' for your configuration file.")
    print("Please copy this line into 'config.yml' under the 'training' section:")
    print("-" * 50)
    print(f"pos_weight: {pos_weight.numpy().round(4).tolist()}")
    print("-" * 50)


if __name__ == "__main__":
    # Use the config file to find the data directory
    config = get_config()
    base_dir = config['preprocessed_data_dir']
    sources = config['training_data_sources']

    # Handle 'all' keyword to automatically include all subdirectories
    if "all" in sources:
        sources = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Detected 'all' keyword. Using all available data sources: {sources}")

    train_data_paths = [os.path.join(base_dir, source, 'train') for source in sources]
    
    calculate_weights(train_data_paths) 