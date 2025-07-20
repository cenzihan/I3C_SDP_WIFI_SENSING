import os
import torch
import glob
from tqdm import tqdm
import argparse
import sys
from torch.utils.data import DataLoader

# Add project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import get_config
from src.dataset import PreprocessedCSIDataset

def calculate_weights(config: dict):
    """
    Calculates the pos_weight for BCEWithLogitsLoss by iterating through
    the training dataset partition, according to the configured split strategy.
    """
    base_dir = config['preprocessed_data_dir']
    sources = config['training_data_sources']
    strategy = config.get('data_split', {}).get('strategy', 'preprocess')

    if "all" in sources:
        sources = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Detected 'all' keyword. Using all available data sources: {sources}")
    
    # --- Determine the training dataset based on the split strategy ---
    train_dataset = None
    if strategy == 'preprocess':
        print("Strategy 'preprocess': Calculating weights from pre-split 'train' directories.")
        train_dirs = [os.path.join(base_dir, source, 'train') for source in sources]
        train_dataset = PreprocessedCSIDataset(train_dirs)
    
    elif strategy == 'on_the_fly':
        print("Strategy 'on_the_fly': Calculating weights from the training portion of a new random split.")
        all_data_dirs = []
        for source in sources:
            all_data_dirs.append(os.path.join(base_dir, source, 'train'))
            all_data_dirs.append(os.path.join(base_dir, source, 'val'))
            
        full_dataset = PreprocessedCSIDataset(all_data_dirs)
        
        val_split = config['data_split']['val_size']
        train_size = int((1.0 - val_split) * len(full_dataset))
        
        generator = torch.Generator().manual_seed(config['seed'])
        train_subset, _ = torch.utils.data.random_split(
            full_dataset, [train_size, len(full_dataset) - train_size], generator=generator
        )
        train_dataset = train_subset
    else:
        raise ValueError(f"Unknown data split strategy: '{strategy}'")

    if not train_dataset or len(train_dataset) == 0:
        print("Error: Training dataset is empty. Cannot calculate weights.")
        return
        
    print(f"Analyzing {len(train_dataset)} training samples to calculate pos_weight...")

    num_classes = 3 
    pos_counts = torch.zeros(num_classes)
    neg_counts = torch.zeros(num_classes)

    # Use a DataLoader to iterate through the dataset efficiently
    data_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['num_workers'])

    for _, labels in tqdm(data_loader, desc="Calculating Class Weights"):
        # Ensure labels are float for sum, even if they are loaded as something else
        labels = labels.float()
        pos_counts += labels.sum(axis=0)
        neg_counts += (labels.shape[0] - labels.sum(axis=0))

    print("\n--- Calculation Complete ---")
    print(f"Total samples processed for weight calculation: {len(train_dataset)}")
    print(f"Positive counts per class: {pos_counts.numpy()}")
    print(f"Negative counts per class: {neg_counts.numpy()}")

    # Avoid division by zero
    pos_counts[pos_counts == 0] = 1
    pos_weight = neg_counts / pos_counts

    print("\n--- Results ---")
    print("Calculated 'pos_weight' for your configuration file.")
    print("Please copy this line into 'config.yml' under the 'training' section:")
    print("-" * 50)
    print(f"pos_weight: {pos_weight.numpy().round(4).tolist()}")
    print("-" * 50)


def main():
    config = get_config()
    calculate_weights(config)

if __name__ == "__main__":
    main() 