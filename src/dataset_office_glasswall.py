import logging
import os
from typing import Dict, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, ConcatDataset

# Add project root to sys.path to allow imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import PreprocessedCSIDataset

def create_office_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation dataloaders for the Office Scenario.
    Supports 'preprocess', 'on_the_fly', and 'group_level_random_split' strategies.
    """
    base_dir = config['preprocessed_data_dir']
    strategy = config['data_split']['strategy']
    batch_size = config['training']['batch_size']
    num_workers = config.get('num_workers', 4)

    logging.info(f"--- Loading Office Scenario Data (Strategy: {strategy}) ---")
    
    # In this new setup, all strategies will rely on the pre-split train/val dirs
    # created by the preprocess_office script.
    
    if strategy == 'preprocess':
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'val')
        
        train_dataset = PreprocessedCSIDataset([train_dir])
        val_dataset = PreprocessedCSIDataset([val_dir])

    elif strategy == 'on_the_fly':
        all_dirs = [os.path.join(base_dir, d) for d in ['train', 'val']]
        full_dataset = PreprocessedCSIDataset(all_dirs)
        
        val_split = config['data_split']['val_size']
        train_size = int((1.0 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        generator = torch.Generator().manual_seed(config['seed'])
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    elif strategy == 'group_level_random_split':
        all_dirs = [os.path.join(base_dir, d) for d in ['train', 'val']]
        all_files_dataset = PreprocessedCSIDataset(all_dirs)
        all_files = all_files_dataset.sample_files
        
        file_groups = defaultdict(list)
        for f_path in all_files:
            filename = os.path.basename(f_path)
            parts = filename.replace('sample_', '').replace('.pt', '').split('_')
            group_key = '_'.join(parts[:-1])
            file_groups[group_key].append(f_path)
            
        train_subsets, val_subsets = [], []
        val_split = config['data_split']['val_size']
        generator = torch.Generator().manual_seed(config['seed'])
        
        for group_key, files_in_group in file_groups.items():
            if len(files_in_group) < 2: continue
            
            group_dataset = PreprocessedCSIDataset([])
            group_dataset.sample_files = sorted(files_in_group)
            
            train_size = int((1.0 - val_split) * len(group_dataset))
            val_size = len(group_dataset) - train_size

            if train_size == 0 or val_size == 0:
                train_subsets.append(group_dataset)
                continue
                
            group_train_subset, group_val_subset = torch.utils.data.random_split(
                group_dataset, [train_size, val_size], generator=generator
            )
            train_subsets.append(group_train_subset)
            val_subsets.append(group_val_subset)

        train_dataset = ConcatDataset(train_subsets)
        val_dataset = ConcatDataset(val_subsets)
        
    else:
        raise ValueError(f"Unknown data split strategy for office scenario: '{strategy}'")

    logging.info(f"Total training samples: {len(train_dataset)}")
    logging.info(f"Total validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader 