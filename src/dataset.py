import logging
import os
import re
from typing import List, Dict, Tuple, Optional
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from collections import defaultdict

# A global flag to ensure the debug print happens only once
_HAS_PARSED_NON_ZERO = False

def parse_csi_line(line: str) -> Optional[Tuple[float, np.ndarray]]:
    """
    Parses a single line of CSI data. It takes the dB value of the absolute
    real and imaginary parts separately and scales them to a [0, 1] range.
    The final feature map has a shape of (8, 250).
    """
    CSI_LEN_80M = 1000
    CSI_LEN_160M = 992
    
    parts = line.strip().split()
    if len(parts) < 8:
        return None

    try:
        h, m, s_part = parts[0], parts[1], parts[2]
        total_seconds = float(h) * 3600.0 + float(m) * 60.0 + float(s_part)

        csi_features_str = parts[8:]
        
        if len(csi_features_str) == CSI_LEN_160M:
            csi_features_str.extend(['0+0j'] * (CSI_LEN_80M - CSI_LEN_160M))
        
        if len(csi_features_str) != CSI_LEN_80M:
            return None

        csi_complex = np.array([np.complex128(s.replace('j', 'J')) for s in csi_features_str])
        
        streams = csi_complex.reshape(4, 250)
        
        real_parts = np.real(streams)
        imag_parts = np.imag(streams)

        # --- dB Conversion and Scaling for both parts ---
        MIN_DB = -120  # A more sensitive min for components
        MAX_DB = -40   # A more sensitive max for components

        # Process real part
        abs_real = np.abs(real_parts)
        db_real = 20 * np.log10(abs_real + 1e-12) # Add larger epsilon
        scaled_real = np.clip((db_real - MIN_DB) / (MAX_DB - MIN_DB), 0, 1)

        # Process imaginary part
        abs_imag = np.abs(imag_parts)
        db_imag = 20 * np.log10(abs_imag + 1e-12)
        scaled_imag = np.clip((db_imag - MIN_DB) / (MAX_DB - MIN_DB), 0, 1)
        
        feature_map = np.vstack((scaled_real, scaled_imag)).astype(np.float32)
        
        return total_seconds, feature_map

    except (ValueError, IndexError):
        return None


class PreprocessedCSIDataset(Dataset):
    """
    A lightweight dataset that loads preprocessed tensor files (.pt) from one or more directories.
    """
    def __init__(self, data_dirs: List[str]):
        self.sample_files = []
        
        # If data_dirs is not provided, we assume manual population of sample_files later.
        if not data_dirs:
            return # Just create an empty dataset object
        
        for data_dir in data_dirs:
            files = sorted(glob.glob(os.path.join(data_dir, "sample_*.pt")))
            if not files:
                logging.warning(f"No preprocessed .pt files found in {data_dir}. Skipping.")
            self.sample_files.extend(files)

        if not self.sample_files:
            raise FileNotFoundError(f"No preprocessed .pt files found in any of the specified directories: {data_dirs}. "
                                    f"Please run the preprocessing script first.")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        return torch.load(self.sample_files[idx])


def find_file_groups(data_dir: str, scenes: List[str]) -> List[Dict[str, str]]:
    """
    Scans directories using os.walk to find paired data and ground truth files.
    This approach is more robust to variations in directory depth.
    """
    file_groups = []
    
    for scene_name in scenes:
        scene_path = os.path.join(data_dir, scene_name)
        if not os.path.isdir(scene_path):
            logging.warning(f"Scene directory not found: {scene_path}")
            continue

        logging.info(f"Scanning for file groups in: {scene_path}")
        
        for dirpath, dirnames, filenames in os.walk(scene_path):
            if 'RoomAData' in dirnames:
                room_a_data_dir = os.path.join(dirpath, 'RoomAData')
                
                # Based on RoomAData, try to find its siblings and parent folders
                # The structure is assumed to be .../experiment_name/RoomAData
                # and .../experiment_name_RoomB/RoomBData
                
                experiment_dir_a = os.path.dirname(room_a_data_dir) # e.g., .../2to2_Scene1_GreenPlantFan_1
                room_a_base_dir = os.path.dirname(experiment_dir_a) # e.g., .../Scene1_RoomA
                
                if not room_a_base_dir.endswith("_RoomA"):
                    continue
                
                # Construct path for RoomB
                scene_base_name = os.path.basename(room_a_base_dir).replace("_RoomA", "")
                room_b_base_dir = os.path.join(os.path.dirname(room_a_base_dir), f"{scene_base_name}_RoomB")
                experiment_dir_b = os.path.join(room_b_base_dir, os.path.basename(experiment_dir_a))
                
                room_b_data_dir = os.path.join(experiment_dir_b, 'RoomBData')
                gt_a_dir = os.path.join(experiment_dir_a, 'RoomAGroundtruth')
                gt_b_dir = os.path.join(experiment_dir_b, 'RoomBGroundtruth')
                gt_lr_dir = os.path.join(experiment_dir_a, 'LivingRoomGroundtruth') # LR truth is with RoomA

                if not all(os.path.isdir(d) for d in [room_b_data_dir, gt_a_dir, gt_b_dir, gt_lr_dir]):
                    continue
                
                # Now we have the directories, let's match the files
                for data_file_a_name in os.listdir(room_a_data_dir):
                    match = re.match(r'(\d{4})_(\d{2})\.txt', data_file_a_name)
                    if not match: continue
                    
                    loc_code_str, rep_code = match.groups()
                    loc_code = int(loc_code_str)

                    # Determine the file naming rule based on the first two digits
                    if 1100 <= loc_code < 1200: # Home_Scene1 logic (11xx)
                        interference_code = loc_code % 10
                        room_b_loc_code = loc_code - interference_code + 10 + interference_code
                        living_room_loc_code = loc_code - interference_code + 20 + interference_code
                    elif 1200 <= loc_code < 1300: # Home_Scene2 logic (12xx)
                        interference_code = loc_code % 10
                        room_b_loc_code = loc_code - interference_code + 10 + interference_code
                        living_room_loc_code = loc_code - interference_code + 20 + interference_code
                    else:
                        continue # Skip if the code is not in a known range

                    # Construct corresponding filenames
                    room_b_data_name = f"{room_b_loc_code:04d}_{rep_code}.txt"
                    room_a_gt_name = f"groundtruth_{data_file_a_name}"
                    room_b_gt_name = f"groundtruth_{room_b_data_name}"
                    living_room_gt_name = f"groundtruth_{living_room_loc_code:04d}_{rep_code}.txt"

                    # Construct full paths
                    path_a_data = os.path.join(room_a_data_dir, data_file_a_name)
                    path_b_data = os.path.join(room_b_data_dir, room_b_data_name)
                    path_a_gt = os.path.join(gt_a_dir, room_a_gt_name)
                    path_b_gt = os.path.join(gt_b_dir, room_b_gt_name)
                    path_lr_gt = os.path.join(gt_lr_dir, living_room_gt_name)
                    
                    all_paths = [path_a_data, path_b_data, path_a_gt, path_b_gt, path_lr_gt]
                    if all(os.path.exists(p) for p in all_paths):
                        key = f"{scene_name}_{os.path.basename(experiment_dir_a)}_{data_file_a_name}"
                        file_groups.append({
                            "key": key, "room_a_data": path_a_data, "room_b_data": path_b_data,
                            "room_a_gt": path_a_gt, "room_b_gt": path_b_gt, "living_room_gt": path_lr_gt,
                        })
    return file_groups


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation dataloaders.
    Supports two strategies defined in config['data_split']['strategy']:
    1. 'preprocess': Loads from pre-split 'train' and 'val' directories.
    2. 'on_the_fly': Loads all data and performs a random split.
    """
    base_dir = config['preprocessed_data_dir']
    sources = config['training_data_sources']
    strategy = config.get('data_split', {}).get('strategy', 'preprocess')
    batch_size = config['training']['batch_size']
    num_workers = config['num_workers']

    if "all" in sources:
        sources = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        logging.info(f"Detected 'all' keyword. Using all available data sources: {sources}")

    if strategy == 'preprocess':
        logging.info("Using 'preprocess' split strategy. Loading from pre-split train/val directories.")
        train_dirs = [os.path.join(base_dir, source, 'train') for source in sources]
        val_dirs = [os.path.join(base_dir, source, 'val') for source in sources]

        train_dataset = PreprocessedCSIDataset(train_dirs)
        val_dataset = PreprocessedCSIDataset(val_dirs)
    
    elif strategy == 'on_the_fly':
        logging.info("Using 'on_the_fly' split strategy. Loading all data for a new random split.")
        # Load data from both train and val directories for a full dataset
        all_data_dirs = []
        for source in sources:
            all_data_dirs.append(os.path.join(base_dir, source, 'train'))
            all_data_dirs.append(os.path.join(base_dir, source, 'val'))

        full_dataset = PreprocessedCSIDataset(all_data_dirs)
        
        # Perform the random split
        val_split = config['data_split']['val_size']
        train_size = int((1.0 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        generator = torch.Generator().manual_seed(config['seed'])
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
    elif strategy == 'group_level_random_split':
        logging.info("Using 'group_level_random_split' strategy. Splitting within each file group.")
        
        # 1. Load all files from the specified sources
        all_files = []
        for source in sources:
            source_dir = os.path.join(base_dir, source)
            for split in ['train', 'val']:
                split_dir = os.path.join(source_dir, split)
                if os.path.isdir(split_dir):
                    all_files.extend([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.pt')])

        if not all_files:
            raise ValueError("No preprocessed .pt files found for the specified sources.")

        # 2. Group files by their original file group key
        file_groups = defaultdict(list)
        for f_path in all_files:
            # Filename format: sample_{key}_{index}.pt
            # We want to extract the {key} part.
            filename = os.path.basename(f_path)
            parts = filename.replace('sample_', '').replace('.pt', '').split('_')
            group_key = '_'.join(parts[:-1]) # Everything except the last part (the index)
            file_groups[group_key].append(f_path)

        # 3. Split each group and collect subsets
        train_subsets, val_subsets = [], []
        val_split = config['data_split']['val_size']
        generator = torch.Generator().manual_seed(config['seed'])
        
        for group_key, files_in_group in file_groups.items():
            if len(files_in_group) < 2: continue # Skip groups with too few samples to split
            
            # Create a dataset for this specific group
            group_dataset = PreprocessedCSIDataset([]) # Create an empty dataset
            group_dataset.sample_files = sorted(files_in_group) # Manually assign its files

            train_size = int((1.0 - val_split) * len(group_dataset))
            val_size = len(group_dataset) - train_size
            
            # Ensure we can actually split it
            if train_size == 0 or val_size == 0:
                logging.warning(f"Group '{group_key}' with {len(files_in_group)} samples is too small to be split. Adding all to training set.")
                train_subsets.append(group_dataset)
                continue

            group_train_subset, group_val_subset = torch.utils.data.random_split(
                group_dataset, [train_size, val_size], generator=generator
            )
            train_subsets.append(group_train_subset)
            val_subsets.append(group_val_subset)

        if not train_subsets:
            raise ValueError("No data groups were large enough to be split.")

        train_dataset = ConcatDataset(train_subsets)
        val_dataset = ConcatDataset(val_subsets)
    else:
        raise ValueError(f"Unknown data split strategy: '{strategy}'")

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