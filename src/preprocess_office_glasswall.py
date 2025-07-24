#!/usr/bin/env python

import os
import re
import glob
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to sys.path to allow imports when run as a script
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import get_config

# --- Constants ---
CSI_COLUMN_COUNT = 256  # 4 * 64
TOTAL_COLUMN_COUNT = 268 # 3 (time) + 2 (rssi) + 1 (mcs) + 2 (gain) + 256 (csi) = 264. Let's stick to 268 as described.
SECONDS_PER_LABEL = 1.0
PACKETS_PER_WINDOW = 100
SECONDS_PER_FILE = 121

def find_office_file_groups(data_root):
    """
    Finds groups of 5 data files and their corresponding 5 label files for the office scenario.
    Only considers 20MHz configurations.
    """
    logging.info("--- Scanning for Office Scenario file groups (20MHz only) ---")
    file_groups = []
    
    data_base_path = os.path.join(data_root, "Data")
    label_base_path = os.path.join(data_root, "Groundtruth")

    # Find all potential experiment directories
    for mount_type in os.listdir(data_base_path): # ceiling_mount, wall_mount
        mount_path = os.path.join(data_base_path, mount_type)
        if not os.path.isdir(mount_path): continue

        for config_name in os.listdir(mount_path): # 20MHzConfig1, 20MHzConfig1fixgain, etc.
            if "20MHz" not in config_name:
                continue # Skip non-20MHz folders
            
            config_path = os.path.join(mount_path, config_name)
            if not os.path.isdir(config_path): continue

            for condition_name in os.listdir(config_path): # Fan, No_Interference
                condition_path = os.path.join(config_path, condition_name)
                data_dir = os.path.join(condition_path, "Processed_Data")

                if os.path.isdir(data_dir):
                    # Check if we have 1m.txt to 5m.txt
                    data_files = [os.path.join(data_dir, f"{i}m.txt") for i in range(1, 6)]
                    if not all(os.path.exists(p) for p in data_files):
                        continue

                    # Construct corresponding label paths
                    label_dir = os.path.join(label_base_path, mount_type, config_name, condition_name)
                    # Note: Label path does not have "Processed_Data" subdir
                    label_files = [os.path.join(label_dir, f"{i}m.txt") for i in range(1, 6)]
                    if not all(os.path.exists(p) for p in label_files):
                        logging.warning(f"Found data files in {data_dir}, but missing corresponding label files in {label_dir}. Skipping group.")
                        continue
                    
                    group_key = f"office_{mount_type}_{config_name}_{condition_name}"
                    file_groups.append({
                        "key": group_key,
                        "data_files": data_files,
                        "label_files": label_files
                    })
    
    logging.info(f"Found {len(file_groups)} file groups to process.")
    return file_groups

def parse_and_process_csi(csi_data_str):
    """
    Parses a string of 256 CSI values, converts them to complex numbers,
    separates real/imag parts, and normalizes them into a dB-scaled tensor.
    Returns a tensor of shape (8, 64).
    """
    # Assuming csi_data_str is a list of strings
    csi_complex = np.array([complex(s.replace('i', 'j')) for s in csi_data_str])
    
    streams = csi_complex.reshape(4, 64)
    
    real_parts = np.real(streams)
    imag_parts = np.imag(streams)

    # dB Conversion and Scaling
    # --- MODIFICATION: Adjusted the dB range based on data analysis ---
    MIN_DB = -95
    MAX_DB = 5

    def to_db_and_scale(arr):
        # Prevent log(0)
        arr[arr == 0] = 1e-12
        db_arr = 20 * np.log10(np.abs(arr))
        return np.clip((db_arr - MIN_DB) / (MAX_DB - MIN_DB), 0, 1)

    scaled_real = to_db_and_scale(real_parts)
    scaled_imag = to_db_and_scale(imag_parts)
    
    # Shape (8, 64): 4 real streams stacked on top of 4 imag streams
    feature_map = np.vstack((scaled_real, scaled_imag)).astype(np.float32)
    return feature_map


def process_file_group(group, output_base_dir, split_name):
    """
    Processes one group of 5 data files and 5 label files.
    Generates 121 samples of shape (40, 100, 64) and saves them.
    """
    try:
        # 1. Load all data and labels for the group
        all_data_packets = []
        for data_file in group['data_files']:
            # Read and filter lines that have the correct number of columns
            with open(data_file, 'r') as f:
                packets = [line.strip().split() for line in f if len(line.strip().split()) == TOTAL_COLUMN_COUNT]
                all_data_packets.append(packets)

        all_labels = []
        for label_file in group['label_files']:
            with open(label_file, 'r') as f:
                labels = f.read().strip().split()
                all_labels.append(labels)
        
        # 2. Process each of the 121 time windows
        samples_created = 0
        for t in range(SECONDS_PER_FILE):
            # For each 1-second window (t)
            
            # 2a. Gather packets and labels for this window
            window_packets_per_file = []
            for file_packets in all_data_packets:
                current_file_window = []
                for pkt in file_packets:
                    # pkt[2] is the seconds part of the timestamp
                    if int(float(pkt[2])) == t:
                        current_file_window.append(pkt)
                window_packets_per_file.append(current_file_window)

            window_labels = [labels[t] for labels in all_labels if t < len(labels)]
            
            # We need a label from each of the 5 files
            if len(window_labels) != 5:
                continue

            # 2b. Process the 5 files' data for this window
            processed_files_in_window = []
            for file_packets in window_packets_per_file:
                # Pad or truncate to PACKETS_PER_WINDOW
                padded_packets = np.zeros((PACKETS_PER_WINDOW, CSI_COLUMN_COUNT), dtype=np.float32)
                
                # Take the CSI part (last 256 columns)
                csi_part = [p[-CSI_COLUMN_COUNT:] for p in file_packets]
                
                num_to_copy = min(len(csi_part), PACKETS_PER_WINDOW)
                if num_to_copy > 0:
                    csi_block = np.array(csi_part[:num_to_copy], dtype=str)
                    
                    # Process each packet in the block
                    processed_block = np.array([parse_and_process_csi(row) for row in csi_block])
                    # After processing, shape is (num_to_copy, 8, 64)
                    
                    # We need to reshape this back for padding, which is complex.
                    # Let's simplify: pad the processed data instead.
                    
                # Let's re-think the processing pipeline for efficiency.
                # It's better to pad first, then process.
                
                # Simplified pipeline:
                # 1. Get raw CSI strings for the window
                raw_csi_in_window = [p[-CSI_COLUMN_COUNT:] for p in file_packets]
                
                # 2. Pad/truncate the list of raw CSI strings
                padded_raw_csi = raw_csi_in_window[:PACKETS_PER_WINDOW]
                while len(padded_raw_csi) < PACKETS_PER_WINDOW:
                    padded_raw_csi.append(['0+0j'] * CSI_COLUMN_COUNT) # Pad with zero-valued strings
                
                # 3. Process the padded list
                processed_window = np.array([parse_and_process_csi(row) for row in padded_raw_csi])
                # Shape is now (100, 8, 64)
                processed_files_in_window.append(processed_window)

            # 2c. Concatenate the 5 processed files to get the final sample
            # List of 5 arrays, each of shape (100, 8, 64)
            # We want to stack along the '8' dimension to get (40, 100, 64)
            final_sample_x = np.concatenate([p.transpose(1, 0, 2) for p in processed_files_in_window], axis=0)
            
            # 2d. Create label tensor
            final_sample_y = np.array(window_labels, dtype=np.float32)

            # 2e. Save the sample
            output_dir = os.path.join(output_base_dir, split_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Use a unique name for the sample
            sample_unique_id = f"{group['key']}_{t}"
            save_path = os.path.join(output_dir, f"sample_{sample_unique_id}.pt")
            torch.save((torch.from_numpy(final_sample_x), torch.from_numpy(final_sample_y)), save_path)
            samples_created += 1
            
        return samples_created, group['key']
    except Exception as e:
        logging.error(f"Error processing group {group.get('key', 'unknown')}. Error: {e}", exc_info=True)
        return 0, group.get('key', 'unknown')


def main():
    """Main function to run the preprocessing."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Use a fixed config for this specific preprocessing script
    output_dir = "datasets/predata/Office_Glass_Wall"
    dataset_root = "datasets/OfficeScenario/Glass_Wall_Scenario"
    
    file_groups = find_office_file_groups(dataset_root)
    
    if not file_groups:
        logging.error("No file groups found. Exiting.")
        return

    # --- Use config for splitting strategy ---
    config = get_config()
    strategy = config.get('data_split', {}).get('strategy', 'preprocess')
    
    if strategy != 'preprocess':
        logging.warning(f"This script is designed for 'preprocess' split strategy, but found '{strategy}'. "
                        f"It will perform a file-group-level split and save to train/val directories.")
    
    val_size = config['data_split'].get('val_size', 0.2)
    seed = config.get('seed', 42)
    
    train_groups, val_groups = train_test_split(
        file_groups, test_size=val_size, random_state=seed
    )
    
    # Assign split info to each group for the worker function
    for group in train_groups: group['split'] = 'train'
    for group in val_groups: group['split'] = 'val'
    all_groups_with_split = train_groups + val_groups

    # Use ProcessPoolExecutor for parallel processing
    num_workers = os.cpu_count()
    total_samples = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file_group, group, output_dir, group['split']) for group in all_groups_with_split]
        
        with tqdm(total=len(all_groups_with_split), desc="Processing File Groups") as pbar:
            for future in as_completed(futures):
                samples_processed, group_key = future.result()
                if samples_processed > 0:
                    total_samples += samples_processed
                else:
                    logging.warning(f"Group {group_key} produced 0 samples.")
                pbar.update(1)

    logging.info(f"--- Preprocessing Finished ---")
    logging.info(f"Total samples created: {total_samples}")
    logging.info(f"Preprocessed data saved in: {output_dir}")


if __name__ == "__main__":
    main() 