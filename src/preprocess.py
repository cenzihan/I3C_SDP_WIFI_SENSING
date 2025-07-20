import logging
import os
import torch
from tqdm import tqdm
import numpy as np

from .config import get_config
from .dataset import find_file_groups, parse_csi_line
from sklearn.model_selection import train_test_split
import multiprocessing as mp

def process_group(args):
    """
    Processes a single file group. This function is designed to be called by a multiprocessing pool.
    """
    group, config, scene_output_dir = args
    split_name = group['split']
    
    try:
        # Load and immediately binarize the labels
        gt_a = (np.loadtxt(group['room_a_gt'], dtype=np.int64) > 0).astype(np.int64)
        gt_b = (np.loadtxt(group['room_b_gt'], dtype=np.int64) > 0).astype(np.int64)
        gt_lr = (np.loadtxt(group['living_room_gt'], dtype=np.int64) > 0).astype(np.int64)

        with open(group['room_a_data'], 'r') as f_a, open(group['room_b_data'], 'r') as f_b:
            packets_a = [p for p in (parse_csi_line(line) for line in f_a) if p]
            packets_b = [p for p in (parse_csi_line(line) for line in f_b) if p]

        if not packets_a or not packets_b:
            return 0 # Return 0 samples processed

        max_ts_data = max(packets_a[-1][0], packets_b[-1][0]) if packets_a and packets_b else 0
        max_ts_label = max(len(gt_a), len(gt_b), len(gt_lr)) * 2
        duration = max(max_ts_data, float(max_ts_label))

        samples_processed = 0
        for i in range(int(duration // 2)):
            start_time, end_time = i * 2.0, (i + 1) * 2.0
            
            # Packet Assembly for Room A
            padded_a = np.zeros((config['max_packets_per_interval'], 8, 250), dtype=np.float32)
            packet_idx = 0
            for ts, feat in packets_a:
                if start_time <= ts < end_time:
                    if packet_idx < config['max_packets_per_interval']:
                        padded_a[packet_idx] = feat
                        packet_idx += 1
            
            # Packet Assembly for Room B
            padded_b = np.zeros((config['max_packets_per_interval'], 8, 250), dtype=np.float32)
            packet_idx = 0
            for ts, feat in packets_b:
                if start_time <= ts < end_time:
                    if packet_idx < config['max_packets_per_interval']:
                        padded_b[packet_idx] = feat
                        packet_idx += 1

            fused_data = np.concatenate((padded_a, padded_b), axis=1)
            data_tensor = torch.from_numpy(fused_data.transpose(1, 0, 2).copy())

            label_a = gt_a[i] if i < len(gt_a) else 0
            label_b = gt_b[i] if i < len(gt_b) else 0
            label_lr = gt_lr[i] if i < len(gt_lr) else 0
            label_tensor = torch.from_numpy(np.array([label_a, label_b, label_lr], dtype=np.float32))
            
            # Use a unique name for the sample to avoid collisions
            sample_unique_id = f"{group['key']}_{i}"
            save_path = os.path.join(scene_output_dir, split_name, f"sample_{sample_unique_id}.pt")
            torch.save((data_tensor, label_tensor), save_path)
            samples_processed += 1
        
        return samples_processed

    except Exception as e:
        logging.warning(f"Error processing group {group.get('key', 'unknown')}. Error: {e}. Skipping.")
        return 0


def process_and_save_data_parallel(config):
    """
    Finds raw data files for each scene, splits them, and processes them in parallel,
    saving results into scene-specific subdirectories.
    """
    logging.info("--- Starting Parallel Data Preprocessing ---")
    
    base_output_dir = config['preprocessed_data_dir']
    scenes = config['scenes_to_process']
    num_workers = config.get('num_workers', os.cpu_count())

    for scene in scenes:
        logging.info(f"--- Processing Scene: {scene} ---")
        
        scene_output_dir = os.path.join(base_output_dir, scene)
        os.makedirs(os.path.join(scene_output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(scene_output_dir, 'val'), exist_ok=True)

        file_groups = find_file_groups(config['dataset_root'], [scene])
        if not file_groups:
            logging.warning(f"No file groups found for scene '{scene}'. Skipping.")
            continue

        logging.info(f"Found {len(file_groups)} file groups for '{scene}'.")
        
        train_groups, val_groups = train_test_split(
            file_groups, test_size=0.2, random_state=config['seed']
        )
        
        # Add split info to each group for the worker function
        for group in train_groups: group['split'] = 'train'
        for group in val_groups: group['split'] = 'val'
            
        all_groups = train_groups + val_groups
        
        tasks = [(group, config, scene_output_dir) for group in all_groups]

        total_samples = 0
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=len(tasks), desc=f"Processing {scene}") as pbar:
                for samples_processed in pool.imap_unordered(process_group, tasks):
                    total_samples += samples_processed
                    pbar.update()

        logging.info(f"Finished processing scene '{scene}'. Total samples created: {total_samples}")

    logging.info("--- All scenes processed. ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = get_config()
    # Using the parallel version by default now
    process_and_save_data_parallel(config) 