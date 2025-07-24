#!/usr/bin/env python

import os
import sys
import glob
import logging
import random
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_data_for_analysis(preprocessed_dir, scene, num_samples=1000):
    """
    Loads a subset of the preprocessed data for feature analysis.
    """
    logging.info(f"Loading data from scene: {scene}...")
    
    # We are interested in the training data for feature analysis
    data_path = os.path.join(preprocessed_dir, scene, 'train')
    
    if not os.path.isdir(data_path):
        logging.error(f"Directory not found: {data_path}. Please run preprocessing first.")
        return None, None
        
    all_files = glob.glob(os.path.join(data_path, "*.pt"))
    if not all_files:
        logging.error(f"No .pt files found in {data_path}.")
        return None, None
        
    # Randomly sample a subset of files to speed up analysis
    sample_files = random.sample(all_files, min(num_samples, len(all_files)))
    
    all_tensors_a = []
    all_tensors_b = []
    all_labels = []
    
    for file_path in tqdm(sample_files, desc=f"Loading {len(sample_files)} samples"):
        try:
            (tensor_a, tensor_b), label_tensor = torch.load(file_path)
            
            # Flatten the tensors for sklearn compatibility
            all_tensors_a.append(tensor_a.numpy().flatten())
            all_tensors_b.append(tensor_b.numpy().flatten())
            all_labels.append(label_tensor.numpy())
        except Exception as e:
            logging.warning(f"Skipping file {os.path.basename(file_path)} due to error: {e}")
            
    if not all_tensors_a:
        logging.error("Could not load any valid data.")
        return None, None

    return np.array(all_tensors_a), np.array(all_tensors_b), np.array(all_labels)


def analyze_and_print_weights(data_a, data_b, labels):
    """
    Calculates and prints the initial weights using mutual information.
    """
    # The labels are structured as [Room A, Room B, Living Room]
    labels_a = labels[:, 0]
    labels_b = labels[:, 1]
    labels_lr = labels[:, 2]
    
    tasks = {
        "Predict Room A": labels_a,
        "Predict Room B": labels_b,
        "Predict Living Room": labels_lr
    }
    
    initial_weights = {}

    for task_name, task_labels in tasks.items():
        logging.info(f"--- Analyzing features for task: {task_name} ---")
        
        # Calculate Mutual Information for tensor A
        logging.info("Calculating MI for Room A data...")
        mi_a = mutual_info_classif(data_a, task_labels, discrete_features=False)
        avg_mi_a = np.mean(mi_a)
        
        # Calculate Mutual Information for tensor B
        logging.info("Calculating MI for Room B data...")
        mi_b = mutual_info_classif(data_b, task_labels, discrete_features=False)
        avg_mi_b = np.mean(mi_b)
        
        # Normalize the MI scores to get weights that sum to 1
        total_mi = avg_mi_a + avg_mi_b
        if total_mi > 0:
            weight_a = avg_mi_a / total_mi
            weight_b = avg_mi_b / total_mi
        else:
            weight_a = 0.5
            weight_b = 0.5 # Default to equal weights if no information is found
            
        initial_weights[task_name] = {'weight_a': weight_a, 'weight_b': weight_b}

        print(f"\nResults for: {task_name}")
        print(f"  - Avg. MI(Room A Data, Label): {avg_mi_a:.4f}")
        print(f"  - Avg. MI(Room B Data, Label): {avg_mi_b:.4f}")
        print(f"  - Normalized Initial Weight for Room A input: {weight_a:.4f}")
        print(f"  - Normalized Initial Weight for Room B input: {weight_b:.4f}")
        
    return initial_weights

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # We'll analyze Home_Scene1 data as it's representative
    # Ensure you have run preprocess.py with the dual-tensor output format first
    preprocessed_dir = "datasets/predata/"
    scene_to_analyze = "Home_Scene1"
    
    data_a, data_b, labels = load_data_for_analysis(preprocessed_dir, scene_to_analyze)
    
    if data_a is not None:
        analyze_and_print_weights(data_a, data_b, labels)
    
if __name__ == "__main__":
    main() 