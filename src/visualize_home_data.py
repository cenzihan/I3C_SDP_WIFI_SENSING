#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import random
import re

# Add project root to sys.path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Using the original find_file_groups from the home dataset processing
from src.dataset import find_file_groups

def parse_data_file(file_path):
    """
    Parses a raw data file to extract timestamps and RSSI.
    """
    timestamps = []
    rssi_values = []

    if not os.path.exists(file_path):
        logging.warning(f"Data file not found: {file_path}. Skipping.")
        return np.array([]), np.array([])

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 1008:
                continue
            
            try:
                h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                time = h * 3600 + m * 60 + s
                timestamps.append(time)
                rssi = float(parts[3])
                rssi_values.append(rssi)
            except (ValueError, IndexError):
                continue

    return np.array(timestamps), np.array(rssi_values)

def parse_gt_file(file_path):
    """
    Parses a groundtruth file, loading 0, 1, and 2 as they are.
    """
    if not os.path.exists(file_path):
        logging.warning(f"Ground truth file not found: {file_path}. Skipping.")
        return np.array([]), np.array([])

    with open(file_path, 'r') as f:
        labels_str = f.read().strip().split()
        if not labels_str:
            return np.array([],), np.array([])
        # Load labels as integers, preserving 0, 1, 2
        labels = np.array(labels_str, dtype=int)
    
    gt_time = np.arange(len(labels) + 1) * 2.0
    return gt_time, np.append(labels, labels[-1])

def create_visualization(group, output_dir):
    """
    Creates and saves a single visualization for a given file group.
    """
    logging.info(f"Visualizing data for group: {group['key']}")

    ts_a, rssi_a = parse_data_file(group['room_a_data'])
    ts_b, rssi_b = parse_data_file(group['room_b_data'])
    gt_time_a, labels_a = parse_gt_file(group['room_a_gt'])
    gt_time_b, labels_b = parse_gt_file(group['room_b_gt'])
    gt_time_lr, labels_lr = parse_gt_file(group['living_room_gt'])

    if ts_a.size == 0 and ts_b.size == 0:
        logging.warning(f"No data to plot for group {group['key']}. Skipping.")
        return

    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.6, 1]) 
    
    ax1 = plt.subplot(gs[0])
    if ts_a.size > 0: ax1.plot(ts_a, rssi_a, label='Room A RSSI', color='deepskyblue', alpha=0.9)
    if ts_b.size > 0: ax1.plot(ts_b, rssi_b, label='Room B RSSI', color='tomato', alpha=0.9)
    ax1.set_title(f'Raw RSSI Variation - {group["key"]}', fontsize=16)
    ax1.set_ylabel('RSSI (Raw Value)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xticklabels([])

    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.set_title('Ground Truth: Person Count (Gantt-style)', fontsize=16)
    ax2.set_ylabel('Location')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Living Room', 'Room B', 'Room A'])
    ax2.set_ylim(-0.5, 2.5)
    
    def plot_gantt_chart(ax, gt_time, labels, y_pos, colors, labels_legend):
        if gt_time.size == 0: return
        # Create bars for each state (1 and 2)
        for state, color, label_text in zip([1, 2], colors, labels_legend):
            intervals = []
            start_time = None
            for i in range(len(labels)):
                if labels[i] == state and start_time is None:
                    start_time = gt_time[i]
                elif labels[i] != state and start_time is not None:
                    intervals.append((start_time, gt_time[i] - start_time))
                    start_time = None
            if start_time is not None:
                 intervals.append((start_time, gt_time[-1] - start_time))
            if intervals:
                ax.broken_barh(intervals, (y_pos - 0.4, 0.8), facecolors=color, label=label_text)

    plot_gantt_chart(ax2, gt_time_a, labels_a, 2, ['mediumseagreen', 'darkgreen'], ['Room A (1p)', 'Room A (2p)'])
    plot_gantt_chart(ax2, gt_time_b, labels_b, 1, ['mediumpurple', 'darkviolet'], ['Room B (1p)', 'Room B (2p)'])
    plot_gantt_chart(ax2, gt_time_lr, labels_lr, 0, ['sandybrown', 'saddlebrown'], ['Living Room (1p)', 'Living Room (2p)'])
    
    ax2.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5)
    ax2.legend(loc='center right')
    ax2.set_xticklabels([])

    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.set_title('Normalized RSSI vs. Ground Truth Person Count', fontsize=16)
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Normalized Value / Person Count')
    ax3.set_yticks([0, 0.5, 1, 1.5, 2]) # Adjust y-ticks for clarity

    combined_rssi = np.concatenate([rssi_a, rssi_b])
    if combined_rssi.size > 0:
        rssi_min, rssi_max = np.min(combined_rssi), np.max(combined_rssi)
        if rssi_max > rssi_min:
            norm_rssi_a = (rssi_a - rssi_min) / (rssi_max - rssi_min)
            norm_rssi_b = (rssi_b - rssi_min) / (rssi_max - rssi_min)
            if ts_a.size > 0: ax3.plot(ts_a, norm_rssi_a, label='Norm. RSSI (A)', color='deepskyblue', alpha=0.6)
            if ts_b.size > 0: ax3.plot(ts_b, norm_rssi_b, label='Norm. RSSI (B)', color='tomato', alpha=0.6)

    if gt_time_a.size > 0: ax3.step(gt_time_a, labels_a, where='post', label='GT (Room A)', color='green', linewidth=1.5)
    if gt_time_b.size > 0: ax3.step(gt_time_b, labels_b, where='post', label='GT (Room B)', color='purple', linewidth=1.5)
    if gt_time_lr.size > 0: ax3.step(gt_time_lr, labels_lr, where='post', label='GT (Living Room)', color='orange', linewidth=1.5)

    ax3.legend()
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', group['key'])
    save_path = os.path.join(output_dir, f"home_vis_multiclass_{safe_key}.png")
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved visualization to: {save_path}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logging.info("Scanning for file groups...")
    config = {'dataset_root': 'datasets/', 'scenes_to_process': ['Home_Scene1', 'Home_Scene2']}
    file_groups = find_file_groups(config['dataset_root'], config['scenes_to_process'])
    
    if not file_groups:
        logging.error("No file groups found. Check dataset paths and scene names.")
        return
        
    num_to_visualize = min(10, len(file_groups))
    selected_groups = random.sample(file_groups, num_to_visualize)
    
    logging.info(f"Selected {len(selected_groups)} random groups to visualize.")
    
    output_dir = "results/home_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Outputting images to: {output_dir}")
    
    for group in selected_groups:
        try:
            create_visualization(group, output_dir)
        except Exception as e:
            logging.error(f"Failed to process group {group['key']}. Error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 