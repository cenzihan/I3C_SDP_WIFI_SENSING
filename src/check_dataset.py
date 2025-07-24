import os
from collections import Counter

def analyze_data_file(file_path):
    """Analyzes a CSI data file (.txt)."""
    print(f"--- Analyzing Data File: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"  [!] Error: File not found.\n")
        return
        
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("  [!] Error: File is empty.\n")
            return

        line_count = len(lines)
        first_line_points = lines[0].strip().split()
        num_columns = len(first_line_points)
        
        print(f"  [*] Total Packets (Lines): {line_count}")
        print(f"  [*] Data Points per Packet (Columns): {num_columns}")

        # Infer bandwidth based on the number of CSI data points
        if num_columns == 1008:
            print("  [*] Inferred Format: Correct for 5G_80M")
        elif num_columns == 1000:
            print("  [*] Inferred Format: Correct for 5G_160M")
        else:
            print(f"  [!] Warning: Unexpected number of columns ({num_columns}). Check data format.")
            
        print(f"\n  [*] Sample and length check for the first 10 packets:")
        for i, line in enumerate(lines[:10]):
            parts = line.strip().split()
            print(f"      Line {i+1:03d}: Length = {len(parts):<5} | Sample: {' '.join(parts[:13])} ...")
        
        if line_count > 10:
            print(f"\n  [*] Sample and length check for the last 10 packets:")
            # Handle cases where the file has 11-19 lines etc.
            start_index = max(10, line_count - 10)
            for i, line in enumerate(lines[start_index:], start=start_index):
                parts = line.strip().split()
                print(f"      Line {i+1:03d}: Length = {len(parts):<5} | Sample: {' '.join(parts[:13])} ...")

    except Exception as e:
        print(f"  [!] An error occurred: {e}")
    
    print("-" * (len(file_path) + 20) + "\n")


def analyze_groundtruth_file(file_path):
    """Analyzes a Groundtruth label file (.txt)."""
    print(f"--- Analyzing Groundtruth File: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"  [!] Error: File not found.\n")
        return

    try:
        with open(file_path, 'r') as f:
            content = f.read()
            labels = content.strip().split()
            
            if not labels or (len(labels) == 1 and not labels[0]):
                print("  [!] Error: File is empty or contains no labels.\n")
                return

            total_labels = len(labels)
            label_distribution = Counter(labels)

        print(f"  [*] Total Labels: {total_labels}")
        print(f"  [*] Label Distribution:")
        print(f"      - '0' (no person): {label_distribution.get('0', 0)} times")
        print(f"      - '1' (person present): {label_distribution.get('1', 0)} times")
        
        print(f"\n  [*] Sample of labels (first 20):")
        print(f"      {' '.join(labels[:20])} ...")

    except Exception as e:
        print(f"  [!] An error occurred: {e}")

    print("-" * (len(file_path) + 20) + "\n")


def main():
    # --- Pre-defined list of files to check ---
    files_to_check = [
        # OfficeScenario Glass_Wall_Scenario
        "datasets/OfficeScenario/Glass_Wall_Scenario/Data/ceiling_mount/20MHzConfig1/Fan/Processed_Data/1m.txt",
        "datasets/OfficeScenario/Glass_Wall_Scenario/Data/ceiling_mount/20MHzConfig1/Fan/Processed_Data/2m.txt",
        "datasets/OfficeScenario/Glass_Wall_Scenario/Data/ceiling_mount/20MHzConfig1/Fan/Processed_Data/3m.txt",
        "datasets/OfficeScenario/Glass_Wall_Scenario/Data/ceiling_mount/20MHzConfig1/Fan/Processed_Data/4m.txt",
        # "datasets/OfficeScenario/Glass_Wall_Scenario/Data/ceiling_mount/80MHzConfig2fixgain/Fan/Processed_Data/1m.txt",
        # "datasets/OfficeScenario/Glass_Wall_Scenario/Data/ceiling_mount/80MHzConfig2fixgain/Fan/Processed_Data/2m.txt",
        # "datasets/OfficeScenario/Glass_Wall_Scenario/Data/ceiling_mount/80MHzConfig2fixgain/Fan/Processed_Data/3m.txt",
        "datasets/OfficeScenario/Glass_Wall_Scenario/Groundtruth/ceiling_mount/20MHzConfig2/Fan/1m.txt",
        "datasets/OfficeScenario/Glass_Wall_Scenario/Groundtruth/ceiling_mount/20MHzConfig2/Fan/2m.txt",
        
        # "datasets/OfficeScenario/Glass_Wall_Scenario/Groundtruth/ceiling_mount/80MHzConfig2fixgain/Fan/1m.txt",

        # OfficeScenario Meeting_Room_Scene_2
        # "datasets/OfficeScenario/Meeting_Room_Scene_2/Room1/Groundtruth/Stream1x4_Config1/001_0_0000_3.txt",
        # "datasets/OfficeScenario/Meeting_Room_Scene_2/Room1/Stream1x4_Config1/Processed_Data/001_0_0000_3.txt",
        # "datasets/OfficeScenario/Meeting_Room_Scene_2/Room1/Groundtruth/Stream2x4_Config1/101_0_0000_3.txt",
        # "datasets/OfficeScenario/Meeting_Room_Scene_2/Room1/Stream2x4_Config1/Processed_Data/101_0_0000_3.txt",
        
        
        
    ]

    print("===== Starting Dataset Integrity Check =====\n")

    for file_path in files_to_check:
        # Determine file type based on its path and analyze it
        if "Data" in file_path:
            analyze_data_file(file_path)
        elif "Groundtruth" in file_path:
            analyze_groundtruth_file(file_path)
        else:
            print(f"--- Skipping file with unknown type: {file_path} ---")
            print("Path does not contain 'Data' or 'Groundtruth'.\n")

    print("===== Dataset Integrity Check Finished =====")


if __name__ == "__main__":
    main() 