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
            # Read the first line to check format
            first_line = f.readline()
            if not first_line:
                print("  [!] Error: File is empty.\n")
                return

            data_points = first_line.strip().split()
            num_columns = len(data_points)
            
            # Efficiently count lines and get the last line
            line_count = 1
            last_line = first_line
            for line in f:
                last_line = line
                line_count += 1
        
        last_line_data_points = last_line.strip().split()

        print(f"  [*] Total Packets (Lines): {line_count}")
        print(f"  [*] Data Points per Packet (Columns): {num_columns}")

        # Infer bandwidth based on the number of CSI data points
        if num_columns == 1008:
            print("  [*] Inferred Format: Correct for 5G_80M")
        elif num_columns == 1000:
            print("  [*] Inferred Format: Correct for 5G_160M")
        else:
            print(f"  [!] Warning: Unexpected number of columns ({num_columns}). Check data format.")
            
        print(f"\n  [*] Sample of the first packet (first 10 data points):")
        print(f"      {' '.join(data_points[:10])} ...")
        
        print(f"\n  [*] Sample of the last packet (first 10 data points):")
        print(f"      {' '.join(last_line_data_points[:10])} ...")

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
        # Room A Data and Groundtruth
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomA/2to2_Scene3_GreenPlantFan_1/RoomAData/1101_01.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomA/2to2_Scene3_GreenPlantFan_1/RoomAData/1101_02.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomA/2to2_Scene3_GreenPlantFan_1/RoomAData/1101_03.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomA/2to2_Scene3_GreenPlantFan_1/LivingRoomGroundtruth/groundtruth_1121_01.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomA/2to2_Scene3_GreenPlantFan_1/RoomAGroundtruth/groundtruth_1101_01.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomA/2to2_Scene3_GreenPlantFan_1/RoomAGroundtruth/groundtruth_1101_02.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomA/2to2_Scene3_GreenPlantFan_1/RoomAGroundtruth/groundtruth_1101_03.txt",

        # Room B Data and Groundtruth
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomB/2to2_Scene3_GreenPlantFan_1/RoomBData/1111_01.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomB/2to2_Scene3_GreenPlantFan_1/RoomBData/1111_02.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomB/2to2_Scene3_GreenPlantFan_1/RoomBData/1111_03.txt",
        "datasets/Home_Scene1/Home_Suite1/layout_1/5G_80M/Scene3_RoomB/2to2_Scene3_GreenPlantFan_1/RoomBGroundtruth/groundtruth_1111_01.txt"
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