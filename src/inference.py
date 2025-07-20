import torch
import numpy as np
import os
import argparse
from .model import SimpleTransformer
from .dataset import WiFiSensingDataset # For creating a single instance

def run_inference(config, model_path, data_file_path):
    """
    Runs inference on a single data file using a trained model.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model ---
    model = SimpleTransformer(
        input_dim=config['input_dim'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    )
    
    # Load the trained model weights
    # Need to handle DataParallel wrapper if the model was trained on multiple GPUs
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()

    # --- Data ---
    # We will fake a dataset with a single file to process it
    # Note: For real inference, you might want a simpler data loading function
    print("Loading and preparing data for inference...")
    inference_dataset = WiFiSensingDataset(data_files=[data_file_path], label_files=["dummy_label.txt"])
    
    # We'll process sequence by sequence for this example
    predictions = []
    
    with torch.no_grad():
        for sequence, _ in inference_dataset:
            sequence = sequence.unsqueeze(0).to(device) # Add batch dimension
            output = model(sequence)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted.item())

    print("\n--- Inference Results ---")
    print(f"File: {os.path.basename(data_file_path)}")
    print(f"Total sequences processed: {len(predictions)}")
    
    # You can add more sophisticated analysis here, like showing a timeline of predictions
    print(f"Sample predictions (first 10): {predictions[:10]}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained WiFi Sensing model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth) file.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file (.txt) to run inference on.')
    
    # We need the config to build the model structure
    # In a real system, you might save the config with the model
    parser.add_argument('--config_path', type=str, default='config.yml', help='Path to the YAML configuration file used for training.')
    
    args = parser.parse_args()

    # Create a dummy label file as the dataset class expects one
    with open("dummy_label.txt", "w") as f:
        f.write("0 1 0 1") # Content doesn't matter

    # Load config to get model parameters
    from .config import get_config
    config = get_config()
        
    run_inference(config, args.model_path, args.data_file)
    
    # Clean up dummy file
    os.remove("dummy_label.txt")

if __name__ == "__main__":
    main() 