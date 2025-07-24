import torch
import numpy as np
import os
import argparse
from .model_office_glasswall import get_model
from .preprocess_office_glasswall import parse_office_csi_line
from .config import get_config

def load_and_preprocess_single_file(file_path, config):
    """
    Loads and preprocesses a single office scenario data file into a tensor
    that can be fed into the model.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, 'r') as f:
        packets = [p for p in (parse_office_csi_line(line, config) for line in f) if p is not None]

    if not packets:
        raise ValueError("No valid packets could be parsed from the file.")

    # Convert list of (timestamp, feature_map) to just a list of feature_maps
    feature_maps = [p[1] for p in packets]
    
    # Stack them into a single numpy array
    all_features = np.stack(feature_maps, axis=0) # -> (num_packets, channels, subcarriers)

    # Pad or truncate to the required number of packets
    num_packets = all_features.shape[0]
    max_packets = config['max_packets_per_interval']
    
    if num_packets > max_packets:
        padded_features = all_features[:max_packets, :, :]
    else:
        pad_width = ((0, max_packets - num_packets), (0, 0), (0, 0))
        padded_features = np.pad(all_features, pad_width, mode='constant', constant_values=0)

    # Transpose to match model input: (channels, seq_len, features)
    # The model expects (C, H, W) which is (input_channels, max_packets, feature_dim)
    model_input = padded_features.astype(np.float32)

    # Convert to tensor and add batch dimension
    return torch.from_numpy(model_input).unsqueeze(0)


def run_inference(config, model_path, data_file_path):
    """
    Runs inference on a single data file using a trained office glasswall model.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and config.get('gpus') else "cpu")
    print(f"Using device: {device}")

    # --- Model ---
    model = get_model(config)
    
    # Load the trained model weights
    print(f"Loading model from: {model_path}")
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
    print(f"Loading and preparing data for inference from: {data_file_path}")
    try:
        input_tensor = load_and_preprocess_single_file(data_file_path, config)
        input_tensor = input_tensor.to(device)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error processing data file: {e}")
        return

    # --- Inference ---
    print("\n--- Running Inference ---")
    with torch.no_grad():
        output = model(input_tensor)
        
        # Apply sigmoid to get probabilities and threshold at 0.5
        probs = torch.sigmoid(output)
        predictions = (probs > 0.5).int()

    print("\n--- Inference Results ---")
    print(f"File: {os.path.basename(data_file_path)}")
    print(f"Model Output (Logits): {output.cpu().numpy().flatten()}")
    print(f"Predicted Probabilities: {probs.cpu().numpy().flatten()}")
    print(f"Final Predictions (Threshold > 0.5): {predictions.cpu().numpy().flatten()}")
    
    # You can map the prediction indices to class names if you have them
    # e.g., class_names = ['Person Behind Glass', 'Person Behind Wall', ...]
    # predicted_labels = [class_names[i] for i, p in enumerate(predictions.flatten()) if p == 1]
    # print(f"Predicted Labels: {predicted_labels}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Office Glasswall model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth) file, e.g., models/glass_wall/model.pth')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file (.txt) to run inference on.')
    
    # The config file for this specific model is essential
    parser.add_argument('--config_path', type=str, default='config_office_glasswall.yml', help='Path to the YAML configuration file.')
    
    args = parser.parse_args()

    # Load config to get model and preprocessing parameters
    config = get_config(default_config_path=args.config_path)
        
    run_inference(config, args.model_path, args.data_file)

if __name__ == "__main__":
    main() 