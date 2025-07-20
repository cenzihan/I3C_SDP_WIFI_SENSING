import yaml
import argparse
import os

def get_config():
    """
    Loads configuration from a YAML file and merges it with command-line arguments.
    Command-line arguments override YAML file settings.
    """
    parser = argparse.ArgumentParser(
        description="Transformer-based model for WiFi Sensing Human Presence Detection."
    )
    
    # --- Path and Data Arguments ---
    parser.add_argument('--config_path', type=str, default='config.yml', help='Path to the YAML configuration file.')
    parser.add_argument('--dataset_root', type=str, help='Root directory of the dataset.')
    parser.add_argument('--project_name', type=str, help='Name of the project for logging purposes.')
    parser.add_argument('--scenes_to_process', nargs='+', help='List of scene folders to process.')
    
    # --- Model Hyperparameters ---
    parser.add_argument('--model_name', type=str, help='Name of the model architecture to use.')
    parser.add_argument('--input_dim', type=int, help='Input dimension for the model (e.g., number of CSI features).')
    parser.add_argument('--num_classes', type=int, help='Number of output classes.')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads in the Transformer.')
    parser.add_argument('--num_layers', type=int, help='Number of layers in the Transformer encoder.')
    parser.add_argument('--embed_dim', type=int, help='Embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, help='Dimension of the feedforward network in the Transformer.')
    parser.add_argument('--dropout', type=float, help='Dropout rate.')

    # --- Training Parameters ---
    parser.add_argument('--epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training and validation.')
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use (e.g., "AdamW", "SGD").')
    parser.add_argument('--loss_function', type=str, help='Loss function to use.')
    parser.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler (e.g., "StepLR", "CosineAnnealingLR").')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for the optimizer.')
    parser.add_argument('--patience', type=int, help='Patience for early stopping.')

    # --- System and Hardware ---
    parser.add_argument('--gpus', type=str, help='GPUs to use (e.g., "0,1,2,3").')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading.')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Load config from YAML file
    config = {}
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

    # Merge YAML config with command-line arguments
    # CLI arguments take precedence
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
            
    return config 