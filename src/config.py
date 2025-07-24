import yaml
import argparse
import os

def get_config(default_config_path=None):
    """
    Loads configuration from a YAML file and merges it with command-line arguments.
    Command-line arguments override YAML file settings.
    """
    parser = argparse.ArgumentParser(
        description="Transformer-based model for WiFi Sensing Human Presence Detection."
    )
    
    # --- Path and Data Arguments ---
    parser.add_argument('--config_path', type=str, default=default_config_path, help='Path to the YAML configuration file.')
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

    # Determine the final config path
    config_path_to_load = args.config_path
    if not config_path_to_load:
        # If no default was provided and no CLI arg was given, fall back to the main config.
        config_path_to_load = 'config.yml'

    # Load config from YAML file
    config = {}
    if os.path.exists(config_path_to_load):
        with open(config_path_to_load, 'r') as f:
            try:
                config = yaml.safe_load(f)
                # Store the actual path used for loading
                config['config_path_loaded'] = config_path_to_load
            except yaml.YAMLError as exc:
                print(exc)
    else:
        print(f"Warning: Config file not found at '{config_path_to_load}'. Using command-line args only.")


    # Merge YAML config with command-line arguments
    # CLI arguments take precedence
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cli_args)
            
    return config 