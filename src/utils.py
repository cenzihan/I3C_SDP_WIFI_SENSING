import torch
import numpy as np
import random
import os
import logging

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger(log_path, name="wifi-sensing"):
    """Initializes a logger to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers if they don't exist
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def parse_csi_data(line):
    """Parses a single line of CSI data and handles complex numbers."""
    parts = line.strip().split()
    
    # Timestamp, RSSI, MCS, Gain (first 8 elements)
    metadata = [float(p) for p in parts[:8]]
    
    # CSI data (complex numbers)
    csi_flat = []
    csi_str_parts = parts[8:]
    for part in csi_str_parts:
        try:
            # Handle 'j' as imaginary unit
            c = complex(part.replace('j', 'j'))
            csi_flat.extend([c.real, c.imag])
        except ValueError:
            # If conversion fails, maybe it's already a float?
            # Or handle error appropriately
            csi_flat.extend([0.0, 0.0]) # Placeholder for malformed data
            
    return metadata + csi_flat 