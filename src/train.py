import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from tqdm import tqdm
import numpy as np

from .config import get_config
from .utils import set_seed, get_logger
from .model import get_model
from .dataset import create_dataloaders
from .losses import get_loss_function


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device, epoch, logger):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # --- Calculate Accuracy Metrics ---
    
    # 1. Overall Accuracy (Element-wise)
    # This checks how many individual labels (out of samples * num_classes) are correct.
    overall_accuracy = (all_preds == all_labels).mean() * 100
    
    # 2. Exact Match Ratio (Row-wise)
    # This checks how many full rows (samples) are predicted perfectly.
    exact_matches = np.all(all_preds == all_labels, axis=1).sum()
    exact_match_ratio = (exact_matches / len(all_labels)) * 100
    
    logger.info(
        f"Epoch {epoch+1} - Validation Loss: {avg_loss:.4f}, "
        f"Overall Accuracy: {overall_accuracy:.2f}%, "
        f"Exact Match Ratio: {exact_match_ratio:.2f}%"
    )
    return avg_loss, overall_accuracy, exact_match_ratio

def main():
    config = get_config()
    set_seed(config['seed'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"training/{config['project_name']}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = get_logger(os.path.join(log_dir, "training.log"))
    writer = SummaryWriter(log_dir)
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("--- Starting Data Loading ---")
    train_loader, val_loader = create_dataloaders(config)
    logger.info("--- Data Loading Finished ---")
    
    model = get_model(config)
    
    if torch.cuda.device_count() > 1 and config['gpus']:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        
    model.to(device)

    # Create loss function from config
    criterion = get_loss_function(config)
    criterion.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    logger.info("--- Starting Training ---")
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        val_loss, val_accuracy, val_exact_match = validate_one_epoch(model, val_loader, criterion, device, epoch, logger)
        
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation_overall', val_accuracy, epoch)
        writer.add_scalar('Accuracy/validation_exact_match', val_exact_match, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs("models", exist_ok=True)
            model_save_path = os.path.join("models", f"{config['project_name']}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Validation loss improved. Saved model to {model_save_path}")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config['training']['patience']:
            logger.info(f"Early stopping triggered after {config['training']['patience']} epochs with no improvement.")
            break
            
    writer.close()
    logger.info("--- Training Finished ---")

if __name__ == "__main__":
    main() 