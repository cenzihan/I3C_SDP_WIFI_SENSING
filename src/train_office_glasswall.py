import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

# --- MODIFIED IMPORTS for Office Scenario ---
from src.config import get_config # Re-used
from src.utils import set_seed, get_logger # Re-used
from src.model_office_glasswall import get_office_model as get_model # MODIFIED
from src.dataset_office_glasswall import create_office_dataloaders as create_dataloaders # MODIFIED
from src.losses import get_loss_function # Re-used


def log_confusion_matrix(y_true, y_pred, epoch, writer, results_dir, class_names=None):
    """
    Calculates, logs, and saves the multilabel confusion matrix.
    """
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        # For office, we can have fixed names
        class_names = [f"{i+1}m" for i in range(cm.shape[0])]

    for i, (matrix, class_name) in enumerate(zip(cm, class_names)):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax.set_title(f'Confusion Matrix for {class_name} (Epoch {epoch+1})')
        plt.tight_layout()
        
        # Save figure to results directory
        fig_path = os.path.join(results_dir, f"cm_epoch_{epoch+1}_class_{class_name}.png")
        plt.savefig(fig_path)
        plt.close(fig) # Close figure to free memory
        
        # Log to TensorBoard
        writer.add_figure(f"ConfusionMatrix/{class_name}", fig, global_step=epoch)


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

def validate_one_epoch(model, dataloader, criterion, device, epoch, logger, writer, results_dir, log_every_n_epochs):
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
    overall_accuracy = (all_preds == all_labels).mean() * 100
    
    # 2. Exact Match Ratio (Row-wise)
    exact_matches = np.all(all_preds == all_labels, axis=1).sum()
    exact_match_ratio = (exact_matches / len(all_labels)) * 100
    
    logger.info(
        f"Epoch {epoch+1} - Validation Loss: {avg_loss:.4f}, "
        f"Overall Accuracy: {overall_accuracy:.2f}%, "
        f"Exact Match Ratio: {exact_match_ratio:.2f}%"
    )

    # --- Generate and Save Confusion Matrix ---
    if (epoch + 1) % log_every_n_epochs == 0:
        logger.info(f"Epoch {epoch+1}: Logging confusion matrix.")
        log_confusion_matrix(all_labels, all_preds, epoch, writer, results_dir)

    return avg_loss, overall_accuracy, exact_match_ratio

def main():
    # --- MODIFICATION: Point to the new config file by default ---
    # By passing the default path to get_config, we ensure it's loaded correctly.
    # This can still be overridden from the shell script with --config_path.
    config = get_config(default_config_path='config_office_glasswall.yml')
        
    set_seed(config['seed'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Use project_name from the new config and add the 'glass_wall' subdirectory
    run_name = f"{config['project_name']}/{timestamp}"
    
    # --- MODIFICATION: Hardcode 'glass_wall' subdirectory ---
    output_subdir = "glass_wall"
    log_dir = f"training/{output_subdir}/{run_name}"
    results_dir = f"results/{output_subdir}/{run_name}"
    model_dir = f"models/{output_subdir}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    logger = get_logger(os.path.join(log_dir, "training.log"), name=config['project_name'])
    writer = SummaryWriter(log_dir)
    
    logger.info("--- Office Scenario Training ---")
    # Correctly log the path of the config file that was actually loaded.
    logger.info(f"Loading configuration from: {config.get('config_path_loaded', 'N/A')}")

    # Correctly log all configuration values from the dictionary.
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
        
    # --- GPU Setup: Sanitize the GPU string from config ---
    gpu_ids_str = config.get('gpus')
    if gpu_ids_str:
        cleaned_gpu_ids = "".join(str(gpu_ids_str).replace("，", ",").split())
        os.environ["CUDA_VISIBLE_DEVICES"] = cleaned_gpu_ids
        logger.info(f"Sanitized and set CUDA_VISIBLE_DEVICES to: '{cleaned_gpu_ids}'")

    device = torch.device("cuda" if torch.cuda.is_available() and gpu_ids_str else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("--- Starting Data Loading ---")
    train_loader, val_loader = create_dataloaders(config)
    logger.info("--- Data Loading Finished ---")
    
    model = get_model(config)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        
    model.to(device)

    criterion = get_loss_function(config)
    criterion.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    # --- MODIFICATION: Add logic for ReduceLROnPlateau scheduler ---
    scheduler_name = config['training'].get('lr_scheduler', 'cosine')
    if scheduler_name.lower() == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=10, # Number of epochs with no improvement after which learning rate will be reduced.
            verbose=True
        )
        logger.info("Using ReduceLROnPlateau learning rate scheduler.")
    else: # Default to CosineAnnealingLR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
        logger.info("Using CosineAnnealingLR learning rate scheduler.")


    best_val_loss = float('inf')
    epochs_no_improve = 0
    log_every_n_epochs = config['training'].get('log_every_n_epochs', 1)
    
    logger.info("--- Starting Training ---")
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        val_loss, val_accuracy, val_exact_match = validate_one_epoch(
            model, val_loader, criterion, device, epoch, logger, writer, results_dir, log_every_n_epochs
        )
        
        # --- MODIFICATION: Step scheduler based on its type ---
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation_overall', val_accuracy, epoch)
        writer.add_scalar('Accuracy/validation_exact_match', val_exact_match, epoch)
        # --- MODIFICATION: Get LR from optimizer, which is compatible with all schedulers ---
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            model_filename = config['training']['model_save_name'].format(
                project_name=config['project_name'],
                model_name=config['model']['name'],
                timestamp=timestamp
            )
            # --- MODIFICATION: Use the new model_dir ---
            model_save_path = os.path.join(model_dir, model_filename)
            
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