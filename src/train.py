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

from .config import get_config
from .utils import set_seed, get_logger
from .model import get_model
from .dataset import create_dataloaders
from .losses import get_loss_function
from .model import MultiTaskTransformer # Added import for MultiTaskTransformer


def log_confusion_matrix(y_true, y_pred, epoch, writer, results_dir, class_names=None):
    """
    Calculates, logs, and saves the multilabel confusion matrix.
    """
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    for i, (matrix, class_name) in enumerate(zip(cm, class_names)):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax.set_title(f'Confusion Matrix for {class_name} (Epoch {epoch+1})')
        plt.tight_layout()
        
        # Save figure to results directory
        fig_path = os.path.join(results_dir, f"cm_epoch_{epoch+1}_class_{i}.png")
        plt.savefig(fig_path)
        plt.close(fig) # Close figure to free memory
        
        # Log to TensorBoard
        writer.add_figure(f"ConfusionMatrix/{class_name}", fig, global_step=epoch)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, is_multitask=False):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    # --- MODIFICATION: Handle both single and multi-criterion ---
    if is_multitask:
        criterion_a, criterion_b, criterion_lr = criterion
    else:
        criterion_a, criterion_b, criterion_lr = criterion, criterion, criterion

    for inputs, labels in progress_bar:
        # --- MODIFICATION: Unpack the dual-stream inputs ---
        inputs_a, inputs_b = inputs
        inputs_a, inputs_b, labels = inputs_a.to(device), inputs_b.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # --- MODIFICATION FOR MULTI-TASK ---
        # Model now returns three separate outputs
        if is_multitask:
            output_a, output_b, output_lr = model(inputs_a, inputs_b)
            
            # Split labels for each task
            label_a, label_b, label_lr = labels[:, 0].unsqueeze(1), labels[:, 1].unsqueeze(1), labels[:, 2].unsqueeze(1)
            
            # Calculate loss for each task with its specific criterion
            loss_a = criterion_a(output_a, label_a)
            loss_b = criterion_b(output_b, label_b)
            loss_lr = criterion_lr(output_lr, label_lr)
            
            # Combine losses (simple sum for now)
            loss = loss_a + loss_b + loss_lr
        else: # Original logic for single-output models
            # --- FIX: Combine the two input streams for single-stream models ---
            inputs_combined = torch.cat((inputs_a, inputs_b), dim=1)
            outputs = model(inputs_combined)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # --- MODIFICATION: Log adaptive weights to progress bar ---
        if isinstance(model, MultiTaskTransformer) or (hasattr(model, 'module') and isinstance(model.module, MultiTaskTransformer)):
            # Handle DataParallel wrapper
            inner_model = model.module if hasattr(model, 'module') else model
            
            # Access weights via the new properties
            w1 = inner_model.weights_task1
            w2 = inner_model.weights_task2
            w3 = inner_model.weights_task3
            
            weights_log = {
                'W_a1': f"{w1[0].item():.2f}", 'W_b1': f"{w1[1].item():.2f}",
                'W_a2': f"{w2[0].item():.2f}", 'W_b2': f"{w2[1].item():.2f}",
                'W_a3': f"{w3[0].item():.2f}", 'W_b3': f"{w3[1].item():.2f}",
            }
            progress_bar.set_postfix(loss=loss.item(), **weights_log)
        else:
            progress_bar.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device, epoch, logger, writer, results_dir, log_every_n_epochs, is_multitask=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    
    # --- MODIFICATION: Handle both single and multi-criterion ---
    if is_multitask:
        criterion_a, criterion_b, criterion_lr = criterion
    else:
        criterion_a, criterion_b, criterion_lr = criterion, criterion, criterion
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            # --- MODIFICATION: Unpack and move dual-stream inputs to device ---
            inputs_a, inputs_b = inputs
            inputs_a, inputs_b, labels = inputs_a.to(device), inputs_b.to(device), labels.to(device)
            
            # --- MODIFICATION FOR MULTI-TASK ---
            if is_multitask:
                output_a, output_b, output_lr = model(inputs_a, inputs_b)
                
                label_a, label_b, label_lr = labels[:, 0].unsqueeze(1), labels[:, 1].unsqueeze(1), labels[:, 2].unsqueeze(1)

                loss_a = criterion_a(output_a, label_a)
                loss_b = criterion_b(output_b, label_b)
                loss_lr = criterion_lr(output_lr, label_lr)
                
                loss = loss_a + loss_b + loss_lr
                
                # Combine outputs for metric calculation
                outputs_combined = torch.cat([output_a, output_b, output_lr], dim=1)
                
            else: # Original logic for single-output models
                # --- FIX: Combine the two input streams for single-stream models ---
                inputs_combined = torch.cat((inputs_a, inputs_b), dim=1)
                outputs_combined = model(inputs_combined)
                loss = criterion(outputs_combined, labels)

            total_loss += loss.item()
            
            # --- MODIFICATION: Calculate predictions for multi-task output ---
            # Combine outputs and labels to calculate overall metrics
            
            preds = (torch.sigmoid(outputs_combined) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # --- MODIFICATION: Log final weights after validation epoch ---
    if isinstance(model, MultiTaskTransformer) or (hasattr(model, 'module') and isinstance(model.module, MultiTaskTransformer)):
        inner_model = model.module if hasattr(model, 'module') else model
        
        # Access weights via the new properties
        w1 = inner_model.weights_task1
        w2 = inner_model.weights_task2
        w3 = inner_model.weights_task3
        
        logger.info(
            f"Epoch {epoch+1} - Final Weights: "
            f"TaskA(W_a:{w1[0].item():.3f}, W_b:{w1[1].item():.3f}), "
            f"TaskB(W_a:{w2[0].item():.3f}, W_b:{w2[1].item():.3f}), "
            f"TaskLR(W_a:{w3[0].item():.3f}, W_b:{w3[1].item():.3f})"
        )

    # --- Calculate Accuracy Metrics ---
    
    # 1. Overall Accuracy (Element-wise)
    overall_accuracy = (all_preds == all_labels).mean() * 100
    
    # 2. Exact Match Ratio (Row-wise)
    exact_matches = np.all(all_preds == all_labels, axis=1).sum()
    exact_match_ratio = (exact_matches / len(all_labels)) * 100
    
    # --- MODIFICATION: Calculate and log per-task accuracy ---
    log_message = (
        f"Epoch {epoch+1} - Validation Loss: {avg_loss:.4f}, "
        f"Overall Accuracy: {overall_accuracy:.2f}%, "
        f"Exact Match Ratio: {exact_match_ratio:.2f}%"
    )

    if is_multitask:
        # Assuming the order is Room A, Room B, Living Room
        task_names = ["RoomA", "RoomB", "LivingRoom"]
        per_task_accuracies = (all_preds == all_labels).mean(axis=0) * 100
        
        accuracy_log = []
        for i, task_name in enumerate(task_names):
            # Also log to TensorBoard
            writer.add_scalar(f'Accuracy/validation_{task_name}', per_task_accuracies[i], epoch)
            accuracy_log.append(f"Acc_{task_name}: {per_task_accuracies[i]:.2f}%")
        
        log_message += ", " + ", ".join(accuracy_log)

    logger.info(log_message)

    # --- Generate and Save Confusion Matrix ---
    if (epoch + 1) % log_every_n_epochs == 0:
        logger.info(f"Epoch {epoch+1}: Logging confusion matrix.")
        # Assuming class names can be retrieved or are predefined.
        # For now, we'll use generic names.
        # You might want to pass class names from your dataset configuration.
        log_confusion_matrix(all_labels, all_preds, epoch, writer, results_dir)

    return avg_loss, overall_accuracy, exact_match_ratio

def main():
    config = get_config()
    set_seed(config['seed'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # --- MODIFICATION: Use model name in the run name for better organization ---
    model_name = config['model']['name']
    run_name = f"{model_name}/{timestamp}"
    
    log_dir = f"training/{run_name}"
    results_dir = f"results/{run_name}" # Create a corresponding results directory
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True) # Create the results directory
    
    logger = get_logger(os.path.join(log_dir, "training.log"))
    writer = SummaryWriter(log_dir)
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # --- GPU Setup: Sanitize the GPU string from config ---
    gpu_ids_str = config.get('gpus')
    if gpu_ids_str:
        # Remove all whitespace and replace full-width commas with standard commas
        cleaned_gpu_ids = "".join(str(gpu_ids_str).replace("，", ",").split())
        os.environ["CUDA_VISIBLE_DEVICES"] = cleaned_gpu_ids
        logger.info(f"Sanitized and set CUDA_VISIBLE_DEVICES to: '{cleaned_gpu_ids}'")

    device = torch.device("cuda" if torch.cuda.is_available() and gpu_ids_str else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("--- Starting Data Loading ---")
    train_loader, val_loader = create_dataloaders(config)
    logger.info("--- Data Loading Finished ---")
    
    model = get_model(config)
    
    # DataParallel will automatically use all GPUs made visible by CUDA_VISIBLE_DEVICES
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        
    model.to(device)

    # --- MODIFICATION: Create loss function based on model type ---
    is_multitask = isinstance(model.module if hasattr(model, 'module') else model, MultiTaskTransformer)
    
    if is_multitask:
        logger.info("Multi-task model detected. Creating separate loss functions for each head.")
        pos_weights = config['training']['pos_weight']
        if len(pos_weights) != 3:
            raise ValueError("pos_weight must contain exactly 3 values for the multi-task model.")
        
        # Create a separate criterion for each task, with its own pos_weight
        criterion_a = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weights[0]], device=device))
        criterion_b = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weights[1]], device=device))
        criterion_lr = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weights[2]], device=device))
        
        criterion = (criterion_a, criterion_b, criterion_lr) # Pass as a tuple
    else:
        logger.info("Single-task model detected. Using global loss function from config.")
        criterion = get_loss_function(config)
        criterion.to(device)
    
    # --- MODIFICATION: Conditionally set different LR for adaptive weights ---
    if is_multitask and config['training'].get('use_adaptive_weights', True):
        logger.info("Setting up optimizer with different learning rates for adaptive weights.")
        
        inner_model = model.module if hasattr(model, 'module') else model
        
        # Identify the adaptive weight parameters (logits)
        adaptive_weights_params_ids = {id(p) for p in [
            inner_model.task1_logits, 
            inner_model.task2_logits, 
            inner_model.task3_logits
        ]}
        
        # Separate parameters into two groups
        base_params = [p for p in inner_model.parameters() if id(p) not in adaptive_weights_params_ids]
        adaptive_weights_params = [p for p in inner_model.parameters() if id(p) in adaptive_weights_params_ids]

        # Get base learning rate from config
        base_lr = config['training']['learning_rate']
        
        # Get a multiplier for the adaptive weights' LR
        lr_multiplier = config['training'].get('adaptive_lr_multiplier', 10.0)
        adaptive_lr = base_lr * lr_multiplier
        
        logger.info(f"Base LR: {base_lr}, Adaptive Weights LR: {adaptive_lr} (Multiplier: {lr_multiplier}x)")

        param_groups = [
            {'params': base_params},
            {'params': adaptive_weights_params, 'lr': adaptive_lr}
        ]
        
        optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=config['training']['weight_decay'])

    else: # Original optimizer setup for other models or when adaptive weights are off
        if is_multitask:
             logger.info("Adaptive weights are disabled. Using a single learning rate for all parameters.")
        optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    log_every_n_epochs = config['training'].get('log_every_n_epochs', 1) # Default to 1 if not set
    
    logger.info("--- Starting Training ---")
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, is_multitask)
        val_loss, val_accuracy, val_exact_match = validate_one_epoch(
            model, val_loader, criterion, device, epoch, logger, writer, results_dir, log_every_n_epochs, is_multitask
        )
        
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
            
            # Format the model filename using placeholders from config
            model_filename = config['training']['model_save_name'].format(
                project_name=config['project_name'],
                model_name=config['model']['name'],
                timestamp=timestamp
            )
            model_save_path = os.path.join("models", model_filename)
            
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