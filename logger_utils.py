import os
import csv
import json
from pathlib import Path
import logging
from datetime import datetime

def setup_logger(run_id, split, log_dir='result/log'):
    """
    Sets up a logger that will write to a file in the log directory
    
    Args:
        run_id (str): The run ID
        split (int): The split number
        log_dir (str): Directory for logs
    
    Returns:
        logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create the log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{run_id}_split{split+1}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure logger
    logger = logging.getLogger(f"{run_id}_split{split+1}")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    return logger

def log_train_metrics(logger, epoch, metrics, log_to_console=True):
    """
    Log training metrics for an epoch
    
    Args:
        logger: Logger instance
        epoch (int): Current epoch number
        metrics (dict): Dictionary containing metrics
        log_to_console (bool): Whether to also print to console
    """
    log_message = f"Epoch {epoch} - Training - "
    
    # Add work function metrics
    log_message += f"WF-MAE: {metrics.get('wf_mae', 0):.4f}, WF-Weight: {metrics.get('wf_weight', 0):.4f}, "
    
    # Add band gap metrics
    log_message += f"BG-MAE: {metrics.get('bg_mae', 0):.4f}, BG-Weight: {metrics.get('bg_weight', 0):.4f}, "
    
    # Add classification metrics if present
    if 'cls_acc' in metrics:
        log_message += f"CLS-ACC: {metrics.get('cls_acc', 0):.4f}, CLS-Loss: {metrics.get('cls_loss', 0):.4f}, "
    
    # Add metal/semiconductor metrics if present
    if 'metal_loss' in metrics and 'sc_loss' in metrics:
        log_message += f"Metal-Loss: {metrics.get('metal_loss', 0):.4f}, SC-Loss: {metrics.get('sc_loss', 0):.4f}, "
    
    # Add auxiliary task metrics if present
    if 'aux1_mae' in metrics:
        log_message += f"AUX1-MAE: {metrics.get('aux1_mae', 0):.4f}, AUX1-Loss: {metrics.get('aux1_loss', 0):.4f}, AUX1-Weight: {metrics.get('aux1_weight', 0):.4f}, "
    
    if 'aux2_mae' in metrics:
        log_message += f"AUX2-MAE: {metrics.get('aux2_mae', 0):.4f}, AUX2-Loss: {metrics.get('aux2_loss', 0):.4f}, AUX2-Weight: {metrics.get('aux2_weight', 0):.4f}, "
    
    # Add total loss
    log_message += f"Total-Loss: {metrics.get('total_loss', 0):.4f}"
    
    # Log to file
    logger.info(log_message)
    
    # Also print to console if requested
    if log_to_console:
        print(log_message)

def log_val_metrics(logger, epoch, metrics, phase="Validation", log_to_console=True):
    """
    Log validation/test metrics for an epoch
    
    Args:
        logger: Logger instance
        epoch (int): Current epoch number
        metrics (dict): Dictionary containing metrics
        phase (str): Either "Validation" or "Test"
        log_to_console (bool): Whether to also print to console
    """
    log_message = f"Epoch {epoch} - {phase} - "
    
    # Add work function metrics
    log_message += f"WF-MAE: {metrics.get('wf_mae', 0):.4f}, WF-R2: {metrics.get('wf_r2', 0):.4f}, "
    
    # Add band gap metrics
    log_message += f"BG-MAE: {metrics.get('bg_mae', 0):.4f}, BG-R2: {metrics.get('bg_r2', 0):.4f}, "
    
    # Add classification metrics if present
    if 'cls_acc' in metrics:
        log_message += f"CLS-ACC: {metrics.get('cls_acc', 0):.4f}, CLS-Loss: {metrics.get('cls_loss', 0):.4f}, "
    
    # Add metal/semiconductor metrics if present
    if 'metal_loss' in metrics and 'sc_loss' in metrics:
        log_message += f"Metal-Loss: {metrics.get('metal_loss', 0):.4f}, SC-Loss: {metrics.get('sc_loss', 0):.4f}, "
    
    # Add auxiliary task metrics if present
    if 'aux1_mae' in metrics:
        log_message += f"AUX1-MAE: {metrics.get('aux1_mae', 0):.4f}, AUX1-Loss: {metrics.get('aux1_loss', 0):.4f}, AUX1-Weight: {metrics.get('aux1_weight', 0):.4f}, "
    
    if 'aux2_mae' in metrics:
        log_message += f"AUX2-MAE: {metrics.get('aux2_mae', 0):.4f}, AUX2-Loss: {metrics.get('aux2_loss', 0):.4f}, AUX2-Weight: {metrics.get('aux2_weight', 0):.4f}, "
    
    # Add total loss
    log_message += f"Total-Loss: {metrics.get('total_loss', 0):.4f}"
    
    # Log to file
    logger.info(log_message)
    
    # Also print to console if requested
    if log_to_console:
        print(log_message)