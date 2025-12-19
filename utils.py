"""
Utility functions for Bangla Cyberbullying Detection
Includes seed setting, model metrics calculation, and formatting helpers
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to: {seed}")


def get_model_metrics(model):
    """
    Calculate model size and parameter counts.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary containing total parameters, trainable parameters, and model size in MB
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    print(f"\n📐 Model Metrics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {size_mb:.2f} MB")
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(size_mb, 2)
    }


def print_experiment_header(config):
    """
    Print formatted experiment header with configuration details.
    
    Args:
        config: Configuration object with experiment parameters
    """
    print("\n" + "="*70)
    print(f"🧪 EXPERIMENT: {config.author_name}")
    print("="*70)
    print(f"   Model: {config.model_path}")
    print(f"   Batch Size: {config.batch}")
    print(f"   Learning Rate: {config.lr}")
    print(f"   Max Epochs: {config.epochs}")
    print(f"   Max Length: {config.max_length}")
    print(f"   Freeze Base: {config.freeze_base}")
    print(f"   Stratification: {config.stratification_type}")
    print(f"   K-Folds: {config.num_folds}")
    print(f"   Dropout: {config.dropout}")
    print(f"   Weight Decay: {config.weight_decay}")
    print(f"   Warmup Ratio: {config.warmup_ratio}")
    print(f"   Gradient Clip Norm: {config.gradient_clip_norm}")
    print(f"   Early Stopping Patience: {config.early_stopping_patience}")
    print(f"   MLflow Experiment: {config.mlflow_experiment_name}")
    print("="*70 + "\n")


def print_fold_summary(fold_num, best_metrics, best_epoch):
    """
    Print summary of fold performance.
    
    Args:
        fold_num (int): Fold number (0-indexed)
        best_metrics (dict): Best metrics achieved in this fold
        best_epoch (int): Epoch number where best performance was achieved
    """
    print("\n" + "-"*70)
    print(f"📋 FOLD {fold_num + 1} SUMMARY")
    print("-"*70)
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best F1 (weighted): {best_metrics['f1_weighted']:.4f}")
    print(f"   Best F1 (macro): {best_metrics['f1_macro']:.4f}")
    print(f"   Best Accuracy (exact match): {best_metrics['accuracy']:.4f}")
    print(f"   Best Per-Label Accuracy: {best_metrics['per_label_accuracy']:.4f}")
    print(f"   Best Hamming Loss: {best_metrics['hamming_loss']:.4f}")
    print("-"*70 + "\n")


def print_experiment_summary(best_fold_idx, best_fold_metrics, model_metrics):
    """
    Print final experiment summary.
    
    Args:
        best_fold_idx (int): Index of best performing fold
        best_fold_metrics (dict): Metrics from the best fold
        model_metrics (dict): Model size and parameter information
    """
    print("\n" + "="*70)
    print("🏆 EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n   Best performing fold: Fold {best_fold_idx + 1}")
    print(f"\n   📊 Validation Metrics:")
    print(f"      Accuracy (exact match): {best_fold_metrics['accuracy']:.4f}")
    print(f"      Per-Label Accuracy: {best_fold_metrics['per_label_accuracy']:.4f}")
    print(f"      Hamming Loss: {best_fold_metrics['hamming_loss']:.4f}")
    print(f"      F1 (weighted): {best_fold_metrics['f1_weighted']:.4f}")
    print(f"      F1 (macro): {best_fold_metrics['f1_macro']:.4f}")
    print(f"      Precision (weighted): {best_fold_metrics['precision_weighted']:.4f}")
    print(f"      Precision (macro): {best_fold_metrics['precision_macro']:.4f}")
    print(f"      Recall (weighted): {best_fold_metrics['recall_weighted']:.4f}")
    print(f"      Recall (macro): {best_fold_metrics['recall_macro']:.4f}")
    
    if model_metrics:
        print(f"\n   🤖 Model Information:")
        print(f"      Model size: {model_metrics['model_size_mb']} MB")
        print(f"      Total parameters: {model_metrics['total_parameters']:,}")
        print(f"      Trainable parameters: {model_metrics['trainable_parameters']:,}")
    
    print("="*70)


def format_time(seconds):
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device_info():
    """
    Get information about available compute devices.
    
    Returns:
        dict: Device information including type, name, and memory
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {
            'device': device,
            'type': 'cuda',
            'name': gpu_name,
            'memory_gb': round(gpu_memory, 2)
        }
    else:
        return {
            'device': torch.device('cpu'),
            'type': 'cpu',
            'name': 'CPU',
            'memory_gb': None
        }


def print_device_info():
    """Print information about the compute device being used."""
    info = get_device_info()
    
    print(f"\n💻 Device Information:")
    print(f"   Type: {info['type'].upper()}")
    print(f"   Name: {info['name']}")
    if info['memory_gb']:
        print(f"   Memory: {info['memory_gb']} GB")
    
    return info['device']
