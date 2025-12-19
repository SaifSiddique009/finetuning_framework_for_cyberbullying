"""
Configuration module for Bangla Cyberbullying Detection
Includes all hyperparameters, experiment settings, and deployment options

Features:
- Training parameters (batch size, learning rate, epochs)
- Model configuration (dropout, max_length, freeze options)
- K-fold cross-validation settings
- MLflow experiment tracking
- HuggingFace Hub deployment options
- Mixed precision training toggle
"""

import argparse


def parse_arguments():
    """
    Parse command-line arguments for experiment configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Transformer models for multi-label Bangla cyberbullying detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # =========================================================================
    # BASIC TRAINING PARAMETERS
    # =========================================================================
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size for training and evaluation.')
    
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate for optimizer.')
    
    parser.add_argument('--epochs', type=int, default=15,
                       help='Maximum number of training epochs.')
    
    # =========================================================================
    # DATASET AND MODEL PARAMETERS
    # =========================================================================
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the CSV dataset file.')
    
    parser.add_argument('--model_path', type=str, default='sagorsarker/bangla-bert-base',
                       help='Pre-trained model name or path. Supports any HuggingFace transformer model.')
    
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for tokenization.')
    
    # =========================================================================
    # TRAINING CONFIGURATION
    # =========================================================================
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of folds for K-Fold cross-validation.')
    
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze the base transformer layers during fine-tuning (feature extraction mode).')
    
    # =========================================================================
    # REPRODUCIBILITY AND STRATIFICATION
    # =========================================================================
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility.')
    
    parser.add_argument('--stratification_type', type=str, default='multilabel',
                       choices=['multilabel', 'multiclass', 'none'],
                       help='Type of stratification for K-fold splitting. '
                            'multilabel: preserves distribution across all labels (requires iterative-stratification), '
                            'multiclass: uses primary label for stratification, '
                            'none: no stratification (regular K-fold).')
    
    # =========================================================================
    # EXPERIMENT TRACKING
    # =========================================================================
    parser.add_argument('--author_name', type=str, required=True,
                       help='Author name for MLflow run tagging and identification.')
    
    parser.add_argument('--mlflow_experiment_name', type=str, default='Bangla-Cyberbullying-Detection',
                       help='MLflow experiment name for tracking.')
    
    # =========================================================================
    # REGULARIZATION AND OPTIMIZATION
    # =========================================================================
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for the classification head.')
    
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW optimizer.')
    
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Ratio of total steps for learning rate warmup.')
    
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                       help='Maximum norm for gradient clipping.')
    
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Number of epochs without improvement before early stopping.')
    
    # =========================================================================
    # MIXED PRECISION TRAINING (NEW - from Codebase 1)
    # =========================================================================
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable Automatic Mixed Precision (AMP) training. '
                            'By default, AMP is enabled on CUDA devices for faster training.')
    
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable dataset caching. By default, tokenized datasets are cached '
                            'to speed up subsequent runs.')
    
    # =========================================================================
    # MODEL SAVING AND DEPLOYMENT (NEW - from Codebase 2)
    # =========================================================================
    parser.add_argument('--save_model_dir', type=str, default='./saved_models',
                       help='Directory to save trained models in HuggingFace format.')
    
    parser.add_argument('--no_save_model', action='store_true',
                       help='Disable automatic model saving after training.')
    
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push the model to HuggingFace Hub after training.')
    
    parser.add_argument('--hub_repo_name', type=str, default=None,
                       help='HuggingFace Hub repository name (e.g., username/model-name). '
                            'Required if --push_to_hub is set.')
    
    parser.add_argument('--hub_private', action='store_true',
                       help='Create a private repository on HuggingFace Hub.')
    
    # =========================================================================
    # OUTPUT DIRECTORIES
    # =========================================================================
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save CSV metrics and other outputs.')
    
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Directory to store cached datasets.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    if args.batch <= 0:
        raise ValueError("Batch size must be positive")
    
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if args.num_folds < 2:
        raise ValueError("Number of folds must be at least 2")
    
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("Dropout must be between 0 and 1")
    
    if args.warmup_ratio < 0 or args.warmup_ratio > 1:
        raise ValueError("Warmup ratio must be between 0 and 1")
    
    if args.push_to_hub and not args.hub_repo_name:
        raise ValueError("--hub_repo_name is required when --push_to_hub is set")
    
    return args


def print_config(config):
    """
    Print configuration in a formatted way.
    
    Args:
        config: Configuration namespace
    """
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    print("\n📊 Training Parameters:")
    print(f"  Batch Size: {config.batch}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Max Epochs: {config.epochs}")
    print(f"  Early Stopping Patience: {config.early_stopping_patience}")
    
    print("\n🤖 Model Parameters:")
    print(f"  Model: {config.model_path}")
    print(f"  Max Sequence Length: {config.max_length}")
    print(f"  Freeze Base: {config.freeze_base}")
    print(f"  Dropout: {config.dropout}")
    
    print("\n⚙️ Optimizer Parameters:")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Warmup Ratio: {config.warmup_ratio}")
    print(f"  Gradient Clip Norm: {config.gradient_clip_norm}")
    
    print("\n🔬 Experiment Parameters:")
    print(f"  Author: {config.author_name}")
    print(f"  K-Folds: {config.num_folds}")
    print(f"  Stratification: {config.stratification_type}")
    print(f"  Random Seed: {config.seed}")
    print(f"  MLflow Experiment: {config.mlflow_experiment_name}")
    
    print("\n🚀 Performance Options:")
    print(f"  Mixed Precision (AMP): {'Disabled' if config.no_amp else 'Enabled (auto)'}")
    print(f"  Dataset Caching: {'Disabled' if config.no_cache else 'Enabled'}")
    
    print("\n💾 Saving Options:")
    print(f"  Save Model: {'Disabled' if config.no_save_model else 'Enabled'}")
    print(f"  Save Directory: {config.save_model_dir}")
    print(f"  Push to Hub: {config.push_to_hub}")
    if config.push_to_hub:
        print(f"  Hub Repo: {config.hub_repo_name}")
        print(f"  Private Repo: {config.hub_private}")
    
    print("\n📁 Data Parameters:")
    print(f"  Dataset Path: {config.dataset_path}")
    print(f"  Output Directory: {config.output_dir}")
    print(f"  Cache Directory: {config.cache_dir}")
    
    print("="*70 + "\n")


def get_config_dict(config):
    """
    Convert config namespace to dictionary for logging.
    
    Args:
        config: Configuration namespace
        
    Returns:
        dict: Configuration as dictionary
    """
    return {
        'batch_size': config.batch,
        'learning_rate': config.lr,
        'num_epochs': config.epochs,
        'num_folds': config.num_folds,
        'max_length': config.max_length,
        'freeze_base': config.freeze_base,
        'dropout': config.dropout,
        'weight_decay': config.weight_decay,
        'warmup_ratio': config.warmup_ratio,
        'gradient_clip_norm': config.gradient_clip_norm,
        'early_stopping_patience': config.early_stopping_patience,
        'author_name': config.author_name,
        'model_path': config.model_path,
        'seed': config.seed,
        'stratification_type': config.stratification_type,
        'mixed_precision': not config.no_amp,
        'dataset_caching': not config.no_cache,
    }
