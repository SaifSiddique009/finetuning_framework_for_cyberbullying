#!/usr/bin/env python3
"""
Main entry point for Bangla Cyberbullying Detection fine-tuning
Orchestrates the complete training pipeline with K-fold cross-validation

Usage:
    python main.py --author_name "your_name" --dataset_path "path/to/data.csv" [options]

Example:
    python main.py --author_name "saif" --dataset_path "./data.csv" --batch 32 --lr 2e-5 --epochs 15
"""

import torch
from transformers import AutoTokenizer
import warnings
import sys

import data
import train
from config import parse_arguments, print_config
from utils import set_seed, print_experiment_header, print_device_info


def main():
    """
    Main function that orchestrates the training pipeline.
    """
    # =========================================================================
    # STARTUP
    # =========================================================================
    print("\n" + "="*70)
    print("🔥 BANGLA CYBERBULLYING DETECTION - FINE-TUNING PIPELINE")
    print("="*70)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # =========================================================================
    # PARSE CONFIGURATION
    # =========================================================================
    try:
        config = parse_arguments()
    except Exception as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)
    
    # =========================================================================
    # SET RANDOM SEED (MUST be done before any random operations)
    # =========================================================================
    set_seed(config.seed)
    
    # =========================================================================
    # SETUP DEVICE
    # =========================================================================
    device = print_device_info()
    
    # =========================================================================
    # PRINT CONFIGURATION
    # =========================================================================
    print_config(config)
    
    # =========================================================================
    # INITIALIZE TOKENIZER
    # =========================================================================
    print(f"\n📝 Loading tokenizer: {config.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        print(f"   ✓ Tokenizer loaded successfully")
        print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"\n❌ Error loading tokenizer: {e}")
        print("   Make sure the model name/path is correct and you have internet connection.")
        sys.exit(1)
    
    # =========================================================================
    # LOAD AND PREPROCESS DATA
    # =========================================================================
    print(f"\n📂 Loading dataset from: {config.dataset_path}")
    try:
        comments, labels = data.load_and_preprocess_data(config.dataset_path)
        print(f"\n   ✓ Successfully loaded {len(comments):,} samples with {len(data.LABEL_COLUMNS)} labels")
    except FileNotFoundError:
        print(f"\n❌ Error: Dataset file not found at {config.dataset_path}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # =========================================================================
    # PRINT EXPERIMENT HEADER
    # =========================================================================
    print_experiment_header(config)
    
    # =========================================================================
    # CHECK STRATIFICATION PACKAGE
    # =========================================================================
    if config.stratification_type == 'multilabel':
        try:
            import iterstrat
            print("✓ iterative-stratification package found. Multi-label stratification will be used.")
        except ImportError:
            print("\n⚠️  WARNING: iterative-stratification package not found.")
            print("   Install it with: pip install iterative-stratification")
            print("   Falling back to regular K-fold splitting.")
            config.stratification_type = 'none'
    
    # =========================================================================
    # RUN K-FOLD CROSS-VALIDATION TRAINING
    # =========================================================================
    print("\n" + "="*70)
    print("🚀 STARTING K-FOLD CROSS-VALIDATION TRAINING")
    print("="*70)
    
    try:
        train.run_kfold_training(config, comments, labels, tokenizer, device)
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(0)
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU out of memory error!")
        print("   Try one of the following:")
        print(f"   • Reduce batch size (current: {config.batch})")
        print(f"   • Reduce sequence length (current: {config.max_length})")
        print("   • Use --freeze_base to reduce trainable parameters")
        print("   • Use gradient accumulation (not yet implemented)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
