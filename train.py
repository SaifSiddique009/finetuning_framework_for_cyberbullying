"""
Training module with comprehensive features for Bangla Cyberbullying Detection

MERGED FEATURES:
- Mixed Precision Training (AMP) for faster GPU training
- Dataset Caching for faster iteration
- CSV Export for portable metrics
- Unweighted Validation Loss (best practice for fair evaluation)
- HuggingFace Model Saving for deployment
- HuggingFace Hub Push support
- MLflow Experiment Tracking
- Early Stopping with patience
- K-fold Cross-validation with stratification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import mlflow
import pickle
import os
import time
import pandas as pd

import data
from model import TransformerMultiLabelClassifier, save_model_for_huggingface
from utils import get_model_metrics, print_fold_summary, print_experiment_summary


# =============================================================================
# DATASET CACHING (from Codebase 1)
# =============================================================================

def cache_dataset(comments, labels, tokenizer, max_length, cache_file):
    """
    Cache dataset to avoid reprocessing on repeated runs.
    
    Args:
        comments: Array of text comments
        labels: Array of labels
        tokenizer: Tokenizer for text encoding
        max_length: Maximum sequence length
        cache_file: Path to cache file
        
    Returns:
        CyberbullyingDataset: Cached or newly created dataset
    """
    if os.path.exists(cache_file):
        print(f"  📦 Loading cached dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"  🔄 Creating and caching dataset to {cache_file}")
    dataset = data.CyberbullyingDataset(comments, labels, tokenizer, max_length)
    
    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics for multi-label classification.
    Includes both weighted and macro averages for better insight into model performance.
    
    Args:
        y_true: Ground truth labels (n_samples, n_labels)
        y_pred: Predicted probabilities (n_samples, n_labels) (thresholded at 0.5)
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 1. Subset accuracy (exact match - HARSH)
    subset_accuracy = accuracy_score(y_true, y_pred_binary)

    # 2. Per-Label accuracy (element-wise - FORGIVING)
    per_label_accuracy = 1 - hamming_loss(y_true, y_pred_binary)

    # 3. Hamming loss (inverse of per-label accuracy)
    hamming = hamming_loss(y_true, y_pred_binary)
    
    return {
        'accuracy': subset_accuracy,                       # HARSH - exact match only
        'per_label_accuracy': per_label_accuracy,          # FORGIVING - element wise match
        'hamming_loss': hamming,                           # Error rate (lower is better)
        
        # Weighted metrics - gives more importance to frequent classes
        'precision_weighted': precision_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        
        # Macro metrics - treats all classes equally (better for imbalance insight)
        'precision_macro': precision_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
    }


# =============================================================================
# TRAINING EPOCH (with AMP support from Codebase 1)
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None, max_norm=1.0, scaler=None):
    """
    Train the model for one epoch and calculate training metrics.
    
    Args:
        model: The transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run training on
        class_weights: Optional class weights for imbalanced data
        max_norm: Maximum gradient norm for clipping
        scaler: GradScaler for mixed precision training
        
    Returns:
        dict: Training metrics including loss and performance metrics
    """
    model.train()
    total_loss = 0
    all_train_predictions = []
    all_train_labels = []
    
    # Use mixed precision if scaler is provided
    use_amp = scaler is not None
    
    # Setup loss function with class weights if provided
    if class_weights is not None:
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    else:
        loss_fct = nn.BCEWithLogitsLoss()
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast('cuda', enabled=use_amp):
            outputs = model(input_ids, attention_mask=attention_mask, labels=None)
            loss = loss_fct(outputs['logits'], labels)
        
        # Mixed precision backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        
        # Accumulate predictions for metrics calculation (detached from computation graph)
        with torch.no_grad():
            predictions = torch.sigmoid(outputs['logits'])
            all_train_predictions.extend(predictions.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
    
    # Calculate training metrics
    avg_loss = total_loss / len(dataloader)
    train_metrics = calculate_metrics(np.array(all_train_labels), np.array(all_train_predictions))
    train_metrics['loss'] = avg_loss
    
    return train_metrics


# =============================================================================
# EVALUATION (with unweighted loss from Codebase 2 - best practice)
# =============================================================================

def evaluate_model(model, dataloader, device, use_amp=False):
    """
    Evaluate the model on validation data.
    
    NOTE: Uses UNWEIGHTED loss for fair evaluation (best practice from Codebase 2)
    
    Args:
        model: The transformer model
        dataloader: Validation data loader
        device: Device to run evaluation on
        use_amp: Whether to use mixed precision
        
    Returns:
        dict: Validation metrics including loss and performance metrics
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # Use UNWEIGHTED loss for fair evaluation (best practice)
    loss_fct = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Mixed precision inference
            with autocast('cuda', enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask, labels=None)
                loss = loss_fct(outputs['logits'], labels)
            
            total_loss += loss.item()
            
            predictions = torch.sigmoid(outputs['logits'])
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
    metrics['loss'] = avg_loss
    
    return metrics


# =============================================================================
# EPOCH METRICS PRINTING
# =============================================================================

def print_epoch_metrics(epoch, num_epochs, fold, num_folds, train_metrics, val_metrics, best_f1, best_epoch):
    """
    Print comprehensive epoch metrics in a formatted way.
    """
    print("\n" + "="*70)
    print(f"📈 Epoch {epoch+1}/{num_epochs} | Fold {fold+1}/{num_folds}")
    print("="*70)
    
    print("\n📊 TRAINING:")
    print(f"   Loss: {train_metrics['loss']:.4f}")
    print(f"   Accuracy (exact match): {train_metrics['accuracy']:.4f}")
    print(f"   Per-Label Accuracy: {train_metrics['per_label_accuracy']:.4f}")
    print(f"   Hamming Loss: {train_metrics['hamming_loss']:.4f}")
    print(f"   Precision (weighted/macro): {train_metrics['precision_weighted']:.4f} / {train_metrics['precision_macro']:.4f}")
    print(f"   Recall (weighted/macro): {train_metrics['recall_weighted']:.4f} / {train_metrics['recall_macro']:.4f}")
    print(f"   F1 (weighted/macro): {train_metrics['f1_weighted']:.4f} / {train_metrics['f1_macro']:.4f}")
    
    print("\n📊 VALIDATION:")
    print(f"   Loss: {val_metrics['loss']:.4f}")
    print(f"   Accuracy (exact match): {val_metrics['accuracy']:.4f}")
    print(f"   Per-Label Accuracy: {val_metrics['per_label_accuracy']:.4f}")
    print(f"   Hamming Loss: {val_metrics['hamming_loss']:.4f}")
    print(f"   Precision (weighted/macro): {val_metrics['precision_weighted']:.4f} / {val_metrics['precision_macro']:.4f}")
    print(f"   Recall (weighted/macro): {val_metrics['recall_weighted']:.4f} / {val_metrics['recall_macro']:.4f}")
    print(f"   F1 (weighted/macro): {val_metrics['f1_weighted']:.4f} / {val_metrics['f1_macro']:.4f}")
    
    print(f"\n⭐ Best F1 so far: {best_f1:.4f} (Epoch {best_epoch})")
    print("="*70)


# =============================================================================
# CSV EXPORT (from Codebase 1)
# =============================================================================

def save_metrics_to_csv(fold_results, best_fold_idx, best_fold_metrics, config, output_dir='./outputs'):
    """
    Save fold summary and best metrics to CSV files.
    
    Args:
        fold_results: List of metrics dictionaries for each fold
        best_fold_idx: Index of the best fold
        best_fold_metrics: Metrics dictionary for the best fold
        config: Configuration object
        output_dir: Directory to save CSV files
        
    Returns:
        tuple: (fold_summary_path, best_metrics_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name_safe = config.model_path.replace('/', '_')
    
    # Fold Summary CSV
    fold_summary_filename = f'fold_summary_{model_name_safe}_batch{config.batch}_lr{config.lr}_epochs{config.epochs}_{timestamp}.csv'
    fold_summary_path = os.path.join(output_dir, fold_summary_filename)
    
    summary_data = {
        'Fold': [f'Fold {i+1}' for i in range(config.num_folds)],
        'Best Epoch': [fr['best_epoch'] for fr in fold_results],
        'Val Loss': [fr['loss'] for fr in fold_results],
        'Val Accuracy': [fr['accuracy'] for fr in fold_results],
        'Val Per-Label Accuracy': [fr['per_label_accuracy'] for fr in fold_results],
        'Val Hamming Loss': [fr['hamming_loss'] for fr in fold_results],
        'Val Precision (weighted)': [fr['precision_weighted'] for fr in fold_results],
        'Val Recall (weighted)': [fr['recall_weighted'] for fr in fold_results],
        'Val F1 (weighted)': [fr['f1_weighted'] for fr in fold_results],
        'Val Precision (macro)': [fr['precision_macro'] for fr in fold_results],
        'Val Recall (macro)': [fr['recall_macro'] for fr in fold_results],
        'Val F1 (macro)': [fr['f1_macro'] for fr in fold_results],
        'Train Loss': [fr['train_loss'] for fr in fold_results],
        'Train Accuracy': [fr['train_accuracy'] for fr in fold_results],
        'Train F1 (weighted)': [fr['train_f1_weighted'] for fr in fold_results],
        'Train F1 (macro)': [fr['train_f1_macro'] for fr in fold_results],
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add Mean and Std rows
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    mean_row = summary_df[numeric_cols].mean()
    std_row = summary_df[numeric_cols].std()
    
    summary_df.loc['Mean'] = mean_row
    summary_df.loc['Std'] = std_row
    summary_df.loc['Mean', 'Fold'] = 'Mean'
    summary_df.loc['Std', 'Fold'] = 'Std'
    
    summary_df.to_csv(fold_summary_path, index=False)
    print(f"   ✓ Fold summary saved to: {fold_summary_path}")
    
    # Best Metrics CSV
    best_metrics_filename = f'best_metrics_{model_name_safe}_batch{config.batch}_lr{config.lr}_epochs{config.epochs}_{timestamp}.csv'
    best_metrics_path = os.path.join(output_dir, best_metrics_filename)
    
    best_metrics_data = {
        'Best Fold': [f'Fold {best_fold_idx+1}'],
        'Best Epoch': [best_fold_metrics['best_epoch']],
        'Val Loss': [best_fold_metrics['loss']],
        'Val Accuracy': [best_fold_metrics['accuracy']],
        'Val Per-Label Accuracy': [best_fold_metrics['per_label_accuracy']],
        'Val Hamming Loss': [best_fold_metrics['hamming_loss']],
        'Val Precision (weighted)': [best_fold_metrics['precision_weighted']],
        'Val Recall (weighted)': [best_fold_metrics['recall_weighted']],
        'Val F1 (weighted)': [best_fold_metrics['f1_weighted']],
        'Val Precision (macro)': [best_fold_metrics['precision_macro']],
        'Val Recall (macro)': [best_fold_metrics['recall_macro']],
        'Val F1 (macro)': [best_fold_metrics['f1_macro']],
        'Train Loss': [best_fold_metrics['train_loss']],
        'Train Accuracy': [best_fold_metrics['train_accuracy']],
        'Train F1 (weighted)': [best_fold_metrics['train_f1_weighted']],
        'Train F1 (macro)': [best_fold_metrics['train_f1_macro']],
    }
    
    best_metrics_df = pd.DataFrame(best_metrics_data)
    best_metrics_df.to_csv(best_metrics_path, index=False)
    print(f"   ✓ Best metrics saved to: {best_metrics_path}")
    
    return fold_summary_path, best_metrics_path


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def run_kfold_training(config, comments, labels, tokenizer, device):
    """
    Run K-fold cross-validation training with all enhanced features.
    
    Features:
    - Mixed Precision Training (AMP)
    - Dataset Caching
    - CSV Export
    - HuggingFace Model Saving
    - MLflow Tracking
    
    Args:
        config: Configuration object with all hyperparameters
        comments: Array of text comments
        labels: Array of multi-label targets
        tokenizer: Tokenizer for text encoding
        device: Device to run training on
    """
    # =========================================================================
    # SETUP DIRECTORIES
    # =========================================================================
    output_dir = config.output_dir
    cache_dir = config.cache_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # =========================================================================
    # SETUP MIXED PRECISION (from Codebase 1)
    # =========================================================================
    use_amp = device.type == 'cuda' and not config.no_amp
    scaler = GradScaler() if use_amp else None
    
    print("\n" + "="*70)
    print("🚀 TRAINING CONFIGURATION")
    print("="*70)
    
    if use_amp:
        print(f"   ✓ Mixed Precision Training: ENABLED (GPU detected)")
    else:
        if config.no_amp:
            print(f"   ✗ Mixed Precision Training: DISABLED (--no_amp flag)")
        else:
            print(f"   ✗ Mixed Precision Training: DISABLED (CPU mode)")
    
    use_cache = not config.no_cache
    if use_cache:
        print(f"   ✓ Dataset Caching: ENABLED")
        print(f"   📁 Cache directory: {os.path.abspath(cache_dir)}")
    else:
        print(f"   ✗ Dataset Caching: DISABLED (--no_cache flag)")
    
    print(f"   📁 Output directory: {os.path.abspath(output_dir)}")
    print("="*70)
    
    # =========================================================================
    # SETUP MLFLOW
    # =========================================================================
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    with mlflow.start_run(run_name=f"{config.author_name}_batch{config.batch}_lr{config.lr}_epochs{config.epochs}"):
        
        # Log run ID
        run_id = mlflow.active_run().info.run_id
        print(f"\n📊 MLflow Run ID: {run_id}")
        
        # Log all configuration parameters
        mlflow.log_params({
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
            'mixed_precision': use_amp,
            'dataset_caching': use_cache,
        })
        
        # =====================================================================
        # PREPARE K-FOLD SPLITS
        # =====================================================================
        kfold_splits = data.prepare_kfold_splits(
            comments, labels, 
            num_folds=config.num_folds,
            stratification_type=config.stratification_type,
            seed=config.seed
        )
        
        # Store results for each fold
        fold_results = []
        best_fold_model = None
        best_fold_idx = -1
        best_overall_f1 = 0
        model_metrics = None
        
        # =====================================================================
        # K-FOLD TRAINING LOOP
        # =====================================================================
        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            print(f"\n{'='*70}")
            print(f"📂 FOLD {fold + 1}/{config.num_folds}")
            print('='*70)
            
            # Split data for current fold
            train_comments, val_comments = comments[train_idx], comments[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            print(f"   Training samples: {len(train_comments)}")
            print(f"   Validation samples: {len(val_comments)}")
            
            # Calculate class weights for imbalanced data
            class_weights = data.calculate_class_weights(train_labels)
            
            # -----------------------------------------------------------------
            # CREATE DATASETS (with optional caching)
            # -----------------------------------------------------------------
            if use_cache:
                train_cache_path = os.path.join(cache_dir, f'train_cache_fold{fold}.pkl')
                val_cache_path = os.path.join(cache_dir, f'val_cache_fold{fold}.pkl')
                
                train_dataset = cache_dataset(
                    train_comments, train_labels, tokenizer, 
                    config.max_length, train_cache_path
                )
                val_dataset = cache_dataset(
                    val_comments, val_labels, tokenizer, 
                    config.max_length, val_cache_path
                )
            else:
                print("  🔄 Creating datasets (caching disabled)")
                train_dataset = data.CyberbullyingDataset(
                    train_comments, train_labels, tokenizer, config.max_length
                )
                val_dataset = data.CyberbullyingDataset(
                    val_comments, val_labels, tokenizer, config.max_length
                )
            
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch, 
                shuffle=True, num_workers=2, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch, 
                shuffle=False, num_workers=2, pin_memory=True
            )
            
            # -----------------------------------------------------------------
            # INITIALIZE MODEL
            # -----------------------------------------------------------------
            model = TransformerMultiLabelClassifier(
                config.model_path, 
                len(data.LABEL_COLUMNS), 
                dropout=config.dropout
            )
            
            if config.freeze_base:
                model.freeze_base_layers()
            
            model.to(device)
            
            # Get model metrics (only once per experiment)
            if fold == 0:
                model_metrics = get_model_metrics(model)
                mlflow.log_metrics({
                    'total_parameters': model_metrics['total_parameters'],
                    'trainable_parameters': model_metrics['trainable_parameters'],
                    'model_size_mb': model_metrics['model_size_mb']
                })
            
            # -----------------------------------------------------------------
            # SETUP OPTIMIZER AND SCHEDULER
            # -----------------------------------------------------------------
            optimizer = AdamW(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay, 
                eps=1e-8
            )
            total_steps = len(train_loader) * config.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(config.warmup_ratio * total_steps), 
                num_training_steps=total_steps
            )
            
            # -----------------------------------------------------------------
            # TRAINING LOOP
            # -----------------------------------------------------------------
            best_f1 = 0
            best_metrics = {}
            best_epoch = 0
            patience = config.early_stopping_patience
            patience_counter = 0
            
            for epoch in range(config.epochs):
                # Train for one epoch (with AMP if enabled)
                train_metrics = train_epoch(
                    model, train_loader, optimizer, scheduler, device, 
                    class_weights, max_norm=config.gradient_clip_norm,
                    scaler=scaler
                )
                
                # Evaluate on validation set (with unweighted loss)
                val_metrics = evaluate_model(
                    model, val_loader, device, use_amp=use_amp
                )
                               
                # Check if this is the best epoch for this fold
                if val_metrics['f1_weighted'] > best_f1:
                    best_f1 = val_metrics['f1_weighted']
                    best_metrics = val_metrics.copy()
                    best_metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
                    best_epoch = epoch + 1
                    patience_counter = 0
                    
                    # Save model if this fold is the best overall
                    if best_f1 > best_overall_f1:
                        best_overall_f1 = best_f1
                        best_fold_idx = fold
                        best_fold_model = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Print comprehensive metrics
                print_epoch_metrics(
                    epoch, config.epochs, fold, config.num_folds, 
                    train_metrics, val_metrics, best_f1, best_epoch
                )

                # Early stopping
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Store best metrics for this fold
            best_metrics['best_epoch'] = best_epoch
            fold_results.append(best_metrics)
            
            # Log best metrics for this fold to MLflow
            for metric_name, metric_value in best_metrics.items():
                if not metric_name.startswith('train_'):
                    mlflow.log_metric(f"fold_{fold+1}_best_{metric_name}", metric_value)
            
            # Print fold summary
            print_fold_summary(fold, best_metrics, best_epoch)
        
        # =====================================================================
        # POST-TRAINING: AGGREGATE RESULTS
        # =====================================================================
        best_fold_metrics = fold_results[best_fold_idx]
        
        # Log best overall metrics
        mlflow.log_metric('best_fold_index', best_fold_idx + 1)
        
        for metric_name in ['accuracy', 'per_label_accuracy', 'precision_weighted', 'precision_macro', 
                           'recall_weighted', 'recall_macro', 'f1_weighted', 'f1_macro']:
            best_value = max([fold_result[metric_name] for fold_result in fold_results])
            mlflow.log_metric(f'best_{metric_name}', best_value)
        
        best_loss = min([fold_result['loss'] for fold_result in fold_results])
        mlflow.log_metric('best_loss', best_loss)
        
        best_hamming_loss = min([fold_result['hamming_loss'] for fold_result in fold_results])
        mlflow.log_metric('best_hamming_loss', best_hamming_loss)
        
        # =====================================================================
        # SAVE METRICS TO CSV (from Codebase 1)
        # =====================================================================
        print(f"\n{'='*70}")
        print("💾 SAVING METRICS TO CSV")
        print('='*70)
        
        try:
            fold_summary_path, best_metrics_path = save_metrics_to_csv(
                fold_results, best_fold_idx, best_fold_metrics, config, output_dir
            )
            
            mlflow.log_artifact(fold_summary_path)
            mlflow.log_artifact(best_metrics_path)
            print(f"   ✓ CSV files logged to MLflow")
        except Exception as e:
            print(f"   ✗ Error saving CSV files: {e}")
        
        # =====================================================================
        # SAVE MODEL IN HUGGINGFACE FORMAT (from Codebase 2)
        # =====================================================================
        if not config.no_save_model and best_fold_model is not None:
            print(f"\n{'='*70}")
            print("💾 SAVING MODEL IN HUGGINGFACE FORMAT")
            print('='*70)
            
            # Create a new model instance and load the best weights
            final_model = TransformerMultiLabelClassifier(
                config.model_path, 
                len(data.LABEL_COLUMNS), 
                dropout=config.dropout
            )
            final_model.load_state_dict(best_fold_model)
            
            # Generate save path
            os.makedirs(config.save_model_dir, exist_ok=True)
            save_dir = os.path.join(
                config.save_model_dir,
                f"{config.author_name}_fold{best_fold_idx+1}_f1_{best_overall_f1:.4f}"
            )
            
            # Save model
            save_model_for_huggingface(
                model=final_model,
                tokenizer=tokenizer,
                save_path=save_dir,
                model_metrics=best_fold_metrics,
                training_config=config
            )
            
            mlflow.log_param('saved_model_path', save_dir)
            mlflow.log_artifacts(save_dir, artifact_path="huggingface_model")
            
            # -----------------------------------------------------------------
            # PUSH TO HUGGINGFACE HUB (from Codebase 2)
            # -----------------------------------------------------------------
            if config.push_to_hub and config.hub_repo_name:
                print(f"\n{'='*70}")
                print("📤 PUSHING MODEL TO HUGGINGFACE HUB")
                print('='*70)
                
                try:
                    from huggingface_hub import HfApi, create_repo
                    api = HfApi()
                    
                    # Create repo if it doesn't exist
                    try:
                        create_repo(
                            config.hub_repo_name, 
                            private=config.hub_private, 
                            exist_ok=True
                        )
                    except Exception as e:
                        print(f"   Repository creation note: {e}")
                    
                    print(f"   Uploading to: {config.hub_repo_name}")
                    api.upload_folder(
                        folder_path=save_dir,
                        repo_id=config.hub_repo_name,
                        commit_message=f"Upload model - F1: {best_overall_f1:.4f}"
                    )
                    print(f"   ✅ Model pushed to: https://huggingface.co/{config.hub_repo_name}")
                    mlflow.log_param('huggingface_repo', config.hub_repo_name)
                    
                except ImportError:
                    print("   ⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub")
                except Exception as e:
                    print(f"   ⚠️  Failed to push to HuggingFace Hub: {e}")
        
        # =====================================================================
        # PRINT FINAL SUMMARY
        # =====================================================================
        print_experiment_summary(best_fold_idx, best_fold_metrics, model_metrics)
        
        print(f"\n{'='*70}")
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print('='*70)
        print(f"   📁 CSV metrics saved in: {os.path.abspath(output_dir)}")
        if use_cache:
            print(f"   📁 Cache files saved in: {os.path.abspath(cache_dir)}")
        if not config.no_save_model:
            print(f"   📁 Model saved in: {os.path.abspath(config.save_model_dir)}")
        print(f"   📊 MLflow Run ID: {run_id}")
        print(f"\n   To view results in MLflow UI:")
        print(f"   $ mlflow ui")
        print(f"   Then open: http://localhost:5000")
        print('='*70 + "\n")
