"""
Data loading and preprocessing module with stratified K-fold support
Supports both multi-label and multi-class stratification

Features:
- Multi-label dataset handling for cyberbullying detection
- Stratified K-fold splitting (multilabel, multiclass, or none)
- Class weight calculation for imbalanced data
- Efficient tokenization with HuggingFace tokenizers
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold


# Label columns for the cyberbullying detection task
LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']


class CyberbullyingDataset(Dataset):
    """
    PyTorch Dataset for cyberbullying detection.
    Handles tokenization and label conversion for multi-label classification.
    """
    
    def __init__(self, comments, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            comments: Array of text comments
            labels: Array of multi-label targets
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.comments = comments
        self.labels = labels.astype(np.float32) if isinstance(labels, np.ndarray) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        labels = self.labels[idx]

        # Tokenize the comment
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the dataset from CSV.
    
    Args:
        dataset_path (str): Path to the CSV file
        
    Returns:
        tuple: (comments array, labels array)
        
    Expected CSV format:
        comment,bully,sexual,religious,threat,spam
        "some text",1,0,0,0,0
        ...
    """
    df = pd.read_csv(dataset_path)
    
    # Drop unnecessary columns if present
    columns_to_drop = [col for col in ['Gender', 'Profession'] if col in df.columns]
    df_clean = df.drop(columns_to_drop, axis=1) if columns_to_drop else df
    
    # Ensure required columns exist
    if 'comment' not in df_clean.columns:
        raise ValueError("Dataset must contain a 'comment' column")
    
    for col in LABEL_COLUMNS:
        if col not in df_clean.columns:
            raise ValueError(f"Missing label column: {col}")
    
    comments = df_clean['comment'].values
    labels = df_clean[LABEL_COLUMNS].values
    
    # Print dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(comments)}")
    print(f"   Number of labels: {len(LABEL_COLUMNS)}")
    print(f"\n   Label distribution:")
    for i, col in enumerate(LABEL_COLUMNS):
        positive_count = np.sum(labels[:, i])
        percentage = (positive_count / len(labels)) * 100
        print(f"   • {col}: {int(positive_count)}/{len(labels)} ({percentage:.2f}% positive)")
    
    # Multi-label statistics
    label_counts = np.sum(labels, axis=1)
    print(f"\n   Samples per label count:")
    for i in range(int(label_counts.max()) + 1):
        count = np.sum(label_counts == i)
        if count > 0:
            print(f"   • {i} labels: {count} samples ({count/len(labels)*100:.2f}%)")
    
    return comments, labels


def prepare_kfold_splits(comments, labels, num_folds=5, stratification_type='multilabel', seed=42):
    """
    Prepare K-fold cross-validation splits with optional stratification.
    
    Args:
        comments: Array of text comments
        labels: Array of multi-label targets
        num_folds (int): Number of folds for cross-validation
        stratification_type (str): Type of stratification ('multilabel', 'multiclass', 'none')
        seed (int): Random seed for reproducibility
        
    Returns:
        generator: K-fold split indices
    """
    
    if stratification_type == 'multilabel':
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
            print(f"\n✓ Using MultilabelStratifiedKFold with {num_folds} folds")
            print("  (Preserves label distribution across all labels)")
            
            kfold = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            return kfold.split(comments, labels)
            
        except ImportError:
            print("\n⚠️  WARNING: iterative-stratification not installed.")
            print("   Install with: pip install iterative-stratification")
            print("   Falling back to regular KFold")
            stratification_type = 'none'
    
    if stratification_type == 'multiclass':
        print(f"\n✓ Using StratifiedKFold with {num_folds} folds (based on primary label)")
        
        # For multi-class stratification, we use the primary label (most severe)
        # Priority order: threat > sexual > religious > bully > spam
        primary_labels = np.zeros(len(labels), dtype=int)
        
        for i in range(len(labels)):
            if labels[i, 3] == 1:  # threat
                primary_labels[i] = 4
            elif labels[i, 1] == 1:  # sexual
                primary_labels[i] = 3
            elif labels[i, 2] == 1:  # religious
                primary_labels[i] = 2
            elif labels[i, 0] == 1:  # bully
                primary_labels[i] = 1
            elif labels[i, 4] == 1:  # spam
                primary_labels[i] = 5
            else:  # no label
                primary_labels[i] = 0
        
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return kfold.split(comments, primary_labels)
    
    else:  # stratification_type == 'none'
        print(f"\n✓ Using regular KFold with {num_folds} folds (no stratification)")
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return kfold.split(comments)


def calculate_class_weights(labels):
    """
    Calculate class weights for handling imbalanced data.
    Uses inverse frequency weighting: weight = negative_count / positive_count
    
    Args:
        labels: Array of multi-label targets
        
    Returns:
        torch.FloatTensor: Class weights for each label
    """
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts
    
    # Avoid division by zero
    weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    
    print("\n⚖️  Class weights for handling imbalance:")
    for i, col in enumerate(LABEL_COLUMNS):
        print(f"   • {col}: {weights[i]:.3f}")
    
    return torch.FloatTensor(weights)


def get_label_names():
    """Return the list of label column names."""
    return LABEL_COLUMNS.copy()


def create_sample_dataset(output_path='sample_dataset.csv', num_samples=100):
    """
    Create a sample dataset for testing purposes.
    
    Args:
        output_path (str): Path to save the sample CSV
        num_samples (int): Number of samples to generate
    """
    import random
    
    # Sample Bangla comments (placeholder)
    sample_comments = [
        "এটি একটি সাধারণ মন্তব্য",
        "তুমি খুব খারাপ",
        "এই পোস্টটি স্প্যাম",
        "ধর্মীয় বিদ্বেষমূলক মন্তব্য",
        "তোমাকে মেরে ফেলব",
    ]
    
    data = []
    for i in range(num_samples):
        comment = random.choice(sample_comments) + f" {i}"
        labels = [random.randint(0, 1) for _ in LABEL_COLUMNS]
        data.append([comment] + labels)
    
    df = pd.DataFrame(data, columns=['comment'] + LABEL_COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to: {output_path}")
    
    return output_path
