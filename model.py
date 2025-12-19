"""
Generic Transformer-based Multi-Label Classifier
Supports any transformer model (BERT, RoBERTa, XLM-RoBERTa, etc.) through AutoModel

Features:
- HuggingFace-compatible save/load (save_pretrained, from_pretrained)
- Inference with configurable threshold (predict method)
- Layer freezing for feature extraction
- Automatic model card generation for HuggingFace Hub
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer


# Label columns for the cyberbullying detection task
LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']


class TransformerMultiLabelClassifier(nn.Module):
    """
    Generic transformer-based multi-label classifier.
    Works with any transformer model from HuggingFace (BERT, RoBERTa, XLM-RoBERTa, etc.)
    """
    
    def __init__(self, model_name, num_labels, dropout=0.1, classifier_hidden_size=256):
        """
        Initialize the multi-label classifier.
        
        Args:
            model_name (str): Name or path of pre-trained transformer model
            num_labels (int): Number of labels for multi-label classification
            dropout (float): Dropout rate for regularization
            classifier_hidden_size (int): Hidden size for classifier intermediate layer
        """
        super(TransformerMultiLabelClassifier, self).__init__()
        
        # Store configuration for later saving
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_rate = dropout
        self.classifier_hidden_size = classifier_hidden_size
        
        # Auto-detect and load any transformer model
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get the hidden size from the model's config
        self.encoder_config = AutoConfig.from_pretrained(model_name)
        hidden_size = self.encoder_config.hidden_size
        
        # Classification head with intermediate layer for better feature extraction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels)
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            labels: Ground truth labels (optional, for loss calculation)
            
        Returns:
            dict: Dictionary containing loss (if labels provided) and logits
        """
        # Get encoder outputs (works for BERT, RoBERTa, XLM-RoBERTa, etc.)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract CLS token representation (first token)
        # This pattern works for most transformer models
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier(cls_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}
    
    def predict(self, input_ids, attention_mask=None, threshold=0.5):
        """
        Make predictions with sigmoid activation and thresholding.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            threshold: Classification threshold (default 0.5)
            
        Returns:
            dict: Dictionary containing probabilities and binary predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs['logits'])
            predictions = (probabilities > threshold).int()
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'logits': outputs['logits']
        }
    
    def freeze_base_layers(self):
        """
        Freeze encoder parameters for feature extraction.
        This prevents updating the pre-trained weights during training.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Count frozen parameters for logging
        frozen_params = sum(p.numel() for p in self.encoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n🔒 Frozen {frozen_params:,} parameters out of {total_params:,} total parameters")
        print(f"   Trainable parameters: {total_params - frozen_params:,}")
    
    def unfreeze_base_layers(self):
        """
        Unfreeze encoder parameters (useful for fine-tuning after feature extraction).
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n🔓 All parameters unfrozen. Trainable parameters: {trainable_params:,}")

    def save_pretrained(self, save_directory, tokenizer=None):
        """
        Save model in HuggingFace-compatible format for deployment.
        
        Args:
            save_directory (str): Directory to save the model
            tokenizer: Optional tokenizer to save alongside the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. Save the classifier configuration
        classifier_config = {
            "base_model_name": self.model_name,
            "num_labels": self.num_labels,
            "dropout": self.dropout_rate,
            "classifier_hidden_size": self.classifier_hidden_size,
            "label_names": LABEL_COLUMNS,
            "model_type": "transformer_multilabel_classifier",
            "architectures": ["TransformerMultiLabelClassifier"]
        }
        
        config_path = os.path.join(save_directory, "classifier_config.json")
        with open(config_path, 'w') as f:
            json.dump(classifier_config, f, indent=2)
        
        # 2. Save the base encoder with its original config
        encoder_dir = os.path.join(save_directory, "encoder")
        self.encoder.save_pretrained(encoder_dir)
        
        # 3. Save the classifier head weights separately
        classifier_weights = {
            'classifier': self.classifier.state_dict()
        }
        classifier_path = os.path.join(save_directory, "classifier_head.pt")
        torch.save(classifier_weights, classifier_path)
        
        # 4. Save complete model state dict (for easy loading)
        full_model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), full_model_path)
        
        # 5. Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)
        
        # 6. Create model card
        self._create_model_card(save_directory)
        
        print(f"\n✅ Model saved to {save_directory}")
        print(f"   📄 classifier_config.json: Model configuration")
        print(f"   📁 encoder/: Base transformer encoder")
        print(f"   📄 classifier_head.pt: Classification head weights")
        print(f"   📄 pytorch_model.bin: Complete model state dict")
        if tokenizer:
            print(f"   📄 tokenizer files: Tokenizer configuration")
    
    def _create_model_card(self, save_directory):
        """Create a model card (README.md) for HuggingFace Hub."""
        model_card = f"""---
language:
  - bn
license: mit
tags:
  - text-classification
  - multi-label-classification
  - bangla
  - cyberbullying
  - bert
  - pytorch
datasets:
  - custom
metrics:
  - f1
  - accuracy
pipeline_tag: text-classification
---

# Bangla Cyberbullying Detection Model

This model is fine-tuned for multi-label classification to detect cyberbullying in Bangla text.

## Model Details

- **Base Model:** {self.model_name}
- **Task:** Multi-label text classification
- **Labels:** {', '.join(LABEL_COLUMNS)}
- **Number of Labels:** {self.num_labels}
- **Classifier Hidden Size:** {self.classifier_hidden_size}
- **Dropout:** {self.dropout_rate}

## Usage

### Installation

```bash
pip install torch transformers
```

### Loading and Inference

```python
from model import TransformerMultiLabelClassifier
from transformers import AutoTokenizer
import torch

# Load the model
model = TransformerMultiLabelClassifier.from_pretrained("path/to/saved/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/model")

# Prepare input
text = "আপনার বাংলা টেক্সট এখানে"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

# Get predictions
outputs = model.predict(inputs['input_ids'], inputs['attention_mask'])
    
probabilities = outputs['probabilities'][0]
predictions = outputs['predictions'][0]

labels = ['bully', 'sexual', 'religious', 'threat', 'spam']
for label, prob, pred in zip(labels, probabilities, predictions):
    status = "✓ Detected" if pred else "✗ Not detected"
    print(f"{{label}}: {{prob:.4f}} ({{status}})")
```

### Using with Pipeline (Alternative)

```python
# For batch inference
texts = ["টেক্সট ১", "টেক্সট ২", "টেক্সট ৩"]
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs = model.predict(inputs['input_ids'], inputs['attention_mask'])
```

## Labels

| Label | Description |
|-------|-------------|
| bully | General bullying content |
| sexual | Sexual harassment or inappropriate content |
| religious | Religious hate or discrimination |
| threat | Threatening content |
| spam | Spam or irrelevant content |

## Training

This model was trained using:
- K-fold cross-validation with multi-label stratification
- AdamW optimizer with linear warmup
- Mixed precision training (AMP)
- Early stopping based on weighted F1 score

## Citation

If you use this model, please cite:

```bibtex
@misc{{bangla-cyberbullying-detection,
  author = {{Your Name}},
  title = {{Bangla Cyberbullying Detection Model}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/your-username/your-model}}
}}
```

## Limitations

- Trained specifically on Bangla text
- Performance may vary on out-of-domain text
- Multi-label threshold of 0.5 used by default (can be adjusted)
- May not generalize well to code-mixed text (Bangla + English)
"""
        readme_path = os.path.join(save_directory, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
    
    @classmethod
    def from_pretrained(cls, model_directory, device=None):
        """
        Load a model from a saved directory or HuggingFace Hub.
        
        Args:
            model_directory (str): Directory containing saved model or HuggingFace repo ID
            device: Device to load the model to (optional)
            
        Returns:
            TransformerMultiLabelClassifier: Loaded model
        """
        # Load classifier configuration
        config_path = os.path.join(model_directory, "classifier_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if encoder is saved separately or we should use base model name
        encoder_dir = os.path.join(model_directory, "encoder")
        if os.path.exists(encoder_dir):
            model_name = encoder_dir
        else:
            model_name = config['base_model_name']
        
        # Create model instance
        model = cls(
            model_name=model_name,
            num_labels=config['num_labels'],
            dropout=config['dropout'],
            classifier_hidden_size=config.get('classifier_hidden_size', 256)
        )
        
        # Load full state dict if available
        full_model_path = os.path.join(model_directory, "pytorch_model.bin")
        if os.path.exists(full_model_path):
            state_dict = torch.load(full_model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
        else:
            # Load classifier head separately
            classifier_path = os.path.join(model_directory, "classifier_head.pt")
            if os.path.exists(classifier_path):
                classifier_weights = torch.load(classifier_path, map_location='cpu', weights_only=True)
                model.classifier.load_state_dict(classifier_weights['classifier'])
        
        if device:
            model = model.to(device)
        
        model.eval()
        print(f"\n✅ Model loaded from {model_directory}")
        
        return model


def save_model_for_huggingface(model, tokenizer, save_path, model_metrics=None, training_config=None):
    """
    Convenience function to save model with all necessary files for HuggingFace Hub.
    
    Args:
        model: TransformerMultiLabelClassifier instance
        tokenizer: HuggingFace tokenizer
        save_path (str): Directory to save the model
        model_metrics (dict): Optional metrics to include in model card
        training_config: Optional training configuration
    """
    model.save_pretrained(save_path, tokenizer=tokenizer)
    
    # Save training metrics
    if model_metrics:
        metrics_path = os.path.join(save_path, "training_metrics.json")
        # Convert any non-serializable types
        serializable_metrics = {}
        for k, v in model_metrics.items():
            if isinstance(v, (int, float, str, bool)):
                serializable_metrics[k] = v
            else:
                serializable_metrics[k] = str(v)
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"   📄 training_metrics.json: Training metrics")
    
    # Save training config
    if training_config:
        config_dict = vars(training_config) if hasattr(training_config, '__dict__') else training_config
        train_config_path = os.path.join(save_path, "training_config.json")
        
        # Convert any non-serializable types
        serializable_config = {}
        for k, v in config_dict.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
        
        with open(train_config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        print(f"   📄 training_config.json: Training configuration")


def get_label_columns():
    """Return the list of label column names."""
    return LABEL_COLUMNS.copy()
