# 🔥 Bangla Cyberbullying Detection

[![Model](https://img.shields.io/badge/Model-BanglaBERT-blue)](https://huggingface.co/sagorsarker/bangla-bert-base)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)

A production-ready framework for fine-tuning transformer models on Bangla cyberbullying detection with multi-label classification.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **Mixed Precision Training** | Up to 2x faster training with AMP on GPU |
| 📦 **Dataset Caching** | Skip tokenization on repeated runs |
| 📊 **CSV Export** | Portable metrics without MLflow |
| 🤗 **HuggingFace Integration** | Save/load models in HF format |
| 📤 **One-Click Deploy** | Push to HuggingFace Hub |
| 🔬 **MLflow Tracking** | Full experiment tracking |
| ⚖️ **Class Weighting** | Handle imbalanced datasets |
| 🎯 **Multi-label Support** | 5 cyberbullying categories |
| 🔄 **K-Fold CV** | Robust model evaluation |

## 📋 Labels

| Label | Description |
|-------|-------------|
| `bully` | General bullying content |
| `sexual` | Sexual harassment or inappropriate content |
| `religious` | Religious hate or discrimination |
| `threat` | Threatening content |
| `spam` | Spam or irrelevant content |

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

```python
# 1. Clone the repository
!git clone https://github.com/YOUR_USERNAME/bangla-cyberbullying-detection.git
%cd bangla-cyberbullying-detection

# 2. Install dependencies
!pip install -q torch transformers scikit-learn pandas numpy tqdm mlflow iterative-stratification huggingface_hub

# 3. Upload your dataset or use sample
# (Upload data.csv to Colab files)

# 4. Run training
!python main.py \
    --author_name "your_name" \
    --dataset_path "./data.csv" \
    --batch 32 \
    --lr 2e-5 \
    --epochs 15 \
    --model_path "sagorsarker/bangla-bert-base"
```

### Option 2: Kaggle

```python
# 1. Add the dataset to your Kaggle notebook

# 2. Clone and install
!git clone https://github.com/YOUR_USERNAME/bangla-cyberbullying-detection.git
%cd bangla-cyberbullying-detection
!pip install -q transformers mlflow iterative-stratification huggingface_hub

# 3. Run training (Kaggle has PyTorch pre-installed)
!python main.py \
    --author_name "your_name" \
    --dataset_path "/kaggle/input/your-dataset/data.csv" \
    --batch 32 \
    --lr 2e-5 \
    --epochs 15
```

### Option 3: Local Machine

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/bangla-cyberbullying-detection.git
cd bangla-cyberbullying-detection

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training
python main.py \
    --author_name "your_name" \
    --dataset_path "./data.csv" \
    --batch 32 \
    --lr 2e-5 \
    --epochs 15
```

## 📖 Usage Guide

### Training

#### Basic Training
```bash
python main.py \
    --author_name "your_name" \
    --dataset_path "path/to/data.csv" \
    --batch 32 \
    --lr 2e-5 \
    --epochs 15
```

#### Full Configuration
```bash
python main.py \
    --author_name "saif" \
    --dataset_path "./data.csv" \
    --model_path "sagorsarker/bangla-bert-base" \
    --batch 32 \
    --lr 2e-5 \
    --epochs 15 \
    --num_folds 5 \
    --max_length 128 \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --gradient_clip_norm 1.0 \
    --early_stopping_patience 5 \
    --stratification_type multilabel \
    --seed 42 \
    --save_model_dir "./saved_models" \
    --output_dir "./outputs" \
    --mlflow_experiment_name "Bangla-Cyberbullying"
```

#### Disable Features (for debugging)
```bash
# Disable AMP (use FP32)
python main.py ... --no_amp

# Disable caching
python main.py ... --no_cache

# Disable model saving
python main.py ... --no_save_model
```

#### Push to HuggingFace Hub
```bash
# During training
python main.py ... --push_to_hub --hub_repo_name "username/model-name"

# After training (separate script)
python upload_to_hub.py --model_path ./saved_models/your_model --repo_name "username/model-name"
```

### Inference

#### Single Text
```bash
python inference.py \
    --model_path ./saved_models/your_model \
    --text "আপনার বাংলা টেক্সট এখানে"
```

#### Batch Inference
```bash
python inference.py \
    --model_path ./saved_models/your_model \
    --input_file texts.txt \
    --output_file predictions.csv
```

#### Interactive Mode
```bash
python inference.py \
    --model_path ./saved_models/your_model \
    --interactive
```

#### Python API
```python
from model import TransformerMultiLabelClassifier
from transformers import AutoTokenizer
import torch

# Load model
model = TransformerMultiLabelClassifier.from_pretrained("./saved_models/your_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_models/your_model")

# Predict
text = "আপনার বাংলা টেক্সট"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs = model.predict(inputs['input_ids'], inputs['attention_mask'])

# Get results
labels = ['bully', 'sexual', 'religious', 'threat', 'spam']
probs = outputs['probabilities'][0]
preds = outputs['predictions'][0]

for label, prob, pred in zip(labels, probs, preds):
    print(f"{label}: {prob:.4f} ({'Detected' if pred else 'Not detected'})")
```

## 📁 Project Structure

```
bangla-cyberbullying-detection/
├── main.py              # Entry point for training
├── config.py            # Configuration and argument parsing
├── data.py              # Data loading and preprocessing
├── model.py             # Model definition with HF integration
├── train.py             # Training logic with all features
├── utils.py             # Utility functions
├── inference.py         # Production inference script
├── upload_to_hub.py     # HuggingFace Hub upload script
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── DEPLOYMENT_GUIDE.md  # Deployment documentation
```

## ⚙️ Command-Line Arguments

### Training Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch` | 32 | Batch size |
| `--lr` | 2e-5 | Learning rate |
| `--epochs` | 15 | Maximum epochs |
| `--early_stopping_patience` | 5 | Early stopping patience |

### Model Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | sagorsarker/bangla-bert-base | Pre-trained model |
| `--max_length` | 128 | Max sequence length |
| `--dropout` | 0.1 | Dropout rate |
| `--freeze_base` | False | Freeze base layers |

### Optimizer Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--weight_decay` | 0.01 | Weight decay |
| `--warmup_ratio` | 0.1 | Warmup ratio |
| `--gradient_clip_norm` | 1.0 | Gradient clipping |

### Experiment Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--author_name` | Required | Your name (for tracking) |
| `--num_folds` | 5 | K-fold CV folds |
| `--stratification_type` | multilabel | Stratification method |
| `--seed` | 42 | Random seed |

### Performance Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--no_amp` | False | Disable mixed precision |
| `--no_cache` | False | Disable dataset caching |

### Deployment Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--save_model_dir` | ./saved_models | Model save directory |
| `--no_save_model` | False | Disable model saving |
| `--push_to_hub` | False | Push to HuggingFace Hub |
| `--hub_repo_name` | None | HuggingFace repo name |
| `--hub_private` | False | Make repo private |

## 📊 Dataset Format

Your CSV should have the following columns:

```csv
comment,bully,sexual,religious,threat,spam
"কিছু বাংলা টেক্সট",1,0,0,0,0
"আরেকটি টেক্সট",0,1,0,0,0
...
```

## 📈 Viewing Results

### MLflow UI
```bash
# Navigate to project directory
cd bangla-cyberbullying-detection

# Start MLflow UI
mlflow ui

# Open in browser
# http://localhost:5000
```

### CSV Files
After training, find metrics in:
- `outputs/fold_summary_*.csv` - Per-fold results with mean/std
- `outputs/best_metrics_*.csv` - Best model metrics

### Download from Colab
```python
# Zip and download
!zip -r results.zip ./outputs ./saved_models ./mlruns
# Then download from Colab files panel
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Experiment Contribution
1. Pick an experiment configuration
2. Run with your name: `--author_name "your_name"`
3. Share results via PR or Issue

## 📝 Citation

```bibtex
@misc{bangla-cyberbullying-detection,
  author = {Your Name},
  title = {Bangla Cyberbullying Detection with Multi-Label Classification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/bangla-cyberbullying-detection}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BanglaBERT](https://huggingface.co/sagorsarker/bangla-bert-base) by Sagor Sarker
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [MLflow](https://mlflow.org/)
- [iterative-stratification](https://github.com/trent-b/iterative-stratification)
