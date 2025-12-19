# 🚀 Deployment Guide

This guide covers deploying your trained Bangla Cyberbullying Detection model to production.

## Table of Contents

1. [Model Output Structure](#model-output-structure)
2. [HuggingFace Hub Deployment](#huggingface-hub-deployment)
3. [Local API Deployment](#local-api-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Optimization Tips](#optimization-tips)

---

## Model Output Structure

After training, your model is saved in HuggingFace-compatible format:

```
saved_models/
└── your_model_fold1_f1_0.8500/
    ├── classifier_config.json    # Model configuration
    ├── pytorch_model.bin         # Complete model weights
    ├── classifier_head.pt        # Classifier head only
    ├── encoder/                  # Base transformer encoder
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── ...
    ├── tokenizer_config.json     # Tokenizer configuration
    ├── vocab.txt                 # Vocabulary file
    ├── special_tokens_map.json   # Special tokens
    ├── training_metrics.json     # Best fold metrics
    ├── training_config.json      # Training hyperparameters
    └── README.md                 # Auto-generated model card
```

---

## HuggingFace Hub Deployment

### Option 1: During Training

```bash
python main.py \
    --author_name "your_name" \
    --dataset_path "data.csv" \
    --push_to_hub \
    --hub_repo_name "your-username/bangla-cyberbully-detector" \
    --hub_private  # Optional: make repo private
```

### Option 2: After Training (Recommended)

```bash
# 1. Login to HuggingFace
huggingface-cli login

# 2. Upload using provided script
python upload_to_hub.py \
    --model_path ./saved_models/your_model \
    --repo_name "your-username/bangla-cyberbully-detector"
```

### Option 3: Using huggingface-cli

```bash
huggingface-cli upload your-username/model-name ./saved_models/your_model
```

### Option 4: Python API

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./saved_models/your_model",
    repo_id="your-username/bangla-cyberbully-detector",
    commit_message="Upload fine-tuned model"
)
```

---

## Local API Deployment

### FastAPI Server

Create `api.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer
from model import TransformerMultiLabelClassifier, LABEL_COLUMNS

app = FastAPI(
    title="Bangla Cyberbullying Detection API",
    description="Detect cyberbullying in Bangla text",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = "./saved_models/your_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
model = TransformerMultiLabelClassifier.from_pretrained(MODEL_PATH, device=device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"Model loaded on {device}")


class TextInput(BaseModel):
    text: str
    threshold: Optional[float] = 0.5


class BatchInput(BaseModel):
    texts: List[str]
    threshold: Optional[float] = 0.5


class PredictionResult(BaseModel):
    text: str
    predictions: dict
    detected_labels: List[str]


@app.get("/")
def root():
    return {"message": "Bangla Cyberbullying Detection API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "device": device}


@app.post("/predict", response_model=PredictionResult)
def predict(input: TextInput):
    """Predict cyberbullying labels for a single text."""
    try:
        inputs = tokenizer(
            input.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.predict(
            inputs['input_ids'], 
            inputs['attention_mask'],
            threshold=input.threshold
        )
        
        probs = outputs['probabilities'][0].cpu().tolist()
        preds = outputs['predictions'][0].cpu().tolist()
        
        predictions = {}
        detected = []
        
        for label, prob, pred in zip(LABEL_COLUMNS, probs, preds):
            predictions[label] = {
                "probability": round(prob, 4),
                "detected": bool(pred)
            }
            if pred:
                detected.append(label)
        
        return PredictionResult(
            text=input.text,
            predictions=predictions,
            detected_labels=detected
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResult])
def predict_batch(input: BatchInput):
    """Predict cyberbullying labels for multiple texts."""
    results = []
    
    for text in input.texts:
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.predict(
                inputs['input_ids'],
                inputs['attention_mask'],
                threshold=input.threshold
            )
            
            probs = outputs['probabilities'][0].cpu().tolist()
            preds = outputs['predictions'][0].cpu().tolist()
            
            predictions = {}
            detected = []
            
            for label, prob, pred in zip(LABEL_COLUMNS, probs, preds):
                predictions[label] = {
                    "probability": round(prob, 4),
                    "detected": bool(pred)
                }
                if pred:
                    detected.append(label)
            
            results.append(PredictionResult(
                text=text,
                predictions=predictions,
                detected_labels=detected
            ))
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run the API

```bash
# Install dependencies
pip install fastapi uvicorn

# Run server
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000

# API available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "আপনার বাংলা টেক্সট", "threshold": 0.5}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["টেক্সট ১", "টেক্সট ২"], "threshold": 0.5}'
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# Copy application code
COPY *.py .
COPY saved_models/ ./saved_models/

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t bangla-cyberbully-api .

# Run container
docker run -p 8000:8000 bangla-cyberbully-api

# Run with GPU (requires nvidia-docker)
docker run --gpus all -p 8000:8000 bangla-cyberbully-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Cloud Deployment

### AWS Lambda (Serverless)

For serverless deployment, consider using AWS Lambda with a container image or SageMaker for larger models.

### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/bangla-cyberbully-api

# Deploy to Cloud Run
gcloud run deploy bangla-cyberbully-api \
    --image gcr.io/PROJECT_ID/bangla-cyberbully-api \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2
```

### HuggingFace Inference Endpoints

1. Push model to HuggingFace Hub (see above)
2. Go to your model page on HuggingFace
3. Click "Deploy" → "Inference Endpoints"
4. Configure your endpoint:
   - Instance type: CPU or GPU
   - Region: Choose based on latency needs
   - Scaling: Auto-scaling configuration
5. Use the provided API endpoint

---

## Optimization Tips

### 1. Model Quantization

Reduce model size and improve inference speed:

```python
import torch

# Dynamic quantization (CPU)
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(model_quantized.state_dict(), "model_quantized.bin")
```

### 2. ONNX Export

Convert to ONNX for cross-platform deployment:

```python
import torch.onnx

# Dummy input
dummy_input = {
    'input_ids': torch.randint(0, 1000, (1, 128)),
    'attention_mask': torch.ones(1, 128, dtype=torch.long)
}

# Export
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch'}
    }
)
```

### 3. Batching for Production

For high-throughput scenarios, batch requests:

```python
from typing import List
import asyncio

class BatchPredictor:
    def __init__(self, model, tokenizer, batch_size=32, max_wait_ms=100):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
    
    async def predict(self, text: str):
        future = asyncio.Future()
        self.queue.append((text, future))
        
        if len(self.queue) >= self.batch_size:
            await self._process_batch()
        else:
            await asyncio.sleep(self.max_wait_ms / 1000)
            if self.queue:
                await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        if not self.queue:
            return
        
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        texts = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        
        # Process batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.predict(inputs['input_ids'], inputs['attention_mask'])
        
        # Set results
        for i, future in enumerate(futures):
            future.set_result(outputs['predictions'][i])
```

### 4. Caching

For repeated queries, implement caching:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def cached_predict(text_hash: str, threshold: float):
    # This would be called with hash of text
    pass

def predict_with_cache(text: str, threshold: float = 0.5):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return cached_predict(text_hash, threshold)
```

---

## Monitoring

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(input: TextInput):
    logger.info(f"Prediction request received: {len(input.text)} chars")
    # ... prediction logic ...
    logger.info(f"Prediction completed: {detected_labels}")
```

### Metrics (Prometheus)

```python
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
def predict(input: TextInput):
    REQUEST_COUNT.inc()
    with LATENCY.time():
        # ... prediction logic ...

@app.get("/metrics")
def metrics():
    return generate_latest()
```

---

## Security

### Input Validation

```python
from pydantic import BaseModel, validator

class TextInput(BaseModel):
    text: str
    threshold: float = 0.5
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 10000:
            raise ValueError('Text too long (max 10000 chars)')
        return v.strip()
    
    @validator('threshold')
    def valid_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        return v
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
def predict(input: TextInput):
    # ... prediction logic ...
```

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow inference**: Enable GPU, use batching, or quantize model
3. **Model not found**: Check path and ensure all files are present
4. **Tokenizer errors**: Ensure tokenizer files match the model

### Debug Mode

```bash
# Run with debug logging
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from model import TransformerMultiLabelClassifier
model = TransformerMultiLabelClassifier.from_pretrained('./saved_models/your_model')
"
```
