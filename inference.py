#!/usr/bin/env python3
"""
Inference Script for Bangla Cyberbullying Detection Model
Load a trained model and make predictions on new text

Usage:
    # Single text prediction
    python inference.py --model_path ./saved_models/your_model --text "আপনার টেক্সট"
    
    # Batch inference from file
    python inference.py --model_path ./saved_models/your_model --input_file texts.txt --output_file predictions.csv
    
    # Interactive mode
    python inference.py --model_path ./saved_models/your_model --interactive
"""

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

from model import TransformerMultiLabelClassifier, LABEL_COLUMNS


def load_model_and_tokenizer(model_path: str, device: str = None):
    """
    Load the model and tokenizer from a saved directory or HuggingFace Hub.
    
    Args:
        model_path: Path to model directory or HuggingFace repo name
        device: Device to load model on ('cuda', 'cpu', or None for auto)
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📦 Loading model from: {model_path}")
    print(f"   Device: {device}")
    
    # Load model
    model = TransformerMultiLabelClassifier.from_pretrained(model_path, device=device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"   ✓ Model and tokenizer loaded successfully")
    
    return model, tokenizer, device


def predict_single(model, tokenizer, text: str, device: str, threshold: float = 0.5, max_length: int = 128):
    """
    Make prediction for a single text.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        text: Input text
        device: Device to run on
        threshold: Classification threshold
        max_length: Maximum sequence length
    
    Returns:
        dict: Prediction results with probabilities and labels
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    outputs = model.predict(
        inputs['input_ids'],
        inputs['attention_mask'],
        threshold=threshold
    )
    
    # Process results
    probs = outputs['probabilities'][0].cpu().numpy()
    preds = outputs['predictions'][0].cpu().numpy()
    
    results = {
        'text': text,
        'predictions': {},
        'detected_labels': []
    }
    
    for label, prob, pred in zip(LABEL_COLUMNS, probs, preds):
        results['predictions'][label] = {
            'probability': float(prob),
            'detected': bool(pred)
        }
        if pred:
            results['detected_labels'].append(label)
    
    return results


def predict_batch(model, tokenizer, texts: list, device: str, threshold: float = 0.5, 
                  batch_size: int = 32, max_length: int = 128):
    """
    Make predictions for multiple texts.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        texts: List of input texts
        device: Device to run on
        threshold: Classification threshold
        batch_size: Batch size for inference
        max_length: Maximum sequence length
    
    Returns:
        list: List of prediction results
    """
    from tqdm import tqdm
    
    all_results = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        outputs = model.predict(
            inputs['input_ids'],
            inputs['attention_mask'],
            threshold=threshold
        )
        
        # Process batch results
        probs = outputs['probabilities'].cpu().numpy()
        preds = outputs['predictions'].cpu().numpy()
        
        for text, prob_row, pred_row in zip(batch_texts, probs, preds):
            result = {
                'text': text,
                'predictions': {},
                'detected_labels': []
            }
            
            for label, prob, pred in zip(LABEL_COLUMNS, prob_row, pred_row):
                result['predictions'][label] = {
                    'probability': float(prob),
                    'detected': bool(pred)
                }
                if pred:
                    result['detected_labels'].append(label)
            
            all_results.append(result)
    
    return all_results


def format_output(result: dict, verbose: bool = True):
    """Format prediction result for display."""
    output_lines = []
    
    text_display = result['text'][:100] + '...' if len(result['text']) > 100 else result['text']
    output_lines.append(f"\n📝 Text: {text_display}")
    output_lines.append("-" * 60)
    
    if result['detected_labels']:
        labels_str = ', '.join(result['detected_labels'])
        output_lines.append(f"⚠️  Detected: {labels_str}")
    else:
        output_lines.append("✅ No cyberbullying detected")
    
    if verbose:
        output_lines.append("\n   Label Probabilities:")
        for label, data in result['predictions'].items():
            indicator = "🔴" if data['detected'] else "⚪"
            output_lines.append(f"   {indicator} {label}: {data['probability']:.4f}")
    
    return "\n".join(output_lines)


def interactive_mode(model, tokenizer, device, threshold=0.5):
    """Run interactive prediction mode."""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MODE")
    print("="*60)
    print("Enter text to analyze (type 'quit' or 'exit' to stop):")
    print("-"*60)
    
    while True:
        try:
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not text:
                print("   Please enter some text.")
                continue
            
            result = predict_single(model, tokenizer, text, device, threshold)
            print(format_output(result, verbose=True))
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions with trained Bangla Cyberbullying Detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single text prediction
    python inference.py --model_path ./saved_models/model --text "আপনার টেক্সট"
    
    # Batch inference
    python inference.py --model_path ./saved_models/model --input_file texts.txt --output_file predictions.csv
    
    # Interactive mode
    python inference.py --model_path ./saved_models/model --interactive
    
    # Adjust threshold
    python inference.py --model_path ./saved_models/model --text "টেক্সট" --threshold 0.3
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model directory or HuggingFace repo name'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Single text to classify'
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        help='Path to file with texts (one per line) for batch prediction'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to save predictions (CSV or JSON format based on extension)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum sequence length (default: 128)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and not args.input_file and not args.interactive:
        parser.error("One of --text, --input_file, or --interactive is required")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, device, args.threshold)
        return
    
    # Single text prediction
    if args.text:
        result = predict_single(model, tokenizer, args.text, device, args.threshold, args.max_length)
        print(format_output(result, verbose=not args.quiet))
        
        if args.output_file:
            if args.output_file.endswith('.json'):
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                # CSV format
                row = {'text': result['text'], 'detected_labels': ','.join(result['detected_labels'])}
                for label in LABEL_COLUMNS:
                    row[f'{label}_prob'] = result['predictions'][label]['probability']
                    row[f'{label}_detected'] = result['predictions'][label]['detected']
                pd.DataFrame([row]).to_csv(args.output_file, index=False, encoding='utf-8')
            print(f"\n💾 Results saved to: {args.output_file}")
    
    # Batch prediction
    elif args.input_file:
        # Read texts
        print(f"\n📂 Reading texts from: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"   Found {len(texts)} texts to process")
        
        results = predict_batch(
            model, tokenizer, texts, device, 
            args.threshold, args.batch_size, args.max_length
        )
        
        # Display results (if not too many)
        if len(results) <= 10 and not args.quiet:
            for result in results:
                print(format_output(result, verbose=not args.quiet))
        else:
            # Summary
            total_detected = sum(1 for r in results if r['detected_labels'])
            print(f"\n📊 Summary:")
            print(f"   Total texts: {len(results)}")
            print(f"   Texts with cyberbullying: {total_detected} ({total_detected/len(results)*100:.1f}%)")
            
            # Per-label stats
            print(f"\n   Per-label detection:")
            for label in LABEL_COLUMNS:
                count = sum(1 for r in results if r['predictions'][label]['detected'])
                print(f"   • {label}: {count} ({count/len(results)*100:.1f}%)")
        
        # Save to file
        if args.output_file:
            if args.output_file.endswith('.json'):
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                # CSV format
                rows = []
                for result in results:
                    row = {'text': result['text'], 'detected_labels': ','.join(result['detected_labels'])}
                    for label in LABEL_COLUMNS:
                        row[f'{label}_prob'] = result['predictions'][label]['probability']
                        row[f'{label}_detected'] = result['predictions'][label]['detected']
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(args.output_file, index=False, encoding='utf-8')
            
            print(f"\n💾 Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
