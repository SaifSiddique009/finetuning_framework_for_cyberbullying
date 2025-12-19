#!/usr/bin/env python3
"""
HuggingFace Hub Upload Script
Upload your trained Bangla Cyberbullying Detection model to HuggingFace Hub

Usage:
    python upload_to_hub.py --model_path ./saved_models/your_model --repo_name username/model-name

Prerequisites:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Login to HuggingFace: huggingface-cli login
"""

import argparse
import os
from pathlib import Path


def upload_to_huggingface(model_path: str, repo_name: str, private: bool = False, commit_message: str = None):
    """
    Upload a saved model directory to HuggingFace Hub.
    
    Args:
        model_path: Path to the saved model directory
        repo_name: HuggingFace repository name (e.g., 'username/model-name')
        private: Whether to create a private repository
        commit_message: Custom commit message
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ Error: huggingface_hub not installed")
        print("   Install with: pip install huggingface_hub")
        return False
    
    # Validate model path
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        return False
    
    # Check for required files
    required_files = ['classifier_config.json', 'pytorch_model.bin']
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    
    if missing_files:
        print(f"❌ Error: Required files missing: {missing_files}")
        print("   Make sure you're pointing to a valid model directory.")
        return False
    
    api = HfApi()
    
    # Check if user is logged in
    try:
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception:
        print("❌ Not logged in to HuggingFace Hub")
        print("   Run: huggingface-cli login")
        return False
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_name, private=private, exist_ok=True)
        visibility = "private" if private else "public"
        print(f"✅ Repository ready ({visibility}): https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"⚠️  Repository creation note: {e}")
    
    # Upload all files
    print(f"\n📤 Uploading model from {model_path}...")
    
    if commit_message is None:
        commit_message = "Upload Bangla Cyberbullying Detection model"
    
    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            commit_message=commit_message
        )
        
        print(f"\n✅ Model uploaded successfully!")
        print(f"   View at: https://huggingface.co/{repo_name}")
        print(f"\n📝 To use this model:")
        print(f"   from model import TransformerMultiLabelClassifier")
        print(f"   from transformers import AutoTokenizer")
        print(f"")
        print(f"   model = TransformerMultiLabelClassifier.from_pretrained('{repo_name}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{repo_name}')")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload to a public repository
    python upload_to_hub.py --model_path ./saved_models/best_model --repo_name username/bangla-cyberbully
    
    # Upload to a private repository
    python upload_to_hub.py --model_path ./saved_models/best_model --repo_name username/bangla-cyberbully --private
    
    # Custom commit message
    python upload_to_hub.py --model_path ./saved_models/best_model --repo_name username/model --message "v2.0 - improved F1"

Before uploading:
    1. Make sure you're logged in: huggingface-cli login
    2. Your model directory should contain:
       - classifier_config.json
       - pytorch_model.bin
       - encoder/ directory
       - tokenizer files (optional but recommended)
       - README.md (auto-generated model card)
        """
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to the saved model directory'
    )
    
    parser.add_argument(
        '--repo_name', 
        type=str, 
        required=True,
        help='HuggingFace repository name (e.g., username/model-name)'
    )
    
    parser.add_argument(
        '--private', 
        action='store_true',
        help='Create a private repository'
    )
    
    parser.add_argument(
        '--message',
        type=str,
        default=None,
        help='Custom commit message'
    )
    
    args = parser.parse_args()
    
    success = upload_to_huggingface(
        args.model_path, 
        args.repo_name, 
        args.private,
        args.message
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
