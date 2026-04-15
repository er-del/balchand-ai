"""
Setup script to download and configure PIXEL model from HuggingFace Hub.

Usage:
    python setup_hf_model.py                          # Download to local directory
    python setup_hf_model.py --repo-id sage002/pixel  # Use custom repo
    python setup_hf_model.py --cache-dir ./models     # Use custom cache directory
"""

import argparse
from pathlib import Path
import json

try:
    from huggingface_hub import hf_hub_download, repo_info
except ImportError:
    print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
    exit(1)


def download_model(repo_id: str, cache_dir: str | None = None) -> dict:
    """Download PIXEL model files from HuggingFace."""
    
    files_to_download = {
        "checkpoint": "latest.pt",
        "tokenizer_model": "pixel_tokenizer.model",
        "tokenizer_vocab": "pixel_tokenizer.vocab",
        "model_config": "pixel_model_config.json",
        "training_config": "pixel_training_config.json",
        "manifest": "manifest.json",
    }
    
    print(f"📥 Downloading PIXEL model from {repo_id}...")
    print("-" * 60)
    
    downloaded_files = {}
    
    for key, filename in files_to_download.items():
        try:
            print(f"  ⏳ Downloading {filename}...", end=" ", flush=True)
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_folder=cache_dir,
                repo_type="model"
            )
            downloaded_files[key] = str(path)
            print("✓")
        except Exception as e:
            print(f"✗\n     Error: {e}")
    
    print("-" * 60)
    print(f"✅ Downloaded {len(downloaded_files)}/{len(files_to_download)} files\n")
    
    return downloaded_files


def load_model_info(model_config_path: str) -> dict:
    """Load and display model configuration."""
    try:
        with open(model_config_path, 'r') as f:
            config = json.load(f)
        
        print("📋 Model Configuration:")
        print("-" * 60)
        for key, value in config.items():
            if not isinstance(value, (dict, list)):
                print(f"  {key}: {value}")
        print()
        
        return config
    except Exception as e:
        print(f"Warning: Could not load config: {e}\n")
        return {}


def create_inference_example(files: dict, output_file: str = "inference_example.py") -> None:
    """Create example inference script."""
    
    template = '''"""
Example: Load PIXEL model from HuggingFace and generate text.

This script shows how to use the downloaded model files.
"""

from pathlib import Path
from core.checkpoint import CheckpointManager
from tokenizer.manager import PixelTokenizer
from inference.generator import PixelGenerator
from core.types import GenerationRequest
import json

# Paths to downloaded files
CHECKPOINT = "{checkpoint}"
TOKENIZER_MODEL = "{tokenizer_model}"
MODEL_CONFIG = "{model_config}"

def main():
    # Load model config
    with open(MODEL_CONFIG, 'r') as f:
        config_dict = json.load(f)
    
    print("Loading PIXEL model...")
    
    # Load checkpoint
    checkpoint_dir = Path(CHECKPOINT).parent
    manager = CheckpointManager(checkpoint_dir, create=False)
    checkpoint_info = manager.inspect(CHECKPOINT, device="cpu")
    
    # Load tokenizer
    tokenizer = PixelTokenizer.from_pretrained(TOKENIZER_MODEL)
    
    # Create generator
    generator = PixelGenerator(
        checkpoint_info.model_config,
        tokenizer,
        checkpoint_path=CHECKPOINT,
        checkpoint_info=checkpoint_info,
    )
    
    print("✅ Model loaded successfully!\\n")
    
    # Generate text
    prompts = [
        "Write a short paragraph about machine learning.",
        "Explain quantum computing in simple terms.",
        "What is artificial intelligence?",
    ]
    
    for prompt in prompts:
        print(f"📝 Prompt: {prompt}")
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=96,
            temperature=0.8,
            top_p=0.95,
        )
        response = generator.generate(request)
        print(f"✨ Generated: {response.output}\\n")

if __name__ == "__main__":
    main()
'''
    
    script_content = template.format(
        checkpoint=files.get("checkpoint", "path/to/latest.pt"),
        tokenizer_model=files.get("tokenizer_model", "path/to/tokenizer.model"),
        model_config=files.get("model_config", "path/to/model_config.json"),
    )
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    print(f"📄 Created inference example: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download PIXEL model from HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id",
        default="sage002/pixel",
        help="HuggingFace repository ID (default: sage002/pixel)"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for downloaded files (default: HuggingFace default cache)"
    )
    parser.add_argument(
        "--no-example",
        action="store_true",
        help="Skip creating example inference script"
    )
    
    args = parser.parse_args()
    
    # Download model
    files = download_model(args.repo_id, args.cache_dir)
    
    if not files:
        print("❌ No files downloaded. Check your internet connection and repo-id.")
        return
    
    # Show model info
    if "model_config" in files:
        load_model_info(files["model_config"])
    
    # Create example script
    if not args.no_example:
        create_inference_example(files)
    
    # Print usage instructions
    print("🚀 Next steps:")
    print("-" * 60)
    if not args.no_example:
        print("  1. Review inference_example.py")
        print("  2. Run: python inference_example.py")
    print("  3. Or use the downloaded files directly in your code")
    print()
    print("📂 Downloaded files:")
    for key, path in files.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
