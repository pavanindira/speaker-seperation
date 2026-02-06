#!/usr/bin/env python3
"""
Speaker Separator with .env Configuration Support
Improved version that reads settings from .env file
"""

import os
import sys
from pathlib import Path

# Load configuration first
try:
    from config import config, setup_huggingface_auth
    print(f"✓ Loaded config from .env")
    print(f"  Ollama: {config.OLLAMA_URL}")
    print(f"  Model: {config.OLLAMA_MODEL}")
except ImportError:
    print("⚠️  config.py not found - using defaults")
    print("   Create .env file for custom configuration")
    
    # Fallback config class
    class Config:
        OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
        HF_TOKEN = os.getenv('HF_TOKEN', '')
        DEFAULT_NUM_SPEAKERS = int(os.getenv('DEFAULT_NUM_SPEAKERS', '2'))
        DEFAULT_CLUSTERING_METHOD = os.getenv('DEFAULT_CLUSTERING_METHOD', 'gmm')
        MFCC_COEFFICIENTS = int(os.getenv('MFCC_COEFFICIENTS', '20'))
        FRAME_LENGTH = int(os.getenv('FRAME_LENGTH', '2048'))
        HOP_LENGTH = int(os.getenv('HOP_LENGTH', '512'))
    
    config = Config()
    
    def setup_huggingface_auth():
        if config.HF_TOKEN:
            os.environ['HF_TOKEN'] = config.HF_TOKEN
            return True
        return False

# Now import the improved separator
from improved_speaker_separator import ImprovedSpeakerSeparator, analyze_separation_quality

# Re-export with config defaults
def main():
    """Main with .env configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Speaker Separation with .env Configuration"
    )
    parser.add_argument("audio_file", nargs="?", help="Audio file path")
    parser.add_argument(
        "-n", "--num-speakers",
        type=int,
        default=config.DEFAULT_NUM_SPEAKERS,
        help=f"Number of speakers (default from .env: {config.DEFAULT_NUM_SPEAKERS})"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["gmm", "kmeans", "spectral"],
        default=config.DEFAULT_CLUSTERING_METHOD,
        help=f"Clustering method (default from .env: {config.DEFAULT_CLUSTERING_METHOD})"
    )
    parser.add_argument("-o", "--output-dir", default="separated_speakers")
    parser.add_argument("--create-test", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    # Show configuration
    if args.show_config:
        if hasattr(config, 'print_config'):
            config.print_config()
        else:
            print("\nCurrent Configuration:")
            print(f"  Ollama URL: {config.OLLAMA_URL}")
            print(f"  Ollama Model: {config.OLLAMA_MODEL}")
            print(f"  Default Speakers: {config.DEFAULT_NUM_SPEAKERS}")
            print(f"  Clustering Method: {config.DEFAULT_CLUSTERING_METHOD}")
            print(f"  HF Token: {'Set' if config.HF_TOKEN else 'Not set'}")
        return
    
    # Create test
    if args.create_test:
        from improved_speaker_separator import create_test_audio
        create_test_audio()
        return
    
    # Process audio
    if not args.audio_file:
        parser.print_help()
        print("\n💡 Tip: Create .env file to customize settings")
        print("   Copy .env.template to .env and edit")
        return
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {args.audio_file}")
        return
    
    output_dir = Path(args.output_dir) / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    separator = ImprovedSpeakerSeparator(output_dir)
    
    try:
        results = separator.separate_speakers(
            audio_path,
            n_speakers=args.num_speakers,
            method=args.method
        )
        
        if args.analyze:
            analyze_separation_quality(output_dir, args.num_speakers)
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
