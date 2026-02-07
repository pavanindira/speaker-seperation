#!/usr/bin/env python3
"""
Configuration Management
Loads settings from .env file and environment variables
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("    Falling back to environment variables only.")


class Config:
    """Application configuration"""
    
    def __init__(self):
        """Load configuration from .env file and environment"""
        
        # Try to load .env file
        if DOTENV_AVAILABLE:
            env_path = Path('.env')
            if env_path.exists():
                load_dotenv(env_path)
                print(f"✓ Loaded configuration from {env_path}")
            else:
                # Look for .env in parent directories
                current = Path.cwd()
                while current != current.parent:
                    env_file = current / '.env'
                    if env_file.exists():
                        load_dotenv(env_file)
                        print(f"✓ Loaded configuration from {env_file}")
                        break
                    current = current.parent
                else:
                    print("⚠️  No .env file found. Using environment variables and defaults.")
        
        # Ollama Configuration
        self.OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
        
        # HuggingFace Configuration
        self.HF_TOKEN = os.getenv('HF_TOKEN', '')
        
        # API Configuration
        self.API_HOST = os.getenv('API_HOST', '0.0.0.0')
        self.API_PORT = int(os.getenv('API_PORT', '8900'))
        self.API_MAX_FILE_SIZE = int(os.getenv('API_MAX_FILE_SIZE', '104857600'))
        
        # Audio Processing Configuration
        self.DEFAULT_NUM_SPEAKERS = int(os.getenv('DEFAULT_NUM_SPEAKERS', '2'))
        self.DEFAULT_CLUSTERING_METHOD = os.getenv('DEFAULT_CLUSTERING_METHOD', 'gmm')
        self.DEFAULT_VOCAL_METHOD = os.getenv('DEFAULT_VOCAL_METHOD', 'auto')
        
        # File Storage Configuration
        self.UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', 'uploads'))
        self.OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'outputs'))
        self.TEMP_DIR = Path(os.getenv('TEMP_DIR', 'temp'))
        
        # Create directories if they don't exist
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.TEMP_DIR.mkdir(exist_ok=True)
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        # Advanced Speaker Separation Settings
        self.VOICE_ACTIVITY_THRESHOLD = float(os.getenv('VOICE_ACTIVITY_THRESHOLD', '0.3'))
        self.MFCC_COEFFICIENTS = int(os.getenv('MFCC_COEFFICIENTS', '20'))
        self.FRAME_LENGTH = int(os.getenv('FRAME_LENGTH', '2048'))
        self.HOP_LENGTH = int(os.getenv('HOP_LENGTH', '512'))
        
        # GPU Configuration
        self.USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'
        self.GPU_DEVICE = os.getenv('GPU_DEVICE', '')
        
        # Security Configuration    
        cors_origins = os.getenv('CORS_ORIGINS', '*')
        self.CORS_ORIGINS = [origin.strip() for origin in cors_origins.split(',')]
        self.ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
        self.API_KEY = os.getenv('API_KEY', '')
        
        # Development Settings
        self.AUTO_RELOAD = os.getenv('AUTO_RELOAD', 'false').lower() == 'true'
        self.DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    
    def validate(self):
        """Validate critical configuration"""
        issues = []
        
        # Check HuggingFace token if needed
        if not self.HF_TOKEN:
            issues.append("HF_TOKEN not set (required for pyannote.audio)")
        
        # Check if Ollama URL is accessible
        try:
            import requests
            response = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                issues.append(f"Cannot connect to Ollama at {self.OLLAMA_URL}")
        except Exception as e:
            issues.append(f"Ollama connection error: {str(e)}")
        
        return issues
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "=" * 60)
        print("Current Configuration")
        print("=" * 60)
        
        print("\n🤖 Ollama:")
        print(f"  URL: {self.OLLAMA_URL}")
        print(f"  Model: {self.OLLAMA_MODEL}")
        
        print("\n🤗 HuggingFace:")
        if self.HF_TOKEN:
            print(f"  Token: {'*' * 20}{self.HF_TOKEN[-8:]}")
        else:
            print("  Token: Not set")
        
        print("\n🌐 API:")
        print(f"  Host: {self.API_HOST}")
        print(f"  Port: {self.API_PORT}")
        print(f"  Max File Size: {self.API_MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        
        print("\n🎵 Audio Processing:")
        print(f"  Default Speakers: {self.DEFAULT_NUM_SPEAKERS}")
        print(f"  Clustering Method: {self.DEFAULT_CLUSTERING_METHOD}")
        print(f"  Vocal Method: {self.DEFAULT_VOCAL_METHOD}")
        
        print("\n📁 Storage:")
        print(f"  Upload Dir: {self.UPLOAD_DIR}")
        print(f"  Output Dir: {self.OUTPUT_DIR}")
        print(f"  Temp Dir: {self.TEMP_DIR}")
        
        print("\n⚙️  Advanced:")
        print(f"  MFCC Coefficients: {self.MFCC_COEFFICIENTS}")
        print(f"  Frame Length: {self.FRAME_LENGTH}")
        print(f"  Hop Length: {self.HOP_LENGTH}")
        print(f"  Use GPU: {self.USE_GPU}")
        
        print("\n🔒 Security:")
        print(f"  CORS Origins: {', '.join(self.CORS_ORIGINS)}")
        print(f"  Auth Enabled: {self.ENABLE_AUTH}")
        
        print("\n" + "=" * 60 + "\n")


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global config instance"""
    return config


def setup_huggingface_auth():
    """Setup HuggingFace authentication"""
    if not config.HF_TOKEN:
        print("\n⚠️  HuggingFace token not found!")
        print("\nTo use advanced speaker diarization (pyannote.audio):")
        print("  1. Create account at https://huggingface.co")
        print("  2. Get token from https://huggingface.co/settings/tokens")
        print("  3. Accept model license at https://huggingface.co/pyannote/speaker-diarization")
        print("  4. Add token to .env file: HF_TOKEN=your_token_here")
        return False
    
    # Set HuggingFace token for pyannote
    os.environ['HF_TOKEN'] = config.HF_TOKEN
    
    try:
        from huggingface_hub import login
        login(token=config.HF_TOKEN)
        print("✓ HuggingFace authentication successful")
        return True
    except ImportError:
        print("⚠️  huggingface_hub not installed")
        print("   Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"⚠️  HuggingFace authentication failed: {e}")
        return False


if __name__ == "__main__":
    """Test configuration loading"""
    
    # Print config
    config.print_config()
    
    # Validate
    issues = config.validate()
    
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Configuration validated successfully")
    
    # Test HuggingFace auth
    if config.HF_TOKEN:
        print("\nTesting HuggingFace authentication...")
        setup_huggingface_auth()
