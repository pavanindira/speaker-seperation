# Environment Configuration Guide

## Quick Setup

### Step 1: Install Dependencies

```bash
# Install python-dotenv for .env support
pip install python-dotenv huggingface-hub

# Or install everything
pip install -r requirements-complete.txt
```

### Step 2: Create .env File

```bash
# Copy the template
cp .env.template .env

# Edit with your settings
nano .env  # or use your favorite editor
```

### Step 3: Configure Your Settings

```bash
# Minimal .env file
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
HF_TOKEN=your_huggingface_token_here
```

## Configuration Options

### 🤖 Ollama Settings

```bash
# Local Ollama
OLLAMA_URL=http://localhost:11434

# Ollama in Docker Compose
OLLAMA_URL=http://ollama:11434

# Remote Ollama
OLLAMA_URL=http://192.168.1.100:11434

# Model selection
OLLAMA_MODEL=llama3.2
# or: llama2, codellama, mistral, etc.
```

### 🤗 HuggingFace Token

**Required for pyannote.audio (advanced speaker diarization)**

**How to get your token:**

1. **Create account:** https://huggingface.co/join
2. **Get token:** https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "speaker-separation")
   - Select "Read" access
   - Copy the token
3. **Accept license:** https://huggingface.co/pyannote/speaker-diarization
   - Click "Agree and access repository"
4. **Add to .env:**
   ```bash
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### 🎵 Audio Processing

```bash
# Default number of speakers
DEFAULT_NUM_SPEAKERS=2

# Clustering method (gmm, kmeans, spectral)
DEFAULT_CLUSTERING_METHOD=gmm

# MFCC coefficients (more = better separation, slower)
MFCC_COEFFICIENTS=20

# Frame settings
FRAME_LENGTH=2048
HOP_LENGTH=512
```

### 🌐 API Settings

```bash
# API server host and port
API_HOST=0.0.0.0
API_PORT=8900

# Max upload file size (100MB)
API_MAX_FILE_SIZE=104857600

# CORS origins
CORS_ORIGINS=*
# For production: CORS_ORIGINS=https://yourdomain.com
```

### 📁 Storage Paths

```bash
# Default paths
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
TEMP_DIR=temp

# Custom paths
# UPLOAD_DIR=/mnt/storage/uploads
# OUTPUT_DIR=/mnt/storage/outputs
```

## Usage Examples

### With .env File

```bash
# 1. Create .env file
cat > .env << EOF
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
HF_TOKEN=hf_your_token_here
DEFAULT_NUM_SPEAKERS=2
DEFAULT_CLUSTERING_METHOD=gmm
EOF

# 2. Run separator (uses .env automatically)
python speaker_separator_with_env.py conversation.wav

# 3. Show current config
python speaker_separator_with_env.py --show-config
```

### Override .env with Command Line

```bash
# .env has DEFAULT_NUM_SPEAKERS=2
# Override with command line:
python speaker_separator_with_env.py audio.wav --num-speakers 3

# .env has DEFAULT_CLUSTERING_METHOD=gmm
# Override:
python speaker_separator_with_env.py audio.wav --method kmeans
```

### Test Configuration

```bash
# Test if config loads correctly
python config.py

# Output shows:
# ✓ Loaded configuration from .env
# Current Configuration
# 🤖 Ollama:
#   URL: http://localhost:11434
#   Model: llama3.2
# ...
```

## Docker Setup with .env

### Option 1: Pass .env to Docker

```bash
# docker-compose.yml
version: '3.8'

services:
  vocal-api:
    build: .
    env_file:
      - .env
    ports:
      - "${API_PORT}:8900"
```

### Option 2: Environment Variables

```bash
# Pass specific variables
docker run -e OLLAMA_URL=http://ollama:11434 \
           -e HF_TOKEN=$HF_TOKEN \
           vocal-api
```

## Common Configurations

### Configuration 1: Local Development

```bash
# .env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
DEFAULT_NUM_SPEAKERS=2
DEFAULT_CLUSTERING_METHOD=gmm
DEBUG=true
AUTO_RELOAD=true
LOG_LEVEL=DEBUG
```

### Configuration 2: Production API

```bash
# .env
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2
API_HOST=0.0.0.0
API_PORT=8900
CORS_ORIGINS=https://yourdomain.com
ENABLE_AUTH=true
API_KEY=your_secure_api_key_here
DEBUG=false
LOG_LEVEL=INFO
```

### Configuration 3: GPU-Enabled Processing

```bash
# .env
OLLAMA_URL=http://localhost:11434
HF_TOKEN=hf_xxxxxxxxxxxxx
USE_GPU=true
GPU_DEVICE=cuda:0
DEFAULT_VOCAL_METHOD=demucs  # Uses GPU for best quality
```

## Verification

### Check if .env is Loaded

```python
from config import config

# Print all settings
config.print_config()

# Check specific setting
print(f"Ollama URL: {config.OLLAMA_URL}")
print(f"HF Token set: {bool(config.HF_TOKEN)}")
```

### Validate Configuration

```bash
python config.py

# Output:
# ✓ Loaded configuration from .env
# ✓ Configuration validated successfully
# or:
# ⚠️  Configuration Issues:
#   - HF_TOKEN not set (required for pyannote.audio)
#   - Cannot connect to Ollama at http://localhost:11434
```

## Troubleshooting

### .env File Not Found

**Problem:** Settings not loading from .env

**Solution:**
```bash
# Check if .env exists
ls -la .env

# Check if you're in the right directory
pwd

# Create .env from template
cp .env.template .env
```

### HuggingFace Token Not Working

**Problem:** pyannote.audio can't authenticate

**Solution:**
```bash
# 1. Verify token is set
python -c "from config import config; print('Token:', config.HF_TOKEN[:10] if config.HF_TOKEN else 'NOT SET')"

# 2. Test authentication
python config.py

# 3. Check token format (should start with hf_)
# Correct: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Wrong: xxxxxxxxxxxxx (without hf_ prefix)
```

### Ollama Connection Failed

**Problem:** Cannot connect to Ollama

**Solutions:**

```bash
# 1. Check if Ollama is running
curl http://localhost:11434/api/tags

# 2. Check .env URL
cat .env | grep OLLAMA_URL

# 3. If Ollama is in Docker:
OLLAMA_URL=http://ollama:11434  # Docker network
# NOT: http://localhost:11434

# 4. If Ollama is on another machine:
OLLAMA_URL=http://192.168.1.100:11434
```

## Environment Variable Priority

Settings are loaded in this order (later overrides earlier):

1. **Hardcoded defaults** (in config.py)
2. **.env file** (in current or parent directory)
3. **Environment variables** (export OLLAMA_URL=...)
4. **Command line arguments** (--num-speakers 3)

Example:
```bash
# 1. Default: DEFAULT_NUM_SPEAKERS=2
# 2. .env file: DEFAULT_NUM_SPEAKERS=3
# 3. Environment: export DEFAULT_NUM_SPEAKERS=4
# 4. Command line: --num-speakers 5

# Final value: 5 (command line wins)
```

## Security Notes

### ⚠️ Important: Don't Commit .env

```bash
# Add to .gitignore
echo ".env" >> .gitignore

# .env should NEVER be in version control
# It contains secrets (API keys, tokens)
```

### Use .env.template for Sharing

```bash
# Share this (no secrets):
.env.template

# Keep private (has secrets):
.env
```

### Rotate Tokens Regularly

```bash
# 1. Create new HuggingFace token
# 2. Update .env
HF_TOKEN=hf_NEW_TOKEN_HERE

# 3. Revoke old token on HuggingFace
```

## Complete Example

### 1. Setup

```bash
# Install dependencies
pip install python-dotenv huggingface-hub

# Create .env
cp .env.template .env
```

### 2. Configure .env

```bash
# Edit .env
nano .env

# Add your settings:
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
HF_TOKEN=hf_your_actual_token_here
DEFAULT_NUM_SPEAKERS=2
DEFAULT_CLUSTERING_METHOD=gmm
```

### 3. Verify

```bash
# Test configuration
python config.py

# Should show:
# ✓ Loaded configuration from .env
# ✓ Configuration validated successfully
```

### 4. Use

```bash
# Run with .env settings
python speaker_separator_with_env.py conversation.wav

# Show what settings are being used
python speaker_separator_with_env.py --show-config
```

## Next Steps

1. ✅ Create .env file
2. ✅ Add your HuggingFace token
3. ✅ Test configuration: `python config.py`
4. ✅ Run separator: `python speaker_separator_with_env.py audio.wav`

For more help, see:
- `INSTALLATION.md` - Installation guide
- `TROUBLESHOOTING.md` - Common issues
- `.env.template` - All available options
