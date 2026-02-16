FROM python:3.11.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p conversation_output uploads outputs temp logs data

# Note: Application code is mounted via docker-compose.yml volume
# This allows for hot-reloading during development without rebuilding

# Expose port for API
EXPOSE 8900

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8900/health || exit 1

# Run the application
CMD ["python", "final_speaker_api.py", "--host", "0.0.0.0", "--port", "8900"]