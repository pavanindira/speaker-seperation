FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY final_speaker_api.py .
COPY audio_cleaner.py .
COPY final_speaker_frontend.html .
COPY improved_speaker_separator.py .
COPY config.py .
COPY requirements.txt .
COPY .env .
COPY templates ./templates
COPY static ./static


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p conversation_output uploads outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://192.168.1.33:11434
ENV API_PORT=8900

# Expose port for API
EXPOSE 8900

# Default command - API mode
#CMD ["python", "final_speaker_api.py"]
#    CMD curl -f http://localhost:8900/health || exit 1

# Run the application
CMD ["python", "final_speaker_api.py", "--host", "0.0.0.0", "--port", "8900"]