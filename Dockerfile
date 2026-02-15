FROM python:3.11.9-slim

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
COPY templates ./templates
COPY static ./static


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p conversation_output uploads outputs temp

# Expose port for API
EXPOSE 8900

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8900/health || exit 1

# Run the application
CMD ["python", "final_speaker_api.py", "--host", "0.0.0.0", "--port", "8900"]