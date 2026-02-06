# Speaker Separator API Documentation

A REST API service for separating individual speakers from multi-speaker conversation audio using AI-powered diarization.

## Features

- **Multi-speaker separation**: Support for 1-10 speakers
- **Multiple separation methods**: 
  - Clustering (spectral clustering with dual thresholds)
  - Frequency-based separation
  - Auto-detection (uses Ollama to recommend the best method)
- **Output format options**: WAV, MP3, FLAC, OGG
- **AI analysis**: Powered by Ollama for intelligent recommendations and conversation insights
- **Async processing**: Background job processing with status tracking
- **REST API**: Full REST API with Swagger/OpenAPI documentation

## Quick Start

### Using Docker Compose

```bash
# Set Ollama URL environment variable
export OLLAMA_URL=http://192.168.1.33:11434

# Start the API service
docker compose up -d

# API will be available at http://localhost:8000
```

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### 1. Health Check

```bash
GET /
```

Response:
```json
{
  "status": "healthy",
  "service": "Speaker Separator API",
  "version": "1.0.0"
}
```

### 2. Submit Separation Job

```bash
POST /api/v1/separate
```

**Parameters:**
- `file` (required): Audio file (multipart/form-data)
  - Supported formats: MP3, WAV, FLAC, OGG, M4A, MPEG
- `num_speakers` (optional, default: 2): Number of speakers (1-10)
- `method` (optional, default: auto): Separation method
  - `auto`: Let Ollama recommend the best method
  - `clustering`: Spectral clustering-based separation
  - `frequency`: Frequency-based separation
- `output_format` (optional, default: wav): Output audio format
  - `wav`, `mp3`, `flac`, `ogg`
- `ollama_url` (optional): Ollama API URL (uses env var if not provided)

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/separate \
  -F "file=@conversation.mp3" \
  -F "num_speakers=2" \
  -F "method=auto" \
  -F "output_format=wav"
```

**Response:**
```json
{
  "job_id": "f594cdba-f2a6-40a7-9f82-ed9570cf2794",
  "status": "pending",
  "created_at": "2026-02-05T11:38:26.079072",
  "filename": "conversation.mp3",
  "num_speakers": 2,
  "method": "auto",
  "output_format": "wav",
  "error": null
}
```

### 3. Get Job Status

```bash
GET /api/v1/status/{job_id}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/status/f594cdba-f2a6-40a7-9f82-ed9570cf2794
```

**Response:**
```json
{
  "job_id": "f594cdba-f2a6-40a7-9f82-ed9570cf2794",
  "status": "completed",
  "created_at": "2026-02-05T11:38:26.079072",
  "filename": "conversation.mp3",
  "num_speakers": 2,
  "method": "clustering",
  "output_format": "wav",
  "error": null
}
```

**Status values:**
- `pending`: Job queued, waiting to start
- `processing`: Currently processing the audio
- `completed`: Successfully completed
- `failed`: Job failed (check `error` field)

### 4. Get Job Results

```bash
GET /api/v1/results/{job_id}
```

Returns complete job results including separated speaker files and Ollama analysis.

**Example:**
```bash
curl http://localhost:8000/api/v1/results/f594cdba-f2a6-40a7-9f82-ed9570cf2794
```

**Response:**
```json
{
  "job_id": "f594cdba-f2a6-40a7-9f82-ed9570cf2794",
  "status": "completed",
  "created_at": "2026-02-05T11:38:26.079072",
  "filename": "conversation.mp3",
  "num_speakers": 2,
  "method": "clustering",
  "output_format": "wav",
  "speakers": {
    "speaker_1": {
      "path": "outputs/f594cdba-f2a6-40a7-9f82-ed9570cf2794/speaker_1.wav",
      "url": "/api/v1/download/f594cdba-f2a6-40a7-9f82-ed9570cf2794/speaker_1",
      "speaking_time": 219.14
    },
    "speaker_2": {
      "path": "outputs/f594cdba-f2a6-40a7-9f82-ed9570cf2794/speaker_2.wav",
      "url": "/api/v1/download/f594cdba-f2a6-40a7-9f82-ed9570cf2794/speaker_2",
      "speaking_time": 25.92
    }
  },
  "ollama_analysis": "Analysis and recommendations from Ollama..."
}
```

### 5. Download Speaker File

```bash
GET /api/v1/download/{job_id}/{speaker_name}
```

Download the separated audio file for a specific speaker.

**Example:**
```bash
curl -o speaker_1.wav http://localhost:8000/api/v1/download/f594cdba-f2a6-40a7-9f82-ed9570cf2794/speaker_1
```

### 6. List All Jobs

```bash
GET /api/v1/jobs
```

Returns a list of all jobs with their current status.

**Example:**
```bash
curl http://localhost:8000/api/v1/jobs
```

**Response:**
```json
{
  "total": 3,
  "jobs": [
    {
      "job_id": "f594cdba-f2a6-40a7-9f82-ed9570cf2794",
      "status": "completed",
      "created_at": "2026-02-05T11:38:26.079072",
      "filename": "conversation.mp3",
      "num_speakers": 2
    },
    // ... more jobs
  ]
}
```

## Python Client Example

```python
import requests
import time

# API endpoint
api_url = "http://localhost:8000"

# 1. Submit separation job
with open("conversation.mp3", "rb") as f:
    response = requests.post(
        f"{api_url}/api/v1/separate",
        files={"file": f},
        data={
            "num_speakers": 2,
            "method": "auto",
            "output_format": "wav"
        }
    )
    
    job = response.json()
    job_id = job["job_id"]
    print(f"Job submitted: {job_id}")

# 2. Wait for job completion
while True:
    status_response = requests.get(f"{api_url}/api/v1/status/{job_id}")
    status = status_response.json()
    
    if status["status"] == "completed":
        print("Job completed!")
        break
    elif status["status"] == "failed":
        print(f"Job failed: {status['error']}")
        break
    else:
        print(f"Status: {status['status']}")
        time.sleep(5)

# 3. Get results
results = requests.get(f"{api_url}/api/v1/results/{job_id}").json()

print("\nOllama Analysis:")
print(results["ollama_analysis"])

print("\nSeparated speakers:")
for speaker_name, speaker_info in results["speakers"].items():
    print(f"  {speaker_name}: {speaker_info['speaking_time']:.2f}s")

# 4. Download speaker files
for speaker_name in results["speakers"].keys():
    response = requests.get(
        f"{api_url}/api/v1/download/{job_id}/{speaker_name}",
        stream=True
    )
    
    filename = f"{speaker_name}_{job_id}.wav"
    with open(filename, "wb") as f:
        f.write(response.content)
    
    print(f"Downloaded: {filename}")
```

## JavaScript/Node.js Client Example

```javascript
const fetch = require('node-fetch');
const fs = require('fs');
const FormData = require('form-data');

const apiUrl = "http://localhost:8000";

async function separateSpeakers(audioFilePath, numSpeakers = 2) {
  // 1. Submit job
  const form = new FormData();
  form.append('file', fs.createReadStream(audioFilePath));
  form.append('num_speakers', numSpeakers);
  form.append('method', 'auto');
  form.append('output_format', 'wav');

  const submitResponse = await fetch(`${apiUrl}/api/v1/separate`, {
    method: 'POST',
    body: form
  });

  const job = await submitResponse.json();
  const jobId = job.job_id;
  console.log(`Job submitted: ${jobId}`);

  // 2. Poll for completion
  let status = 'pending';
  while (status !== 'completed' && status !== 'failed') {
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    const statusResponse = await fetch(`${apiUrl}/api/v1/status/${jobId}`);
    const statusData = await statusResponse.json();
    status = statusData.status;
    
    console.log(`Status: ${status}`);
  }

  if (status === 'failed') {
    throw new Error(`Job failed`);
  }

  // 3. Get results
  const resultsResponse = await fetch(`${apiUrl}/api/v1/results/${jobId}`);
  const results = await resultsResponse.json();

  console.log("\nOllama Analysis:");
  console.log(results.ollama_analysis);

  // 4. Download files
  for (const [speakerName, speakerInfo] of Object.entries(results.speakers)) {
    const downloadUrl = `${apiUrl}${speakerInfo.url}`;
    const fileResponse = await fetch(downloadUrl);
    const fileBuffer = await fileResponse.buffer();
    
    const filename = `${speakerName}_${jobId}.wav`;
    fs.writeFileSync(filename, fileBuffer);
    console.log(`Downloaded: ${filename}`);
  }
}

// Usage
separateSpeakers('conversation.mp3', 2);
```

## Configuration

### Environment Variables

```bash
# Ollama API URL
export OLLAMA_URL=http://192.168.1.33:11434

# API port
export API_PORT=8000

# Optional: Ollama model (default: llama3.2)
export OLLAMA_MODEL=llama3.2
```

### Docker Compose Overrides

```bash
# Start with custom Ollama URL
OLLAMA_URL=http://custom-ollama:11434 docker compose up -d

# Start with custom port
API_PORT=9000 docker compose up -d
```

## Separation Methods

### 1. Clustering (Spectral Clustering)

**Best for:** Distinct vocal characteristics between speakers

**Process:**
1. Extracts MFCC (Mel-Frequency Cepstral Coefficients) features
2. Uses dual thresholds (RMS + Spectral flux) for voice activity detection
3. Applies spectral clustering to group frames by speaker
4. Reconstructs individual speaker audio streams

**Pros:**
- Fast processing
- Works well with clear speaker distinctions
- Minimal computational overhead

**Cons:**
- Struggles with overlapping speech
- May not work well with very similar voices

### 2. Frequency-Based Separation

**Best for:** Speakers with different pitch ranges

**Process:**
1. Computes STFT (Short-Time Fourier Transform)
2. Splits frequency spectrum into lower and higher bands
3. Reconstructs audio from each frequency band separately

**Pros:**
- Effective for naturally separated pitch ranges
- Simple and predictable

**Cons:**
- Only works for 2 speakers
- Less accurate than clustering with good models

### 3. Auto (Ollama-Recommended)

**Best for:** When you're unsure which method to use

**Process:**
1. Analyzes audio characteristics
2. Asks Ollama to recommend the best method
3. Applies the recommended approach

**Benefits:**
- AI-guided selection
- Adapts to audio characteristics
- Includes pre and post-processing analysis

## Ollama Integration

The API uses Ollama for:

1. **Method recommendation**: Analyzes audio and suggests clustering vs frequency separation
2. **Post-processing analysis**:
   - Identifies dominant speaker
   - Analyzes speaking patterns
   - Provides quality improvement recommendations
   - Detects audio issues

### Ollama Setup

```bash
# Install Ollama
https://ollama.ai

# Pull recommended model
ollama pull llama3.2

# Start Ollama server
ollama serve

# Ollama will be available at http://localhost:11434
```

For Docker networking, use your actual IP instead of localhost:
```bash
export OLLAMA_URL=http://192.168.1.33:11434
```

## Performance Considerations

### Processing Time (on average hardware)

- **2 speakers, 3-5 min audio**: ~15-30 seconds
- **3 speakers, 3-5 min audio**: ~20-40 seconds
- **Processing is I/O bound** (waiting for Ollama responses)

### Storage

- **Input files**: Stored in `uploads/` directory
- **Output files**: Stored in `outputs/{job_id}/` directory
- **Jobs database**: In-memory (lost on restart)

### Recommendations

- Use **clustering** for fast processing
- Use **frequency** for pitch-distinct speakers
- Use **auto** for best quality (slower due to Ollama analysis)

## Error Handling

### Common Errors

**400 - File type not supported**
```json
{
  "detail": "File type not supported. Allowed: .mp3, .wav, .flac, .ogg, .m4a, .mpeg"
}
```
Solution: Use supported audio formats

**404 - Job not found**
```json
{
  "detail": "Job xyz not found"
}
```
Solution: Verify job_id is correct

**400 - Job not completed**
```json
{
  "detail": "Job not completed. Status: processing"
}
```
Solution: Wait for job to complete before downloading

**500 - Ollama connection failed**
The job continues without AI analysis but still completes.

## Troubleshooting

### API not responding

```bash
# Check if container is running
docker compose ps

# View logs
docker compose logs -f speaker-separator

# Restart service
docker compose restart
```

### Ollama connection issues

```bash
# Verify Ollama is accessible
curl http://192.168.1.33:11434/api/tags

# Check network connectivity
docker exec speaker-separator-api ping 192.168.1.33
```

### Jobs not completing

```bash
# Check container logs for errors
docker compose logs speaker-separator

# Verify sufficient disk space for outputs
df -h
```

## Advanced Usage

### Bulk Processing

```bash
import requests
import glob

api_url = "http://localhost:8000"

# Submit multiple jobs
job_ids = []
for audio_file in glob.glob("audio_files/*.mp3"):
    with open(audio_file, "rb") as f:
        response = requests.post(
            f"{api_url}/api/v1/separate",
            files={"file": f},
            data={"num_speakers": 2}
        )
        job_ids.append(response.json()["job_id"])

# Monitor all jobs
import time
while True:
    all_jobs = requests.get(f"{api_url}/api/v1/jobs").json()
    statuses = [j["status"] for j in all_jobs["jobs"]]
    
    if all(s in ["completed", "failed"] for s in statuses):
        break
    
    print(f"Pending: {statuses.count('pending')}, Processing: {statuses.count('processing')}")
    time.sleep(10)
```

## License

MIT

## Support

For issues, feature requests, or contributions, please refer to the project repository.
