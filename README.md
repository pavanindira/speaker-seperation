# Speaker Separation Guide - Separating Individual Voices

## Overview

This system can separate individual speakers from conversation audio, allowing you to:
- Extract each person's voice separately
- Analyze speaking patterns (who talks more, when, etc.)
- Process multi-speaker recordings (meetings, interviews, podcasts)
- Get AI insights about the conversation dynamics

## Quick Start

### Installation

```bash
# Install dependencies
pip install librosa soundfile numpy scipy scikit-learn

# Optional: For advanced diarization
pip install pyannote.audio

# Start your system
python speaker_separator.py --install
```

### Basic Usage

```bash
# Separate 2 speakers from a conversation
python speaker_separator.py conversation.wav

# Specify number of speakers
python speaker_separator.py meeting.mp3 --num-speakers 3

# Use specific method
python speaker_separator.py interview.wav --method clustering

# Without Ollama AI
python speaker_separator.py discussion.wav --no-ollama
```

## How It Works

### Method 1: Clustering (Default - Best for Most Cases)

**How it works:**
1. Extracts MFCC (Mel-frequency cepstral coefficients) features
2. Detects voice activity (removes silence)
3. Clusters similar voice features using K-means
4. Separates audio based on cluster assignments

**Best for:**
- 2-5 speakers
- Clear speech with minimal overlap
- Interviews, podcasts, meetings
- When speakers have distinct voice characteristics

**Pros:**
- Works without internet/models
- Fast processing
- No training data needed

**Cons:**
- Less accurate with overlapping speech
- May struggle if voices are very similar
- Needs clear audio quality

**Example Output:**
```
conversation_output/
├── speaker_1.wav  (Person A: 45.2s speaking time)
├── speaker_2.wav  (Person B: 38.7s speaking time)
└── speaker_3.wav  (Person C: 22.1s speaking time)
```

### Method 2: Frequency Separation

**How it works:**
1. Splits audio into frequency bands
2. Assigns lower frequencies to one speaker
3. Assigns higher frequencies to another
4. Reconstructs separated audio

**Best for:**
- 2 speakers only
- Male + Female speakers (different pitch ranges)
- When speakers have very different voice tones
- Quick separation without complex analysis

**Pros:**
- Very fast
- Works when voices have different pitch
- Simple approach

**Cons:**
- Only works for 2 speakers
- Requires different vocal ranges
- Lower quality than clustering

### Method 3: PyAnnote (Advanced - Best Quality)

**How it works:**
1. Uses deep learning models
2. Advanced speaker embeddings
3. Neural network-based diarization
4. State-of-the-art accuracy

**Best for:**
- Professional use
- Complex conversations
- Overlapping speech
- High accuracy requirements

**Pros:**
- Best accuracy
- Handles overlapping speech
- Works with similar voices

**Cons:**
- Requires model download (~300MB)
- Slower processing
- Needs HuggingFace account/token

**Setup:**
```bash
# Install
pip install pyannote.audio

# Get HuggingFace token
# 1. Create account: https://huggingface.co
# 2. Accept license: https://huggingface.co/pyannote/speaker-diarization
# 3. Get token: https://huggingface.co/settings/tokens

# Set token
export HF_TOKEN=your_token_here

# Use pyannote
python speaker_separator.py audio.wav --method pyannote
```

## With Ollama AI Integration

### What Ollama Does

1. **Pre-Analysis:**
   - Recommends best separation method
   - Identifies potential challenges
   - Suggests audio quality improvements

2. **Post-Analysis:**
   - Identifies dominant speaker
   - Analyzes speaking patterns
   - Provides conversation insights
   - Suggests improvements

### Example Ollama Analysis

```
🤖 Ollama's Recommendation:
For this 2-speaker conversation with clear distinct voices, 
clustering method will work best. The speakers have different 
pitch ranges which should help with accurate separation.

[After processing...]

🤖 Conversation Analysis:
Speaker 1 dominates the conversation (60% speaking time), suggesting 
an interview format. Both speakers have minimal overlap, indicating 
good turn-taking. The audio quality is excellent for separation.
```

## Real-World Examples

### Example 1: Podcast Interview (2 speakers)

```bash
# Input: podcast_episode.mp3 (30 minutes)
python speaker_separator.py podcast_episode.mp3 --num-speakers 2

# Output:
# ✓ Speaker 1: 18m 30s (Host)
# ✓ Speaker 2: 11m 10s (Guest)
# AI: Host dominates with 62% speaking time, typical interview format
```

### Example 2: Business Meeting (4 speakers)

```bash
# Input: team_meeting.wav (45 minutes)
python speaker_separator.py team_meeting.wav --num-speakers 4

# Output:
# ✓ Speaker 1: 15m 20s
# ✓ Speaker 2: 12m 45s
# ✓ Speaker 3: 10m 15s
# ✓ Speaker 4: 6m 40s
# AI: Speaker 1 leads the discussion. Minimal overlap detected.
```

### Example 3: Male + Female Discussion (2 speakers)

```bash
# Use frequency method for different pitch ranges
python speaker_separator.py discussion.mp3 --method frequency

# Output:
# ✓ Speaker (lower frequency): male_voice.wav
# ✓ Speaker (higher frequency): female_voice.wav
```

## API Endpoints

### Separate Speakers via API

**POST** `/api/v1/separate-speakers`

```bash
# Upload and separate
curl -X POST "http://localhost:8000/api/v1/separate-speakers?num_speakers=2&method=clustering" \
  -F "file=@conversation.wav"

# Response:
{
  "job_id": "abc-123",
  "status": "pending",
  "num_speakers": 2,
  "method": "clustering"
}

# Check status
curl http://localhost:8000/api/v1/jobs/abc-123

# Download separated speakers
curl -O http://localhost:8000/api/v1/download/abc-123/speaker_1
curl -O http://localhost:8000/api/v1/download/abc-123/speaker_2
```

### Python Client

```python
import requests
import time

# Upload conversation
with open('conversation.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/separate-speakers',
        files={'file': f},
        params={
            'num_speakers': 2,
            'method': 'clustering',
            'use_ollama': True
        }
    )

job = response.json()
job_id = job['job_id']

# Wait for completion
while True:
    status = requests.get(f'http://localhost:8000/api/v1/jobs/{job_id}')
    job = status.json()
    
    if job['status'] == 'completed':
        print("Separation complete!")
        
        # Show AI analysis
        if job.get('ollama_recommendation'):
            print(f"\nAI Analysis:\n{job['ollama_recommendation']}")
        
        # Download each speaker
        for speaker_id in range(1, job['num_speakers'] + 1):
            speaker_url = f"http://localhost:8000/api/v1/download/{job_id}/speaker_{speaker_id}"
            audio = requests.get(speaker_url)
            
            with open(f'speaker_{speaker_id}.wav', 'wb') as f:
                f.write(audio.content)
            
            print(f"✓ Downloaded speaker_{speaker_id}.wav")
        
        break
    
    time.sleep(2)
```

## Output Format

### Directory Structure

```
conversation_output/
└── conversation_name/
    ├── speaker_1.wav       # First speaker's audio
    ├── speaker_2.wav       # Second speaker's audio
    ├── speaker_3.wav       # Third speaker's audio (if applicable)
    └── analysis.json       # Metadata and statistics
```

### Statistics Provided

```json
{
  "num_speakers": 2,
  "total_duration": 180.5,
  "method": "clustering",
  "speakers": {
    "speaker_1": {
      "speaking_time": 95.3,
      "speaking_percentage": 52.8,
      "file": "speaker_1.wav"
    },
    "speaker_2": {
      "speaking_time": 85.2,
      "speaking_percentage": 47.2,
      "file": "speaker_2.wav"
    }
  },
  "ollama_analysis": "Speaker 1 is more dominant..."
}
```

## Tips for Best Results

### Audio Quality

✅ **Good:**
- Clear recording
- Minimal background noise
- Speakers don't talk over each other
- Consistent volume levels
- WAV or FLAC format

❌ **Avoid:**
- Heavy background music
- Multiple simultaneous speakers
- Low-quality microphones
- Highly compressed audio (low bitrate MP3)
- Echo or reverb

### Speaker Count

- **2-3 speakers**: Best results with any method
- **4-5 speakers**: Use clustering or pyannote
- **6+ speakers**: Recommended to use pyannote for accuracy

### Pre-processing Tips

```bash
# If audio has noise, clean it first:
# 1. Use noise reduction tools
# 2. Normalize volume
# 3. Remove background music

# Example with ffmpeg:
ffmpeg -i noisy.wav -af "highpass=f=200,lowpass=f=3000" clean.wav
```

## Troubleshooting

### Poor Separation Quality

**Problem:** Speakers are mixed together in output

**Solutions:**
1. Try different method (clustering vs frequency)
2. Ensure correct speaker count
3. Check audio quality
4. Use pyannote for better accuracy

```bash
# Try frequency method if clustering fails
python speaker_separator.py audio.wav --method frequency

# Use pyannote for best quality
python speaker_separator.py audio.wav --method pyannote
```

### Wrong Speaker Count

**Problem:** Don't know how many speakers

**Solution:** Let AI analyze first

```bash
# Start with auto-detection
python speaker_separator.py audio.wav --num-speakers 2

# Check output, adjust if needed
python speaker_separator.py audio.wav --num-speakers 3
```

### Speakers Assigned Incorrectly

**Problem:** Speaker 1 and 2 are swapped

**Solution:** This is normal - assignment is arbitrary. Just rename files:

```bash
mv speaker_1.wav person_A.wav
mv speaker_2.wav person_B.wav
```

## Advanced: Multi-Speaker Identification

### Future Feature: Speaker Identification

Coming soon - system will identify who is speaking:

```python
# Future API
POST /api/v1/identify-speakers
{
  "audio_file": "meeting.wav",
  "speaker_profiles": {
    "john": "john_voice_sample.wav",
    "sarah": "sarah_voice_sample.wav"
  }
}

# Response:
{
  "speaker_1": "john",      # 45% speaking time
  "speaker_2": "sarah",     # 35% speaking time  
  "speaker_3": "unknown"    # 20% speaking time
}
```

### How to Prepare

To enable speaker identification in the future:

1. **Collect voice samples** - Get 10-30 second samples of each person
2. **Label your data** - Name files clearly (john_sample.wav)
3. **Store profiles** - Keep samples organized
4. **Train or use embeddings** - System will match voices

## Use Cases

### 1. Podcast Production
- Separate host and guests
- Edit individual tracks
- Balance audio levels
- Remove one person's mistakes

### 2. Meeting Transcription
- Generate separate transcripts per speaker
- Track who said what
- Analyze participation
- Create meeting minutes

### 3. Interview Analysis
- Study interviewer vs interviewee patterns
- Calculate talk time ratio
- Identify interruptions
- Analyze conversation flow

### 4. Content Creation
- Extract specific voices from group recordings
- Create highlight reels
- Mix tracks independently
- Professional post-production

### 5. Research & Analysis
- Conversation analysis studies
- Linguistic research
- Communication patterns
- Social interaction studies

## Performance Expectations

| Scenario | Processing Time | Quality |
|----------|----------------|---------|
| 2 speakers, 5 min, clustering | ~10 seconds | Good |
| 3 speakers, 10 min, clustering | ~20 seconds | Good |
| 2 speakers, 5 min, frequency | ~5 seconds | Fair |
| 4 speakers, 30 min, pyannote | ~5 minutes | Excellent |

## Comparison with Commercial Tools

| Feature | This System | Commercial Tools |
|---------|------------|------------------|
| Cost | Free | $10-100/month |
| Privacy | Local processing | Cloud upload |
| Accuracy | Good-Excellent | Excellent |
| Speed | Fast | Variable |
| Customization | Full control | Limited |
| AI Integration | Ollama included | Extra cost |

## Next Steps

1. **Try it out**: Process a sample conversation
2. **Experiment**: Test different methods
3. **Integrate**: Add to your workflow
4. **Scale**: Deploy as API for team use
5. **Extend**: Add speaker identification

Happy speaker separating! 🎤
