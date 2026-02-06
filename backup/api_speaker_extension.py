"""
Enhanced Vocal Isolation API with Speaker Separation
Includes both vocal isolation and speaker diarization capabilities
"""

# Add to existing vocal_isolation_api.py

import os
from enum import Enum
from sklearn.cluster import KMeans

class SeparationType(str, Enum):
    VOCAL = "vocal"  # Separate vocals from music
    SPEAKER = "speaker"  # Separate individual speakers


class SpeakerSeparationRequest(BaseModel):
    num_speakers: int = 2
    method: str = "clustering"  # clustering, frequency, pyannote


# Add new endpoint to the API

@app.post("/api/v1/separate-speakers", response_model=JobResponse)
async def separate_speakers(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_speakers: int = Query(2, ge=2, le=10),
    method: str = Query("clustering", regex="^(clustering|frequency|pyannote)$"),
    use_ollama: bool = Query(True),
    ollama_model: str = Query("llama3.2")
):
    """
    Separate individual speakers from conversation audio
    
    - **file**: Audio file with multiple speakers
    - **num_speakers**: Number of speakers to separate (2-10)
    - **method**: Separation method (clustering, frequency, pyannote)
    - **use_ollama**: Use Ollama for AI analysis
    """
    
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create job record
    jobs_db[job_id] = {
        'job_id': job_id,
        'status': JobStatus.PENDING,
        'created_at': datetime.utcnow().isoformat(),
        'filename': file.filename,
        'separation_type': 'speaker',
        'num_speakers': num_speakers,
        'method': method,
        'audio_analysis': None,
        'ollama_recommendation': None,
        'speaker_files': {},
        'error': None
    }
    
    # Start background processing for speaker separation
    background_tasks.add_task(
        process_speaker_separation_job,
        job_id,
        upload_path,
        num_speakers,
        method,
        use_ollama,
        ollama_model
    )
    
    return JobResponse(**jobs_db[job_id])


async def process_speaker_separation_job(
    job_id: str,
    audio_path: Path,
    num_speakers: int,
    method: str,
    use_ollama: bool,
    ollama_model: str
):
    """Background task for speaker separation"""
    
    try:
        jobs_db[job_id]['status'] = JobStatus.PROCESSING
        
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Basic analysis
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.mean(rms) * 0.5
        voice_activity = rms > threshold
        speaking_time = np.sum(voice_activity) / sr
        
        jobs_db[job_id]['audio_analysis'] = {
            'sample_rate': int(sr),
            'duration': float(duration),
            'speaking_time': float(speaking_time)
        }
        
        # Get Ollama recommendation
        if use_ollama:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            assistant = OllamaAssistant(model=ollama_model, ollama_url=ollama_url)
            recommendation = await assistant.ask(f"""
Conversation audio with {num_speakers} speakers, {duration:.1f}s duration.
Recommend best separation method and explain why (2 sentences max).""")
            
            jobs_db[job_id]['ollama_recommendation'] = recommendation
        
        # Create output directory
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        
        # Perform separation based on method
        if method == "clustering":
            result = separate_by_clustering(y, sr, num_speakers, job_output_dir)
        elif method == "frequency":
            result = separate_by_frequency(y, sr, job_output_dir)
        else:
            raise ValueError(f"Method {method} not implemented")
        
        # Update job with results
        jobs_db[job_id]['speaker_files'] = result
        jobs_db[job_id]['status'] = JobStatus.COMPLETED
        
    except Exception as e:
        jobs_db[job_id]['status'] = JobStatus.FAILED
        jobs_db[job_id]['error'] = str(e)


def separate_by_clustering(y: np.ndarray, sr: int, n_speakers: int, output_dir: Path) -> Dict:
    """Separate speakers using MFCC + K-means clustering"""
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Frame parameters
    hop_length = 512
    frame_length = 2048
    
    # Voice activity detection
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.percentile(rms, 30)
    voice_frames = rms > threshold
    
    # Normalize MFCC
    mfcc_norm = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-10)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    labels = np.zeros(len(rms), dtype=int)
    labels[voice_frames] = kmeans.fit_predict(mfcc_norm[:, voice_frames].T)
    
    # Separate and save
    results = {}
    
    for speaker_id in range(n_speakers):
        speaker_mask = labels == speaker_id
        time_mask = np.repeat(speaker_mask, hop_length)[:len(y)]
        speaker_audio = y * time_mask
        
        # Save
        output_path = output_dir / f"speaker_{speaker_id + 1}.wav"
        sf.write(output_path, speaker_audio, sr)
        
        speaking_time = np.sum(speaker_mask) * hop_length / sr
        
        results[f"speaker_{speaker_id + 1}"] = {
            "path": str(output_path),
            "url": f"/api/v1/download/{output_dir.name}/speaker_{speaker_id + 1}",
            "speaking_time": float(speaking_time)
        }
    
    return results


def separate_by_frequency(y: np.ndarray, sr: int, output_dir: Path) -> Dict:
    """Separate speakers by frequency characteristics"""
    
    D = librosa.stft(y)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Split frequency bands
    n_freq_bins = magnitude.shape[0]
    mid_point = n_freq_bins // 2
    
    # Lower and upper frequency speakers
    lower_mask = np.zeros_like(magnitude)
    lower_mask[:mid_point, :] = magnitude[:mid_point, :]
    
    upper_mask = np.zeros_like(magnitude)
    upper_mask[mid_point:, :] = magnitude[mid_point:, :]
    
    # Reconstruct
    lower_audio = librosa.istft(lower_mask * np.exp(1j * phase))
    upper_audio = librosa.istft(upper_mask * np.exp(1j * phase))
    
    # Save
    results = {}
    
    for idx, (name, audio) in enumerate([("lower", lower_audio), ("higher", upper_audio)]):
        output_path = output_dir / f"speaker_{name}_freq.wav"
        sf.write(output_path, audio, sr)
        
        results[f"speaker_{idx + 1}"] = {
            "path": str(output_path),
            "url": f"/api/v1/download/{output_dir.name}/speaker_{name}_freq"
        }
    
    return results
