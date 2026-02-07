#!/usr/bin/env python3
"""
Complete Speaker Separation API - Final Version
Full workflow: Upload → Diagnose → Separate → Clean → Feedback
"""

import os
import sys
import shutil
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

# Install FastAPI dependencies if needed
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form, Request
    from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
except ImportError:
    print("Installing FastAPI and dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "fastapi", "uvicorn[standard]", "python-multipart", "aiofiles"])
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form, Request
    from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn

try:
    import requests
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Installing audio processing libraries...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "requests", "librosa", "soundfile", "numpy"])
    import requests
    import librosa
    import soundfile as sf
    import numpy as np

from pydantic import BaseModel

# Load configuration
try:
    from config import config
    print(f"✓ Loaded config: Ollama at {config.OLLAMA_URL}")
except ImportError:
    class Config:
        OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
        API_HOST = '0.0.0.0'
        API_PORT = 8900
    config = Config()

# ============================================================================
# Configuration
# ============================================================================

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# ============================================================================
# Models
# ============================================================================

class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    DIAGNOSED = "diagnosed"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SEPARATING = "separating"
    SEPARATED = "separated"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    FAILED = "failed"


class DiagnosticReport(BaseModel):
    duration: float
    sample_rate: int
    speaking_time: float
    silence_time: float
    estimated_speakers: int
    mean_pitch: float
    pitch_range: List[float]
    audio_quality: str
    issues: List[str]
    recommendations: List[str]
    ollama_analysis: Optional[str] = None


class SeparationConfig(BaseModel):
    num_speakers: int = 2
    method: str = "gmm"


class CleaningConfig(BaseModel):
    remove_silence: bool = True
    reduce_noise: bool = True
    remove_clicks: bool = True
    normalize: bool = True


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    filename: str
    progress: int = 0  # 0-100 (UI progress indicator)
    progress_message: Optional[str] = None
    diagnostic_report: Optional[DiagnosticReport] = None
    separation_config: Optional[SeparationConfig] = None
    separated_files: Optional[Dict[str, str]] = None
    cleaned_files: Optional[Dict[str, str]] = None
    user_feedback: Optional[str] = None
    ollama_feedback_response: Optional[str] = None
    error: Optional[str] = None


class UserAction(BaseModel):
    action: str  # "proceed", "adjust", "cancel"
    num_speakers: Optional[int] = None
    method: Optional[str] = None


class UserFeedback(BaseModel):
    feedback: str
    rating: Optional[int] = None  # 1-5 stars


# ============================================================================
# In-memory job storage
# ============================================================================

jobs_db = {}


def _set_job_progress(job_id: str, progress: int, message: Optional[str] = None):
    """Best-effort in-memory progress updates for UI polling."""
    job = jobs_db.get(job_id)
    if not job:
        return
    job["progress"] = int(max(0, min(100, progress)))
    if message is not None:
        job["progress_message"] = message


# ============================================================================
# Import processing modules
# ============================================================================

# Try to import external modules, fall back to inline implementations
SEPARATOR_AVAILABLE = False
CLEANER_AVAILABLE = False

try:
    import sys
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from improved_speaker_separator_fixed import ImprovedSpeakerSeparator
    SEPARATOR_AVAILABLE = True
    print("✓ Loaded ImprovedSpeakerSeparator")
except Exception as e:
    print(f"⚠️  Could not load improved_speaker_separator: {e}")
    print("   Using inline implementation")

try:
    from audio_cleaner import AudioCleaner, clean_separated_speakers as clean_speakers_external
    CLEANER_AVAILABLE = True
    print("✓ Loaded AudioCleaner")
except Exception as e:
    print(f"⚠️  Could not load audio_cleaner: {e}")
    print("   Using inline implementation")

# ============================================================================
# Inline Implementations (fallback if modules not available)
# ============================================================================

if not SEPARATOR_AVAILABLE:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from scipy.ndimage import median_filter
    from scipy.signal import medfilt
    
    class ImprovedSpeakerSeparator:
        """Inline speaker separator implementation"""
        
        def __init__(self, output_dir: Path):
            self.output_dir = output_dir
            self.output_dir.mkdir(exist_ok=True, parents=True)
        
        def separate_speakers(self, audio_path: Path, n_speakers: int = 2, method: str = 'gmm'):
            """Separate speakers from audio file"""
            
            print(f"\nSeparating {n_speakers} speakers using {method}...")
            
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=16000)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Voice activity detection
            hop_length = 512
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            threshold = np.percentile(rms, 40)
            voice_activity = rms > threshold
            
            # Normalize features
            mfcc_norm = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-10)
            
            # Get features for active frames
            min_len = min(mfcc_norm.shape[1], len(voice_activity))
            mfcc_norm = mfcc_norm[:, :min_len]
            voice_activity = voice_activity[:min_len]
            
            active_frames = np.where(voice_activity)[0]
            
            if len(active_frames) < n_speakers * 10:
                raise ValueError(f"Not enough voice activity detected. Only {len(active_frames)} frames found.")
            
            active_features = mfcc_norm[:, active_frames].T
            
            # Cluster
            scaler = StandardScaler()
            active_features_scaled = scaler.fit_transform(active_features)
            
            labels = np.zeros(mfcc_norm.shape[1], dtype=int)
            
            if method == 'gmm':
                gmm = GaussianMixture(n_components=n_speakers, covariance_type='full', n_init=10, random_state=42)
                active_labels = gmm.fit_predict(active_features_scaled)
            else:  # kmeans
                kmeans = KMeans(n_clusters=n_speakers, n_init=20, random_state=42)
                active_labels = kmeans.fit_predict(active_features_scaled)
            
            labels[active_frames] = active_labels
            labels = medfilt(labels.astype(float), kernel_size=11).astype(int)
            
            # Separate and save
            results = {'speakers': {}}
            
            for speaker_id in range(n_speakers):
                speaker_mask = labels == speaker_id
                time_mask = np.repeat(speaker_mask, hop_length)[:len(y)]
                
                # Smooth mask
                from scipy.signal import windows
                window_size = 441
                smooth_mask = np.convolve(time_mask.astype(float), windows.hann(window_size), mode='same')
                smooth_mask = np.clip(smooth_mask, 0, 1)
                
                speaker_audio = y * smooth_mask
                
                # Normalize
                if np.max(np.abs(speaker_audio)) > 0:
                    speaker_audio = speaker_audio / np.max(np.abs(speaker_audio)) * 0.9
                
                # Save
                output_path = self.output_dir / f"speaker_{speaker_id + 1}.wav"
                sf.write(output_path, speaker_audio, sr)
                
                speaking_time = np.sum(speaker_mask) * hop_length / sr
                
                results['speakers'][f'speaker_{speaker_id + 1}'] = {
                    'path': str(output_path),
                    'speaking_time': float(speaking_time)
                }
            
            print(f"✓ Separated into {n_speakers} speakers")
            return results


if not CLEANER_AVAILABLE:
    from scipy import signal
    
    class AudioCleaner:
        """Inline audio cleaner implementation"""
        
        def __init__(self, sr: int = 16000):
            self.sr = sr
        
        def clean_audio(self, audio_path: Path, output_path: Path,
                       remove_silence: bool = True,
                       reduce_noise: bool = True,
                       remove_clicks: bool = True,
                       normalize: bool = True) -> dict:
            """Clean audio file"""
            
            print(f"\nCleaning: {audio_path.name}")
            
            y, sr = librosa.load(str(audio_path), sr=self.sr)
            original_duration = len(y) / sr
            
            # Remove clicks
            if remove_clicks:
                median_filtered = signal.medfilt(y, kernel_size=5)
                diff = np.abs(y - median_filtered)
                std = np.std(diff)
                clicks = diff > (3.0 * std)
                y[clicks] = median_filtered[clicks]
            
            # Reduce noise (simple spectral subtraction)
            if reduce_noise:
                D = librosa.stft(y)
                magnitude, phase = librosa.magphase(D)
                
                rms = librosa.feature.rms(y=y)[0]
                noise_frames = int(0.5 * sr / 512)
                quietest_idx = np.argpartition(rms, noise_frames)[:noise_frames]
                
                noise_magnitude = np.mean(magnitude[:, quietest_idx], axis=1, keepdims=True)
                clean_magnitude = magnitude - 2.0 * noise_magnitude
                clean_magnitude = np.maximum(clean_magnitude, 0.02 * magnitude)
                
                D_clean = clean_magnitude * phase
                y = librosa.istft(D_clean)
                
                if len(y) < original_duration * sr:
                    y = np.pad(y, (0, int(original_duration * sr) - len(y)))
                else:
                    y = y[:int(original_duration * sr)]
            
            # Remove silence
            if remove_silence:
                rms = librosa.feature.rms(y=y)[0]
                threshold = np.percentile(rms, 40)
                voice_activity = rms > threshold
                
                from scipy.ndimage import median_filter as med_filt
                voice_activity = med_filt(voice_activity.astype(float), size=5) > 0.5
                
                # Expand with padding
                padding_frames = int(0.1 * sr / 512)
                padded_activity = np.copy(voice_activity)
                for i in range(len(voice_activity)):
                    if voice_activity[i]:
                        start = max(0, i - padding_frames)
                        end = min(len(voice_activity), i + padding_frames + 1)
                        padded_activity[start:end] = True
                
                sample_mask = np.repeat(padded_activity, 512)[:len(y)]
                y = y[sample_mask]
            
            # Normalize
            if normalize:
                peak = np.max(np.abs(y))
                if peak > 0:
                    target_peak = 10 ** (-3.0 / 20)
                    y = y * (target_peak / peak)
            
            # Save
            sf.write(output_path, y, sr)
            
            return {
                'original_duration': original_duration,
                'output_duration': len(y) / sr,
                'output_file': str(output_path)
            }
    
    def clean_separated_speakers(input_dir: Path, output_dir: Path = None, **kwargs):
        """Clean all separated speaker files"""
        
        if output_dir is None:
            output_dir = input_dir / "cleaned"
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        speaker_files = list(input_dir.glob("speaker_*.wav"))
        
        if not speaker_files:
            return {}
        
        cleaner = AudioCleaner()
        results = {}
        
        for audio_file in sorted(speaker_files):
            output_file = output_dir / f"{audio_file.stem}_cleaned.wav"
            stats = cleaner.clean_audio(audio_file, output_file, **kwargs)
            results[audio_file.stem] = stats
        
        return results
    
    clean_speakers_external = clean_separated_speakers
else:
    # `clean_speakers_external` is already set by the successful import above:
    # from audio_cleaner import ... clean_separated_speakers as clean_speakers_external
    pass


# ============================================================================
# Diagnostic Module
# ============================================================================

async def diagnose_audio(audio_path: Path) -> DiagnosticReport:
    """Comprehensive audio diagnostic"""
    
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=None, duration=180)  # First 3 minutes
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Voice activity detection
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    threshold = np.percentile(rms, 40)
    voice_activity = rms > threshold
    speaking_time = np.sum(voice_activity) * hop_length / sr
    silence_time = duration - speaking_time
    
    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    mean_pitch = float(np.mean(pitch_values)) if pitch_values else 0.0
    pitch_range = [float(np.min(pitch_values)), float(np.max(pitch_values))] if pitch_values else [0.0, 0.0]
    
    # Estimate number of speakers from pitch clustering
    estimated_speakers = 2  # Default
    if len(pitch_values) > 100:
        from sklearn.cluster import KMeans
        pitch_array = np.array(pitch_values).reshape(-1, 1)
        
        for n_clusters in [2, 3, 4]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pitch_array)
            unique, counts = np.unique(labels, return_counts=True)
            min_cluster_size = np.min(counts)
            if min_cluster_size > len(pitch_values) * 0.1:
                estimated_speakers = n_clusters
                break
    
    # Assess audio quality
    issues = []
    recommendations = []
    
    peak = np.max(np.abs(y))
    rms_level = np.sqrt(np.mean(y ** 2))
    
    if peak < 0.01:
        issues.append("Very low audio level")
        recommendations.append("Increase audio volume/gain")
    
    if speaking_time < 10:
        issues.append("Less than 10 seconds of speech detected")
        recommendations.append("Use longer audio file (30+ seconds)")
    
    std_pitch = np.std(pitch_values) if pitch_values else 0
    if std_pitch < 20:
        issues.append("Low pitch variation - might be single speaker")
        recommendations.append("Verify audio has multiple distinct speakers")
    
    if duration < 20:
        issues.append("Audio too short")
        recommendations.append("Use audio at least 30 seconds long")
    
    # Overall quality assessment
    if not issues:
        audio_quality = "excellent"
    elif len(issues) <= 2:
        audio_quality = "good"
    else:
        audio_quality = "poor"
    
    # Get Ollama analysis
    ollama_analysis = None
    try:
        ollama_analysis = await get_ollama_diagnostic_analysis(
            duration, speaking_time, estimated_speakers, mean_pitch, audio_quality, issues
        )
    except Exception as e:
        print(f"Ollama analysis failed: {e}")
    
    return DiagnosticReport(
        duration=duration,
        sample_rate=int(sr),
        speaking_time=speaking_time,
        silence_time=silence_time,
        estimated_speakers=estimated_speakers,
        mean_pitch=mean_pitch,
        pitch_range=pitch_range,
        audio_quality=audio_quality,
        issues=issues,
        recommendations=recommendations,
        ollama_analysis=ollama_analysis
    )


async def get_ollama_diagnostic_analysis(duration, speaking_time, estimated_speakers, 
                                        mean_pitch, quality, issues) -> str:
    """Get Ollama's analysis of the diagnostic"""
    
    url = f"{config.OLLAMA_URL}/api/generate"
    
    prompt = f"""Audio file diagnostic analysis:
- Duration: {duration:.1f} seconds
- Speaking time: {speaking_time:.1f} seconds
- Estimated speakers: {estimated_speakers}
- Mean pitch: {mean_pitch:.1f} Hz
- Quality: {quality}
- Issues: {', '.join(issues) if issues else 'None'}

Provide a brief analysis (2-3 sentences):
1. Is this suitable for speaker separation?
2. What method would work best (GMM, K-means, or Spectral clustering)?
3. Any concerns or recommendations?"""

    data = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: requests.post(url, json=data, timeout=60)
    )
    
    if response.status_code == 200:
        return response.json()['response']
    
    return None


async def get_ollama_feedback_response(feedback: str, job_info: dict) -> str:
    """Get Ollama's response to user feedback"""
    
    url = f"{config.OLLAMA_URL}/api/generate"
    
    prompt = f"""User provided feedback on speaker separation results:

Job details:
- Number of speakers: {job_info.get('num_speakers', 'N/A')}
- Method used: {job_info.get('method', 'N/A')}
- Files cleaned: {job_info.get('cleaned', False)}

User feedback: "{feedback}"

Analyze the feedback and provide:
1. What the issue might be
2. Suggested improvements or next steps
3. If they should try a different method

Keep response brief and actionable (3-4 sentences)."""

    data = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: requests.post(url, json=data, timeout=60)
    )
    
    if response.status_code == 200:
        return response.json()['response']
    
    return "Thank you for your feedback. We'll use this to improve the service."


# ============================================================================
# Background Processing Tasks
# ============================================================================

async def process_separation_task(job_id: str, audio_path: Path, 
                                  num_speakers: int, method: str):
    """Background task to separate speakers"""
    
    try:
        jobs_db[job_id]['status'] = JobStatus.SEPARATING
        _set_job_progress(job_id, 35, "Preparing separation…")
        
        # Create output directory
        job_output_dir = OUTPUT_DIR / job_id / "separated"
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate speakers
        separator = ImprovedSpeakerSeparator(job_output_dir)
        _set_job_progress(job_id, 45, "Separating speakers…")

        # Progress callback called from worker thread
        def _progress_callback(pct: int, message: Optional[str] = None):
            try:
                _set_job_progress(job_id, pct, message)
            except Exception:
                pass

        # Load and process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: separator.separate_speakers(audio_path, num_speakers, method, progress_callback=_progress_callback)
        )
        _set_job_progress(job_id, 65, "Finalizing outputs…")
        
        # Build file URLs
        separated_files = {}
        for speaker, info in results['speakers'].items():
            separated_files[speaker] = f"/api/v1/download/{job_id}/separated/{speaker}.wav"
        
        jobs_db[job_id]['separated_files'] = separated_files
        jobs_db[job_id]['separation_results'] = results
        jobs_db[job_id]['status'] = JobStatus.SEPARATED
        _set_job_progress(job_id, 70, "Separation complete.")
        
    except Exception as e:
        jobs_db[job_id]['status'] = JobStatus.FAILED
        jobs_db[job_id]['error'] = str(e)
        _set_job_progress(job_id, 100, "Failed.")


async def process_cleaning_task(job_id: str, cleaning_config: CleaningConfig):
    """Background task to clean separated audio"""
    
    try:
        jobs_db[job_id]['status'] = JobStatus.CLEANING
        _set_job_progress(job_id, 75, "Cleaning audio…")
        
        separated_dir = OUTPUT_DIR / job_id / "separated"
        cleaned_dir = OUTPUT_DIR / job_id / "cleaned"
        
        # Clean files in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: clean_speakers_external(
                separated_dir,
                cleaned_dir,
                remove_silence=cleaning_config.remove_silence,
                reduce_noise=cleaning_config.reduce_noise,
                remove_clicks=cleaning_config.remove_clicks,
                normalize=cleaning_config.normalize
            )
        )
        
        # Build file URLs
        cleaned_files = {}
        for speaker in results.keys():
            cleaned_files[f"{speaker}_cleaned"] = f"/api/v1/download/{job_id}/cleaned/{speaker}_cleaned.wav"
        
        jobs_db[job_id]['cleaned_files'] = cleaned_files
        jobs_db[job_id]['cleaning_results'] = results
        jobs_db[job_id]['status'] = JobStatus.COMPLETED
        _set_job_progress(job_id, 100, "Done.")
        
    except Exception as e:
        jobs_db[job_id]['status'] = JobStatus.FAILED
        jobs_db[job_id]['error'] = str(e)
        _set_job_progress(job_id, 100, "Failed.")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Speaker Separation API - Complete Workflow",
    description="Upload → Diagnose → Separate → Clean → Feedback",
    version="2.0.0"
)

# Jinja frontend (UI)
# Resolve paths relative to this file so it works in Docker and any CWD.
_BASE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _BASE_DIR / "templates"
_STATIC_DIR = _BASE_DIR / "static"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static assets for UI (safe even if unused)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Speaker Separation API",
        "version": "2.0.0",
        "workflow": [
            "1. POST /api/v1/upload - Upload audio file",
            "2. GET /api/v1/jobs/{job_id}/diagnose - Get diagnostic report",
            "3. POST /api/v1/jobs/{job_id}/proceed - Confirm and separate",
            "4. GET /api/v1/jobs/{job_id} - Check status",
            "5. POST /api/v1/jobs/{job_id}/clean - Clean separated files",
            "6. GET /api/v1/download/{job_id}/{type}/{file} - Download",
            "7. POST /api/v1/jobs/{job_id}/feedback - Submit feedback"
        ],
        "docs": "/docs"
    }


@app.get("/final_speaker_frontend.html")
async def serve_frontend():
    """Legacy path: redirect to the new Jinja UI."""
    return RedirectResponse(url="/ui", status_code=302)


@app.get("/ui", response_class=HTMLResponse)
async def ui_index(request: Request):
    """Jinja UI: upload page."""
    # Recent jobs are in-memory only (per running server).
    recent_jobs = sorted(
        jobs_db.values(),
        key=lambda j: j.get("created_at", ""),
        reverse=True,
    )[:20]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "recent_jobs": recent_jobs,
        },
    )


@app.post("/ui/upload")
async def ui_upload_audio(request: Request, file: UploadFile = File(...)):
    """Jinja UI: handle upload and redirect to job page."""
    # Validate file type (same as API)
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {file_ext}")

    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")

    jobs_db[job_id] = {
        'job_id': job_id,
        'status': JobStatus.UPLOADED,
        'created_at': datetime.utcnow().isoformat(),
        'filename': file.filename,
        'progress': 10,
        'progress_message': 'Uploaded. Ready to diagnose.',
        'file_path': str(upload_path),
        'diagnostic_report': None,
        'separated_files': None,
        'cleaned_files': None,
        'user_feedback': None,
        'ollama_feedback_response': None,
        'error': None
    }

    return RedirectResponse(url=f"/ui/jobs/{job_id}", status_code=303)


@app.get("/ui/jobs/{job_id}", response_class=HTMLResponse)
async def ui_job(request: Request, job_id: str):
    """Jinja UI: job detail page (JS will poll the API)."""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")

    return templates.TemplateResponse(
        "job.html",
        {
            "request": request,
            "job_id": job_id,
            "job": jobs_db[job_id],
        },
    )


@app.get("/health")
async def health():
    """Health check"""
    ollama_status = "disconnected"
    try:
        response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "error"
    except:
        pass
    
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "ollama_url": config.OLLAMA_URL
    }


@app.post("/api/v1/upload", response_model=JobResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Step 1: Upload audio file
    Returns job_id and starts diagnostic
    """
    
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {file_ext}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save file
    upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")
    
    # Create job
    jobs_db[job_id] = {
        'job_id': job_id,
        'status': JobStatus.UPLOADED,
        'created_at': datetime.utcnow().isoformat(),
        'filename': file.filename,
        'progress': 10,
        'progress_message': 'Uploaded. Ready to diagnose.',
        'file_path': str(upload_path),
        'diagnostic_report': None,
        'separated_files': None,
        'cleaned_files': None,
        'user_feedback': None,
        'ollama_feedback_response': None,
        'error': None
    }
    
    return JobResponse(**jobs_db[job_id])


@app.get("/api/v1/jobs/{job_id}/diagnose", response_model=JobResponse)
async def get_diagnostic(job_id: str):
    """
    Step 2: Get diagnostic report
    Analyzes audio and provides recommendations
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    # Run diagnostic if not done
    if not job['diagnostic_report']:
        try:
            _set_job_progress(job_id, 15, "Diagnosing audio…")
            audio_path = Path(job['file_path'])
            diagnostic = await diagnose_audio(audio_path)
            
            jobs_db[job_id]['diagnostic_report'] = diagnostic.dict()
            jobs_db[job_id]['status'] = JobStatus.DIAGNOSED
            _set_job_progress(job_id, 25, "Diagnosis complete.")
            
        except Exception as e:
            jobs_db[job_id]['error'] = f"Diagnostic failed: {str(e)}"
            _set_job_progress(job_id, 100, "Failed.")
            raise HTTPException(500, f"Diagnostic failed: {e}")
    
    return JobResponse(**jobs_db[job_id])


@app.post("/api/v1/jobs/{job_id}/proceed", response_model=JobResponse)
async def proceed_with_separation(
    job_id: str,
    background_tasks: BackgroundTasks,
    action: UserAction
):
    """
    Step 3: User confirms and starts separation
    User can adjust num_speakers and method
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    if action.action == "cancel":
        jobs_db[job_id]['status'] = JobStatus.FAILED
        jobs_db[job_id]['error'] = "Cancelled by user"
        return JobResponse(**jobs_db[job_id])
    
    # Get separation config
    diagnostic = job['diagnostic_report']
    num_speakers = action.num_speakers or diagnostic['estimated_speakers']
    method = action.method or "gmm"
    
    jobs_db[job_id]['separation_config'] = {
        'num_speakers': num_speakers,
        'method': method
    }
    
    # Start separation in background
    _set_job_progress(job_id, 30, "Queued for separation…")
    background_tasks.add_task(
        process_separation_task,
        job_id,
        Path(job['file_path']),
        num_speakers,
        method
    )
    
    jobs_db[job_id]['status'] = JobStatus.SEPARATING
    
    return JobResponse(**jobs_db[job_id])


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Step 4: Get job status
    Poll this to check progress
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    return JobResponse(**jobs_db[job_id])


@app.get("/api/v1/jobs/{job_id}/stream")
async def stream_job_status(job_id: str):
    """
    Real-time job status using Server-Sent Events (SSE)
    Client opens EventSource connection; server streams updates as job progresses
    """
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    async def event_generator():
        """Stream job status updates"""
        last_progress = -1
        last_status = None
        
        while True:
            job = jobs_db.get(job_id)
            if not job:
                break
            
            # Send update if progress or status changed
            progress = job.get('progress', 0)
            status = job.get('status')
            message = job.get('progress_message', '')
            
            if progress != last_progress or status != last_status:
                yield f"data: {json.dumps({'progress': progress, 'status': status, 'message': message})}\n\n"
                last_progress = progress
                last_status = status
            
            # Stop if job completed/failed
            if status in [JobStatus.COMPLETED, JobStatus.SEPARATED, JobStatus.FAILED]:
                # Send final update with full job data
                yield f"data: {json.dumps(jobs_db[job_id])}\n\n"
                break
            
            await asyncio.sleep(0.5)  # Update every 500ms
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )



@app.post("/api/v1/jobs/{job_id}/clean", response_model=JobResponse)
async def clean_separated_files(
    job_id: str,
    background_tasks: BackgroundTasks,
    cleaning_config: CleaningConfig
):
    """
    Step 5: Clean separated audio files
    Removes silence, noise, clicks, normalizes
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    if job['status'] != JobStatus.SEPARATED:
        raise HTTPException(400, "Job not in separated state")
    
    # Start cleaning
    background_tasks.add_task(
        process_cleaning_task,
        job_id,
        cleaning_config
    )
    
    jobs_db[job_id]['status'] = JobStatus.CLEANING
    
    return JobResponse(**jobs_db[job_id])


@app.get("/api/v1/download/{job_id}/{file_type}/{filename}")
async def download_file(job_id: str, file_type: str, filename: str, inline: bool = Query(False)):
    """
    Step 6: Download files
    file_type: 'separated' or 'cleaned'
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    file_path = OUTPUT_DIR / job_id / file_type / filename
    
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    
    # If `filename` is provided to FileResponse, Starlette sets Content-Disposition as attachment.
    # For inline playback in the UI, optionally override to `inline`.
    headers = None
    if inline:
        headers = {"Content-Disposition": f'inline; filename="{filename}"'}

    return FileResponse(file_path, media_type="audio/wav", filename=filename, headers=headers)


@app.get("/api/v1/jobs/{job_id}/original")
async def stream_original_audio(job_id: str):
    """Stream the original uploaded file inline (for UI preview)."""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    job = jobs_db[job_id]
    file_path = Path(job["file_path"])
    if not file_path.exists():
        raise HTTPException(404, "Original file not found")

    ext = file_path.suffix.lower()
    media_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    media_type = media_map.get(ext, "application/octet-stream")
    headers = {"Content-Disposition": f'inline; filename="{job.get("filename","audio"+ext)}"'}
    return FileResponse(file_path, media_type=media_type, headers=headers)


@app.post("/api/v1/jobs/{job_id}/feedback", response_model=JobResponse)
async def submit_feedback(job_id: str, feedback: UserFeedback):
    """
    Step 7: Submit user feedback
    Gets Ollama's analysis and suggestions
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    jobs_db[job_id]['user_feedback'] = feedback.feedback
    jobs_db[job_id]['user_rating'] = feedback.rating
    
    # Get Ollama's response to feedback
    try:
        job_info = {
            'num_speakers': job.get('separation_config', {}).get('num_speakers'),
            'method': job.get('separation_config', {}).get('method'),
            'cleaned': job.get('cleaned_files') is not None
        }
        
        ollama_response = await get_ollama_feedback_response(
            feedback.feedback,
            job_info
        )
        
        jobs_db[job_id]['ollama_feedback_response'] = ollama_response
        
    except Exception as e:
        print(f"Ollama feedback response failed: {e}")
    
    return JobResponse(**jobs_db[job_id])


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and all associated files"""
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    # Delete files
    for ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
        upload_path = UPLOAD_DIR / f"{job_id}{ext}"
        if upload_path.exists():
            upload_path.unlink()
    
    output_path = OUTPUT_DIR / job_id
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Remove from database
    del jobs_db[job_id]
    
    return {"message": "Job deleted successfully"}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker Separation API Server")
    parser.add_argument("--host", default=config.API_HOST)
    parser.add_argument("--port", type=int, default=config.API_PORT)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Speaker Separation API - Complete Workflow")
    print("=" * 60)
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"Ollama: {config.OLLAMA_URL}")
    print("=" * 60)
    print("\nWorkflow:")
    print("  1. Upload file → /api/v1/upload")
    print("  2. Get diagnosis → /api/v1/jobs/{id}/diagnose")
    print("  3. Confirm separation → /api/v1/jobs/{id}/proceed")
    print("  4. Check status → /api/v1/jobs/{id}")
    print("  5. Clean files → /api/v1/jobs/{id}/clean")
    print("  6. Download → /api/v1/download/{id}/{type}/{file}")
    print("  7. Submit feedback → /api/v1/jobs/{id}/feedback")
    print("=" * 60)
    
    uvicorn.run(
        "final_speaker_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
