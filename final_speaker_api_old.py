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
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("Installing FastAPI and dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "fastapi", "uvicorn[standard]", "python-multipart", "aiofiles"])
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
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
        API_PORT = 8000
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


# ============================================================================
# Import processing modules
# ============================================================================

# We'll import the actual processing functions
from improved_speaker_separator import ImprovedSpeakerSeparator
from audio_cleaner import AudioCleaner, clean_separated_speakers


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
        
        # Create output directory
        job_output_dir = OUTPUT_DIR / job_id / "separated"
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate speakers
        separator = ImprovedSpeakerSeparator(job_output_dir)
        
        # Load and process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: separator.separate_speakers(audio_path, num_speakers, method)
        )
        
        # Build file URLs
        separated_files = {}
        for speaker, info in results['speakers'].items():
            separated_files[speaker] = f"/api/v1/download/{job_id}/separated/{speaker}.wav"
        
        jobs_db[job_id]['separated_files'] = separated_files
        jobs_db[job_id]['separation_results'] = results
        jobs_db[job_id]['status'] = JobStatus.SEPARATED
        
    except Exception as e:
        jobs_db[job_id]['status'] = JobStatus.FAILED
        jobs_db[job_id]['error'] = str(e)


async def process_cleaning_task(job_id: str, cleaning_config: CleaningConfig):
    """Background task to clean separated audio"""
    
    try:
        jobs_db[job_id]['status'] = JobStatus.CLEANING
        
        separated_dir = OUTPUT_DIR / job_id / "separated"
        cleaned_dir = OUTPUT_DIR / job_id / "cleaned"
        
        # Clean files in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: clean_separated_speakers(
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
        
    except Exception as e:
        jobs_db[job_id]['status'] = JobStatus.FAILED
        jobs_db[job_id]['error'] = str(e)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Speaker Separation API - Complete Workflow",
    description="Upload → Diagnose → Separate → Clean → Feedback",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Serve the frontend HTML"""
    frontend_path = Path("final_speaker_frontend.html")
    if frontend_path.exists():
        return FileResponse(frontend_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")


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
            audio_path = Path(job['file_path'])
            diagnostic = await diagnose_audio(audio_path)
            
            jobs_db[job_id]['diagnostic_report'] = diagnostic.dict()
            jobs_db[job_id]['status'] = JobStatus.DIAGNOSED
            
        except Exception as e:
            jobs_db[job_id]['error'] = f"Diagnostic failed: {str(e)}"
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
    background_tasks.add_task(
        process_separation_task,
        job_id,
        Path(job['file_path']),
        num_speakers,
        method
    )
    
    jobs_db[job_id]['status'] = JobStatus.SEPARATING
    
    return JobResponse(**jobs_db[job_id])


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Step 4: Get job status
    Poll this to check progress
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    return JobResponse(**jobs_db[job_id])


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
async def download_file(job_id: str, file_type: str, filename: str):
    """
    Step 6: Download files
    file_type: 'separated' or 'cleaned'
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    file_path = OUTPUT_DIR / job_id / file_type / filename
    
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )


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
