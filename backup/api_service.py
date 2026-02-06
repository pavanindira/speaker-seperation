#!/usr/bin/env python3
"""
FastAPI service for Speaker Separation
Provides REST API endpoints for speaker diarization and separation
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from speaker_separator import ConversationProcessor, OllamaAssistant

# Audio format conversion
try:
    import pydub
    AUDIO_CONVERSION_AVAILABLE = True
except ImportError:
    AUDIO_CONVERSION_AVAILABLE = False


class OutputFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class SeparationMethod(str, Enum):
    AUTO = "auto"
    CLUSTERING = "clustering"
    FREQUENCY = "frequency"
    DEMUCS = "demucs"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    filename: str
    num_speakers: int
    method: str
    output_format: str
    error: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Speaker Separator API",
    description="Separate individual speakers from multi-speaker conversation audio",
    version="1.0.0"
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job database
jobs_db: Dict[str, dict] = {}


def convert_audio_format(input_path: Path, output_path: Path, target_format: str) -> bool:
    """Convert audio to target format"""
    if not AUDIO_CONVERSION_AVAILABLE:
        return False
    
    try:
        from pydub import AudioSegment
        
        # Load audio
        audio = AudioSegment.from_file(str(input_path))
        
        # Export to target format
        if target_format == "mp3":
            audio.export(str(output_path), format="mp3", bitrate="192k")
        elif target_format == "flac":
            audio.export(str(output_path), format="flac")
        elif target_format == "ogg":
            audio.export(str(output_path), format="ogg")
        elif target_format == "wav":
            audio.export(str(output_path), format="wav")
        else:
            return False
        
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False


async def process_separation_job(
    job_id: str,
    audio_path: Path,
    num_speakers: int,
    method: str,
    output_format: str,
    ollama_url: str
):
    """Background task for processing speaker separation"""
    try:
        jobs_db[job_id]['status'] = JobStatus.PROCESSING
        
        # Initialize Ollama assistant
        assistant = OllamaAssistant(ollama_url=ollama_url)
        
        # Process conversation
        processor = ConversationProcessor(ollama_assistant=assistant)
        result = processor.process_conversation(
            audio_path,
            num_speakers=num_speakers,
            method=method
        )
        
        # Create output directory for job
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        
        # Convert audio files if needed
        speakers_info = {}
        for speaker_name, speaker_info in result.get("speakers", {}).items():
            original_path = Path(speaker_info["path"])
            
            if output_format == "wav":
                # Already WAV, just copy
                output_path = job_output_dir / f"{speaker_name}.wav"
                shutil.copy(original_path, output_path)
            else:
                # Convert to target format
                output_path = job_output_dir / f"{speaker_name}.{output_format}"
                success = convert_audio_format(original_path, output_path, output_format)
                if not success:
                    # Fallback to copy if conversion fails
                    shutil.copy(original_path, output_path.with_suffix('.wav'))
            
            speakers_info[speaker_name] = {
                "path": str(output_path),
                "url": f"/api/v1/download/{job_id}/{speaker_name}",
                "speaking_time": speaker_info.get("speaking_time", 0)
            }
        
        jobs_db[job_id]['status'] = JobStatus.COMPLETED
        jobs_db[job_id]['speakers'] = speakers_info
        jobs_db[job_id]['output_format'] = output_format
        jobs_db[job_id]['ollama_analysis'] = result.get('ollama_analysis')
        
    except Exception as e:
        jobs_db[job_id]['status'] = JobStatus.FAILED
        jobs_db[job_id]['error'] = str(e)
        print(f"Job {job_id} failed: {e}")


@app.get("/", tags=["Health"])
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "service": "Speaker Separator API",
        "version": "1.0.0"
    }


@app.post("/api/v1/separate", response_model=JobResponse, tags=["Separation"])
async def separate_speakers(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_speakers: int = Query(2, ge=1, le=10),
    method: str = Query("auto", regex="^(auto|clustering|frequency|demucs)$"),
    output_format: str = Query("wav", regex="^(wav|mp3|flac|ogg)$"),
    ollama_url: Optional[str] = Query(None)
):
    """
    Separate individual speakers from conversation audio
    
    - **file**: Audio file (mp3, wav, flac, ogg, m4a)
    - **num_speakers**: Number of speakers to separate (1-10)
    - **method**: Separation method (auto, clustering, frequency)
    - **output_format**: Output audio format (wav, mp3, flac, ogg)
    - **ollama_url**: Ollama API URL (uses env var if not provided)
    """
    
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.mpeg'}
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
    
    # Get Ollama URL
    if not ollama_url:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # Create job record
    jobs_db[job_id] = {
        'job_id': job_id,
        'status': JobStatus.PENDING,
        'created_at': datetime.utcnow().isoformat(),
        'filename': file.filename,
        'num_speakers': num_speakers,
        'method': method,
        'output_format': output_format,
        'speakers': {},
        'ollama_analysis': None,
        'error': None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_separation_job,
        job_id,
        upload_path,
        num_speakers,
        method,
        output_format,
        ollama_url
    )
    
    return JobResponse(**jobs_db[job_id])


@app.get("/api/v1/status/{job_id}", response_model=JobResponse, tags=["Status"])
async def get_job_status(job_id: str):
    """Get the status of a separation job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobResponse(**jobs_db[job_id])


@app.get("/api/v1/download/{job_id}/{speaker_name}", tags=["Download"])
async def download_speaker_file(job_id: str, speaker_name: str):
    """Download a separated speaker audio file"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    
    if job['status'] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    if speaker_name not in job['speakers']:
        raise HTTPException(
            status_code=404,
            detail=f"Speaker {speaker_name} not found in job results"
        )
    
    file_path = job['speakers'][speaker_name]['path']
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"{job_id}_{speaker_name}.wav"
    )


@app.get("/api/v1/results/{job_id}", tags=["Results"])
async def get_job_results(job_id: str):
    """Get full results of a completed job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    
    if job['status'] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    return {
        "job_id": job_id,
        "status": job['status'],
        "created_at": job['created_at'],
        "filename": job['filename'],
        "num_speakers": job['num_speakers'],
        "method": job['method'],
        "output_format": job['output_format'],
        "speakers": job['speakers'],
        "ollama_analysis": job['ollama_analysis']
    }


@app.get("/api/v1/jobs", tags=["Jobs"])
async def list_jobs():
    """List all jobs with their current status"""
    return {
        "total": len(jobs_db),
        "jobs": [
            {
                "job_id": job_id,
                "status": job['status'],
                "created_at": job['created_at'],
                "filename": job['filename'],
                "num_speakers": job['num_speakers']
            }
            for job_id, job in jobs_db.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
