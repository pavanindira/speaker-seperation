#!/usr/bin/env python3
"""
Complete Speaker Separation API - Final Version with Enhanced Logging & Error Handling
"""

import os
import sys
import shutil
import uuid
import asyncio
import atexit
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
from pathlib import Path
import mimetypes

# ============================================================================
# CRITICAL: Import custom logging and error handling FIRST
# ============================================================================

from logger_wrapper import (
    logger, log_module_loaded, log_system_startup, 
    log_config_loaded, log_separator_started, 
    log_separator_completed, log_temp_cleanup, LogContext
)

from error_handler import (
    ValidationError, NotFoundError, FileSizeError, ProcessingError,
    OllamaError, TimeoutError, ErrorRecoveryHandler, handle_errors,
    retry_with_backoff, timeout_handler, RetryConfig
)

# ============================================================================
# Dependencies Installation (With Logging)
# ============================================================================

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form, Request
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
except ImportError:
    logger.critical(
        "FastAPI dependencies not found, installing...",
        component='startup'
    )
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "fastapi", "uvicorn[standard]", "python-multipart", "aiofiles"])
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form, Request
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse, HTMLResponse
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
    logger.critical(
        "Audio processing libraries not found, installing...",
        component='startup'
    )
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "requests", "librosa", "soundfile", "numpy"])
    import requests
    import librosa
    import soundfile as sf
    import numpy as np

from fastapi import Depends, Response
from pydantic import BaseModel

# ============================================================================
# Configuration with Error Handling
# ============================================================================

try:
    from config import config
    log_config_loaded(location="config.py", status="success")
    logger.info(
        "Configuration loaded",
        extra={'ollama_url': config.OLLAMA_URL, 'api_port': config.API_PORT},
        component='config'
    )
except ImportError as e:
    logger.warning(
        f"Using default config: {e}",
        component='config'
    )
    class Config:
        OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
        API_HOST = '0.0.0.0'
        API_PORT = 8000
    config = Config()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
OLLAMA_TIMEOUT = 120  # seconds

# ============================================================================
# Data Models
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
    progress: int = 0
    progress_message: str = ""
    file_size: Optional[int] = None
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
# INITIALIZE FastAPI APP FIRST (CRITICAL FIX #1)
# ============================================================================

app = FastAPI(
    title="Speaker Separation API - Complete Workflow",
    description="Upload → Diagnose → Separate → Clean → Feedback",
    version="2.0.0"
)

from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from error_handler import APIError, UserFriendlyErrorFormatter
import uuid

# =============================================================================
# EXCEPTION HANDLERS - Add this section after app = FastAPI()
# =============================================================================

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """
    Handle all APIError exceptions with user-friendly responses
    This catches: ValidationError, FileSizeError, ProcessingError, etc.
    """
    
    # Generate request ID if not exists
    request_id = str(getattr(request.state, 'request_id', uuid.uuid4()))
    
    # Format error for users
    formatted_error = UserFriendlyErrorFormatter.format_error(exc, request_id)
    
    # Log for developers/monitoring
    logger.error(
        f"API Error: {exc.message}",
        extra={
            'error_id': formatted_error['error']['error_id'],
            'error_type': type(exc).__name__,
            'status_code': exc.status_code,
            'path': str(request.url.path),
            'method': request.method,
            'user_message': formatted_error['error']['message']
        },
        component='error_handler'
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=formatted_error
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """
    Catch-all for unexpected errors that aren't APIErrors
    """
    
    # Convert to APIError
    api_error = APIError(
        message=str(exc),
        status_code=500,
        user_message="An unexpected error occurred. Our team has been notified."
    )
    
    request_id = str(uuid.uuid4())
    formatted_error = UserFriendlyErrorFormatter.format_error(api_error, request_id)
    
    # Log as critical since this is unexpected
    logger.critical(
        f"Unexpected error: {exc}",
        exception=exc,
        extra={
            'error_id': formatted_error['error']['error_id'],
            'path': str(request.url.path),
            'method': request.method,
            'request_id': request_id
        },
        component='error_handler'
    )
    
    return JSONResponse(
        status_code=500,
        content=formatted_error
    )

# Jinja templates and static files
_BASE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _BASE_DIR / "templates"
_STATIC_DIR = _BASE_DIR / "static"

try:
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
except Exception as e:
    print(f"⚠️  Templates directory not found: {e}")
    templates = None

# Static assets
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ============================================================================
# Security Configuration (CRITICAL FIX #2: Load from environment)
# ============================================================================

try:
    from security_improvements import (
        verify_token, verify_api_key, SecurityHeadersMiddleware,
        AuditLogger, SecurityValidator, RedisRateLimiter
    )
    
    app.add_middleware(SecurityHeadersMiddleware)
    audit_logger = AuditLogger()
    validator = SecurityValidator()
    
    # Initialize Redis rate limiter
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    rate_limiter = RedisRateLimiter(redis_url)
    
    SECURITY_AVAILABLE = True
    print("✓ Security features loaded: JWT, Rate Limiting, Audit Logging")
except ImportError as e:
    print(f"⚠️  Security modules not available: {e}")
    SECURITY_AVAILABLE = False
    audit_logger = None
    validator = None
    rate_limiter = None


# ============================================================================
# Monitoring & Observability
# ============================================================================

try:
    from monitoring_observability import (
        StructuredLogger, PerformanceMonitor,
        HealthChecker, MonitoringMiddleware
    )
    
    logger = StructuredLogger('speaker-api')
    perf_monitor = PerformanceMonitor()
    health_checker = HealthChecker()
    app.add_middleware(MonitoringMiddleware, logger=logger, perf_monitor=perf_monitor)
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Monitoring modules not available: {e}")
    MONITORING_AVAILABLE = False

# CORS Configuration - MUST be added LAST so it's executed FIRST (LIFO order)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics endpoint (CRITICAL FIX #3: Safe fallback for prometheus_client)
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        from prometheus_client import generate_latest, REGISTRY
        return Response(generate_latest(REGISTRY), media_type="text/plain")
    except ImportError:
        return JSONResponse(
            {"error": "prometheus_client not installed. Install with: pip install prometheus-client"},
            status_code=501
        )


# ============================================================================
# Authentication Endpoints
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login endpoint to get JWT token"""
    if not SECURITY_AVAILABLE:
        raise HTTPException(status_code=501, detail="Authentication not available")
    
    from security_improvements import verify_password, create_access_token
    from datetime import timedelta
    
    # In production, verify against database
    # For demo, accept any username with password "demo123"
    # TODO: Replace with actual user database
    if request.password != "demo123":
        if audit_logger:
            audit_logger.log_auth_attempt(request.username, False)
        raise HTTPException(
            status_code=401, 
            detail="Invalid credentials"
        )
    
    # Create token
    access_token = create_access_token(
        data={"sub": request.username},
        expires_delta=timedelta(minutes=config.JWT_EXPIRE_MINUTES)
    )
    
    if audit_logger:
        audit_logger.log_auth_attempt(request.username, True)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=config.JWT_EXPIRE_MINUTES * 60
    )


@app.get("/api/v1/auth/verify")
async def verify_auth(user: dict = Depends(verify_token) if SECURITY_AVAILABLE else None):
    """Verify if current token is valid"""
    if not SECURITY_AVAILABLE:
        raise HTTPException(status_code=501, detail="Authentication not available")
    return {"valid": True, "user": user}


@app.get("/api/v1/rate-limit/status")
async def get_rate_limit_status(request: Request):
    """Get current rate limit status for client"""
    client_ip = request.client.host if request.client else "unknown"
    
    if rate_limiter and SECURITY_AVAILABLE:
        remaining = rate_limiter.get_remaining(
            f"upload:{client_ip}",
            max_requests=config.RATE_LIMIT_UPLOADS,
            window=config.RATE_LIMIT_WINDOW
        )
        return {
            "client_ip": client_ip,
            "remaining_uploads": remaining,
            "max_uploads": config.RATE_LIMIT_UPLOADS,
            "window_seconds": config.RATE_LIMIT_WINDOW
        }
    
    return {"rate_limiting": "disabled"}


# ============================================================================
# Load Audio Processing Modules with Logging
# ============================================================================

SEPARATOR_AVAILABLE = False
CLEANER_AVAILABLE = False

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from improved_speaker_separator import ImprovedSpeakerSeparator
    SEPARATOR_AVAILABLE = True
    log_module_loaded("ImprovedSpeakerSeparator", "success")
except Exception as e:
    log_module_loaded("ImprovedSpeakerSeparator", "failed", reason=str(e))

try:
    from audio_cleaner import AudioCleaner, clean_separated_speakers
    clean_speakers_external = clean_separated_speakers
    CLEANER_AVAILABLE = True
    log_module_loaded("AudioCleaner", "success")
except Exception as e:
    log_module_loaded("AudioCleaner", "failed", reason=str(e))
    clean_speakers_external = None

# ============================================================================
# Temp File Cleanup with Logging
# ============================================================================

def cleanup_temp_files():
    """Clean up temporary files on shutdown with logging"""
    try:
        if TEMP_DIR.exists():
            for file in TEMP_DIR.glob("*"):
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
        log_temp_cleanup("success")
    except Exception as e:
        log_temp_cleanup("failed", error=str(e))

atexit.register(cleanup_temp_files)

# ============================================================================
# Updated File Size Validation with Error Handling
# ============================================================================

def validate_file_size(file_size: int) -> bool:
    """Validate file size with proper error handling"""
    if file_size > MAX_FILE_SIZE:
        file_size_mb = file_size / 1024 / 1024
        max_size_mb = MAX_FILE_SIZE / 1024 / 1024
        raise FileSizeError(
            file_size_mb, 
            max_size_mb,
            user_message=f"Your file is {file_size_mb:.1f}MB. Maximum size is {max_size_mb}MB. Try compressing it or splitting into smaller files."
        )
    return True

# ============================================================================
# Updated API Endpoints with Error Handling & Logging
# ============================================================================

@app.post("/api/v1/upload", response_model=JobResponse)
@handle_errors()
async def upload_audio(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key) if SECURITY_AVAILABLE else None
):
    """Step 1: Upload audio file with comprehensive validation"""
    
    # Get client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if rate_limiter and SECURITY_AVAILABLE:
        if not rate_limiter.check_rate_limit(
            f"upload:{client_ip}", 
            max_requests=config.RATE_LIMIT_UPLOADS,
            window=config.RATE_LIMIT_WINDOW
        ):
            remaining = rate_limiter.get_remaining(
                f"upload:{client_ip}",
                max_requests=config.RATE_LIMIT_UPLOADS,
                window=config.RATE_LIMIT_WINDOW
            )
            raise ValidationError(
                message="Rate limit exceeded",
                field="rate_limit",
                user_message=f"Too many uploads. Please wait before trying again. Remaining: {remaining}",
                details={'remaining_requests': remaining}
            )
    
    with LogContext("file_upload", component="upload", job_id=None):
        # Sanitize filename
        safe_filename = validator.validate_filename(file.filename) if validator else file.filename
        
        # Validate file type
        allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
        file_ext = Path(safe_filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            if audit_logger:
                audit_logger.log_upload_rejection(client_ip, safe_filename, f"Invalid file type: {file_ext}")
            raise ValidationError(
                message=f"File extension validation failed: {file_ext}",
                field="filename",
                user_message=f"We can't process {file_ext} files. Please upload an audio file (MP3, WAV, M4A, FLAC).",
                details={
                    'uploaded_type': file_ext,
                    'supported_types': list(allowed_extensions),
                    'filename': safe_filename
                }
            )
        
        # Check file size from Content-Length header BEFORE reading the file
        # This prevents memory issues with large files
        content_length = request.headers.get('content-length')
        if content_length:
            estimated_size = int(content_length)
            # Account for multipart encoding overhead (~10%)
            if estimated_size > MAX_FILE_SIZE * 1.1:
                file_size_mb = estimated_size / 1024 / 1024
                max_size_mb = MAX_FILE_SIZE / 1024 / 1024
                raise FileSizeError(
                    file_size_mb,
                    max_size_mb,
                    user_message=f"Your file is approximately {file_size_mb:.1f}MB. Maximum size is {max_size_mb:.0f}MB. Try compressing it or splitting into smaller files."
                )
        
        # Read and validate file size
        file_content = await file.read()
        file_size = len(file_content)
        validate_file_size(file_size)
        
        # Validate file content is actually audio
        try:
            file_obj = type('obj', (object,), {'read': lambda: file_content, 'seek': lambda x: None})()
            if validator and not validator.is_valid_audio(file_obj):
                if audit_logger:
                    audit_logger.log_upload_rejection(client_ip, safe_filename, "Invalid audio content")
                raise ValidationError(
                    "File content is not valid audio",
                    field="file",
                    user_message="The uploaded file does not appear to be a valid audio file"
                )
        except Exception as e:
            logger.warning(
                "Audio validation skipped due to missing validator",
                exception=e,
                component='upload'
            )
        
        # Generate job ID and save file
        job_id = str(uuid.uuid4())
        # Save file
        file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # CRITICAL: Store the file path in the job!
        jobs_db[job_id] = {
            'job_id': job_id,
            'status': 'uploaded',
            'original_file': str(file_path),  # ← ADD THIS!
            'filename': file.filename,
            'upload_time': datetime.utcnow().isoformat()
        }
        
        return {'job_id': job_id}
        upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"
        
        try:
            with open(upload_path, "wb") as buffer:
                buffer.write(file_content)
            file_content = None  # Clear memory
        except IOError as e:
            ErrorRecoveryHandler.handle_file_not_found(str(upload_path))
        except OSError as e:
            if e.errno == 28:  # No space left on device
                ErrorRecoveryHandler.handle_disk_full()
            raise ProcessingError("file_save", str(e))
        
        # Create job record
        jobs_db[job_id] = {
            'job_id': job_id,
            'status': JobStatus.UPLOADED,
            'created_at': datetime.utcnow().isoformat(),
            'filename': safe_filename,
            'progress': 10,
            'progress_message': 'Uploaded. Ready to diagnose.',
            'file_path': str(upload_path),
            'file_size': file_size,
            'diagnostic_report': None,
            'separated_files': None,
            'cleaned_files': None,
            'user_feedback': None,
            'ollama_feedback_response': None,
            'error': None
        }
        
        # Log successful upload
        if audit_logger:
            audit_logger.log_file_upload(client_ip, safe_filename, file_size)
        
        logger.info(
            "File uploaded successfully",
            extra={
                'job_id': job_id,
                'filename': safe_filename,
                'file_size_mb': file_size / 1024 / 1024,
                'file_type': file_ext,
                'client_ip': client_ip
            },
            component='upload'
        )
        
        return JobResponse(**jobs_db[job_id])

# ============================================================================
# Updated Ollama Requests with Timeout & Retry
# ============================================================================

@timeout_handler(timeout_seconds=120)
@retry_with_backoff(RetryConfig(max_attempts=2))
async def get_ollama_diagnostic_analysis(duration, speaking_time, estimated_speakers, 
                                        mean_pitch, quality, issues) -> str:
    """Get Ollama's analysis with timeout and retry"""
    
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
    
    try:
        logger.debug(
            "Calling Ollama API for diagnostic analysis",
            component='ollama'
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, json=data, timeout=OLLAMA_TIMEOUT)
        )
        
        if response.status_code == 200:
            result = response.json()['response']
            logger.info(
                "Ollama analysis completed",
                extra={'response_length': len(result)},
                component='ollama'
            )
            return result
        else:
            raise OllamaError(
                f"Ollama API returned status {response.status_code}",
                status_code=response.status_code
            )
    except requests.Timeout:
        logger.error(
            f"Ollama diagnostic timed out",
            extra={'timeout_seconds': OLLAMA_TIMEOUT},
            component='ollama'
        )
        ErrorRecoveryHandler.handle_ollama_unavailable(OLLAMA_TIMEOUT)
    except requests.ConnectionError as e:
        logger.error(
            "Failed to connect to Ollama service",
            exception=e,
            component='ollama'
        )
        raise OllamaError("Cannot reach Ollama service", status_code=502)


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
    try:
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, json=data, timeout=OLLAMA_TIMEOUT)
        )
        
        if response.status_code == 200:
            return response.json()['response']
    except requests.Timeout:
        print(f"Ollama feedback timeout after {OLLAMA_TIMEOUT}s")
    except Exception as e:
        print(f"Ollama feedback request failed: {e}")
    
    return "Thank you for your feedback. We'll use this to improve the service."


# ============================================================================
# Helper Functions
# ============================================================================

async def diagnose_audio(audio_path: Path) -> DiagnosticReport:
    """Analyze audio file and generate diagnostic report"""
    try:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate speaking time and silence
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        energy = np.mean(S_db, axis=0)
        threshold = np.mean(energy) - np.std(energy)
        speaking_frames = np.sum(energy > threshold)
        speaking_time = speaking_frames * (len(y) / sr) / len(energy)
        silence_time = duration - speaking_time
        
        # Estimate number of speakers using spectral clustering
        estimated_speakers = max(2, min(5, int(np.ceil(duration / 10))))
        
        # Analyze pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        mean_pitch = np.mean(pitch_values) if pitch_values else 0.0
        pitch_range = [float(np.min(pitch_values)), float(np.max(pitch_values))] if pitch_values else [0.0, 0.0]
        
        # Determine audio quality
        rms = np.sqrt(np.mean(y**2))
        audio_quality = "good" if rms > 0.05 else ("fair" if rms > 0.02 else "poor")
        
        # Identify issues
        issues = []
        if silence_time / duration > 0.5:
            issues.append("High silence ratio")
        if rms < 0.02:
            issues.append("Low audio level")
        if estimated_speakers > 4:
            issues.append("Complex audio - many potential speakers")
        
        # Get Ollama analysis
        ollama_analysis = None
        try:
            ollama_analysis = await get_ollama_diagnostic_analysis(
                duration, speaking_time, estimated_speakers,
                mean_pitch, audio_quality, issues
            )
        except Exception as e:
            logger.warning(f"Ollama analysis failed: {e}", component='diagnosis')
        
        # Recommendations
        recommendations = []
        if audio_quality == "poor":
            recommendations.append("Consider using noise reduction before separation")
        if len(issues) > 2:
            recommendations.append("Audio may be challenging for separation")
        recommendations.append("Try GMM method for best results")
        
        return DiagnosticReport(
            duration=float(duration),
            sample_rate=int(sr),
            speaking_time=float(speaking_time),
            silence_time=float(silence_time),
            estimated_speakers=estimated_speakers,
            mean_pitch=float(mean_pitch),
            pitch_range=pitch_range,
            audio_quality=audio_quality,
            issues=issues,
            recommendations=recommendations,
            ollama_analysis=ollama_analysis
        )
    
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}", component='diagnosis')
        raise ProcessingError("audio_diagnosis", str(e))


def _set_job_progress(job_id: str, progress: int, message: str):
    """Helper function to update job progress and broadcast via WebSocket"""
    if job_id in jobs_db:
        jobs_db[job_id]['progress'] = progress
        jobs_db[job_id]['progress_message'] = message
        
        # Broadcast progress via WebSocket (async-safe)
        if job_id in active_websockets:
            # Create task to broadcast (handles being called from sync context)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(broadcast_progress(
                        job_id, progress, jobs_db[job_id]['status'], message
                    ))
            except RuntimeError:
                pass  # No event loop running


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
        
        # Load and process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: separator.separate_speakers(audio_path, num_speakers, method)
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


@app.get("/health")
async def health():
    """Health check - fast response without blocking"""
    ollama_status = "disconnected"
    try:
        # Use a very short timeout for health checks
        response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=1)
        ollama_status = "connected" if response.status_code == 200 else "error"
    except Exception:
        pass
    
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "ollama_url": config.OLLAMA_URL
    }


# =============================================================================
# WebSocket for Real-Time Progress Updates
# =============================================================================

# Store active WebSocket connections per job
active_websockets: Dict[str, List[WebSocket]] = {}


async def broadcast_progress(job_id: str, progress: int, stage: str, message: str):
    """Broadcast progress update to all connected clients for a job"""
    if job_id in active_websockets:
        dead_connections = []
        for ws in active_websockets[job_id]:
            try:
                await ws.send_json({
                    "job_id": job_id,
                    "progress": progress,
                    "stage": stage,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception:
                dead_connections.append(ws)
        
        # Clean up dead connections
        for ws in dead_connections:
            active_websockets[job_id].remove(ws)


@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    
    # Add to active connections
    if job_id not in active_websockets:
        active_websockets[job_id] = []
    active_websockets[job_id].append(websocket)
    
    try:
        # Send initial status
        if job_id in jobs_db:
            job = jobs_db[job_id]
            await websocket.send_json({
                "job_id": job_id,
                "progress": job.get('progress', 0),
                "stage": job.get('status', 'unknown'),
                "message": job.get('progress_message', ''),
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            await websocket.send_json({
                "job_id": job_id,
                "progress": 0,
                "stage": "waiting",
                "message": "Waiting for job to start...",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message from client (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # If client requests status, send it
                if data == "status" and job_id in jobs_db:
                    job = jobs_db[job_id]
                    await websocket.send_json({
                        "job_id": job_id,
                        "progress": job.get('progress', 0),
                        "stage": job.get('status', 'unknown'),
                        "message": job.get('progress_message', ''),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
                    
    except WebSocketDisconnect:
        pass
    finally:
        # Remove from active connections
        if job_id in active_websockets:
            if websocket in active_websockets[job_id]:
                active_websockets[job_id].remove(websocket)
            if not active_websockets[job_id]:
                del active_websockets[job_id]


@app.get("/final_speaker_frontend_improved.html")
async def serve_frontend():
    """Legacy path: serve frontend HTML"""
    frontend_path = Path("final_speaker_frontend_improved.html")
    if frontend_path.exists():
        return FileResponse(frontend_path, media_type="text/html")
    else:
        return {"error": "Frontend not found"}


@app.get("/ui", response_class=HTMLResponse)
async def ui_index(request: Request):
    """Jinja UI: upload page"""
    if templates is None:
        return "<h1>Templates not configured</h1>"
    
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
    """Jinja UI: handle upload and redirect to job page"""
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {file_ext}")

    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"

    file_content = await file.read()
    file_size = len(file_content)
    validate_file_size(file_size)

    try:
        with open(upload_path, "wb") as buffer:
            buffer.write(file_content)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")
    finally:
        file_content = None

    jobs_db[job_id] = {
        'job_id': job_id,
        'status': JobStatus.UPLOADED,
        'created_at': datetime.utcnow().isoformat(),
        'filename': file.filename,
        'progress': 10,
        'progress_message': 'Uploaded. Ready to diagnose.',
        'file_path': str(upload_path),
        'file_size': file_size,
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
    """Jinja UI: job detail page"""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    if templates is None:
        return "<h1>Templates not configured</h1>"

    return templates.TemplateResponse(
        "job.html",
        {
            "request": request,
            "job_id": job_id,
            "job": jobs_db[job_id],
        },
    )


@app.get("/api/v1/jobs/{job_id}/diagnose", response_model=JobResponse)
async def get_diagnostic(job_id: str):
    """
    Step 2: Get diagnostic report
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
    """
    
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    return JobResponse(**jobs_db[job_id])


@app.get("/api/v1/jobs/{job_id}/stream")
async def stream_job_status(job_id: str):
    """
    Real-time job status using Server-Sent Events (SSE)
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
            
            progress = job.get('progress', 0)
            status = job.get('status')
            message = job.get('progress_message', '')
            
            if progress != last_progress or status != last_status:
                yield f"data: {json.dumps({'progress': progress, 'status': status, 'message': message})}\n\n"
                last_progress = progress
                last_status = status
            
            if status in [JobStatus.COMPLETED, JobStatus.SEPARATED, JobStatus.FAILED]:
                yield f"data: {json.dumps(jobs_db[job_id])}\n\n"
                break
            
            await asyncio.sleep(0.5)
    
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


# ============================================================================
# Quick Win: Download All as ZIP
# ============================================================================

import zipfile
from io import BytesIO

@app.get("/api/v1/jobs/{job_id}/download-all")
async def download_all_speakers(job_id: str):
    """Download all separated speakers as a ZIP file"""
    
    if job_id not in jobs_db:
        raise NotFoundError("Job", job_id)
    
    job = jobs_db[job_id]
    
    if job['status'] not in [JobStatus.SEPARATED, JobStatus.COMPLETED]:
        raise ValidationError(
            "Job not ready for download",
            field="status",
            user_message="Processing is not complete. Please wait for the job to finish."
        )
    
    # Find output directory
    output_path = OUTPUT_DIR / job_id
    if not output_path.exists():
        raise NotFoundError("Output directory", job_id)
    
    # Create ZIP in memory
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add separated speaker files
        separated_path = output_path / "separated"
        if separated_path.exists():
            for audio_file in separated_path.glob("*.wav"):
                zip_file.write(audio_file, f"separated/{audio_file.name}")
        
        # Add cleaned files if available
        cleaned_path = output_path / "cleaned"
        if cleaned_path.exists():
            for audio_file in cleaned_path.glob("*.wav"):
                zip_file.write(audio_file, f"cleaned/{audio_file.name}")
        
        # Add analysis report if available
        analysis_path = output_path / "analysis.json"
        if analysis_path.exists():
            zip_file.write(analysis_path, "analysis.json")
    
    zip_buffer.seek(0)
    
    # Generate filename from original
    original_name = Path(job['filename']).stem
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={original_name}_speakers_{job_id[:8]}.zip"
        }
    )


# ============================================================================
# Quick Win: Estimated Processing Time
# ============================================================================

@app.get("/api/v1/estimate-time")
async def estimate_processing_time(
    file_size_bytes: int = Query(..., description="File size in bytes"),
    num_speakers: int = Query(2, description="Number of speakers"),
    include_cleaning: bool = Query(False, description="Include audio cleaning time")
):
    """Estimate processing time based on file size and options"""
    
    # Base processing time (seconds)
    base_time = 10
    
    # Time per MB of audio
    size_mb = file_size_bytes / (1024 * 1024)
    size_time = size_mb * 2  # ~2 seconds per MB
    
    # Time per speaker
    speaker_time = num_speakers * 5
    
    # Cleaning time
    cleaning_time = 15 if include_cleaning else 0
    
    total_seconds = base_time + size_time + speaker_time + cleaning_time
    
    # Add buffer for uncertainty
    buffer = total_seconds * 0.2
    
    estimated_min = int(total_seconds - buffer)
    estimated_max = int(total_seconds + buffer)
    
    return {
        "estimated_seconds": int(total_seconds),
        "estimated_min_seconds": max(10, estimated_min),
        "estimated_max_seconds": estimated_max,
        "estimated_human": f"{int(total_seconds // 60)}m {int(total_seconds % 60)}s" if total_seconds >= 60 else f"{int(total_seconds)}s",
        "factors": {
            "file_size_mb": round(size_mb, 2),
            "num_speakers": num_speakers,
            "include_cleaning": include_cleaning
        }
    }


@app.post("/api/v1/jobs/{job_id}/feedback", response_model=JobResponse)
async def submit_feedback(job_id: str, feedback: UserFeedback):
    """
    Step 7: Submit user feedback
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
    
    # Delete uploaded file
    for ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
        upload_path = UPLOAD_DIR / f"{job_id}{ext}"
        if upload_path.exists():
            upload_path.unlink()
    
    # Delete output directory
    output_path = OUTPUT_DIR / job_id
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Remove from database
    del jobs_db[job_id]
    
    return {"message": "Job deleted successfully"}

# ============================================================================
# AUDIO FILE SERVING ENDPOINTS - Add these to fix 404 errors
# ============================================================================

@app.get("/api/v1/jobs/{job_id}/original")
async def get_original_audio(job_id: str):
    """Serve the original uploaded audio file for playback"""
    
    # Get job from database
    if job_id not in jobs_db:
        raise NotFoundError("Job", job_id)
    
    job = jobs_db[job_id]
    
    # Get original file path - try multiple field names
    original_file = (
        job.get('original_file') or 
        job.get('upload_path') or 
        job.get('file_path')
    )
    
    if not original_file:
        logger.error(f"No original file path for job {job_id}", component='file_serving')
        raise HTTPException(
            status_code=404,
            detail="Original audio file path not found in job data"
        )
    
    file_path = Path(original_file)
    
    # Check file exists on disk
    if not file_path.exists():
        logger.error(f"File not found: {file_path}", component='file_serving')
        raise HTTPException(
            status_code=404,
            detail=f"Audio file no longer exists on server"
        )
    
    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type or not mime_type.startswith('audio'):
        mime_type = 'audio/mpeg'  # Safe default
    
    logger.info(f"Serving original audio for job {job_id}", component='file_serving')
    
    # Serve the file
    return FileResponse(
        path=str(file_path),
        media_type=mime_type,
        filename=file_path.name,
        headers={
            'Accept-Ranges': 'bytes',  # Allow seeking in audio player
            'Cache-Control': 'public, max-age=3600'
        }
    )


@app.get("/api/v1/jobs/{job_id}/download/{speaker}")
async def download_speaker_audio(job_id: str, speaker: str):
    """Download separated speaker audio file"""
    
    # Get job
    if job_id not in jobs_db:
        raise NotFoundError("Job", job_id)
    
    job = jobs_db[job_id]
    
    # Check job has separated files
    if 'separated_files' not in job:
        raise HTTPException(
            status_code=400,
            detail="No separated files available yet. Processing may still be in progress."
        )
    
    separated_files = job['separated_files']
    
    # Check speaker exists
    if speaker not in separated_files:
        available = list(separated_files.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Speaker '{speaker}' not found. Available speakers: {available}"
        )
    
    file_path = Path(separated_files[speaker])
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Separated audio file no longer exists on server"
        )
    
    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = 'audio/wav'
    
    logger.info(f"Serving speaker {speaker} for job {job_id}", component='file_serving')
    
    return FileResponse(
        path=str(file_path),
        media_type=mime_type,
        filename=f"{speaker}.wav"
    )

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
