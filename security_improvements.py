#!/usr/bin/env python3
"""
Enhanced Security Features for Speaker Separation API
"""

import os
from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib
import jwt
from fastapi import HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt

# =============================================================================
# 1. JWT TOKEN AUTHENTICATION (CRITICAL FIX: Load SECRET_KEY from environment)
# =============================================================================

# Load .env file first before checking environment variables
try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Load JWT secret from environment, or generate and warn if not found
_ENV_SECRET = os.getenv('JWT_SECRET_KEY')
if _ENV_SECRET:
    SECRET_KEY = _ENV_SECRET
    print("✓ Loaded JWT_SECRET_KEY from environment")
else:
    # Generate a key for this session, but warn user
    SECRET_KEY = secrets.token_urlsafe(32)
    print("⚠️  WARNING: JWT_SECRET_KEY not set in environment!")
    print("   Token validation will fail on server restart.")
    print("   Set JWT_SECRET_KEY in .env file to fix this.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token for user authentication"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token from Authorization header"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")


# =============================================================================
# 2. API KEY AUTHENTICATION
# =============================================================================

# Load API key from environment
_API_KEY = os.getenv('API_KEY', '')
_ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'


def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Verify API key from X-API-Key header"""
    if not _ENABLE_AUTH:
        # Authentication disabled, allow all requests
        return True
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Pass X-API-Key header."
        )
    
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True


# =============================================================================
# 2. PASSWORD HASHING (For user management)
# =============================================================================

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


# =============================================================================
# 3. API KEY ROTATION (For service-to-service communication)
# =============================================================================

class APIKeyManager:
    """Manage API keys with rotation and expiry"""
    
    def __init__(self):
        self.active_keys = {}
        self.revoked_keys = set()
    
    def generate_api_key(self, user_id: str, expires_days: int = 90) -> dict:
        """Generate new API key with expiration"""
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        expiry = datetime.utcnow() + timedelta(days=expires_days)
        
        self.active_keys[key_hash] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'expires_at': expiry,
            'last_used': None
        }
        
        return {
            'api_key': key,
            'expires_at': expiry.isoformat(),
            'note': 'Store this key securely - it cannot be retrieved later'
        }
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Check if revoked
        if key_hash in self.revoked_keys:
            return False
        
        # Check if exists and not expired
        if key_hash in self.active_keys:
            key_data = self.active_keys[key_hash]
            
            if datetime.utcnow() > key_data['expires_at']:
                return False
            
            # Update last used
            key_data['last_used'] = datetime.utcnow()
            return True
        
        return False
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.revoked_keys.add(key_hash)
        if key_hash in self.active_keys:
            del self.active_keys[key_hash]


# =============================================================================
# 4. RATE LIMITING (Enhanced with Redis support)
# =============================================================================

from collections import defaultdict
from threading import Lock
from typing import Optional

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis not installed. Using in-memory rate limiting.")
    print("   Install with: pip install redis")


class RedisRateLimiter:
    """Redis-backed rate limiter for distributed systems"""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        self.redis_client = None
        self.fallback_limiter = None
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                # Test connection
                self.redis_client.ping()
                print("✓ Redis rate limiter connected")
            except Exception as e:
                print(f"⚠️  Redis connection failed: {e}")
                print("   Falling back to in-memory rate limiting")
                self.redis_client = None
        
        if not self.redis_client:
            self.fallback_limiter = EnhancedRateLimiter()
    
    def check_rate_limit(self, key: str, max_requests: int = 10, window: int = 3600) -> bool:
        """Check if request is within rate limit"""
        if self.redis_client:
            return self._check_redis(key, max_requests, window)
        return self.fallback_limiter.check_rate_limit(key, 'free')
    
    def _check_redis(self, key: str, max_requests: int, window: int) -> bool:
        """Redis-based rate limiting with sliding window"""
        try:
            current_time = datetime.utcnow().timestamp()
            window_start = current_time - window
            
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
            
            # Count current entries
            pipe.zcard(f"rate_limit:{key}")
            
            # Add current request
            pipe.zadd(f"rate_limit:{key}", {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(f"rate_limit:{key}", window)
            
            results = pipe.execute()
            current_count = results[1]
            
            return current_count < max_requests
        except Exception as e:
            print(f"⚠️  Redis rate limit error: {e}")
            return True  # Allow on error
    
    def get_remaining(self, key: str, max_requests: int = 10, window: int = 3600) -> int:
        """Get remaining requests for a key"""
        if self.redis_client:
            try:
                current_time = datetime.utcnow().timestamp()
                window_start = current_time - window
                
                # Clean and count
                self.redis_client.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
                current_count = self.redis_client.zcard(f"rate_limit:{key}")
                
                return max(0, max_requests - current_count)
            except Exception:
                return max_requests
        return max_requests


class EnhancedRateLimiter:
    """In-memory rate limiter with different tiers (fallback)"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = Lock()
        
        # Rate limit tiers
        self.tiers = {
            'free': {'uploads': 5, 'window': 3600},      # 5 per hour
            'basic': {'uploads': 20, 'window': 3600},    # 20 per hour
            'premium': {'uploads': 100, 'window': 3600}, # 100 per hour
            'enterprise': {'uploads': 1000, 'window': 3600}  # 1000 per hour
        }
    
    def check_rate_limit(self, user_id: str, tier: str = 'free') -> bool:
        """Check if user is within rate limit"""
        with self.lock:
            now = datetime.utcnow().timestamp()
            limits = self.tiers.get(tier, self.tiers['free'])
            window = limits['window']
            max_requests = limits['uploads']
            
            # Clean old requests
            self.requests[user_id] = [
                ts for ts in self.requests[user_id]
                if now - ts < window
            ]
            
            # Check limit
            if len(self.requests[user_id]) >= max_requests:
                return False
            
            # Add current request
            self.requests[user_id].append(now)
            return True
    
    def get_remaining(self, user_id: str, tier: str = 'free') -> int:
        """Get remaining requests for a user"""
        with self.lock:
            now = datetime.utcnow().timestamp()
            limits = self.tiers.get(tier, self.tiers['free'])
            window = limits['window']
            max_requests = limits['uploads']
            
            # Clean old requests
            self.requests[user_id] = [
                ts for ts in self.requests[user_id]
                if now - ts < window
            ]
            
            return max(0, max_requests - len(self.requests[user_id]))


# =============================================================================
# 5. INPUT VALIDATION & SANITIZATION
# =============================================================================

import re
from pathlib import Path

class SecurityValidator:
    """Validate and sanitize user inputs"""
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[^\w\s\-\.]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename
    
    @staticmethod
    def validate_job_id(job_id: str) -> bool:
        """Validate job ID format"""
        # Should be UUID format
        uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        return bool(re.match(uuid_pattern, job_id))
    
    @staticmethod
    def is_valid_audio(file_content) -> bool:
        """Quick validation of audio file content"""
        # Check magic bytes for common audio formats
        audio_magic_bytes = [
            b'ID3',      # MP3
            b'RIFF',     # WAV
            b'fLaC',     # FLAC
            b'\xff\xfb', # MP3 (MPEG frame sync)
        ]
        
        # Read first few bytes
        beginning = file_content.read(4) if hasattr(file_content, 'read') else file_content[:4]
        file_content.seek(0) if hasattr(file_content, 'seek') else None
        
        return any(beginning.startswith(magic) for magic in audio_magic_bytes)


# =============================================================================
# 6. SECURITY HEADERS
# =============================================================================

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        return response


# =============================================================================
# 7. AUDIT LOGGING
# =============================================================================

import logging
import json

class AuditLogger:
    """Log security-relevant events"""
    
    def __init__(self, log_file: str = 'audit.log'):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_event(self, event_type: str, user_id: str, details: dict):
        """Log security event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'details': details
        }
        self.logger.info(json.dumps(event))
    
    def log_auth_attempt(self, user_id: str, success: bool, ip_address: str = "unknown"):
        """Log authentication attempt"""
        self.log_event('auth_attempt', user_id, {
            'success': success,
            'ip_address': ip_address
        })
    
    def log_upload_attempt(self, user_id: str, filename: str):
        """Log upload attempt"""
        self.log_event('upload_attempt', user_id, {
            'filename': filename
        })
    
    def log_upload_rejection(self, user_id: str, filename: str, reason: str):
        """Log upload rejection"""
        self.log_event('upload_rejection', user_id, {
            'filename': filename,
            'reason': reason
        })
    
    def log_file_upload(self, user_id: str, filename: str, size_bytes: int):
        """Log successful file upload"""
        self.log_event('file_upload', user_id, {
            'filename': filename,
            'size_bytes': size_bytes
        })
    
    def log_suspicious_activity(self, user_id: str, activity: str, details: dict):
        """Log suspicious activity"""
        self.log_event('suspicious_activity', user_id, {
            'activity': activity,
            'details': details
        })


# =============================================================================
# USAGE EXAMPLE IN FASTAPI
# =============================================================================

"""
from fastapi import FastAPI, Depends
from security_improvements import (
    verify_token, SecurityHeadersMiddleware, 
    AuditLogger, SecurityValidator
)

app = FastAPI()

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)

# Initialize services
audit_logger = AuditLogger()
validator = SecurityValidator()

@app.post("/api/v1/secure-upload")
async def secure_upload(
    file: UploadFile,
    user_data: dict = Depends(verify_token)
):
    # Validate filename
    safe_filename = validator.validate_filename(file.filename)
    
    # Log upload
    audit_logger.log_file_upload(
        user_data['sub'],
        safe_filename,
        file.size
    )
    
    # Process upload...
    return {"message": "Upload successful"}
"""
