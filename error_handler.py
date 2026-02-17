#!/usr/bin/env python3
"""
Comprehensive Error Handling & Recovery System
Provides error wrapping, retry logic, and graceful degradation
"""

import asyncio
from functools import wraps
from typing import Optional, Callable, Any, Type, List, Dict
from datetime import datetime, timedelta
from enum import Enum
from logger_wrapper import logger
import os

# ============================================================================
# Error Types
# ============================================================================

class APIError(Exception):
    """Base API error"""
    def __init__(self, message: str, status_code: int = 500, 
                 user_message: Optional[str] = None, details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.user_message = user_message or message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIError):
    """Input validation errors (400)"""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        kwargs['status_code'] = 400
        kwargs['details'] = {'field': field}
        super().__init__(message, **kwargs)


class AuthenticationError(APIError):
    """Authentication/Authorization errors (401/403)"""
    def __init__(self, message: str, status_code: int = 401, **kwargs):
        kwargs['status_code'] = status_code
        super().__init__(message, **kwargs)


class NotFoundError(APIError):
    """Resource not found (404)"""
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        kwargs['status_code'] = 404
        message = f"{resource_type} not found: {resource_id}"
        super().__init__(message, **kwargs)


class FileSizeError(APIError):
    """File too large (413)"""
    def __init__(self, file_size_mb: float, max_size_mb: int, **kwargs):
        kwargs['status_code'] = 413
        message = f"File size ({file_size_mb:.1f}MB) exceeds maximum ({max_size_mb}MB)"
        super().__init__(message, **kwargs)


class ProcessingError(APIError):
    """Audio processing errors (500)"""
    def __init__(self, operation: str, reason: str, **kwargs):
        kwargs['status_code'] = 500
        message = f"{operation} failed: {reason}"
        super().__init__(message, **kwargs)


class OllamaError(APIError):
    """Ollama service errors (502/503)"""
    def __init__(self, message: str, status_code: int = 502, **kwargs):
        kwargs['status_code'] = status_code
        super().__init__(message, **kwargs)


class TimeoutError(APIError):
    """Request timeout (504)"""
    def __init__(self, operation: str, timeout_seconds: int, **kwargs):
        kwargs['status_code'] = 504
        message = f"{operation} timed out after {timeout_seconds}s"
        super().__init__(message, **kwargs)


# ============================================================================
# Retry Configuration
# ============================================================================

class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(self, max_attempts: int = 3, initial_delay: float = 1.0,
                 max_delay: float = 30.0, backoff_multiplier: float = 2.0,
                 retryable_exceptions: Optional[List[Type[Exception]]] = None):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retryable_exceptions = retryable_exceptions or [
            IOError, TimeoutError, ConnectionError, OSError
        ]
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff"""
        delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_delay)


# ============================================================================
# Decorators
# ============================================================================

def handle_errors(status_ok: int = 200):
    """Decorator for error handling in async functions - re-raises exceptions for FastAPI handlers"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except APIError as e:
                logger.error(
                    f"API Error in {func.__name__}",
                    exception=e,
                    extra={
                        'status_code': e.status_code,
                        'message': e.message,
                        'details': e.details
                    },
                    component='error_handler'
                )
                # Re-raise so FastAPI exception handlers can format the response
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error in {func.__name__}",
                    exception=e,
                    extra={'error_type': type(e).__name__},
                    component='error_handler'
                )
                # Re-raise so FastAPI exception handlers can format the response
                raise
        return wrapper
    return decorator


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """Decorator for automatic retry with exponential backoff"""
    config = config or RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(config.retryable_exceptions) as e:
                    if attempt >= config.max_attempts:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}",
                            exception=e,
                            extra={'attempts': attempt},
                            component='retry'
                        )
                        raise
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} (delay: {delay}s)",
                        extra={'delay_seconds': delay, 'attempt': attempt},
                        component='retry'
                    )
                    await asyncio.sleep(delay)
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(config.retryable_exceptions) as e:
                    if attempt >= config.max_attempts:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}",
                            exception=e,
                            extra={'attempts': attempt},
                            component='retry'
                        )
                        raise
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} (delay: {delay}s)",
                        extra={'delay_seconds': delay, 'attempt': attempt},
                        component='retry'
                    )
                    asyncio.sleep(delay)
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
        
        return wrapper
    return decorator


def timeout_handler(timeout_seconds: float = 30.0):
    """Decorator for timeout handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError as e:
                logger.error(
                    f"Timeout in {func.__name__}",
                    exception=e,
                    extra={'timeout_seconds': timeout_seconds},
                    component='timeout'
                )
                raise TimeoutError(func.__name__, int(timeout_seconds))
        
        return async_wrapper
    return decorator


# ============================================================================
# Error Recovery Helper Functions
# ============================================================================

class ErrorRecoveryHandler:
    """Handle common error recovery scenarios"""
    
    @staticmethod
    def handle_file_not_found(file_path: str) -> None:
        """Handle missing file"""
        logger.error(
            f"File not found: {file_path}",
            extra={'file_path': file_path},
            component='file_system'
        )
        raise NotFoundError("File", file_path)
    
    @staticmethod
    def handle_permission_error(operation: str, path: str) -> None:
        """Handle permission denied"""
        logger.error(
            f"Permission denied for {operation} on {path}",
            extra={'operation': operation, 'path': path},
            component='file_system'
        )
        raise APIError(
            f"Permission denied: {operation}",
            status_code=403,
            user_message="You don't have permission to perform this operation"
        )
    
    @staticmethod
    def handle_disk_full() -> None:
        """Handle out of disk space"""
        logger.critical(
            "Disk space full",
            extra={'available_space': 0},
            component='system'
        )
        raise APIError(
            "Server storage is full",
            status_code=507,
            user_message="Server is out of storage space. Please try again later."
        )
    
    @staticmethod
    def handle_memory_error() -> None:
        """Handle out of memory"""
        logger.critical(
            "Out of memory",
            component='system'
        )
        raise APIError(
            "Server out of memory",
            status_code=500,
            user_message="Server is out of memory. Please try with a smaller file."
        )
    
    @staticmethod
    def handle_ollama_unavailable(timeout_seconds: int = 120) -> None:
        """Handle Ollama service unavailable"""
        logger.error(
            f"Ollama service unavailable (timeout: {timeout_seconds}s)",
            extra={'timeout_seconds': timeout_seconds},
            component='ollama'
        )
        raise OllamaError(
            "Ollama service is unavailable",
            status_code=503,
        )


# ============================================================================
# Safe Execution Wrappers
# ============================================================================

class SafeExecutor:
    """Execute operations with comprehensive error handling"""
    
    @staticmethod
    async def execute_with_fallback(
        primary: Callable,
        fallback: Callable,
        operation_name: str
    ) -> Any:
        """Execute primary operation, fall back if it fails"""
        try:
            logger.debug(
                f"Executing {operation_name} (primary)",
                extra={'operation': operation_name},
                component='executor'
            )
            return await primary()
        except Exception as e:
            logger.warning(
                f"Primary {operation_name} failed, using fallback",
                exception=e,
                extra={'operation': operation_name},
                component='executor'
            )
            try:
                return await fallback()
            except Exception as fallback_error:
                logger.error(
                    f"Fallback for {operation_name} also failed",
                    exception=fallback_error,
                    extra={'operation': operation_name},
                    component='executor'
                )
                raise
    
    @staticmethod
    async def execute_with_timeout(
        operation: Callable,
        timeout_seconds: float,
        operation_name: str
    ) -> Any:
        """Execute operation with timeout"""
        try:
            return await asyncio.wait_for(
                operation(),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(
                f"{operation_name} timed out",
                extra={'operation': operation_name, 'timeout_seconds': timeout_seconds},
                component='executor'
            )
            raise TimeoutError(operation_name, int(timeout_seconds))

# =============================================================================
# USER-FRIENDLY ERROR FORMATTER
# Add this entire section to your error_handler.py
# =============================================================================

class UserFriendlyErrorFormatter:
    """Convert technical errors to user-friendly messages"""
    
    ERROR_TEMPLATES = {
        'ValidationError': {
            'icon': '⚠️',
            'title': 'Invalid Input',
            'severity': 'warning',
            'helpful_tips': [
                'Check that your file is a valid audio format',
                'Ensure file size is under 100MB',
                'Make sure the file isn\'t corrupted'
            ]
        },
        'FileSizeError': {
            'icon': '📁',
            'title': 'File Too Large',
            'severity': 'error',
            'helpful_tips': [
                'Maximum file size is 100MB',
                'Try compressing your audio (reduce quality to 128kbps)',
                'Split longer recordings into smaller chunks'
            ]
        },
        'ProcessingError': {
            'icon': '🔧',
            'title': 'Processing Failed',
            'severity': 'error',
            'helpful_tips': [
                'Ensure your audio has clear speech',
                'Try with a different audio file',
                'Check if the file plays correctly on your device'
            ]
        },
        'TimeoutError': {
            'icon': '⏱️',
            'title': 'Taking Too Long',
            'severity': 'warning',
            'helpful_tips': [
                'Large files can take 3-5 minutes to process',
                'Try with a shorter audio clip (under 5 minutes)',
                'Server might be busy - try again in a few minutes'
            ]
        },
        'OllamaError': {
            'icon': '🤖',
            'title': 'AI Analysis Unavailable',
            'severity': 'info',
            'message_override': 'AI recommendations are offline, but speaker separation will work normally.',
            'helpful_tips': [
                'Your audio will still be separated correctly',
                'You won\'t get AI-powered insights right now'
            ]
        },
        'AuthenticationError': {
            'icon': '🔐',
            'title': 'Authentication Failed',
            'severity': 'error',
            'helpful_tips': [
                'Check that your API key is correct',
                'Your session may have expired - try refreshing'
            ]
        },
        'NotFoundError': {
            'icon': '🔍',
            'title': 'Not Found',
            'severity': 'error',
            'helpful_tips': [
                'The file you\'re looking for doesn\'t exist',
                'It may have been deleted or expired',
                'Try uploading your file again'
            ]
        }
    }
    
    @classmethod
    def format_error(cls, error, request_id: Optional[str] = None) -> dict:
        """
        Format any APIError for user-friendly display
        
        Args:
            error: APIError instance
            request_id: Optional request ID for tracking
            
        Returns:
            User-friendly error response dict
        """
        error_type = type(error).__name__
        template = cls.ERROR_TEMPLATES.get(error_type, {
            'icon': '❌',
            'title': 'Something Went Wrong',
            'severity': 'error',
            'helpful_tips': [
                'Try refreshing the page',
                'Upload your file again',
                'Contact support if this persists'
            ]
        })
        
        # Generate unique error ID
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        error_hash = abs(hash(str(error.message) + timestamp)) % 10000
        error_id = f"ERR_{error.status_code}_{error_hash:04d}"
        
        # Determine if user can retry
        can_retry = error.status_code in [408, 429, 500, 502, 503, 504]
        is_temporary = error.status_code in [408, 429, 503, 504]
        
        # Get user-friendly message
        user_message = (
            template.get('message_override') or 
            getattr(error, 'user_message', None) or 
            error.message
        )
        
        # Build response
        response = {
            'success': False,
            'error': {
                'icon': template['icon'],
                'title': template['title'],
                'message': user_message,
                'severity': template['severity'],
                'error_id': error_id,
                'can_retry': can_retry,
                'is_temporary': is_temporary,
                'helpful_tips': template.get('helpful_tips', []),
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': request_id
            },
            'status_code': error.status_code
        }
        
        # Add validation errors if present
        if hasattr(error, 'details') and error.details:
            response['error']['validation_errors'] = error.details
        
        # Show technical details only in debug mode
        if os.getenv('DEBUG') == 'true':
            response['error']['debug'] = {
                'technical_message': str(error.message),
                'error_type': error_type
            }
        
        return response

# ============================================================================
# Usage Examples
# ============================================================================

"""
from error_handler import (
    handle_errors, retry_with_backoff, timeout_handler,
    ValidationError, ProcessingError, ErrorRecoveryHandler,
    SafeExecutor
)

# Example 1: Basic error handling
@app.post("/api/v1/upload")
@handle_errors()
async def upload_audio(file: UploadFile):
    if not file.filename:
        raise ValidationError("Filename required", field="filename")
    return {"status": "ok"}

# Example 2: Retry on failure
@retry_with_backoff(RetryConfig(max_attempts=3))
async def call_external_service():
    # Will retry up to 3 times on network errors
    pass

# Example 3: Timeout handling
@timeout_handler(timeout_seconds=30)
async def long_operation():
    pass

# Example 4: Fallback execution
await SafeExecutor.execute_with_fallback(
    primary=lambda: call_ollama_api(),
    fallback=lambda: get_cached_result(),
    operation_name="AI Analysis"
)
"""