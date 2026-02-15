#!/usr/bin/env python3
"""
Enhanced Logger Wrapper with Structured Logging
Replaces all print() statements with proper logging
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

# ============================================================================
# Log Level Management
# ============================================================================

class LogLevel(str, Enum):
    """Log severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# JSON Formatter for Structured Logging
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for better parsing and monitoring"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from logging extras
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj)


# ============================================================================
# Enhanced Logger Wrapper
# ============================================================================

class AppLogger:
    """Enhanced logger with context support and metrics integration"""
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        """Singleton pattern - only one logger instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the logger"""
        self.base_logger = logging.getLogger('speaker-api')
        self.base_logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if self.base_logger.handlers:
            return
        
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(JSONFormatter())
        self.base_logger.addHandler(console_handler)
        
        # File handler for persistent logs
        log_file = Path('logs') / 'app.log'
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        self.base_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a module/component"""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(f'speaker-api.{name}')
            self._loggers[name].setLevel(logging.DEBUG)
        return self._loggers[name]
    
    @staticmethod
    def _add_extra_fields(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add extra fields to log record"""
        if extra is None:
            return {}
        return {'extra_fields': extra}
    
    def debug(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log debug message"""
        logger = self.get_logger(kwargs.get('component', 'core'))
        extra_dict = self._add_extra_fields(extra or kwargs)
        logger.debug(message, extra=extra_dict if extra_dict else None)
    
    def info(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log info message"""
        logger = self.get_logger(kwargs.get('component', 'core'))
        extra_dict = self._add_extra_fields(extra or kwargs)
        logger.info(message, extra=extra_dict if extra_dict else None)
    
    def warning(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log warning message"""
        logger = self.get_logger(kwargs.get('component', 'core'))
        extra_dict = self._add_extra_fields(extra or kwargs)
        logger.warning(message, extra=extra_dict if extra_dict else None)
    
    def error(self, message: str, exception: Optional[Exception] = None, 
             extra: Optional[Dict] = None, **kwargs):
        """Log error message with exception info"""
        logger = self.get_logger(kwargs.get('component', 'core'))
        extra_dict = self._add_extra_fields(extra or kwargs)
        logger.error(message, exc_info=exception, extra=extra_dict if extra_dict else None)
    
    def critical(self, message: str, exception: Optional[Exception] = None,
                extra: Optional[Dict] = None, **kwargs):
        """Log critical message"""
        logger = self.get_logger(kwargs.get('component', 'core'))
        extra_dict = self._add_extra_fields(extra or kwargs)
        logger.critical(message, exc_info=exception, extra=extra_dict if extra_dict else None)
    
    def log_job_event(self, job_id: str, status: str, message: str, 
                     extra: Optional[Dict] = None):
        """Log job-related event with job_id context"""
        context = {'job_id': job_id, 'status': status}
        if extra:
            context.update(extra)
        self.info(message, extra=context, component='jobs')
    
    def log_audio_processing(self, job_id: str, operation: str, duration: float,
                            success: bool, error: Optional[str] = None):
        """Log audio processing operation"""
        context = {
            'job_id': job_id,
            'operation': operation,
            'duration_seconds': duration,
            'success': success
        }
        if error:
            context['error'] = error
        
        level = 'info' if success else 'error'
        getattr(self, level)(
            f"Audio processing: {operation}",
            extra=context,
            component='audio'
        )
    
    def log_api_request(self, method: str, endpoint: str, status_code: int,
                       duration: float, user_id: Optional[str] = None):
        """Log API request with metrics"""
        context = {
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration_seconds': duration,
            'user_id': user_id or 'anonymous'
        }
        self.info(f"API Request: {method} {endpoint}", extra=context, component='api')
    
    def log_security_event(self, event_type: str, user_id: str, details: Dict):
        """Log security-related event"""
        context = {'event_type': event_type, 'user_id': user_id}
        context.update(details)
        self.warning(f"Security Event: {event_type}", extra=context, component='security')


# ============================================================================
# Global Logger Instance
# ============================================================================

logger = AppLogger()


# ============================================================================
# Convenience Functions (Replace print() calls)
# ============================================================================

def log_system_startup(host: str, port: int, config_source: str = "environment"):
    """Log system startup"""
    logger.info(
        f"Speaker Separation API Starting",
        extra={
            'host': host,
            'port': port,
            'config_source': config_source,
            'python_version': sys.version.split()[0]
        },
        component='startup'
    )


def log_config_loaded(location: str, status: str):
    """Log config loading"""
    message = f"Configuration loaded from {location}" if status == "success" else f"Failed to load config from {location}"
    getattr(logger, 'info' if status == 'success' else 'warning')(
        message,
        extra={'location': location, 'status': status},
        component='config'
    )


def log_module_loaded(module_name: str, status: str, reason: Optional[str] = None):
    """Log module loading"""
    if status == 'success':
        logger.info(
            f"✓ Loaded {module_name}",
            extra={'module': module_name, 'status': status},
            component='modules'
        )
    else:
        logger.warning(
            f"⚠️ Could not load {module_name}" + (f": {reason}" if reason else ""),
            extra={'module': module_name, 'status': status, 'reason': reason},
            component='modules'
        )


def log_separator_started(num_speakers: int, method: str):
    """Log speaker separation start"""
    logger.info(
        f"Separating {num_speakers} speakers using {method}",
        extra={'num_speakers': num_speakers, 'method': method},
        component='separator'
    )


def log_separator_completed(num_speakers: int, duration: float):
    """Log speaker separation completion"""
    logger.info(
        f"✓ Separated into {num_speakers} speakers",
        extra={'num_speakers': num_speakers, 'duration_seconds': duration},
        component='separator'
    )


def log_cleaner_started(filename: str):
    """Log audio cleaner start"""
    logger.info(
        f"Cleaning: {filename}",
        extra={'filename': filename},
        component='cleaner'
    )


def log_temp_cleanup(status: str, error: Optional[str] = None):
    """Log temporary file cleanup"""
    if status == 'success':
        logger.info(
            "✓ Cleaned up temporary files",
            extra={'status': status},
            component='cleanup'
        )
    else:
        logger.error(
            f"⚠️ Error cleaning temp files: {error}",
            extra={'status': status, 'error': error},
            component='cleanup'
        )


# ============================================================================
# Context Managers for Automatic Timing/Logging
# ============================================================================

class LogContext:
    """Context manager for operation timing and logging"""
    
    def __init__(self, operation: str, component: str = 'core', job_id: Optional[str] = None):
        self.operation = operation
        self.component = component
        self.job_id = job_id
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.info(
            f"Starting: {self.operation}",
            extra={'job_id': self.job_id},
            component=self.component
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type:
            logger.error(
                f"Failed: {self.operation} ({duration:.2f}s)",
                exception=exc_val,
                extra={'job_id': self.job_id, 'duration_seconds': duration},
                component=self.component
            )
        else:
            logger.info(
                f"Completed: {self.operation} ({duration:.2f}s)",
                extra={'job_id': self.job_id, 'duration_seconds': duration},
                component=self.component
            )
        
        return False  # Don't suppress exceptions


# ============================================================================
# Usage Example
# ============================================================================

"""
from logger_wrapper import logger, log_module_loaded, LogContext

# Replace print statements:

# OLD: print("✓ Loaded ImprovedSpeakerSeparator")
# NEW:
log_module_loaded("ImprovedSpeakerSeparator", "success")

# OLD: print(f"⚠️  Could not load audio_cleaner: {e}")
# NEW:
log_module_loaded("AudioCleaner", "failed", reason=str(e))

# For timing operations:
with LogContext("audio_processing", component="audio", job_id="uuid-123"):
    # This will automatically log start/end with timing
    separator.separate_speakers(...)
"""