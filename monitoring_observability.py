#!/usr/bin/env python3
"""
Monitoring, Observability, and Alerting System
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import psutil
import json

# =============================================================================
# 1. PROMETHEUS METRICS (Industry Standard)
# =============================================================================

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Install prometheus_client: pip install prometheus-client")

if PROMETHEUS_AVAILABLE:
    # Request metrics
    request_count = Counter(
        'speaker_api_requests_total',
        'Total number of requests',
        ['method', 'endpoint', 'status']
    )
    
    request_duration = Histogram(
        'speaker_api_request_duration_seconds',
        'Request duration in seconds',
        ['method', 'endpoint']
    )
    
    # Processing metrics
    audio_processing_duration = Histogram(
        'audio_processing_duration_seconds',
        'Audio processing duration',
        ['num_speakers', 'method']
    )
    
    audio_file_size = Histogram(
        'audio_file_size_bytes',
        'Uploaded audio file size',
        buckets=[1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]
    )
    
    # System metrics
    active_jobs = Gauge(
        'active_jobs',
        'Number of active processing jobs'
    )
    
    failed_jobs = Counter(
        'failed_jobs_total',
        'Total number of failed jobs',
        ['error_type']
    )
    
    # Business metrics
    speakers_separated = Counter(
        'speakers_separated_total',
        'Total number of speakers separated',
        ['num_speakers']
    )


# =============================================================================
# 2. STRUCTURED LOGGING
# =============================================================================

class StructuredLogger:
    """JSON structured logging for better parsing"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)
    
    class JSONFormatter(logging.Formatter):
        """Format logs as JSON"""
        
        def format(self, record):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add extra fields
            if hasattr(record, 'job_id'):
                log_data['job_id'] = record.job_id
            if hasattr(record, 'user_id'):
                log_data['user_id'] = record.user_id
            if hasattr(record, 'duration_ms'):
                log_data['duration_ms'] = record.duration_ms
            
            return json.dumps(log_data)
    
    def log_request(self, method: str, path: str, status: int, duration_ms: float, user_id: str = None):
        """Log HTTP request"""
        extra = {
            'job_id': None,
            'user_id': user_id,
            'duration_ms': duration_ms
        }
        self.logger.info(
            f"{method} {path} {status} {duration_ms}ms",
            extra=extra
        )
    
    def log_job(self, job_id: str, event: str, details: dict):
        """Log job event"""
        extra = {'job_id': job_id}
        self.logger.info(
            f"Job {event}: {details}",
            extra=extra
        )
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)


# =============================================================================
# 3. PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor system and application performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def get_system_metrics(self) -> dict:
        """Get current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_process_metrics(self) -> dict:
        """Get current process metrics"""
        process = psutil.Process()
        
        return {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=1),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def record_duration(self, operation: str, duration_seconds: float):
        """Record operation duration"""
        self.metrics[f'{operation}_duration'].append(duration_seconds)
    
    def get_statistics(self, operation: str) -> dict:
        """Get statistics for an operation"""
        durations = self.metrics.get(f'{operation}_duration', [])
        
        if not durations:
            return {}
        
        return {
            'count': len(durations),
            'min': min(durations),
            'max': max(durations),
            'avg': sum(durations) / len(durations),
            'p50': sorted(durations)[len(durations) // 2],
            'p95': sorted(durations)[int(len(durations) * 0.95)],
            'p99': sorted(durations)[int(len(durations) * 0.99)]
        }


# =============================================================================
# 4. HEALTH CHECKS
# =============================================================================

@dataclass
class HealthCheck:
    """Health check result"""
    service: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthChecker:
    """Comprehensive health checking"""
    
    def __init__(self):
        self.checks = {}
    
    def check_database(self) -> HealthCheck:
        """Check database connectivity"""
        # If using database
        return HealthCheck('database', 'healthy')
    
    def check_ollama(self, ollama_url: str) -> HealthCheck:
        """Check Ollama service"""
        import requests
        
        start = time.time()
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return HealthCheck('ollama', 'healthy', latency_ms=latency)
            else:
                return HealthCheck('ollama', 'degraded', f'Status: {response.status_code}')
        except Exception as e:
            return HealthCheck('ollama', 'unhealthy', str(e))
    
    def check_disk_space(self, threshold_percent: float = 90) -> HealthCheck:
        """Check disk space"""
        usage = psutil.disk_usage('/')
        
        if usage.percent > threshold_percent:
            return HealthCheck(
                'disk_space',
                'unhealthy',
                f'Disk usage: {usage.percent}%'
            )
        elif usage.percent > threshold_percent - 10:
            return HealthCheck(
                'disk_space',
                'degraded',
                f'Disk usage: {usage.percent}%'
            )
        else:
            return HealthCheck(
                'disk_space',
                'healthy',
                f'Disk usage: {usage.percent}%'
            )
    
    def check_memory(self, threshold_percent: float = 90) -> HealthCheck:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        
        if memory.percent > threshold_percent:
            return HealthCheck(
                'memory',
                'unhealthy',
                f'Memory usage: {memory.percent}%'
            )
        elif memory.percent > threshold_percent - 10:
            return HealthCheck(
                'memory',
                'degraded',
                f'Memory usage: {memory.percent}%'
            )
        else:
            return HealthCheck(
                'memory',
                'healthy',
                f'Memory usage: {memory.percent}%'
            )
    
    def run_all_checks(self, config: dict) -> dict:
        """Run all health checks"""
        checks = {
            'disk_space': self.check_disk_space(),
            'memory': self.check_memory(),
            'ollama': self.check_ollama(config.get('OLLAMA_URL', 'http://localhost:11434'))
        }
        
        # Overall status
        statuses = [check.status for check in checks.values()]
        if 'unhealthy' in statuses:
            overall = 'unhealthy'
        elif 'degraded' in statuses:
            overall = 'degraded'
        else:
            overall = 'healthy'
        
        return {
            'overall': overall,
            'checks': {
                name: {
                    'status': check.status,
                    'message': check.message,
                    'latency_ms': check.latency_ms
                }
                for name, check in checks.items()
            },
            'timestamp': datetime.utcnow().isoformat()
        }


# =============================================================================
# 5. ALERTING SYSTEM
# =============================================================================

class AlertManager:
    """Send alerts when issues are detected"""
    
    def __init__(self):
        self.alert_history = []
    
    def send_alert(self, severity: str, title: str, message: str, details: dict = None):
        """Send alert (implement actual alerting)"""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'severity': severity,
            'title': title,
            'message': message,
            'details': details or {}
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        logging.error(f"ALERT [{severity}]: {title} - {message}")
        
        # TODO: Implement actual alerting
        # - Email via SMTP
        # - Slack webhook
        # - PagerDuty
        # - SMS via Twilio
        
        return alert
    
    def alert_high_error_rate(self, error_count: int, time_window: int):
        """Alert on high error rate"""
        if error_count > 10:  # Threshold
            self.send_alert(
                'high',
                'High Error Rate',
                f'{error_count} errors in {time_window} seconds',
                {'error_count': error_count, 'time_window': time_window}
            )
    
    def alert_disk_space(self, usage_percent: float):
        """Alert on disk space"""
        if usage_percent > 90:
            self.send_alert(
                'critical',
                'Disk Space Critical',
                f'Disk usage at {usage_percent}%',
                {'usage_percent': usage_percent}
            )
    
    def alert_service_down(self, service_name: str, error: str):
        """Alert when service is down"""
        self.send_alert(
            'critical',
            f'Service Down: {service_name}',
            f'Service {service_name} is unavailable: {error}',
            {'service': service_name, 'error': error}
        )


# =============================================================================
# 6. FASTAPI MIDDLEWARE FOR MONITORING
# =============================================================================

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Add monitoring to all requests"""
    
    def __init__(self, app, logger: StructuredLogger, perf_monitor: PerformanceMonitor):
        super().__init__(app)
        self.logger = logger
        self.perf_monitor = perf_monitor
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request
        self.logger.log_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            user_id=getattr(request.state, 'user_id', None)
        )
        
        # Record metrics
        if PROMETHEUS_AVAILABLE:
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration_ms / 1000)
        
        # Add response headers
        response.headers['X-Response-Time'] = f'{duration_ms:.2f}ms'
        
        return response


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
from fastapi import FastAPI
from monitoring import (
    StructuredLogger, PerformanceMonitor, 
    HealthChecker, MonitoringMiddleware
)

app = FastAPI()

# Initialize monitoring
logger = StructuredLogger('speaker-api')
perf_monitor = PerformanceMonitor()
health_checker = HealthChecker()

# Add middleware
app.add_middleware(MonitoringMiddleware, logger=logger, perf_monitor=perf_monitor)

@app.get("/metrics")
async def metrics():
    # Prometheus metrics endpoint
    if PROMETHEUS_AVAILABLE:
        return Response(generate_latest(REGISTRY), media_type="text/plain")
    return {"error": "Prometheus not available"}

@app.get("/health")
async def health():
    return health_checker.run_all_checks({'OLLAMA_URL': 'http://localhost:11434'})

@app.get("/stats")
async def stats():
    return {
        'system': perf_monitor.get_system_metrics(),
        'process': perf_monitor.get_process_metrics(),
        'operations': {
            'upload': perf_monitor.get_statistics('upload'),
            'processing': perf_monitor.get_statistics('processing')
        }
    }
"""
