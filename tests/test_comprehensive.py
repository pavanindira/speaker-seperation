#!/usr/bin/env python3
"""
Comprehensive Test Suite for Speaker Separation API
"""

import pytest
import io
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_client():
    """Create test client"""
    from final_speaker_api import app
    return TestClient(app)


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing"""
    sample_rate = 16000
    duration = 2  # seconds
    
    # Generate simple sine wave
    t = np.linspace(0, duration, sample_rate * duration)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Save to bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    
    return buffer.getvalue()


@pytest.fixture
def auth_headers():
    """Get authentication headers"""
    return {"X-API-Key": "test-token"}


# =============================================================================
# UNIT TESTS - Speaker Separator
# =============================================================================

class TestSpeakerSeparator:
    """Test speaker separation logic"""
    
    def test_initialization(self):
        """Test separator initialization"""
        from improved_speaker_separator import ImprovedSpeakerSeparator
        
        output_dir = Path("test_output")
        separator = ImprovedSpeakerSeparator(output_dir)
        
        assert output_dir.exists()
        assert separator.output_dir == output_dir
    
    def test_method_signature(self):
        """Ensure separate_speakers has correct signature"""
        from improved_speaker_separator import ImprovedSpeakerSeparator
        import inspect
        
        separator = ImprovedSpeakerSeparator(Path("test"))
        sig = inspect.signature(separator.separate_speakers)
        params = list(sig.parameters.keys())
        
        # Critical: No progress_callback parameter
        assert 'progress_callback' not in params
        assert 'audio_path' in params
        assert 'n_speakers' in params
        assert 'method' in params
    
    def test_extract_features(self, sample_audio):
        """Test feature extraction"""
        from improved_speaker_separator import ImprovedSpeakerSeparator
        import librosa
        
        # Load audio
        y, sr = librosa.load(io.BytesIO(sample_audio), sr=16000)
        
        separator = ImprovedSpeakerSeparator(Path("test"))
        features = separator.extract_enhanced_features(y, sr)
        
        assert features.shape[0] > 0  # Has features
        assert features.shape[1] > 0  # Has frames


# =============================================================================
# INTEGRATION TESTS - API Endpoints
# =============================================================================

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_health_check(self, test_client):
        """Test health endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()
    
    def test_upload_without_auth(self, test_client, sample_audio):
        """Test upload fails without authentication"""
        files = {"file": ("test.wav", sample_audio, "audio/wav")}
        response = test_client.post("/api/v1/upload", files=files)
        
        # Should fail without auth
        assert response.status_code == 401
    
    def test_upload_with_auth(self, test_client, sample_audio, auth_headers):
        """Test upload succeeds with authentication"""
        files = {"file": ("test.wav", sample_audio, "audio/wav")}
        response = test_client.post(
            "/api/v1/upload",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "uploaded"
    
    def test_invalid_file_type(self, test_client, auth_headers):
        """Test upload rejects invalid file types"""
        fake_file = b"This is not an audio file"
        files = {"file": ("test.txt", fake_file, "text/plain")}
        
        response = test_client.post(
            "/api/v1/upload",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == 400
    
    def test_file_too_large(self, test_client, auth_headers):
        """Test upload rejects files that are too large"""
        # Create a large fake file
        large_file = b"x" * (101 * 1024 * 1024)  # 101 MB
        files = {"file": ("huge.wav", large_file, "audio/wav")}
        
        response = test_client.post(
            "/api/v1/upload",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == 413  # Payload too large


# =============================================================================
# SECURITY TESTS
# =============================================================================

class TestSecurity:
    """Test security features"""
    
    def test_path_traversal_protection(self, test_client, auth_headers):
        """Test protection against path traversal attacks"""
        malicious_filename = "../../etc/passwd"
        files = {"file": (malicious_filename, b"test", "audio/wav")}
        
        response = test_client.post(
            "/api/v1/upload",
            files=files,
            headers=auth_headers
        )
        
        # Should sanitize filename
        if response.status_code == 200:
            assert "etc" not in response.json().get("filename", "")
    
    def test_rate_limiting(self, test_client, sample_audio, auth_headers):
        """Test rate limiting"""
        files = {"file": ("test.wav", sample_audio, "audio/wav")}
        
        # Make multiple requests
        responses = []
        for i in range(6):  # Limit is 5 per minute
            response = test_client.post(
                "/api/v1/upload",
                files=files,
                headers=auth_headers
            )
            responses.append(response.status_code)
        
        # At least one should be rate limited
        assert 429 in responses  # Too many requests
    
    def test_xss_protection_headers(self, test_client):
        """Test XSS protection headers are present"""
        response = test_client.get("/")
        
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_audio_processing_speed(self, sample_audio):
        """Test audio processing completes in reasonable time"""
        import time
        from improved_speaker_separator import ImprovedSpeakerSeparator
        import tempfile
        
        # Write test audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(sample_audio)
            audio_path = Path(f.name)
        
        try:
            separator = ImprovedSpeakerSeparator(Path("test_output"))
            
            start_time = time.time()
            separator.separate_speakers(audio_path, n_speakers=2, method='gmm')
            duration = time.time() - start_time
            
            # Should complete in under 10 seconds for 2-second audio
            assert duration < 10
        finally:
            audio_path.unlink()
    
    def test_concurrent_uploads(self, test_client, sample_audio, auth_headers):
        """Test handling concurrent uploads"""
        import concurrent.futures
        
        files = {"file": ("test.wav", sample_audio, "audio/wav")}
        
        def upload():
            return test_client.post(
                "/api/v1/upload",
                files=files,
                headers=auth_headers
            )
        
        # Test 5 concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload) for _ in range(5)]
            results = [f.result() for f in futures]
        
        # All should succeed or be rate-limited (not crash)
        for result in results:
            assert result.status_code in [200, 429]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_audio_file(self, test_client, auth_headers):
        """Test handling of empty audio file"""
        files = {"file": ("empty.wav", b"", "audio/wav")}
        response = test_client.post(
            "/api/v1/upload",
            files=files,
            headers=auth_headers
        )
        
        # Should reject empty files
        assert response.status_code in [400, 422]
    
    def test_corrupted_audio_file(self, test_client, auth_headers):
        """Test handling of corrupted audio file"""
        corrupted = b"RIFF" + b"x" * 100  # Fake WAV header
        files = {"file": ("corrupted.wav", corrupted, "audio/wav")}
        
        response = test_client.post(
            "/api/v1/upload",
            files=files,
            headers=auth_headers
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 500]
    
    def test_very_short_audio(self, test_client, auth_headers):
        """Test handling of very short audio (< 1 second)"""
        # Generate 0.5 second audio
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        buffer.seek(0)
        
        files = {"file": ("short.wav", buffer.getvalue(), "audio/wav")}
        response = test_client.post(
            "/api/v1/upload",
            files=files,
            headers=auth_headers
        )
        
        # Should handle or reject gracefully
        assert response.status_code in [200, 400]


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
