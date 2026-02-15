# tests/test_separator.py
import pytest
from pathlib import Path
from improved_speaker_separator import ImprovedSpeakerSeparator
   
def test_separator_initialization():
    output_dir = Path("test_output")
    separator = ImprovedSpeakerSeparator(output_dir)
    assert output_dir.exists()
   
def test_separate_speakers_signature():
    # Ensure method signature matches API expectations
    separator = ImprovedSpeakerSeparator(Path("test"))
    import inspect
    sig = inspect.signature(separator.separate_speakers)
    params = list(sig.parameters.keys())
    assert 'progress_callback' not in params