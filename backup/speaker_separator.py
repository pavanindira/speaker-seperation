#!/usr/bin/env python3
"""
Speaker Diarization and Separation API
Separates audio by individual speakers with AI assistance from Ollama
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import json

# Install required libraries
try:
    import requests
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Installing audio processing libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "requests", "librosa", "soundfile", "numpy", "scipy"])
    import requests
    import librosa
    import soundfile as sf
    import numpy as np

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️  pydub not installed. Install with: pip install pydub")


class OllamaAssistant:
    """Ollama assistant for audio analysis"""
    
    def __init__(self, model="llama3.2", ollama_url="http://192.168.1.33:11434"):
        self.model = model
        self.ollama_url = ollama_url
    
    def ask(self, prompt: str, system_prompt: str = None) -> str:
        """Ask Ollama a question"""
        url = f"{self.ollama_url}/api/generate"
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "You are an expert in audio processing and speaker separation.",
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            if response.status_code == 200:
                return response.json()['response']
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_conversation(self, num_speakers: int, duration: float, 
                           speaker_stats: Dict) -> str:
        """Get analysis of the conversation"""
        prompt = f"""Analyze this conversation recording:

Number of speakers detected: {num_speakers}
Total duration: {duration:.2f} seconds
Speaker statistics:
{json.dumps(speaker_stats, indent=2)}

Provide insights about:
1. Who speaks more (dominant speaker)
2. Speaking patterns observed
3. Recommendations for improving speaker separation quality
4. Any audio quality issues to address

Keep response concise (3-4 sentences)."""

        return self.ask(prompt)


class SpeakerSeparator:
    """Speaker diarization and separation"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_audio(self, audio_path: Path) -> Dict:
        """Analyze audio characteristics"""
        print(f"\nAnalyzing audio: {audio_path}")
        
        y, sr = librosa.load(str(audio_path), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate RMS energy to find speech segments
        rms = librosa.feature.rms(y=y)[0]
        
        # Detect voice activity (simple threshold-based)
        threshold = np.mean(rms) * 0.5
        voice_activity = rms > threshold
        
        # Calculate speaking time
        speaking_time = np.sum(voice_activity) / sr
        
        # Debug info
        rms_mean = np.mean(rms)
        rms_max = np.max(rms)
        rms_std = np.std(rms)
        voice_percentage = (np.sum(voice_activity) / len(rms)) * 100
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Sample Rate: {sr} Hz")
        print(f"  Speaking Time: {speaking_time:.2f}s")
        print(f"  Silence: {duration - speaking_time:.2f}s")
        print(f"  RMS Stats: Mean={rms_mean:.4f}, Max={rms_max:.4f}, Std={rms_std:.4f}")
        print(f"  Voice Frames Detected: {voice_percentage:.2f}% ({np.sum(voice_activity)}/{len(rms)} frames)")
        
        return {
            'sr': sr,
            'duration': duration,
            'speaking_time': speaking_time,
            'audio': y,
            'rms': rms,
            'rms_stats': {
                'mean': float(rms_mean),
                'max': float(rms_max),
                'std': float(rms_std),
                'voice_percentage': float(voice_percentage)
            }
        }
    
    def separate_by_clustering(self, audio_path: Path, n_speakers: int = 2) -> Dict:
        """
        Separate speakers using spectral clustering with aggressive voice extraction
        """
        print(f"\nSeparating {n_speakers} speakers using clustering...")
        
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=16000)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Compute frame-level features
        hop_length = 512
        frame_length = 2048
        
        # Aggressive voice activity detection
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_flux = np.sqrt(np.sum(np.diff(np.abs(librosa.stft(y)))**2, axis=0))
        
        # Multi-criteria voice detection
        rms_threshold = np.percentile(rms, 10)
        spectral_flux_threshold = np.percentile(spectral_flux, 20)
        
        # Use stricter criteria: must pass both thresholds
        rms_frames = rms > rms_threshold
        
        # Match spectral flux length with rms
        if len(spectral_flux) < len(rms):
            spectral_flux = np.pad(spectral_flux, (0, len(rms) - len(spectral_flux)))
        elif len(spectral_flux) > len(rms):
            spectral_flux = spectral_flux[:len(rms)]
        
        spectral_frames = spectral_flux > spectral_flux_threshold
        voice_frames = rms_frames & spectral_frames  # Both must be true
        
        # If too strict, relax to just RMS
        if np.sum(voice_frames) < 20:
            voice_frames = rms_frames
            print("  Warning: Relaxed to RMS-only detection (spectral too strict)")
        
        voice_frame_count = np.sum(voice_frames)
        voice_frame_percentage = (voice_frame_count / len(rms)) * 100
        print(f"  Voice frames for clustering: {voice_frame_percentage:.2f}% ({voice_frame_count}/{len(rms)} frames)")
        print(f"  RMS threshold: {rms_threshold:.6f}, Spectral flux threshold: {spectral_flux_threshold:.6f}")
        
        # Normalize MFCC
        mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-10)
        
        # Cluster frames into speakers using spectral clustering
        from sklearn.cluster import SpectralClustering
        
        # Prepare data for clustering
        X = mfcc_normalized[:, voice_frames].T
        
        # Use spectral clustering with affinity based on euclidean distance
        clustering = SpectralClustering(
            n_clusters=n_speakers,
            affinity='nearest_neighbors',
            n_neighbors=max(5, int(np.sqrt(len(X)))),
            random_state=42,
            n_init=20
        )
        
        cluster_labels = clustering.fit_predict(X)
        labels = np.zeros(len(rms), dtype=int)
        labels[voice_frames] = cluster_labels
        
        print(f"  Clustering algorithm: Spectral Clustering with dual thresholds")
        
        # Separate audio by speaker
        results = {}
        speaker_stats = {}
        
        for speaker_id in range(n_speakers):
            speaker_mask = labels == speaker_id
            
            # Create time-domain mask
            time_mask = np.repeat(speaker_mask, hop_length)
            time_mask = time_mask[:len(y)]
            
            # Apply mask
            speaker_audio = y * time_mask
            
            # Calculate statistics
            speaking_time = np.sum(speaker_mask) * hop_length / sr
            
            speaker_stats[f"speaker_{speaker_id + 1}"] = {
                "speaking_time": float(speaking_time),
                "speaking_percentage": float(speaking_time / (len(y) / sr) * 100)
            }
            
            # Save audio
            output_path = self.output_dir / f"speaker_{speaker_id + 1}.wav"
            sf.write(output_path, speaker_audio, sr)
            
            results[f"speaker_{speaker_id + 1}"] = {
                "path": str(output_path),
                "speaking_time": speaking_time
            }
            
            print(f"  Speaker {speaker_id + 1}: {speaking_time:.2f}s speaking time")
        
        return {
            "speakers": results,
            "stats": speaker_stats,
            "method": "clustering"
        }
    
    def separate_by_frequency(self, audio_path: Path) -> Dict:
        """
        Separate speakers by frequency range (basic method)
        Useful when speakers have different pitch ranges
        """
        print("\nSeparating speakers by frequency characteristics...")
        
        y, sr = librosa.load(str(audio_path), sr=None)
        
        # Compute STFT
        D = librosa.stft(y)
        magnitude = np.abs(D)
        
        # Split frequency bands (rough approximation)
        n_freq_bins = magnitude.shape[0]
        mid_point = n_freq_bins // 2
        
        # Lower frequencies (typically male voices)
        lower_mask = np.zeros_like(magnitude)
        lower_mask[:mid_point, :] = magnitude[:mid_point, :]
        
        # Higher frequencies (typically female/higher voices)
        upper_mask = np.zeros_like(magnitude)
        upper_mask[mid_point:, :] = magnitude[mid_point:, :]
        
        # Reconstruct audio
        phase = np.angle(D)
        
        lower_audio = librosa.istft(lower_mask * np.exp(1j * phase))
        upper_audio = librosa.istft(upper_mask * np.exp(1j * phase))
        
        # Save separated audio
        lower_path = self.output_dir / "speaker_lower_frequency.wav"
        upper_path = self.output_dir / "speaker_higher_frequency.wav"
        
        sf.write(lower_path, lower_audio, sr)
        sf.write(upper_path, upper_audio, sr)
        
        results = {
            "speaker_lower": {"path": str(lower_path)},
            "speaker_higher": {"path": str(upper_path)}
        }
        
        print(f"  Lower frequency speaker: {lower_path}")
        print(f"  Higher frequency speaker: {upper_path}")
        
        return {
            "speakers": results,
            "method": "frequency_separation"
        }

    def separate_by_demucs(self, audio_path: Path, n_speakers: int = 2) -> Dict:
        """
        Separate speakers/music using demucs neural source separation
        Demucs provides high-quality vocal/instrumental/drum/bass separation
        which can be used as preprocessing for speaker diarization
        """
        print(f"\nSeparating audio using demucs neural network...")
        print(f"  Note: demucs performs music/vocal/instrumental separation")
        print(f"        useful for isolating speech content for speaker separation")
        
        try:
            import demucs.separate
            from demucs.pretrained import get_model
        except ImportError:
            return {
                "error": "demucs not installed. Install with: pip install demucs",
                "method": "demucs"
            }
        
        # Create temporary output directory for demucs
        demucs_output = self.output_dir / "demucs_stems"
        demucs_output.mkdir(exist_ok=True)
        
        try:
            # Load audio file
            y, sr = librosa.load(str(audio_path), sr=16000)
            duration = len(y) / sr
            
            print(f"  Audio loaded: {duration:.2f}s at {sr}Hz")
            print(f"  Extracting stems with default demucs model...")
            
            # Save temporary wav for demucs processing
            temp_input = demucs_output / "input_temp.wav"
            sf.write(str(temp_input), y, sr)
            
            # Use demucs to separate stems
            separator = demucs.separate.Separator(
                model=get_model('htdemucs'),
                device='cpu',
                progress=True,
                jobs=1
            )
            
            # Separate the audio
            stems = separator.separate_file(str(temp_input), output_dir=str(demucs_output))
            
            print(f"  Demucs stem separation completed")
            print(f"  Available stems: vocals, drums, bass, other")
            
            # For speaker separation, use vocals stem as primary source
            vocals_path = None
            for stem_file in demucs_output.glob("**/vocals.wav"):
                vocals_path = stem_file
                break
            
            if vocals_path:
                print(f"  Vocals stem isolated: {vocals_path}")
                # Further separate the vocals using clustering
                separator = SpeakerSeparator(self.output_dir)
                
                # Use vocals stem for speaker separation
                vocals_result = separator.separate_by_clustering(vocals_path, n_speakers)
                
                return {
                    "speakers": vocals_result.get("speakers", {}),
                    "stats": vocals_result.get("stats", {}),
                    "method": "demucs_clustering",
                    "demucs_stems_dir": str(demucs_output),
                    "note": "Vocals stem isolated then separated by speaker using clustering"
                }
            else:
                return {
                    "error": "Could not extract vocals stem from demucs",
                    "method": "demucs"
                }
                
        except Exception as e:
            print(f"  Warning: demucs separation failed: {e}")
            return {
                "error": str(e),
                "method": "demucs"
            }


class ConversationProcessor:
    """Main processor for conversation separation"""
    
    def __init__(self, ollama_assistant: Optional[OllamaAssistant] = None):
        self.assistant = ollama_assistant
        self.output_dir = Path("conversation_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def process_conversation(self, audio_path: Path, 
                           num_speakers: int = 2,
                           method: str = "auto") -> Dict:
        """Process conversation and separate speakers"""
        
        print("=" * 60)
        print("Conversation Speaker Separation")
        print("=" * 60)
        
        # Create job-specific output directory
        job_output_dir = self.output_dir / audio_path.stem
        job_output_dir.mkdir(exist_ok=True)
        
        # Initialize separator
        separator = SpeakerSeparator(job_output_dir)
        
        # Analyze audio
        analysis = separator.analyze_audio(audio_path)
        
        # Get Ollama recommendation
        if self.assistant:
            print("\n🤖 Asking Ollama for analysis...")
            
            pre_analysis = self.assistant.ask(f"""
I have a conversation audio file with approximately {num_speakers} speakers.
Duration: {analysis['duration']:.2f} seconds
Speaking time: {analysis['speaking_time']:.2f} seconds

What method would work best for separating the speakers?
- Clustering (MFCC-based separation)
- Frequency separation (good if speakers have different pitch)
- Deep learning (most accurate but slower)

Recommend one method and explain why in 2 sentences.""")
            
            print(f"\nOllama's Recommendation:\n{pre_analysis}\n")
            
            # Auto-select based on recommendation
            if method == "auto":
                if "frequency" in pre_analysis.lower():
                    method = "frequency"
                else:
                    method = "clustering"
        
        # Default method
        if method == "auto":
            method = "clustering"
        
        # Separate speakers
        print(f"\nUsing method: {method}")
        
        try:
            if method == "clustering":
                result = separator.separate_by_clustering(audio_path, num_speakers)
            elif method == "frequency":
                result = separator.separate_by_frequency(audio_path)
            elif method == "demucs":
                result = separator.separate_by_demucs(audio_path, num_speakers)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Get post-processing analysis from Ollama
            if self.assistant and result.get("stats"):
                print("\n🤖 Asking Ollama for conversation analysis...")
                
                post_analysis = self.assistant.analyze_conversation(
                    num_speakers,
                    analysis['duration'],
                    result["stats"]
                )
                
                print(f"\nConversation Analysis:\n{post_analysis}\n")
                result["ollama_analysis"] = post_analysis
            
            print("\n" + "=" * 60)
            print("✓ Separation Complete!")
            print("=" * 60)
            
            print("\nSeparated audio files:")
            for speaker, info in result["speakers"].items():
                print(f"  {speaker}: {info['path']}")
                if 'speaking_time' in info:
                    print(f"    Speaking time: {info['speaking_time']:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"\n✗ Separation failed: {str(e)}")
            raise


def install_dependencies():
    """Install required dependencies"""
    print("\n" + "=" * 60)
    print("Installing Speaker Separation Dependencies")
    print("=" * 60)
    
    packages = [
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "scikit-learn",  # For clustering
        "pydub",         # For audio format conversion
        "requests"       # For Ollama API calls
    ]
    
    print("\nInstalling packages...")
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            print(f"    ✓ {package} installed")
        except:
            print(f"    ✗ {package} failed (may need manual installation)")
    
    print("\n✓ Installation complete!")
    print("\nNote: For AI-powered method recommendations, ensure your Ollama instance")
    print("      is running and accessible via the --ollama-url parameter.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Separate individual speakers from conversation audio"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to conversation audio file"
    )
    parser.add_argument(
        "-n", "--num-speakers",
        type=int,
        default=2,
        help="Number of speakers (default: 2)"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["auto", "clustering", "frequency", "demucs"],
        default="auto",
        help="Separation method (default: auto)"
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Disable Ollama AI assistance"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install required dependencies"
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama API URL"
    )
    
    args = parser.parse_args()
    
    # Handle installation
    if args.install:
        install_dependencies()
        return
    
    # Check if audio file provided
    if not args.audio_file:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("  1. Install dependencies: python speaker_separator.py --install")
        print("  2. Process audio: python speaker_separator.py conversation.wav")
        print("=" * 60)
        return
    
    # Check if file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        return
    
    # Initialize Ollama assistant
    assistant = None
    if not args.no_ollama:
        try:
            assistant = OllamaAssistant(ollama_url=args.ollama_url)
            # Test connection
            test = assistant.ask("Say OK", "Respond with just OK")
            print("✓ Connected to Ollama")
        except Exception as e:
            print(f"⚠️  Could not connect to Ollama: {e}")
            print("  Continuing without AI assistance...")
    
    # Process conversation
    processor = ConversationProcessor(ollama_assistant=assistant)
    
    try:
        result = processor.process_conversation(
            Path(args.audio_file),
            num_speakers=args.num_speakers,
            method=args.method
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
