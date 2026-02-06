#!/usr/bin/env python3
"""
Improved Speaker Separator with Better Speaker Identification
Uses multiple features and improved clustering for better speaker separation
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Install required libraries
try:
    import librosa
    import soundfile as sf
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal import medfilt
except ImportError:
    print("Installing required libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "librosa", "soundfile", "numpy", "scipy"])
    import librosa
    import soundfile as sf
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal import medfilt

try:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
except ImportError:
    print("Installing scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture


class ImprovedSpeakerSeparator:
    """Enhanced speaker separation with better voice identification"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_enhanced_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract multiple acoustic features for better speaker distinction
        """
        print("  Extracting enhanced acoustic features...")
        
        # 1. MFCC (Mel-frequency cepstral coefficients) - voice timbre
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 2. Delta MFCC (rate of change) - speaking dynamics
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 3. Pitch (F0) - voice pitch/fundamental frequency
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches, axis=1)
        
        # 4. Spectral Centroid - brightness of voice
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # 5. Spectral Rolloff - frequency distribution
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # 6. Zero Crossing Rate - voice texture
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # 7. Chroma features - harmonic content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=0)
        
        # Ensure all features have same length
        min_len = min(mfcc.shape[1], len(spectral_centroid), 
                     len(spectral_rolloff), len(zcr), len(chroma_mean))
        
        # Combine all features
        features = np.vstack([
            mfcc[:, :min_len],
            mfcc_delta[:, :min_len],
            spectral_centroid[:min_len].reshape(1, -1),
            spectral_rolloff[:min_len].reshape(1, -1),
            zcr[:min_len].reshape(1, -1),
            chroma_mean[:min_len].reshape(1, -1)
        ])
        
        print(f"    Extracted {features.shape[0]} features across {features.shape[1]} frames")
        
        return features
    
    def detect_voice_activity(self, y: np.ndarray, sr: int, 
                             frame_length: int = 2048, 
                             hop_length: int = 512) -> np.ndarray:
        """
        Improved voice activity detection
        """
        print("  Detecting voice activity...")
        
        # Energy-based detection
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Spectral flatness (distinguishes voice from noise)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        
        # Combine metrics
        # Voice has higher energy and lower flatness
        energy_threshold = np.percentile(rms, 40)
        flatness_threshold = np.percentile(flatness, 60)
        
        voice_activity = (rms > energy_threshold) & (flatness < flatness_threshold)
        
        # Smooth the activity detection
        voice_activity = median_filter(voice_activity.astype(float), size=5) > 0.5
        
        speaking_time = np.sum(voice_activity) * hop_length / sr
        print(f"    Voice activity: {speaking_time:.2f}s / {len(y)/sr:.2f}s")
        
        return voice_activity
    
    def cluster_speakers_advanced(self, features: np.ndarray, 
                                  voice_activity: np.ndarray,
                                  n_speakers: int = 2,
                                  method: str = 'gmm') -> np.ndarray:
        """
        Advanced clustering with multiple methods
        """
        print(f"  Clustering into {n_speakers} speakers using {method}...")
        
        # Only cluster frames with voice activity
        active_frames = np.where(voice_activity)[0]
        
        if len(active_frames) < n_speakers * 10:
            raise ValueError(f"Not enough voice activity detected. Only {len(active_frames)} frames found.")
        
        # Get features for active frames
        active_features = features[:, active_frames].T
        
        # Normalize features
        scaler = StandardScaler()
        active_features_scaled = scaler.fit_transform(active_features)
        
        # Cluster
        labels = np.zeros(features.shape[1], dtype=int)
        
        if method == 'gmm':
            # Gaussian Mixture Model - better for overlapping clusters
            gmm = GaussianMixture(
                n_components=n_speakers, 
                covariance_type='full',
                n_init=10,
                random_state=42
            )
            active_labels = gmm.fit_predict(active_features_scaled)
            
        elif method == 'spectral':
            # Spectral Clustering - better for non-convex clusters
            spectral = SpectralClustering(
                n_clusters=n_speakers,
                affinity='nearest_neighbors',
                random_state=42
            )
            active_labels = spectral.fit_predict(active_features_scaled)
            
        else:  # kmeans
            # K-means - fast and simple
            kmeans = KMeans(
                n_clusters=n_speakers,
                n_init=20,
                random_state=42
            )
            active_labels = kmeans.fit_predict(active_features_scaled)
        
        # Assign labels to all frames
        labels[active_frames] = active_labels
        
        # Smooth labels to reduce rapid switching
        labels = medfilt(labels.astype(float), kernel_size=11).astype(int)
        
        # Print speaker distribution
        for speaker_id in range(n_speakers):
            speaker_frames = np.sum(labels == speaker_id)
            print(f"    Speaker {speaker_id + 1}: {speaker_frames} frames")
        
        return labels
    
    def separate_and_save(self, y: np.ndarray, sr: int, 
                         labels: np.ndarray, hop_length: int,
                         n_speakers: int) -> Dict:
        """
        Separate audio by speaker labels and save
        """
        print("  Separating and saving speaker audio...")
        
        results = {}
        
        for speaker_id in range(n_speakers):
            # Create mask for this speaker
            speaker_mask = labels == speaker_id
            
            # Expand mask to audio samples
            time_mask = np.repeat(speaker_mask, hop_length)
            time_mask = time_mask[:len(y)]
            
            # Apply mask with fade in/out to reduce clicks
            speaker_audio = y.copy()
            
            # Create smooth mask
            smooth_mask = np.zeros(len(y))
            smooth_mask[:len(time_mask)] = time_mask
            
            # Apply smoothing window
            from scipy.signal import windows
            window_size = 441  # 10ms at 44.1kHz
            smooth_mask = np.convolve(smooth_mask, windows.hann(window_size), mode='same')
            smooth_mask = np.clip(smooth_mask, 0, 1)
            
            speaker_audio = speaker_audio * smooth_mask
            
            # Calculate statistics
            speaking_time = np.sum(speaker_mask) * hop_length / sr
            speaking_percentage = speaking_time / (len(y) / sr) * 100
            
            # Normalize audio
            if np.max(np.abs(speaker_audio)) > 0:
                speaker_audio = speaker_audio / np.max(np.abs(speaker_audio)) * 0.9
            
            # Save
            output_path = self.output_dir / f"speaker_{speaker_id + 1}.wav"
            sf.write(output_path, speaker_audio, sr)
            
            results[f"speaker_{speaker_id + 1}"] = {
                "path": str(output_path),
                "speaking_time": float(speaking_time),
                "speaking_percentage": float(speaking_percentage),
                "frame_count": int(np.sum(speaker_mask))
            }
            
            print(f"    Speaker {speaker_id + 1}: {speaking_time:.2f}s ({speaking_percentage:.1f}%)")
        
        return results
    
    def separate_speakers(self, audio_path: Path, 
                         n_speakers: int = 2,
                         method: str = 'gmm') -> Dict:
        """
        Main speaker separation pipeline
        """
        print("\n" + "=" * 60)
        print(f"Speaker Separation: {audio_path.name}")
        print("=" * 60)
        
        # Load audio
        print("\nLoading audio...")
        y, sr = librosa.load(str(audio_path), sr=16000)  # Resample to 16kHz
        duration = len(y) / sr
        print(f"  Duration: {duration:.2f}s")
        print(f"  Sample rate: {sr} Hz")
        
        # Parameters
        hop_length = 512
        frame_length = 2048
        
        # Step 1: Extract features
        features = self.extract_enhanced_features(y, sr)
        
        # Step 2: Detect voice activity
        voice_activity = self.detect_voice_activity(y, sr, frame_length, hop_length)
        
        # Ensure features and voice_activity align
        min_len = min(features.shape[1], len(voice_activity))
        features = features[:, :min_len]
        voice_activity = voice_activity[:min_len]
        
        # Step 3: Cluster speakers
        try:
            labels = self.cluster_speakers_advanced(
                features, voice_activity, n_speakers, method
            )
        except ValueError as e:
            print(f"\nError: {e}")
            print("Try with a longer audio file or check if there's actual speech in the file.")
            raise
        
        # Step 4: Separate and save
        separated_files = self.separate_and_save(y, sr, labels, hop_length, n_speakers)
        
        print("\n" + "=" * 60)
        print("✓ Separation Complete!")
        print("=" * 60)
        
        return {
            "speakers": separated_files,
            "method": method,
            "n_speakers": n_speakers
        }


def analyze_separation_quality(output_dir: Path, n_speakers: int):
    """
    Analyze the quality of speaker separation
    """
    print("\n" + "=" * 60)
    print("Quality Analysis")
    print("=" * 60)
    
    for speaker_id in range(1, n_speakers + 1):
        speaker_file = output_dir / f"speaker_{speaker_id}.wav"
        
        if not speaker_file.exists():
            continue
        
        y, sr = librosa.load(str(speaker_file), sr=None)
        
        # Calculate metrics
        rms = np.sqrt(np.mean(y ** 2))
        peak = np.max(np.abs(y))
        
        # Voice activity
        energy = librosa.feature.rms(y=y)[0]
        voice_activity_ratio = np.sum(energy > np.mean(energy) * 0.3) / len(energy)
        
        print(f"\nSpeaker {speaker_id}:")
        print(f"  RMS Level: {rms:.4f}")
        print(f"  Peak Level: {peak:.4f}")
        print(f"  Voice Activity: {voice_activity_ratio * 100:.1f}%")


def create_test_audio():
    """
    Create a test audio file with 2 simulated speakers
    """
    print("\nCreating test audio file...")
    
    sr = 16000
    duration = 10  # seconds
    
    # Speaker 1: Lower pitch
    t = np.linspace(0, duration, sr * duration)
    freq1 = 150  # Hz
    speaker1 = np.sin(2 * np.pi * freq1 * t)
    
    # Add pauses
    speaker1_mask = np.zeros_like(speaker1)
    speaker1_mask[0:sr*2] = 1  # 0-2s
    speaker1_mask[sr*4:sr*6] = 1  # 4-6s
    speaker1_mask[sr*8:sr*10] = 1  # 8-10s
    speaker1 = speaker1 * speaker1_mask
    
    # Speaker 2: Higher pitch
    freq2 = 250  # Hz
    speaker2 = np.sin(2 * np.pi * freq2 * t)
    
    # Add pauses (different timing)
    speaker2_mask = np.zeros_like(speaker2)
    speaker2_mask[sr*2:sr*4] = 1  # 2-4s
    speaker2_mask[sr*6:sr*8] = 1  # 6-8s
    speaker2 = speaker2 * speaker2_mask
    
    # Mix
    mixed = speaker1 + speaker2
    mixed = mixed / np.max(np.abs(mixed)) * 0.9
    
    # Save
    output_path = Path("test_conversation.wav")
    sf.write(output_path, mixed, sr)
    
    print(f"✓ Test audio created: {output_path}")
    print("  This file has 2 speakers with different pitches")
    print("  Speaker 1 speaks at: 0-2s, 4-6s, 8-10s")
    print("  Speaker 2 speaks at: 2-4s, 6-8s")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Improved Speaker Separation Tool"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to audio file with multiple speakers"
    )
    parser.add_argument(
        "-n", "--num-speakers",
        type=int,
        default=2,
        help="Number of speakers (default: 2)"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["gmm", "kmeans", "spectral"],
        default="gmm",
        help="Clustering method (default: gmm)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="separated_speakers",
        help="Output directory"
    )
    parser.add_argument(
        "--create-test",
        action="store_true",
        help="Create a test audio file"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze separation quality"
    )
    
    args = parser.parse_args()
    
    # Create test audio
    if args.create_test:
        test_file = create_test_audio()
        print(f"\nNow run: python {sys.argv[0]} {test_file}")
        return
    
    # Check if audio file provided
    if not args.audio_file:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("  1. Create test audio: python improved_speaker_separator.py --create-test")
        print("  2. Process audio: python improved_speaker_separator.py conversation.wav")
        print("  3. Try different methods: --method gmm|kmeans|spectral")
        print("=" * 60)
        return
    
    # Check if file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {args.audio_file}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir) / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate speakers
    separator = ImprovedSpeakerSeparator(output_dir)
    
    try:
        results = separator.separate_speakers(
            audio_path,
            n_speakers=args.num_speakers,
            method=args.method
        )
        
        print("\nOutput files:")
        for speaker, info in results.items():
            print(f"  {info['path']}")
        
        # Quality analysis
        if args.analyze:
            analyze_separation_quality(output_dir, args.num_speakers)
        
        print(f"\n💡 Tip: Listen to the separated files and adjust --num-speakers if needed")
        print(f"💡 Try different methods: --method gmm (default), kmeans, or spectral")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check if audio has clear speech (not just noise)")
        print("  2. Try with --num-speakers 2 first")
        print("  3. Ensure audio is at least 10 seconds long")
        print("  4. Try different methods: --method kmeans or --method spectral")
        sys.exit(1)


if __name__ == "__main__":
    main()
