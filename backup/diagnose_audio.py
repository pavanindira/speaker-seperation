#!/usr/bin/env python3
"""
Speaker Separation Diagnostic Tool
Helps identify why speaker separation might not be working
"""

import sys
import subprocess
from pathlib import Path

try:
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Installing required libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "librosa", "numpy", "matplotlib"])
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True


def diagnose_audio(audio_path: Path):
    """
    Comprehensive audio diagnostic
    """
    print("=" * 60)
    print(f"Diagnosing: {audio_path.name}")
    print("=" * 60)
    
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=None)
    duration = len(y) / sr
    
    print(f"\n1. BASIC INFORMATION")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Sample Rate: {sr} Hz")
    print(f"   Samples: {len(y)}")
    
    # Audio level analysis
    rms = np.sqrt(np.mean(y ** 2))
    peak = np.max(np.abs(y))
    
    print(f"\n2. AUDIO LEVELS")
    print(f"   RMS Level: {rms:.4f}")
    print(f"   Peak Level: {peak:.4f}")
    
    if peak < 0.01:
        print("   ⚠️  WARNING: Very low volume! Audio might be too quiet.")
    
    # Voice activity detection
    frame_length = 2048
    hop_length = 512
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    threshold = np.percentile(rms_frames, 40)
    voice_activity = rms_frames > threshold
    speaking_time = np.sum(voice_activity) * hop_length / sr
    silence_time = duration - speaking_time
    
    print(f"\n3. VOICE ACTIVITY")
    print(f"   Speaking Time: {speaking_time:.2f}s ({speaking_time/duration*100:.1f}%)")
    print(f"   Silence Time: {silence_time:.2f}s ({silence_time/duration*100:.1f}%)")
    
    if speaking_time < 5:
        print("   ⚠️  WARNING: Less than 5 seconds of speech detected!")
        print("      Speaker separation needs at least 10 seconds per speaker.")
    
    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get dominant pitches
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if len(pitch_values) > 0:
        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)
        min_pitch = np.min(pitch_values)
        max_pitch = np.max(pitch_values)
        
        print(f"\n4. PITCH ANALYSIS")
        print(f"   Mean Pitch: {mean_pitch:.1f} Hz")
        print(f"   Pitch Range: {min_pitch:.1f} - {max_pitch:.1f} Hz")
        print(f"   Pitch Variation: {std_pitch:.1f} Hz")
        
        if std_pitch < 20:
            print("   ⚠️  WARNING: Very little pitch variation!")
            print("      This might indicate:")
            print("      - Only one speaker")
            print("      - Very similar voices")
            print("      - Poor audio quality")
        
        # Estimate number of speakers from pitch distribution
        from sklearn.cluster import KMeans
        if len(pitch_values) > 100:
            pitch_array = np.array(pitch_values).reshape(-1, 1)
            
            # Try different numbers of clusters
            for n_clusters in [2, 3, 4]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(pitch_array)
                
                # Calculate cluster sizes
                unique, counts = np.unique(labels, return_counts=True)
                
                # Check if clusters are balanced (no cluster < 10% of data)
                min_cluster_size = np.min(counts)
                if min_cluster_size > len(pitch_values) * 0.1:
                    print(f"\n   💡 Pitch analysis suggests {n_clusters} distinct speakers")
                    for i, (cluster_id, count) in enumerate(zip(unique, counts)):
                        cluster_pitches = pitch_array[labels == cluster_id]
                        mean_cluster_pitch = np.mean(cluster_pitches)
                        print(f"      Speaker {i+1}: ~{mean_cluster_pitch:.0f} Hz ({count/len(pitch_values)*100:.1f}%)")
                    break
    
    # Spectral analysis
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(spectral_centroids)
    std_centroid = np.std(spectral_centroids)
    
    print(f"\n5. SPECTRAL CHARACTERISTICS")
    print(f"   Mean Spectral Centroid: {mean_centroid:.1f} Hz")
    print(f"   Spectral Variation: {std_centroid:.1f} Hz")
    
    # MFCC diversity
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_std = np.std(mfcc, axis=1)
    mean_mfcc_std = np.mean(mfcc_std)
    
    print(f"\n6. VOICE CHARACTERISTICS (MFCC)")
    print(f"   MFCC Variation: {mean_mfcc_std:.4f}")
    
    if mean_mfcc_std < 5:
        print("   ⚠️  WARNING: Low MFCC variation!")
        print("      This suggests very similar or monotonous speech.")
    
    # Overall assessment
    print(f"\n7. OVERALL ASSESSMENT")
    
    issues = []
    
    if duration < 20:
        issues.append("Audio too short (< 20 seconds)")
    if speaking_time < 10:
        issues.append("Not enough speech detected")
    if peak < 0.01:
        issues.append("Audio level too low")
    if std_pitch < 20:
        issues.append("Little pitch variation (might be 1 speaker or very similar voices)")
    if mean_mfcc_std < 5:
        issues.append("Low voice characteristic variation")
    
    if issues:
        print("   ⚠️  POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"      - {issue}")
        
        print("\n   RECOMMENDATIONS:")
        if duration < 20:
            print("      → Use longer audio (30+ seconds recommended)")
        if speaking_time < 10:
            print("      → Ensure audio contains clear speech")
        if peak < 0.01:
            print("      → Increase audio volume/gain")
        if std_pitch < 20 or mean_mfcc_std < 5:
            print("      → Verify audio has multiple different speakers")
            print("      → Check if voices are distinct enough")
    else:
        print("   ✓ Audio appears suitable for speaker separation!")
        print("   ✓ No major issues detected")
    
    # Visualization
    print(f"\n8. CREATING VISUALIZATION...")
    
    try:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0])
        axes[0].set_title('Waveform')
        axes[0].set_xlabel('')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('Spectrogram')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # Voice Activity
        time_frames = librosa.frames_to_time(np.arange(len(voice_activity)), sr=sr, hop_length=hop_length)
        axes[2].fill_between(time_frames, 0, voice_activity, alpha=0.5)
        axes[2].set_title('Voice Activity Detection')
        axes[2].set_ylabel('Activity')
        axes[2].set_xlabel('')
        
        # Pitch
        if len(pitch_values) > 0:
            pitch_times = librosa.frames_to_time(np.arange(len(pitches[0])), sr=sr)
            pitch_smooth = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch_smooth.append(pitches[index, t])
            
            axes[3].plot(pitch_times, pitch_smooth)
            axes[3].set_title('Pitch over Time')
            axes[3].set_ylabel('Frequency (Hz)')
            axes[3].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        output_path = audio_path.parent / f"{audio_path.stem}_diagnostic.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved: {output_path}")
        
        # Try to show
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass
        
    except Exception as e:
        print(f"   ⚠️  Could not create visualization: {e}")
    
    print("\n" + "=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnose audio for speaker separation"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file"
    )
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    
    if not audio_path.exists():
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)
    
    diagnose_audio(audio_path)


if __name__ == "__main__":
    main()
