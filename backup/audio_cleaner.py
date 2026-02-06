#!/usr/bin/env python3
"""
Audio Cleaner and Enhancement Module
Cleans separated speaker audio by removing silence, noise, and distortion
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import librosa
    import soundfile as sf
    from scipy import signal
    from scipy.ndimage import median_filter
except ImportError:
    print("Installing required libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "numpy", "librosa", "soundfile", "scipy"])
    import numpy as np
    import librosa
    import soundfile as sf
    from scipy import signal
    from scipy.ndimage import median_filter

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("⚠️  noisereduce not installed (optional for better noise reduction)")
    print("   Install with: pip install noisereduce")


class AudioCleaner:
    """Clean and enhance separated speaker audio"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def remove_silence(self, y: np.ndarray, 
                       threshold_db: float = -40,
                       min_silence_duration: float = 0.3,
                       padding: float = 0.1) -> Tuple[np.ndarray, dict]:
        """
        Remove silence from audio
        
        Args:
            y: Audio signal
            threshold_db: Silence threshold in dB
            min_silence_duration: Minimum silence duration to remove (seconds)
            padding: Padding to keep around speech (seconds)
        
        Returns:
            Cleaned audio and statistics
        """
        print("  Removing silence...")
        
        # Calculate frame-level RMS energy
        frame_length = 2048
        hop_length = 512
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Detect voice activity
        voice_activity = rms_db > threshold_db
        
        # Smooth the detection (remove very short segments)
        min_silence_frames = int(min_silence_duration * self.sr / hop_length)
        voice_activity = median_filter(voice_activity.astype(float), size=min_silence_frames) > 0.5
        
        # Add padding around speech segments
        padding_frames = int(padding * self.sr / hop_length)
        
        # Expand voice activity with padding
        padded_activity = np.copy(voice_activity)
        for i in range(len(voice_activity)):
            if voice_activity[i]:
                start = max(0, i - padding_frames)
                end = min(len(voice_activity), i + padding_frames + 1)
                padded_activity[start:end] = True
        
        # Convert frame-level mask to sample-level
        sample_mask = np.repeat(padded_activity, hop_length)
        sample_mask = sample_mask[:len(y)]
        
        # Apply mask
        y_cleaned = y[sample_mask]
        
        # Statistics
        original_duration = len(y) / self.sr
        cleaned_duration = len(y_cleaned) / self.sr
        removed_duration = original_duration - cleaned_duration
        
        stats = {
            'original_duration': original_duration,
            'cleaned_duration': cleaned_duration,
            'removed_duration': removed_duration,
            'removed_percentage': (removed_duration / original_duration) * 100
        }
        
        print(f"    Original: {original_duration:.2f}s")
        print(f"    Cleaned: {cleaned_duration:.2f}s")
        print(f"    Removed: {removed_duration:.2f}s ({stats['removed_percentage']:.1f}%)")
        
        return y_cleaned, stats
    
    def reduce_noise(self, y: np.ndarray, 
                    method: str = 'spectral',
                    noise_profile_duration: float = 0.5) -> np.ndarray:
        """
        Reduce background noise
        
        Args:
            y: Audio signal
            method: 'spectral' or 'noisereduce'
            noise_profile_duration: Duration of noise profile (seconds)
        
        Returns:
            Denoised audio
        """
        print(f"  Reducing noise (method: {method})...")
        
        if method == 'noisereduce' and NOISEREDUCE_AVAILABLE:
            # Use noisereduce library (better quality)
            y_denoised = nr.reduce_noise(
                y=y,
                sr=self.sr,
                stationary=True,
                prop_decrease=0.8
            )
            
        else:
            # Spectral subtraction method
            # Get noise profile from quietest part
            frame_length = 2048
            hop_length = 512
            
            # Calculate RMS to find quietest segment
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Get noise profile from quietest frames
            noise_frames = int(noise_profile_duration * self.sr / hop_length)
            quietest_idx = np.argpartition(rms, noise_frames)[:noise_frames]
            
            # Compute STFT
            D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
            magnitude, phase = librosa.magphase(D)
            
            # Estimate noise spectrum
            noise_magnitude = np.mean(magnitude[:, quietest_idx], axis=1, keepdims=True)
            
            # Spectral subtraction with over-subtraction factor
            alpha = 2.0  # Over-subtraction factor
            beta = 0.02  # Spectral floor
            
            clean_magnitude = magnitude - alpha * noise_magnitude
            clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
            
            # Reconstruct signal
            D_clean = clean_magnitude * phase
            y_denoised = librosa.istft(D_clean, hop_length=hop_length)
            
            # Match length
            if len(y_denoised) < len(y):
                y_denoised = np.pad(y_denoised, (0, len(y) - len(y_denoised)))
            else:
                y_denoised = y_denoised[:len(y)]
        
        # Calculate noise reduction
        noise_before = np.std(y - signal.medfilt(y, kernel_size=5))
        noise_after = np.std(y_denoised - signal.medfilt(y_denoised, kernel_size=5))
        reduction_db = 20 * np.log10(noise_before / (noise_after + 1e-10))
        
        print(f"    Noise reduced by ~{reduction_db:.1f} dB")
        
        return y_denoised
    
    def remove_clicks_and_pops(self, y: np.ndarray, 
                               threshold: float = 3.0) -> np.ndarray:
        """
        Remove clicks and pops (impulse noise)
        
        Args:
            y: Audio signal
            threshold: Threshold in standard deviations
        
        Returns:
            Cleaned audio
        """
        print("  Removing clicks and pops...")
        
        # Detect outliers using median filter
        window_size = 5
        median_filtered = signal.medfilt(y, kernel_size=window_size)
        
        # Calculate absolute difference
        diff = np.abs(y - median_filtered)
        
        # Detect clicks (outliers)
        std = np.std(diff)
        clicks = diff > (threshold * std)
        
        # Replace clicks with median filtered values
        y_cleaned = np.copy(y)
        y_cleaned[clicks] = median_filtered[clicks]
        
        num_clicks = np.sum(clicks)
        print(f"    Removed {num_clicks} clicks/pops")
        
        return y_cleaned
    
    def reduce_distortion(self, y: np.ndarray, 
                         threshold: float = 0.95) -> np.ndarray:
        """
        Reduce clipping distortion
        
        Args:
            y: Audio signal
            threshold: Clipping threshold (0-1)
        
        Returns:
            Audio with reduced distortion
        """
        print("  Reducing distortion...")
        
        # Detect clipped samples
        clipped = np.abs(y) > threshold
        num_clipped = np.sum(clipped)
        
        if num_clipped > 0:
            print(f"    Found {num_clipped} clipped samples")
            
            # Simple declipping: interpolate clipped regions
            clipped_indices = np.where(clipped)[0]
            
            for idx in clipped_indices:
                # Find boundaries of clipped region
                start = idx
                while start > 0 and clipped[start - 1]:
                    start -= 1
                
                end = idx
                while end < len(y) - 1 and clipped[end + 1]:
                    end += 1
                
                # Interpolate if region is small
                if end - start < 100:  # Don't interpolate long regions
                    if start > 0 and end < len(y) - 1:
                        # Linear interpolation
                        y[start:end+1] = np.linspace(y[start-1], y[end+1], end-start+3)[1:-1]
        else:
            print("    No clipping detected")
        
        return y
    
    def apply_highpass_filter(self, y: np.ndarray, 
                             cutoff: float = 80) -> np.ndarray:
        """
        Apply highpass filter to remove low-frequency rumble
        
        Args:
            y: Audio signal
            cutoff: Cutoff frequency in Hz
        
        Returns:
            Filtered audio
        """
        print(f"  Applying highpass filter (cutoff: {cutoff} Hz)...")
        
        # Design highpass filter
        nyquist = self.sr / 2
        normalized_cutoff = cutoff / nyquist
        
        # Use a high-order Butterworth filter
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        
        # Apply filter (use filtfilt for zero-phase)
        y_filtered = signal.filtfilt(b, a, y)
        
        return y_filtered
    
    def apply_lowpass_filter(self, y: np.ndarray, 
                            cutoff: float = 8000) -> np.ndarray:
        """
        Apply lowpass filter to remove high-frequency hiss
        
        Args:
            y: Audio signal
            cutoff: Cutoff frequency in Hz
        
        Returns:
            Filtered audio
        """
        print(f"  Applying lowpass filter (cutoff: {cutoff} Hz)...")
        
        # Design lowpass filter
        nyquist = self.sr / 2
        normalized_cutoff = cutoff / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            print("    Cutoff too high, skipping")
            return y
        
        # Use a high-order Butterworth filter
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        
        # Apply filter
        y_filtered = signal.filtfilt(b, a, y)
        
        return y_filtered
    
    def normalize_audio(self, y: np.ndarray, 
                       target_level: float = -3.0,
                       method: str = 'peak') -> np.ndarray:
        """
        Normalize audio level
        
        Args:
            y: Audio signal
            target_level: Target level in dB
            method: 'peak' or 'rms'
        
        Returns:
            Normalized audio
        """
        print(f"  Normalizing audio (method: {method}, target: {target_level} dB)...")
        
        if method == 'peak':
            # Peak normalization
            peak = np.max(np.abs(y))
            if peak > 0:
                target_peak = 10 ** (target_level / 20)
                y_normalized = y * (target_peak / peak)
            else:
                y_normalized = y
                
        else:  # rms
            # RMS normalization
            rms = np.sqrt(np.mean(y ** 2))
            if rms > 0:
                target_rms = 10 ** (target_level / 20)
                y_normalized = y * (target_rms / rms)
                
                # Prevent clipping
                peak = np.max(np.abs(y_normalized))
                if peak > 1.0:
                    y_normalized = y_normalized / peak * 0.99
            else:
                y_normalized = y
        
        return y_normalized
    
    def apply_fade(self, y: np.ndarray, 
                   fade_in: float = 0.05,
                   fade_out: float = 0.05) -> np.ndarray:
        """
        Apply fade in/out to prevent clicks at start/end
        
        Args:
            y: Audio signal
            fade_in: Fade in duration (seconds)
            fade_out: Fade out duration (seconds)
        
        Returns:
            Audio with fades
        """
        fade_in_samples = int(fade_in * self.sr)
        fade_out_samples = int(fade_out * self.sr)
        
        # Create fade curves
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        
        # Apply fades
        y_faded = np.copy(y)
        y_faded[:fade_in_samples] *= fade_in_curve
        y_faded[-fade_out_samples:] *= fade_out_curve
        
        return y_faded
    
    def clean_audio(self, audio_path: Path,
                   output_path: Path,
                   remove_silence: bool = True,
                   reduce_noise: bool = True,
                   remove_clicks: bool = True,
                   reduce_distortion: bool = True,
                   apply_filters: bool = True,
                   normalize: bool = True) -> dict:
        """
        Apply all cleaning steps to an audio file
        
        Args:
            audio_path: Input audio file
            output_path: Output audio file
            remove_silence: Remove silence segments
            reduce_noise: Apply noise reduction
            remove_clicks: Remove clicks and pops
            reduce_distortion: Reduce clipping distortion
            apply_filters: Apply highpass/lowpass filters
            normalize: Normalize audio level
        
        Returns:
            Statistics dictionary
        """
        print(f"\nCleaning: {audio_path.name}")
        print("=" * 60)
        
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=self.sr)
        original_duration = len(y) / sr
        
        stats = {
            'original_duration': original_duration,
            'steps_applied': []
        }
        
        # 1. Remove clicks and pops first (before other processing)
        if remove_clicks:
            y = self.remove_clicks_and_pops(y)
            stats['steps_applied'].append('clicks_removed')
        
        # 2. Reduce distortion
        if reduce_distortion:
            y = self.reduce_distortion(y)
            stats['steps_applied'].append('distortion_reduced')
        
        # 3. Apply filters to remove unwanted frequencies
        if apply_filters:
            y = self.apply_highpass_filter(y, cutoff=80)  # Remove rumble
            y = self.apply_lowpass_filter(y, cutoff=8000)  # Remove hiss
            stats['steps_applied'].append('filters_applied')
        
        # 4. Reduce background noise
        if reduce_noise:
            y = self.reduce_noise(y, method='spectral')
            stats['steps_applied'].append('noise_reduced')
        
        # 5. Remove silence
        if remove_silence:
            y, silence_stats = self.remove_silence(y)
            stats.update(silence_stats)
            stats['steps_applied'].append('silence_removed')
        
        # 6. Normalize audio
        if normalize:
            y = self.normalize_audio(y, target_level=-3.0, method='peak')
            stats['steps_applied'].append('normalized')
        
        # 7. Apply fades to prevent clicks
        y = self.apply_fade(y, fade_in=0.01, fade_out=0.05)
        stats['steps_applied'].append('fades_applied')
        
        # Save cleaned audio
        sf.write(output_path, y, self.sr)
        
        stats['output_duration'] = len(y) / sr
        stats['output_file'] = str(output_path)
        
        print(f"\n✓ Cleaned audio saved: {output_path.name}")
        print(f"  Duration: {stats['output_duration']:.2f}s")
        
        return stats


def clean_separated_speakers(input_dir: Path,
                             output_dir: Path = None,
                             **cleaning_options) -> dict:
    """
    Clean all separated speaker files in a directory
    
    Args:
        input_dir: Directory containing separated speaker files
        output_dir: Output directory (default: input_dir/cleaned)
        **cleaning_options: Options to pass to AudioCleaner.clean_audio()
    
    Returns:
        Dictionary with cleaning statistics for each file
    """
    if output_dir is None:
        output_dir = input_dir / "cleaned"
    
    output_dir.mkdir(exist_ok=True)
    
    # Find all speaker audio files
    speaker_files = list(input_dir.glob("speaker_*.wav"))
    
    if not speaker_files:
        print(f"No speaker files found in {input_dir}")
        return {}
    
    print("\n" + "=" * 60)
    print(f"Cleaning {len(speaker_files)} speaker files")
    print("=" * 60)
    
    cleaner = AudioCleaner()
    results = {}
    
    for audio_file in sorted(speaker_files):
        output_file = output_dir / f"{audio_file.stem}_cleaned.wav"
        
        stats = cleaner.clean_audio(
            audio_file,
            output_file,
            **cleaning_options
        )
        
        results[audio_file.stem] = stats
    
    print("\n" + "=" * 60)
    print("✓ All files cleaned")
    print("=" * 60)
    
    # Print summary
    total_original = sum(s['original_duration'] for s in results.values())
    total_cleaned = sum(s['output_duration'] for s in results.values())
    total_removed = total_original - total_cleaned
    
    print(f"\nSummary:")
    print(f"  Total original duration: {total_original:.2f}s")
    print(f"  Total cleaned duration: {total_cleaned:.2f}s")
    print(f"  Total removed: {total_removed:.2f}s ({total_removed/total_original*100:.1f}%)")
    print(f"\nCleaned files saved in: {output_dir}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean and enhance separated speaker audio files"
    )
    parser.add_argument(
        "input",
        help="Input audio file or directory with speaker files"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file or directory"
    )
    parser.add_argument(
        "--no-silence-removal",
        action="store_true",
        help="Don't remove silence"
    )
    parser.add_argument(
        "--no-noise-reduction",
        action="store_true",
        help="Don't reduce noise"
    )
    parser.add_argument(
        "--no-click-removal",
        action="store_true",
        help="Don't remove clicks"
    )
    parser.add_argument(
        "--no-distortion-reduction",
        action="store_true",
        help="Don't reduce distortion"
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Don't apply highpass/lowpass filters"
    )
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Don't normalize audio"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    # Prepare cleaning options
    cleaning_options = {
        'remove_silence': not args.no_silence_removal,
        'reduce_noise': not args.no_noise_reduction,
        'remove_clicks': not args.no_click_removal,
        'reduce_distortion': not args.no_distortion_reduction,
        'apply_filters': not args.no_filters,
        'normalize': not args.no_normalization
    }
    
    if input_path.is_dir():
        # Clean all speaker files in directory
        output_dir = Path(args.output) if args.output else None
        clean_separated_speakers(input_path, output_dir, **cleaning_options)
        
    else:
        # Clean single file
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_cleaned.wav"
        
        cleaner = AudioCleaner()
        cleaner.clean_audio(input_path, output_path, **cleaning_options)


if __name__ == "__main__":
    main()
