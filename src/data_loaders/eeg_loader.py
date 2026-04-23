"""
EEG Data Loader for Preprocessed DMNELF Data

Loads preprocessed simultaneous EEG recordings from microstate_pda pipeline.
Handles BIDS-formatted data structure.

Author: Clemens Bauer
Date: April 2026
"""

import numpy as np
import mne
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd


class EEGLoader:
    """
    Load preprocessed DMNELF EEG data.
    
    Expects BIDS structure:
    data/derivatives/preprocessed/sub-XXX/ses-YY/eeg/
        sub-XXX_ses-YY_task-ZZZZ_run-N_eeg.fif
    """
    
    def __init__(self, data_root: str = "./data"):
        """
        Initialize EEG loader.
        
        Args:
            data_root: Path to data directory (contains derivatives/)
        """
        self.data_root = Path(data_root)
        self.deriv_path = self.data_root / "derivatives" / "preprocessed"
        
        if not self.deriv_path.exists():
            print(f"Warning: {self.deriv_path} not found")
            print("Expected structure: data/derivatives/preprocessed/sub-XXX/...")
    
    def list_subjects(self) -> List[str]:
        """
        List available subjects.
        
        Returns:
            subjects: List of subject IDs (e.g., ['sub-001', 'sub-002'])
        """
        if not self.deriv_path.exists():
            return []
        
        subjects = sorted([
            d.name for d in self.deriv_path.iterdir()
            if d.is_dir() and d.name.startswith('sub-')
        ])
        
        return subjects
    
    def list_runs(self, subject: str, session: str = 'ses-01') -> List[str]:
        """
        List available runs for a subject/session.
        
        Args:
            subject: Subject ID (e.g., 'sub-001')
            session: Session ID (e.g., 'ses-01')
        
        Returns:
            runs: List of run identifiers
        """
        eeg_dir = self.deriv_path / subject / session / "eeg"
        
        if not eeg_dir.exists():
            return []
        
        fif_files = list(eeg_dir.glob("*_eeg.fif"))
        
        runs = []
        for f in fif_files:
            # Extract run number from filename
            parts = f.stem.split('_')
            for part in parts:
                if part.startswith('run-'):
                    runs.append(part)
                    break
        
        return sorted(runs)
    
    def load_run(
        self,
        subject: str,
        session: str = 'ses-01',
        task: str = 'neurofeedback',
        run: str = 'run-01',
        preload: bool = True
    ) -> Optional[mne.io.Raw]:
        """
        Load a single EEG run.
        
        Args:
            subject: Subject ID (e.g., 'sub-001')
            session: Session ID
            task: Task name
            run: Run identifier
            preload: Load data into memory
        
        Returns:
            raw: MNE Raw object, or None if not found
        """
        eeg_file = (
            self.deriv_path / subject / session / "eeg" /
            f"{subject}_{session}_task-{task}_{run}_eeg.fif"
        )
        
        if not eeg_file.exists():
            print(f"File not found: {eeg_file}")
            return None
        
        raw = mne.io.read_raw_fif(eeg_file, preload=preload, verbose=False)
        
        return raw
    
    def extract_timeseries(
        self,
        raw: mne.io.Raw,
        tmin: float = 0.0,
        tmax: Optional[float] = None,
        picks: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Extract EEG time series as numpy array.
        
        Args:
            raw: MNE Raw object
            tmin: Start time in seconds
            tmax: End time in seconds (None = full duration)
            picks: Channel names to extract (None = all)
        
        Returns:
            data: EEG data [n_channels, n_samples]
            sfreq: Sampling frequency in Hz
        """
        if tmax is None:
            tmax = raw.times[-1]
        
        data, times = raw.get_data(
            picks=picks,
            tmin=tmin,
            tmax=tmax,
            return_times=True
        )
        
        return data, raw.info['sfreq']
    
    def get_channel_names(self, raw: mne.io.Raw) -> List[str]:
        """Get list of channel names."""
        return raw.ch_names
    
    def compute_bandpower(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float = 8.0,
        fmax: float = 30.0,
        window_sec: float = 1.0
    ) -> np.ndarray:
        """
        Compute band power in sliding windows.
        
        Args:
            data: EEG data [n_channels, n_samples]
            sfreq: Sampling frequency
            fmin, fmax: Frequency band limits
            window_sec: Window size in seconds
        
        Returns:
            power: Band power time series [n_windows]
        """
        from scipy.signal import butter, sosfiltfilt, welch
        
        # Bandpass filter
        sos = butter(4, [fmin, fmax], btype='bandpass', fs=sfreq, output='sos')
        filtered = sosfiltfilt(sos, data, axis=1)
        
        # Compute power in sliding windows
        window_samples = int(window_sec * sfreq)
        n_windows = data.shape[1] // window_samples
        
        power = np.zeros(n_windows)
        for i in range(n_windows):
            start = i * window_samples
            end = (i + 1) * window_samples
            window = filtered[:, start:end]
            
            # Mean squared amplitude across channels
            power[i] = np.mean(window ** 2)
        
        return power
    
    def load_baseline_run(
        self,
        subject: str,
        session: str = 'ses-01'
    ) -> Optional[mne.io.Raw]:
        """
        Load resting-state baseline run.
        
        Convenience method for accessing baseline data.
        
        Args:
            subject: Subject ID
            session: Session ID
        
        Returns:
            raw: MNE Raw object for baseline run
        """
        # Try common baseline naming conventions
        for task in ['baseline', 'rest', 'resting']:
            for run in ['run-01', 'run-1']:
                raw = self.load_run(subject, session, task, run)
                if raw is not None:
                    return raw
        
        print(f"No baseline run found for {subject} {session}")
        return None


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = EEGLoader(data_root="./data")
    
    # List available subjects
    subjects = loader.list_subjects()
    print(f"Found {len(subjects)} subjects: {subjects[:3]}...")
    
    if subjects:
        # Load first subject's baseline
        subject = subjects[0]
        raw = loader.load_baseline_run(subject)
        
        if raw is not None:
            print(f"\nLoaded {subject}:")
            print(f"  Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  Duration: {raw.times[-1]:.1f} s")
            print(f"  Channels: {len(raw.ch_names)}")
            
            # Extract time series
            data, sfreq = loader.extract_timeseries(raw, tmax=60.0)
            print(f"\nExtracted data shape: {data.shape}")
            
            # Compute band power
            power = loader.compute_bandpower(data, sfreq)
            print(f"Band power trajectory: {power.shape}, mean={np.mean(power):.2e}")
