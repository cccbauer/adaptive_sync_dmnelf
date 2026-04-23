"""
fMRI Data Loader for fMRIPrep-Processed DMNELF Data

Loads BOLD signals from fMRIPrep outputs, extracts ROI time series,
and handles DiFuMo-64 parcellation.

Author: Clemens Bauer
Date: April 2026
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd


class fMRILoader:
    """
    Load fMRIPrep-processed fMRI data for DMNELF.
    
    Handles:
    - MNI-space BOLD NIfTI files
    - Confound regressors
    - DiFuMo-64 parcel extraction
    """
    
    def __init__(self, data_root: str = "./data"):
        """
        Initialize fMRI loader.
        
        Args:
            data_root: Path to data directory
        """
        self.data_root = Path(data_root)
        self.fmriprep_path = self.data_root / "derivatives" / "fmriprep"
        
        # TR (repetition time) for DMNELF - adjust if different
        self.tr = 2.0  # seconds
        
        if not self.fmriprep_path.exists():
            print(f"Warning: {self.fmriprep_path} not found")
    
    def list_subjects(self) -> List[str]:
        """List available subjects in fMRIPrep outputs."""
        if not self.fmriprep_path.exists():
            return []
        
        subjects = sorted([
            d.name for d in self.fmriprep_path.iterdir()
            if d.is_dir() and d.name.startswith('sub-')
        ])
        
        return subjects
    
    def load_bold(
        self,
        subject: str,
        session: str = 'ses-01',
        task: str = 'neurofeedback',
        run: str = 'run-01',
        space: str = 'MNI152NLin2009cAsym'
    ) -> Optional[nib.Nifti1Image]:
        """
        Load preprocessed BOLD NIfTI file.
        
        Args:
            subject: Subject ID
            session: Session ID
            task: Task name
            run: Run identifier
            space: Template space
        
        Returns:
            img: NiBabel image object, or None if not found
        """
        func_dir = self.fmriprep_path / subject / session / "func"
        
        # fMRIPrep naming: sub-X_ses-Y_task-Z_run-N_space-MNI_desc-preproc_bold.nii.gz
        bold_file = (
            func_dir /
            f"{subject}_{session}_task-{task}_{run}_"
            f"space-{space}_desc-preproc_bold.nii.gz"
        )
        
        if not bold_file.exists():
            print(f"BOLD file not found: {bold_file}")
            return None
        
        img = nib.load(bold_file)
        
        return img
    
    def load_confounds(
        self,
        subject: str,
        session: str = 'ses-01',
        task: str = 'neurofeedback',
        run: str = 'run-01'
    ) -> Optional[pd.DataFrame]:
        """
        Load fMRIPrep confound regressors.
        
        Args:
            subject: Subject ID
            session: Session ID
            task: Task name
            run: Run identifier
        
        Returns:
            confounds: DataFrame with confound time series
        """
        func_dir = self.fmriprep_path / subject / session / "func"
        
        confound_file = (
            func_dir /
            f"{subject}_{session}_task-{task}_{run}_desc-confounds_timeseries.tsv"
        )
        
        if not confound_file.exists():
            print(f"Confounds file not found: {confound_file}")
            return None
        
        confounds = pd.read_csv(confound_file, sep='\t')
        
        return confounds
    
    def extract_global_signal(
        self,
        img: nib.Nifti1Image,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract global signal (mean across all voxels).
        
        Args:
            img: BOLD image
            mask: Brain mask (if None, uses all non-zero voxels)
        
        Returns:
            global_signal: Time series [n_volumes]
        """
        data = img.get_fdata()
        
        if mask is None:
            # Use all non-zero voxels
            mask = np.mean(data, axis=-1) > 0
        
        # Mean across masked voxels
        global_signal = np.mean(data[mask], axis=0)
        
        return global_signal
    
    def extract_roi_timeseries(
        self,
        img: nib.Nifti1Image,
        roi_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract mean BOLD signal from an ROI.
        
        Args:
            img: BOLD image [x, y, z, t]
            roi_mask: Binary mask [x, y, z]
        
        Returns:
            timeseries: Mean ROI signal [n_volumes]
        """
        data = img.get_fdata()
        
        # Mean across ROI voxels
        timeseries = np.mean(data[roi_mask > 0], axis=0)
        
        return timeseries
    
    def load_difumo_timeseries(
        self,
        subject: str,
        session: str = 'ses-01',
        task: str = 'neurofeedback',
        run: str = 'run-01',
        n_parcels: int = 64
    ) -> Optional[np.ndarray]:
        """
        Load pre-extracted DiFuMo parcel time series.
        
        Expects file created by microstate_pda preprocessing:
        derivatives/difumo/sub-XXX_ses-YY_task-ZZZZ_run-N_parcels-64.npy
        
        Args:
            subject: Subject ID
            session: Session ID
            task: Task name
            run: Run identifier
            n_parcels: Number of DiFuMo parcels
        
        Returns:
            timeseries: Parcel signals [n_parcels, n_volumes]
        """
        difumo_dir = self.data_root / "derivatives" / "difumo"
        
        ts_file = (
            difumo_dir /
            f"{subject}_{session}_task-{task}_{run}_parcels-{n_parcels}.npy"
        )
        
        if not ts_file.exists():
            print(f"DiFuMo time series not found: {ts_file}")
            return None
        
        timeseries = np.load(ts_file)
        
        return timeseries
    
    def get_pda_components(
        self,
        timeseries: np.ndarray,
        dmn_indices: Optional[List[int]] = None,
        cen_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PDA (Posterior DMN Activity) = CEN - DMN.
        
        Args:
            timeseries: Parcel signals [n_parcels, n_volumes]
            dmn_indices: Parcel indices for DMN (if None, uses defaults)
            cen_indices: Parcel indices for CEN (if None, uses defaults)
        
        Returns:
            dmn_signal: Mean DMN activity [n_volumes]
            cen_signal: Mean CEN activity [n_volumes]
            pda: PDA signal (CEN - DMN) [n_volumes]
        """
        # Default DiFuMo-64 indices (from microstate_pda verification)
        if dmn_indices is None:
            dmn_indices = [2, 3, 9, 13, 20, 25, 28, 32, 38, 41, 52, 55]
        
        if cen_indices is None:
            cen_indices = [0, 5, 11, 14, 16, 22, 29, 35, 39, 44, 47, 50, 57, 62]
        
        # Extract network signals
        dmn_signal = np.mean(timeseries[dmn_indices, :], axis=0)
        cen_signal = np.mean(timeseries[cen_indices, :], axis=0)
        
        # Compute PDA
        pda = cen_signal - dmn_signal
        
        return dmn_signal, cen_signal, pda
    
    def normalize_bold(
        self,
        signal: np.ndarray,
        method: str = 'zscore'
    ) -> np.ndarray:
        """
        Normalize BOLD signal.
        
        Args:
            signal: Raw BOLD time series
            method: 'zscore' or 'percent'
        
        Returns:
            normalized: Normalized signal
        """
        if method == 'zscore':
            # Z-score normalization
            normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        elif method == 'percent':
            # Percent signal change
            normalized = 100 * (signal - np.mean(signal)) / (np.mean(signal) + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def resample_to_eeg(
        self,
        bold_signal: np.ndarray,
        eeg_length: int,
        tr: Optional[float] = None
    ) -> np.ndarray:
        """
        Interpolate BOLD signal to match EEG sampling.
        
        Args:
            bold_signal: BOLD time series [n_volumes]
            eeg_length: Target length (EEG samples)
            tr: Repetition time in seconds (uses self.tr if None)
        
        Returns:
            resampled: Interpolated signal [eeg_length]
        """
        if tr is None:
            tr = self.tr
        
        # Time vectors
        bold_times = np.arange(len(bold_signal)) * tr
        eeg_times = np.linspace(0, bold_times[-1], eeg_length)
        
        # Linear interpolation
        resampled = np.interp(eeg_times, bold_times, bold_signal)
        
        return resampled


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = fMRILoader(data_root="./data")
    
    # List subjects
    subjects = loader.list_subjects()
    print(f"Found {len(subjects)} subjects with fMRI data")
    
    if subjects:
        subject = subjects[0]
        
        # Load BOLD image
        img = loader.load_bold(subject, task='baseline', run='run-01')
        
        if img is not None:
            print(f"\nLoaded {subject} BOLD:")
            print(f"  Shape: {img.shape}")
            print(f"  TR: {loader.tr} s")
            
            # Extract global signal
            global_sig = loader.extract_global_signal(img)
            print(f"  Global signal: {global_sig.shape}")
            
            # Load DiFuMo parcels
            parcels = loader.load_difumo_timeseries(
                subject, task='baseline', run='run-01'
            )
            
            if parcels is not None:
                print(f"\nDiFuMo parcels: {parcels.shape}")
                
                # Compute PDA
                dmn, cen, pda = loader.get_pda_components(parcels)
                print(f"  DMN: {dmn.shape}, mean={np.mean(dmn):.3f}")
                print(f"  CEN: {cen.shape}, mean={np.mean(cen):.3f}")
                print(f"  PDA: {pda.shape}, mean={np.mean(pda):.3f}")
