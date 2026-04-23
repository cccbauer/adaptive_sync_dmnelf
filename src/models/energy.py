"""
Energy Function for Metabolic Cost Modeling

Computes energy consumption E(t) as function of:
- Neural synchronization R(t)
- Transition cost dR/dt
- EEG spectral power
- fMRI BOLD signal

Based on Hall et al. (2025) Equation 3.

Author: Clemens Bauer
Date: April 2026
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnergyParams:
    """Parameters for energy function weighting"""
    alpha: float = 10.01  # Synchronization baseline weight
    beta: float = 5.00    # Transition cost weight
    gamma: float = 3.00   # EEG power weight
    delta: float = 2.00   # fMRI BOLD weight
    
    # HRF parameters for simulated BOLD
    hrf_peak_time: float = 5.0  # seconds
    hrf_undershoot_time: float = 15.0
    hrf_peak_amp: float = 1.0
    hrf_undershoot_amp: float = 0.35


class EnergyFunction:
    """
    Metabolic energy cost function for neural dynamics.
    
    Integrates synchronization state, transition dynamics, and
    multimodal neurophysiology (EEG power, fMRI BOLD).
    """
    
    def __init__(self, params: Optional[EnergyParams] = None):
        """
        Initialize energy function.
        
        Args:
            params: Energy function parameters
        """
        self.params = params if params is not None else EnergyParams()
        self.hrf = self._create_canonical_hrf()
        
    def _create_canonical_hrf(self) -> np.ndarray:
        """
        Create canonical hemodynamic response function.
        
        Double-gamma function modeling BOLD response to neural activity.
        
        Returns:
            hrf: HRF sampled at 100 Hz for 32 seconds
        """
        dt = 0.01  # 100 Hz sampling
        t = np.arange(0, 32, dt)
        
        # Parameters
        t1 = self.params.hrf_peak_time
        t2 = self.params.hrf_undershoot_time
        a1 = self.params.hrf_peak_amp
        a2 = self.params.hrf_undershoot_amp
        
        # Double-gamma function
        gamma1 = (t ** (t1-1)) * np.exp(-t / 1.0) / (np.math.factorial(int(t1)-1))
        gamma2 = (t ** (t2-1)) * np.exp(-t / 1.0) / (np.math.factorial(int(t2)-1))
        
        hrf = a1 * gamma1 - a2 * gamma2
        hrf = hrf / np.max(hrf)  # Normalize
        
        return hrf
    
    def compute_eeg_power(
        self,
        eeg_signal: np.ndarray,
        fs: float = 250.0,
        freq_band: Tuple[float, float] = (8.0, 30.0)
    ) -> float:
        """
        Compute band-limited EEG power.
        
        Args:
            eeg_signal: EEG time series [n_samples] or [n_channels, n_samples]
            fs: Sampling frequency in Hz
            freq_band: (low, high) frequency range in Hz
        
        Returns:
            power: Mean band power in arbitrary units
        """
        if eeg_signal.ndim == 1:
            eeg_signal = eeg_signal[np.newaxis, :]
        
        # Bandpass filter
        sos = signal.butter(4, freq_band, btype='bandpass', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos, eeg_signal, axis=1)
        
        # Compute power (mean squared amplitude)
        power = np.mean(filtered ** 2)
        
        return power
    
    def simulate_bold_from_sync(
        self,
        sync_trajectory: np.ndarray,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        Generate simulated BOLD signal by convolving sync with HRF.
        
        Args:
            sync_trajectory: R(t) synchronization time series [n_steps]
            dt: Time step in seconds
        
        Returns:
            bold_signal: Simulated BOLD response [n_steps]
        """
        # Resample HRF to match sync sampling rate
        hrf_resampled = signal.resample(
            self.hrf,
            int(len(self.hrf) * 0.01 / dt)
        )
        
        # Convolve
        bold_signal = np.convolve(sync_trajectory, hrf_resampled, mode='same')
        
        # Normalize to [0, 1] range
        bold_signal = (bold_signal - np.min(bold_signal)) / (
            np.max(bold_signal) - np.min(bold_signal) + 1e-8
        )
        
        return bold_signal
    
    def compute_energy(
        self,
        sync_trajectory: np.ndarray,
        eeg_power: Optional[np.ndarray] = None,
        bold_signal: Optional[np.ndarray] = None,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        Compute energy consumption E(t) over time.
        
        E(t) = α·R(t) + β·dR/dt + γ·P_EEG(t) + δ·S_fMRI(t)
        
        Args:
            sync_trajectory: R(t) synchronization level [n_steps]
            eeg_power: EEG band power time series [n_steps]
                      If None, uses simulated power
            bold_signal: fMRI BOLD signal [n_steps]
                        If None, simulates from sync using HRF
            dt: Time step in seconds
        
        Returns:
            energy: Energy consumption trajectory [n_steps]
        """
        n_steps = len(sync_trajectory)
        
        # Compute dR/dt using finite differences
        # Handle single-element case
        if n_steps == 1:
            dR_dt = np.array([0.0])
        else:
            dR_dt = np.gradient(sync_trajectory, dt)
        
        # Use simulated EEG power if not provided
        if eeg_power is None:
            # Simple model: power proportional to synchronization
            eeg_power = 0.4 + 0.4 * sync_trajectory + 0.1 * np.random.randn(n_steps)
            eeg_power = np.clip(eeg_power, 0, 1)
        
        # Use simulated BOLD if not provided
        if bold_signal is None:
            bold_signal = self.simulate_bold_from_sync(sync_trajectory, dt)
        
        # Ensure all arrays same length
        min_len = min(len(sync_trajectory), len(eeg_power), len(bold_signal))
        R = sync_trajectory[:min_len]
        dR = dR_dt[:min_len]
        P_eeg = eeg_power[:min_len]
        S_fmri = bold_signal[:min_len]
        
        # Compute energy components
        energy = (
            self.params.alpha * R +
            self.params.beta * np.abs(dR) +
            self.params.gamma * P_eeg +
            self.params.delta * S_fmri
        )
        
        return energy
    
    def compute_energy_from_real_data(
        self,
        sync_trajectory: np.ndarray,
        eeg_data: np.ndarray,
        fmri_data: np.ndarray,
        fs_eeg: float = 250.0,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute energy using real EEG and fMRI measurements.
        
        Args:
            sync_trajectory: Simulated R(t) [n_steps]
            eeg_data: Real EEG time series [n_channels, n_samples]
            fmri_data: Real fMRI BOLD signal [n_volumes]
            fs_eeg: EEG sampling rate in Hz
            dt: Sync trajectory time step
        
        Returns:
            energy: Energy trajectory [n_steps]
            components: Dictionary with individual energy terms
        """
        # Extract EEG power in sliding windows
        window_size = int(fs_eeg * dt)
        n_windows = len(sync_trajectory)
        
        eeg_power_trajectory = np.zeros(n_windows)
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, eeg_data.shape[1])
            
            if end_idx - start_idx > 0:
                window = eeg_data[:, start_idx:end_idx]
                eeg_power_trajectory[i] = self.compute_eeg_power(window, fs_eeg)
        
        # Interpolate fMRI to match sync trajectory resolution
        fmri_interpolated = np.interp(
            np.arange(len(sync_trajectory)),
            np.linspace(0, len(sync_trajectory), len(fmri_data)),
            fmri_data
        )
        
        # Normalize fMRI to [0, 1]
        fmri_norm = (fmri_interpolated - np.min(fmri_interpolated)) / (
            np.max(fmri_interpolated) - np.min(fmri_interpolated) + 1e-8
        )
        
        # Compute total energy
        energy = self.compute_energy(
            sync_trajectory,
            eeg_power=eeg_power_trajectory,
            bold_signal=fmri_norm,
            dt=dt
        )
        
        # Store components for analysis
        dR_dt = np.gradient(sync_trajectory, dt)
        components = {
            'sync_baseline': self.params.alpha * sync_trajectory,
            'transition_cost': self.params.beta * np.abs(dR_dt),
            'eeg_power': self.params.gamma * eeg_power_trajectory,
            'bold_signal': self.params.delta * fmri_norm,
            'total': energy
        }
        
        return energy, components


# Example usage
if __name__ == "__main__":
    # Create energy function
    energy_fn = EnergyFunction()
    
    # Simulate some synchronization trajectory
    t = np.arange(0, 10, 0.01)
    sync_rest = 0.3 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
    sync_focused = 0.8 + 0.05 * np.sin(2 * np.pi * 1.0 * t)
    
    # Compute energy for both states
    energy_rest = energy_fn.compute_energy(sync_rest, dt=0.01)
    energy_focused = energy_fn.compute_energy(sync_focused, dt=0.01)
    
    print(f"Resting state - Mean energy: {np.mean(energy_rest):.3f}")
    print(f"Focused state - Mean energy: {np.mean(energy_focused):.3f}")
    
    # Test BOLD simulation
    bold_sim = energy_fn.simulate_bold_from_sync(sync_focused, dt=0.01)
    print(f"Simulated BOLD range: [{np.min(bold_sim):.3f}, {np.max(bold_sim):.3f}]")
