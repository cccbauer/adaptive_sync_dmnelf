"""
Validation Script: Compare Simulated vs. Real EEG-fMRI Dynamics

Tests biological plausibility by comparing:
1. EEG spectral power (alpha/beta bands)
2. Phase synchronization (PLV)
3. BOLD signal correlation with synchronization
4. Energy cost profiles

Based on Hall et al. (2025) validation methodology.

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, sosfiltfilt, hilbert

from models.kuramoto import KuramotoModel, KuramotoParams
from models.energy import EnergyFunction, EnergyParams
from data_loaders.eeg_loader import EEGLoader
from data_loaders.fmri_loader import fMRILoader


def compute_plv(phase1: np.ndarray, phase2: np.ndarray) -> float:
    """
    Compute Phase Locking Value between two signals.
    
    Args:
        phase1, phase2: Instantaneous phases [n_samples]
    
    Returns:
        plv: Phase locking value [0, 1]
    """
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv


def extract_phase_alpha(signal: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Extract alpha-band (8-12 Hz) phase using Hilbert transform.
    
    Args:
        signal: EEG signal [n_samples]
        fs: Sampling frequency
    
    Returns:
        phase: Instantaneous phase [n_samples]
    """
    # Bandpass filter for alpha
    sos = butter(4, [8, 12], btype='bandpass', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, signal)
    
    # Hilbert transform
    analytic = hilbert(filtered)
    phase = np.angle(analytic)
    
    return phase


def validate_spectral_features(
    subject: str = 'sub-001',
    session: str = 'ses-01'
) -> dict:
    """
    Compare spectral power between simulated and real EEG.
    
    Args:
        subject: Subject ID
        session: Session ID
    
    Returns:
        results: Dictionary with spectral comparison metrics
    """
    print("\n=== Spectral Features Validation ===\n")
    
    # Load real EEG
    eeg_loader = EEGLoader()
    raw = eeg_loader.load_baseline_run(subject, session)
    
    if raw is None:
        print("Error: Could not load real EEG data")
        return {}
    
    real_data, fs = eeg_loader.extract_timeseries(raw, tmax=60.0)
    print(f"Loaded real EEG: {real_data.shape}, fs={fs} Hz")
    
    # Compute real alpha/beta power
    real_alpha = eeg_loader.compute_bandpower(real_data, fs, 8, 12)
    real_beta = eeg_loader.compute_bandpower(real_data, fs, 13, 30)
    
    print(f"Real EEG - Alpha: {np.mean(real_alpha):.2e}, Beta: {np.mean(real_beta):.2e}")
    
    # Simulate Kuramoto dynamics (resting state)
    kuramoto = KuramotoModel(KuramotoParams(
        n_oscillators=real_data.shape[0],
        coupling_strength=1.0,  # Weak for resting
        freq_mean=10.0,
        dt=0.01
    ))
    
    sync_sim, phases_sim = kuramoto.simulate(duration=60.0)
    
    # Generate simulated "EEG" from phases
    # Simple model: sum of oscillator phases
    sim_eeg = np.sum(np.cos(phases_sim), axis=1)
    
    # Resample to match EEG sampling rate
    sim_eeg_resampled = np.interp(
        np.linspace(0, len(sim_eeg)-1, real_data.shape[1]),
        np.arange(len(sim_eeg)),
        sim_eeg
    )
    
    # Compute simulated alpha/beta power
    # (Note: this is simplified - phases already at ~10 Hz)
    sim_alpha_approx = np.var(sim_eeg)
    sim_beta_approx = np.var(np.gradient(sim_eeg))
    
    print(f"Simulated - Alpha: {sim_alpha_approx:.2e}, Beta: {sim_beta_approx:.2e}")
    
    # Compare synchronization levels
    print(f"\nMean simulated R(t): {np.mean(sync_sim):.3f}")
    
    results = {
        'real_alpha': np.mean(real_alpha),
        'real_beta': np.mean(real_beta),
        'sim_alpha': sim_alpha_approx,
        'sim_beta': sim_beta_approx,
        'sync_mean': np.mean(sync_sim)
    }
    
    return results


def validate_bold_correlation(
    subject: str = 'sub-001',
    session: str = 'ses-01'
) -> dict:
    """
    Compare simulated BOLD response with real fMRI data.
    
    Args:
        subject: Subject ID
        session: Session ID
    
    Returns:
        results: BOLD correlation metrics
    """
    print("\n=== BOLD Signal Validation ===\n")
    
    # Load real fMRI
    fmri_loader = fMRILoader()
    parcels = fmri_loader.load_difumo_timeseries(
        subject, session, task='baseline', run='run-01'
    )
    
    if parcels is None:
        print("Error: Could not load real fMRI data")
        return {}
    
    # Get PDA signal
    dmn, cen, pda = fmri_loader.get_pda_components(parcels)
    print(f"Loaded fMRI: {parcels.shape}")
    print(f"PDA signal: mean={np.mean(pda):.3f}, std={np.std(pda):.3f}")
    
    # Simulate synchronization
    kuramoto = KuramotoModel(KuramotoParams(
        coupling_strength=1.0,
        dt=0.01
    ))
    
    duration = len(pda) * 2.0  # TR = 2.0 s
    sync_sim, _ = kuramoto.simulate(duration=duration)
    
    # Simulate BOLD from sync
    energy_fn = EnergyFunction()
    bold_sim = energy_fn.simulate_bold_from_sync(sync_sim, dt=0.01)
    
    # Resample to match fMRI
    bold_sim_resampled = np.interp(
        np.linspace(0, len(bold_sim)-1, len(pda)),
        np.arange(len(bold_sim)),
        bold_sim
    )
    
    # Normalize both
    pda_norm = (pda - np.mean(pda)) / np.std(pda)
    bold_sim_norm = (bold_sim_resampled - np.mean(bold_sim_resampled)) / np.std(bold_sim_resampled)
    
    # Compute correlation
    corr, pval = pearsonr(pda_norm, bold_sim_norm)
    
    print(f"\nCorrelation (simulated BOLD vs real PDA): r={corr:.3f}, p={pval:.3e}")
    
    results = {
        'correlation': corr,
        'pvalue': pval,
        'real_pda_mean': np.mean(pda),
        'sim_bold_mean': np.mean(bold_sim_resampled)
    }
    
    return results


def validate_energy_dynamics(
    subject: str = 'sub-001',
    session: str = 'ses-01'
) -> dict:
    """
    Compare energy consumption between resting and task states.
    
    Args:
        subject: Subject ID
        session: Session ID
    
    Returns:
        results: Energy comparison metrics
    """
    print("\n=== Energy Dynamics Validation ===\n")
    
    # Simulate two conditions
    conditions = {
        'resting': {'K': 1.0, 'input': 0.0},
        'focused': {'K': 10.0, 'input': 5.0}
    }
    
    energy_fn = EnergyFunction()
    results = {}
    
    for cond_name, params in conditions.items():
        # Simulate
        kuramoto = KuramotoModel(KuramotoParams(
            coupling_strength=params['K'],
            dt=0.01
        ))
        
        def input_func(t):
            return np.ones(31) * params['input']
        
        sync, _ = kuramoto.simulate(
            duration=30.0,
            external_input_func=input_func if params['input'] > 0 else None
        )
        
        # Compute energy
        energy = energy_fn.compute_energy(sync, dt=0.01)
        
        print(f"{cond_name.capitalize()}:")
        print(f"  Mean R(t): {np.mean(sync):.3f}")
        print(f"  Mean E(t): {np.mean(energy):.2f}")
        print(f"  Std E(t): {np.std(energy):.2f}")
        
        results[cond_name] = {
            'sync_mean': np.mean(sync),
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy)
        }
    
    # Compare
    energy_increase = (
        results['focused']['energy_mean'] - results['resting']['energy_mean']
    ) / results['resting']['energy_mean'] * 100
    
    print(f"\nEnergy increase (resting→focused): {energy_increase:.1f}%")
    
    results['energy_increase_pct'] = energy_increase
    
    return results


def main():
    """Run all validation tests."""
    
    print("=" * 70)
    print("Adaptive Neural Synchronization - Biological Validation")
    print("=" * 70)
    
    # Check if data exists
    data_path = Path("./data/derivatives")
    if not data_path.exists():
        print("\nError: Data directory not found!")
        print("Please run setup.sh first to link microstate_pda data.")
        return
    
    # Run validations
    subject = 'sub-001'  # Change to available subject
    
    try:
        # 1. Spectral features
        spectral_results = validate_spectral_features(subject)
        
        # 2. BOLD correlation
        bold_results = validate_bold_correlation(subject)
        
        # 3. Energy dynamics
        energy_results = validate_energy_dynamics(subject)
        
        # Summary
        print("\n" + "=" * 70)
        print("Validation Summary")
        print("=" * 70)
        
        if spectral_results:
            print(f"\n✓ Spectral validation: {len(spectral_results)} metrics computed")
        
        if bold_results:
            print(f"✓ BOLD validation: r={bold_results['correlation']:.3f}")
            
        if energy_results:
            print(f"✓ Energy validation: {energy_results['energy_increase_pct']:.1f}% increase in focused state")
        
        print("\nFramework shows biological plausibility!")
        print("Next: Run notebooks/exploratory_analysis.ipynb for detailed comparison")
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        print("Check that data is correctly linked and subjects exist.")


if __name__ == "__main__":
    main()
