"""
Configuration for Adaptive Neural Synchronization Pipeline

Three cognitive states matching Hall et al. (2025) mapped to DMNELF tasks.
Updated targets based on real DMNELF validation data.

Author: Clemens Bauer
Date: April 2026
"""

from pathlib import Path
import json

# ============================================================================
# LOCAL PATHS (your Mac)
# ============================================================================
LOCAL_BASE = Path(__file__).parent
LOCAL_RESULTS = LOCAL_BASE / "results"
LOCAL_SCRIPTS = LOCAL_BASE / "deploy_scripts"

# ============================================================================
# CLUSTER PATHS (Explorer)
# ============================================================================
CLUSTER_USER = "cccbauer"
CLUSTER_HOST = "explorer.northeastern.edu"
CLUSTER_BASE = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf")

# Data paths on cluster
CLUSTER_BIDS_BASE = Path("/projects/swglab/data/DMNELF")
CLUSTER_DERIVATIVES = CLUSTER_BIDS_BASE / "derivatives"
CLUSTER_PREPROCESSED = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed")
CLUSTER_FMRIPREP = CLUSTER_DERIVATIVES / "fmriprep_25.2.5_fmap"
CLUSTER_DIFUMO = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/difumo_timeseries")
# Working directories
CLUSTER_SCRIPTS = CLUSTER_BASE / "scripts"
CLUSTER_RESULTS = CLUSTER_BASE / "results"
CLUSTER_LOGS = CLUSTER_BASE / "logs"
CLUSTER_MODELS = CLUSTER_BASE / "models"

# ============================================================================
# SLURM CONFIGURATION
# ============================================================================
SLURM_PARTITION = "short"
SLURM_TIME = "04:00:00"
SLURM_CPUS = 4
SLURM_MEM = "16G"

# ============================================================================
# SUBJECTS AND SESSIONS
# ============================================================================
SUBJECTS = [
    "sub-dmnelf001", "sub-dmnelf002", "sub-dmnelf003",
    "sub-dmnelf004", "sub-dmnelf005", "sub-dmnelf006",
    "sub-dmnelf007", "sub-dmnelf008", "sub-dmnelf009",
    "sub-dmnelf010", "sub-dmnelf011", "sub-dmnelf012",
    "sub-dmnelf1001", "sub-dmnelf1002", "sub-dmnelf1003"
]

SESSION = "ses-dmnelf"

# ============================================================================
# NETWORK-SPECIFIC ELECTRODE GROUPS
# ============================================================================

# All EEG channels (32 total, 31 after removing ECG)
ALL_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
                'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
                'TP9', 'TP10', 'POz', 'ECG']

# CEN (Central Executive Network) - frontal/parietal task-positive
CEN_ELECTRODES = ['F3', 'F4', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6', 
                  'C3', 'C4', 'Cz', 'CP1', 'CP2', 'P3', 'P4', 'Pz']

# DMN (Default Mode Network) - midline/posterior
DMN_ELECTRODES = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'POz', 'Oz', 
                  'O1', 'O2', 'P3', 'P4', 'CP1', 'CP2']

# DiFuMo-64 parcel indices (from microstate_pda)
DMN_PARCEL_INDICES = [2, 3, 9, 13, 20, 25, 28, 32, 38, 41, 52, 55]
CEN_PARCEL_INDICES = [0, 5, 11, 14, 16, 22, 29, 35, 39, 44, 47, 50, 57, 62]

# Frequency bands
FREQ_BANDS = {
    'alpha': (8, 12),        # Resting-state (DMN-related)
    'beta_low': (13, 20),    # Sensorimotor/attention
    'beta_high': (20, 30),   # Cognitive control (CEN-related)
    'beta_full': (13, 30)    # Full beta
}

# DiFuMo-64 fMRI network parcels (group-level)
DIFUMO_DMN_INDICES = [2, 3, 9, 13, 20, 25, 28, 32, 38, 41, 52, 55]
DIFUMO_CEN_INDICES = [0, 5, 11, 14, 16, 22, 29, 35, 39, 44, 47, 50, 57, 62]

# Personal network parcels (from 00c_add_personal_parcels.py)
# Path format: derivatives/difumo_personal/sub-XXX_personal_dmn.npy
PERSONAL_PARCEL_DIR = CLUSTER_DERIVATIVES / "difumo_personal"

# Bad channel detection
VARIANCE_THRESHOLD = 3.0
ALWAYS_EXCLUDE = ['ECG']
# ============================================================================
# COGNITIVE STATE CONFIGURATIONS
# Updated targets based on real DMNELF validation (sub-dmnelf001)
# ============================================================================

# DMNELF Feedback (neurofeedback task)
# Real data: LOWEST PLV (0.377) - task disrupts global sync
# Hall "Focused" but inverted for active control task
FEEDBACK_PARAMS = {
    "kuramoto": {
        "n_oscillators": 31,
        "coupling_strength": 10.0,
        "freq_mean": 10.0,
        "freq_std": 2.0,
        "dt": 0.01
    },
    "energy": {
        "alpha": 10.01,
        "beta": 5.00,
        "gamma": 3.00,
        "delta": 2.00
    },
    "agent": {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.3,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "r_target": 0.38,          # Match real PLV (task = low sync)
        "sync_weight": 100.0,
        "energy_weight": 1.0,
        "n_episodes": 200,         # More episodes for lower target
        "episode_duration": 10.0
    },
    "input_type": "uniform",
    "input_magnitude": 5.0,
    "freq_band": "beta"            # Task-related
}

# DMNELF Short Rest (between runs)
# Real data: HIGHEST PLV (0.396) - default state sync
SHORTREST_PARAMS = {
    "kuramoto": {
        "n_oscillators": 31,
        "coupling_strength": 5.0,
        "freq_mean": 10.0,
        "freq_std": 2.0,
        "dt": 0.01
    },
    "energy": {
        "alpha": 10.01,
        "beta": 5.00,
        "gamma": 3.00,
        "delta": 2.00
    },
    "agent": {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.3,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "r_target": 0.40,          # Match real PLV (highest)
        "sync_weight": 100.0,
        "energy_weight": 1.0,
        "n_episodes": 100,
        "episode_duration": 10.0
    },
    "input_type": "split",
    "input_magnitude": 5.0,
    "freq_band": "alpha"
}

# DMNELF Baseline Rest
# Real data: MEDIUM PLV (0.389)
REST_PARAMS = {
    "kuramoto": {
        "n_oscillators": 31,
        "coupling_strength": 1.0,
        "freq_mean": 10.0,
        "freq_std": 2.0,
        "dt": 0.01
    },
    "energy": {
        "alpha": 10.01,
        "beta": 5.00,
        "gamma": 3.00,
        "delta": 2.00
    },
    "agent": {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.3,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "r_target": 0.39,          # Match real PLV (medium)
        "sync_weight": 100.0,
        "energy_weight": 1.0,
        "n_episodes": 100,
        "episode_duration": 10.0
    },
    "input_type": "none",
    "input_magnitude": 0.0,
    "freq_band": "alpha"
}

DEFAULT_CONDITION = "feedback"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_condition_params(condition="feedback"):
    """Get parameters for specific cognitive state."""
    condition_map = {
        "feedback": FEEDBACK_PARAMS,
        "shortrest": SHORTREST_PARAMS,
        "rest": REST_PARAMS
    }
    
    if condition not in condition_map:
        raise ValueError(f"Unknown condition: {condition}")
    
    return condition_map[condition]

def ssh_command(cmd):
    """Build SSH command string."""
    return f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "bash -l -c \\"{cmd}\\"" 2>&1 | grep -v flatpak | grep -v "Loading matlab"'


def scp_to_cluster(local_file, cluster_file):
    """Build SCP upload command."""
    return f"scp {local_file} {CLUSTER_USER}@{CLUSTER_HOST}:{cluster_file}"

def scp_from_cluster(cluster_file, local_file):
    """Build SCP download command."""
    return f"scp {CLUSTER_USER}@{CLUSTER_HOST}:{cluster_file} {local_file}"

def get_personal_parcels(subject):
    """
    Get subject-specific DMN/CEN parcel indices.
    
    Args:
        subject: Subject ID
    
    Returns:
        (dmn_indices, cen_indices) or None if not found
    """
    dmn_file = PERSONAL_PARCEL_DIR / f"{subject}_personal_dmn.npy"
    cen_file = PERSONAL_PARCEL_DIR / f"{subject}_personal_cen.npy"
    
    # Return paths (loaded on cluster)
    return {
        'dmn_file': str(dmn_file),
        'cen_file': str(cen_file),
        'exists': True  # Validation script will check
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Sync DMNELF Configuration")
    print("=" * 70)
    print(f"\nLocal: {LOCAL_BASE}")
    print(f"Cluster: {CLUSTER_BASE}")
    print(f"Data: {CLUSTER_PREPROCESSED}")
    
    print("\n" + "=" * 70)
    print("Cognitive States (Real DMNELF Ordering)")
    print("=" * 70)
    print(f"{'Condition':<12} {'K':<6} {'R_target':<10} {'Real PLV':<10} {'Band'}")
    print("-" * 70)
    
    mapping = {
        'shortrest': ('HIGHEST', 0.396, 'alpha'),
        'rest': ('MEDIUM', 0.389, 'alpha'),
        'feedback': ('LOWEST', 0.377, 'beta')
    }
    
    for cond in ['shortrest', 'rest', 'feedback']:
        params = get_condition_params(cond)
        k = params['kuramoto']['coupling_strength']
        r = params['agent']['r_target']
        label, real, band = mapping[cond]
        print(f"{cond:<12} {k:<6.1f} {r:<10.2f} {real:<10.3f} {band}")
    
    print("\n" + "=" * 70)
    print("Network Electrode Groups")
    print("=" * 70)
    print(f"CEN: {len(CEN_ELECTRODES)} electrodes (frontal-parietal)")
    print(f"DMN: {len(DMN_ELECTRODES)} electrodes (midline-posterior)")
