"""
Configuration for Adaptive Neural Synchronization Pipeline

Matches microstate_pda deployment pattern.
Three cognitive states matching Hall et al. (2025) mapped to DMNELF tasks.

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

# Data paths on cluster - ACTUAL DMNELF locations
CLUSTER_BIDS_BASE = Path("/projects/swglab/data/DMNELF")
CLUSTER_DERIVATIVES = CLUSTER_BIDS_BASE / "derivatives"
CLUSTER_PREPROCESSED = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed")  # EEG
CLUSTER_FMRIPREP = CLUSTER_DERIVATIVES / "fmriprep_25.2.5_fmap"  # fMRI
CLUSTER_DIFUMO = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/data/derivatives/difumo")  # From microstate_pda

# Cluster working directories
CLUSTER_SCRIPTS = CLUSTER_BASE / "scripts"
CLUSTER_RESULTS = CLUSTER_BASE / "results"
CLUSTER_LOGS = CLUSTER_BASE / "logs"
CLUSTER_MODELS = CLUSTER_BASE / "models"  # Saved RL agents

# ============================================================================
# SLURM CONFIGURATION
# ============================================================================
SLURM_PARTITION = "short"
SLURM_TIME = "04:00:00"
SLURM_CPUS = 4
SLURM_MEM = "16G"

# ============================================================================
# COGNITIVE STATE CONFIGURATIONS (Hall et al. 2025 → DMNELF mapping)
# ============================================================================

# Hall "Focused" → DMNELF Neurofeedback runs
# Strong coupling + uniform stimulus → high sync (R ≈ 0.8)
FEEDBACK_PARAMS = {
    "kuramoto": {
        "n_oscillators": 31,
        "coupling_strength": 10.0,  # Strong coupling
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
        "r_target": 0.8,           # Hall achieved ~0.8 in focused
        "sync_weight": 100.0,
        "energy_weight": 1.0,
        "n_episodes": 100,
        "episode_duration": 10.0
    },
    "input_type": "uniform",        # Uniform +5 Hz
    "input_magnitude": 5.0
}

# Hall "Multitasking" → DMNELF Short Rest (between feedback runs)
# Moderate coupling + competing stimuli → medium sync (R ≈ 0.5)
SHORTREST_PARAMS = {
    "kuramoto": {
        "n_oscillators": 31,
        "coupling_strength": 5.0,   # Moderate coupling
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
        "r_target": 0.5,           # Hall achieved ~0.5 in multitasking
        "sync_weight": 100.0,
        "energy_weight": 1.0,
        "n_episodes": 100,
        "episode_duration": 10.0
    },
    "input_type": "split",          # ±5 Hz competing
    "input_magnitude": 5.0
}

# Hall "Resting" → DMNELF Baseline Rest
# Weak coupling + no input → low sync (R ≈ 0.3)
REST_PARAMS = {
    "kuramoto": {
        "n_oscillators": 31,
        "coupling_strength": 1.0,   # Weak coupling
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
        "r_target": 0.3,           # Hall achieved ~0.3 in resting
        "sync_weight": 100.0,
        "energy_weight": 1.0,
        "n_episodes": 100,
        "episode_duration": 10.0
    },
    "input_type": "none",           # No external input
    "input_magnitude": 0.0
}

# Default condition (feedback/focused)
DEFAULT_CONDITION = "feedback"

# ============================================================================
# SUBJECTS AND TASKS
# ============================================================================

# Subjects (both naming conventions)
SUBJECTS = [
    "sub-dmnelf1001", "sub-dmnelf1002", "sub-dmnelf1003",
    "sub-dmnelf001", "sub-dmnelf002", "sub-dmnelf003",
    "sub-dmnelf004", "sub-dmnelf005", "sub-dmnelf006",
    "sub-dmnelf007", "sub-dmnelf008", "sub-dmnelf009",
    "sub-dmnelf010", "sub-dmnelf011", "sub-dmnelf012"
]

# DiFuMo network indices (from microstate_pda verification)
DMN_INDICES = [2, 3, 9, 13, 20, 25, 28, 32, 38, 41, 52, 55]
CEN_INDICES = [0, 5, 11, 14, 16, 22, 29, 35, 39, 44, 47, 50, 57, 62]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_condition_params(condition="feedback"):
    """
    Get parameters for specific cognitive state.
    
    Args:
        condition: 'feedback', 'shortrest', or 'rest'
    
    Returns:
        params: Dictionary with kuramoto, energy, agent, input config
    """
    condition_map = {
        "feedback": FEEDBACK_PARAMS,
        "shortrest": SHORTREST_PARAMS,
        "rest": REST_PARAMS
    }
    
    if condition not in condition_map:
        raise ValueError(f"Unknown condition: {condition}. Use: feedback, shortrest, rest")
    
    return condition_map[condition]

def load_json_config():
    """Load config from JSON if exists"""
    json_path = LOCAL_BASE / "config.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def save_json_config(config_dict):
    """Save config to JSON"""
    json_path = LOCAL_BASE / "config.json"
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def get_cluster_path(local_path):
    """Convert local path to cluster path"""
    rel_path = local_path.relative_to(LOCAL_BASE)
    return CLUSTER_BASE / rel_path

def ssh_command(cmd, check_output=False):
    """Build SSH command string for cluster execution"""
    ssh_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "bash -l -c \\"{cmd}\\"" 2>&1 | grep -v flatpak'
    return ssh_cmd

def scp_to_cluster(local_file, cluster_file):
    """Build SCP command to upload to cluster"""
    scp_cmd = f"scp {local_file} {CLUSTER_USER}@{CLUSTER_HOST}:{cluster_file}"
    return scp_cmd

def scp_from_cluster(cluster_file, local_file):
    """Build SCP command to download from cluster"""
    scp_cmd = f"scp {CLUSTER_USER}@{CLUSTER_HOST}:{cluster_file} {local_file}"
    return scp_cmd


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Sync DMNELF Configuration")
    print("=" * 70)
    print(f"\nLocal base: {LOCAL_BASE}")
    print(f"Cluster base: {CLUSTER_BASE}")
    print(f"Data source: {CLUSTER_DATA}")
    
    print("\n" + "=" * 70)
    print("Cognitive State Configurations (Hall et al. 2025 → DMNELF)")
    print("=" * 70)
    
    for condition in ["feedback", "shortrest", "rest"]:
        params = get_condition_params(condition)
        k = params["kuramoto"]["coupling_strength"]
        r_target = params["agent"]["r_target"]
        input_type = params["input_type"]
        
        hall_name = {
            "feedback": "Focused",
            "shortrest": "Multitasking", 
            "rest": "Resting"
        }[condition]
        
        dmnelf_name = {
            "feedback": "Feedback",
            "shortrest": "Short Rest",
            "rest": "Baseline Rest"
        }[condition]
        
        print(f"\n{condition.upper()}:")
        print(f"  Hall: {hall_name:12} | DMNELF: {dmnelf_name}")
        print(f"  K={k:4.1f}, R_target={r_target:.1f}, Input={input_type}")