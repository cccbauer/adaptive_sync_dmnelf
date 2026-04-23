"""
Configuration for Adaptive Neural Synchronization Pipeline

Matches microstate_pda deployment pattern.
Paths for Explorer cluster at Northeastern University.

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

# Data paths on cluster (link to microstate_pda preprocessed data)
CLUSTER_DATA = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/data")
CLUSTER_DERIVATIVES = CLUSTER_DATA / "derivatives"
CLUSTER_PREPROCESSED = CLUSTER_DERIVATIVES / "preprocessed"  # EEG
CLUSTER_FMRIPREP = CLUSTER_DERIVATIVES / "fmriprep"          # fMRI
CLUSTER_DIFUMO = CLUSTER_DERIVATIVES / "difumo"              # DiFuMo parcels

# Cluster working directories
CLUSTER_SCRIPTS = CLUSTER_BASE / "scripts"
CLUSTER_RESULTS = CLUSTER_BASE / "results"
CLUSTER_LOGS = CLUSTER_BASE / "logs"
CLUSTER_MODELS = CLUSTER_BASE / "models"  # Saved RL agents

# ============================================================================
# SLURM CONFIGURATION
# ============================================================================
SLURM_PARTITION = "short"  # or "gpu" for DQN
SLURM_TIME = "04:00:00"    # 4 hours for training
SLURM_CPUS = 4
SLURM_MEM = "16G"
SLURM_ACCOUNT = "swglab"   # Update if different

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# Kuramoto model
KURAMOTO_PARAMS = {
    "n_oscillators": 31,      # Match DMNELF EEG channels
    "coupling_strength": 5.0,  # Default (varies by condition)
    "freq_mean": 10.0,        # Hz
    "freq_std": 2.0,          # Hz
    "dt": 0.01                # Integration timestep
}

# Energy function
ENERGY_PARAMS = {
    "alpha": 10.01,  # Sync baseline weight
    "beta": 5.00,    # Transition cost weight
    "gamma": 3.00,   # EEG power weight
    "delta": 2.00    # fMRI BOLD weight
}

# RL agent
AGENT_PARAMS = {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 0.3,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "r_target": 0.9,        # Target synchronization
    "n_episodes": 100,      # Training episodes
    "episode_duration": 10.0  # seconds per episode
}

# Subjects (from microstate_pda)
SUBJECTS = [
    "sub-dmnelf1001", "sub-dmnelf1002", "sub-dmnelf1003",
    "sub-dmnelf1004", "sub-dmnelf1005", "sub-dmnelf1006",
    "sub-dmnelf1007", "sub-dmnelf1008", "sub-dmnelf1009",
    "sub-dmnelf1010"
]

# Task conditions
TASKS = {
    "baseline": "resting state baseline",
    "neurofeedback": "CEN-DMN neurofeedback",
    "shortrest": "short rest between runs"
}

# DiFuMo network indices (from microstate_pda verification)
DMN_INDICES = [2, 3, 9, 13, 20, 25, 28, 32, 38, 41, 52, 55]
CEN_INDICES = [0, 5, 11, 14, 16, 22, 29, 35, 39, 44, 47, 50, 57, 62]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    ssh_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "bash -l -c \\"{cmd}\\""'
    return ssh_cmd

def scp_to_cluster(local_file, cluster_file):
    """Build SCP command to upload to cluster"""
    scp_cmd = f"scp {local_file} {CLUSTER_USER}@{CLUSTER_HOST}:{cluster_file}"
    return scp_cmd

def scp_from_cluster(cluster_file, local_file):
    """Build SCP command to download from cluster"""
    scp_cmd = f"scp {CLUSTER_USER}@{CLUSTER_HOST}:{cluster_file} {local_file}"
    return scp_cmd


# Print config on import
if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Sync DMNELF Configuration")
    print("=" * 70)
    print(f"\nLocal base: {LOCAL_BASE}")
    print(f"Cluster base: {CLUSTER_BASE}")
    print(f"Data source: {CLUSTER_DATA}")
    print(f"\nSubjects: {len(SUBJECTS)}")
    print(f"Tasks: {list(TASKS.keys())}")
    print(f"\nKuramoto: {KURAMOTO_PARAMS['n_oscillators']} oscillators")
    print(f"RL: {AGENT_PARAMS['n_episodes']} episodes")
