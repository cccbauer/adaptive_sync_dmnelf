# HOW TO — Adaptive Neural Synchronization Pipeline
## RL-Based Neurofeedback Control for DMNELF

**Author:** Clemens C.C. Bauer (cccbauer)  
**Lab:** EPIC Brain Lab, Northeastern University  
**Cluster:** Explorer (explorer.northeastern.edu)  
**Based on:** Hall et al. (2025), Frontiers in Computational Neuroscience

---

## Pipeline Overview

```
Preprocessed EEG/fMRI (from microstate_pda)
           │
           ↓
    Kuramoto Model
    (phase sync)
           │
           ↓
    Energy Function
    (EEG + fMRI cost)
           │
           ↓
    RL Agent Training
    (Q-learning/DQN)
           │
           ↓
  Optimal Control Policy
  (neurofeedback input)
```

---

## Quick Reference

### SSH to cluster
```bash
ssh cccbauer@explorer.northeastern.edu
```

### Check running jobs
```bash
ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer'
```

### Check job log
```bash
ssh cccbauer@explorer.northeastern.edu 'tail -50 /projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/logs/rl_train_JOBID.out'
```

### Cancel job
```bash
ssh cccbauer@explorer.northeastern.edu 'scancel JOBID'
```

---

## Installation

### 1. Clone Repository

```bash
cd ~/Documents/GitHub
# Extract the tarball you downloaded
tar -xzf adaptive_sync_dmnelf.tar.gz
cd adaptive_sync_dmnelf
```

### 2. Verify Configuration

```bash
python config.py
```

Should show:
```
Adaptive Sync DMNELF Configuration
Local base: /Users/anitya/Documents/GitHub/adaptive_sync_dmnelf
Cluster base: /projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf
Data source: /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/data
Subjects: 10
```

### 3. Setup Cluster Environment (First Time Only)

```bash
# SSH to cluster
ssh cccbauer@explorer.northeastern.edu

# Create conda environment
conda create -n adaptive_sync python=3.11 -y
conda activate adaptive_sync

# Install dependencies
conda install numpy scipy pandas matplotlib scikit-learn -c conda-forge -y
pip install nibabel mne nilearn

# Test imports
python -c "import numpy; import scipy; from pathlib import Path; print('✓ All imports successful')"

# Exit
exit
```

---

## Pipeline Scripts

### Script 01: Train RL Agent

**What it does:** Trains Q-learning agent to control Kuramoto synchronization

**Usage:**
```bash
cd ~/Documents/GitHub/adaptive_sync_dmnelf
python deploy_scripts/01_train_rl_agent.py
```

**What happens:**
1. Compiles cluster script
2. Creates directories on cluster
3. Copies Python + SLURM scripts
4. Copies source modules (kuramoto.py, energy.py, agents.py)
5. Submits SLURM job

**Output:**
```
Deploy: Train RL Agent on Cluster
✓ Cluster script compiles
✓ Directories created
✓ Python script copied
✓ SLURM script copied
✓ Source modules copied
✓ Job submitted: 1234567
```

**Monitor:**
```bash
# Check job status
ssh cccbauer@explorer.northeastern.edu 'squeue -j 1234567'

# Watch log in real-time
ssh cccbauer@explorer.northeastern.edu 'tail -f /projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/logs/rl_train_1234567.out'
```

**Expected runtime:** 20-30 minutes (CPU-only, 100 episodes)

---

### Fetch Results

**Download trained models:**
```bash
python deploy_scripts/fetch_results.py --models
```

**Download everything:**
```bash
python deploy_scripts/fetch_results.py --all
```

**Options:**
- `--models` - Trained Q-learning agents (.pkl)
- `--plots` - Validation plots
- `--logs` - SLURM output logs
- `--job-id JOBID` - Fetch logs for specific job
- `--all` - Everything

**Results location:**
```
results/
├── rl_agents/            # Trained models
│   ├── qlearning_agent.pkl
│   └── training_results.pkl
├── training_curves/      # Learning metrics
└── validation_plots/     # Real vs simulated
```

---

## Understanding the Framework

### Kuramoto Model (src/models/kuramoto.py)

Models neuronal phase synchronization:

```
dθᵢ/dt = ωᵢ + (K/N)Σⱼsin(θⱼ - θᵢ) + Iᵢ(t)
```

- **θᵢ**: Phase of oscillator i (neuron/channel)
- **ωᵢ**: Natural frequency (drawn from N(10, 2) Hz)
- **K**: Coupling strength (1=rest, 5=multitask, 10=focused)
- **I(t)**: External input (RL agent's action)
- **R(t)**: Order parameter (synchronization level)

**States:**
- Resting: K=1, no input → R(t) ≈ 0.3
- Focused: K=10, uniform input → R(t) ≈ 0.9
- Multitasking: K=5, competing inputs → R(t) ≈ 0.5

### Energy Function (src/models/energy.py)

Metabolic cost model:

```
E(t) = α·R(t) + β·dR/dt + γ·P_EEG(t) + δ·S_fMRI(t)
```

- **α=10.01**: Baseline sync cost
- **β=5.00**: Transition cost
- **γ=3.00**: EEG power weight
- **δ=2.00**: BOLD signal weight

### RL Agent (src/models/agents.py)

**Q-learning** learns action-value function Q(s, a):

```
Q(s,a) ← Q(s,a) + η[r + γ max_a' Q(s',a') - Q(s,a)]
```

**State:** (R(t), E(t))  
**Action:** External input level [-5, +5] Hz  
**Reward:** -|R_target - R(t)| - E(t)  
**Goal:** Maintain R(t) ≈ 0.9 with minimal energy

---

## Integration with microstate_pda

### Data Flow

```
microstate_pda/
├── data/derivatives/
│   ├── preprocessed/    ← EEG (BCG-corrected, 0.01-20 Hz)
│   ├── fmriprep/        ← BOLD signals
│   └── difumo/          ← DiFuMo-64 parcels
        ↓
    (used by)
        ↓
adaptive_sync_dmnelf/
└── src/data_loaders/    ← Load EEG/fMRI for validation
```

**Key difference:**
- **microstate_pda**: Decodes PDA from microstates (static)
- **adaptive_sync**: Controls synchronization via RL (dynamic)

### Combined Workflow

1. **Preprocess** (microstate_pda): EEG → microstates
2. **Simulate** (adaptive_sync): Kuramoto → sync dynamics
3. **Validate**: Compare simulated vs. real EEG/fMRI
4. **Control**: RL learns optimal neurofeedback input
5. **Apply**: Test policy on real neurofeedback runs

---

## Troubleshooting

### "PATH not found" on anitya

```bash
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin:$PATH"
# Add to ~/.zshrc to make permanent
```

### "Cluster script compilation failed"

Check syntax in `deploy_scripts/01_train_rl_agent.py`:
- No f-strings in CLUSTER_SCRIPT list
- Use + for string concatenation
- Use str() to convert numbers

### "Permission denied" when deploying

```bash
# Make sure SSH key is set up
ssh-copy-id cccbauer@explorer.northeastern.edu
```

### "Module not found" on cluster

```bash
ssh cccbauer@explorer.northeastern.edu
conda activate adaptive_sync
python -c "from models.kuramoto import KuramotoModel"
```

If fails, check paths in cluster script.

### Job stuck in PENDING

```bash
# Check queue position
ssh cccbauer@explorer.northeastern.edu 'squeue -j JOBID -l'

# Cancel and resubmit
ssh cccbauer@explorer.northeastern.edu 'scancel JOBID'
python deploy_scripts/01_train_rl_agent.py
```

---

## Expected Results

After training 100 episodes (30 min):

**Training metrics:**
```
Episode 100: Reward=-2.34, R=0.892, E=2.15
Final mean R(t): 0.892 (target: 0.9)
Final mean E(t): 2.15
Final ε: 0.104 (exploration rate)
```

**Saved files:**
```
models/
├── qlearning_agent.pkl      # 50-100 KB
└── training_results.pkl     # 10-20 KB
```

**Load and test:**
```python
import pickle
from models.agents import QLearningAgent

# Load agent
with open('results/rl_agents/qlearning_agent.pkl', 'rb') as f:
    data = pickle.load(f)
    
agent = QLearningAgent()
agent.q_table = data['q_table']

# Test action selection
R, E = 0.5, 3.0
action_idx, action_val = agent.select_action(R, E)
print(f"Recommended input: {action_val:.2f} Hz")
```

---

## Next Steps

### Extend to DQN

Replace Q-table with neural network for continuous states:
```bash
# Edit deploy_scripts/02_train_dqn.py
# Use PyTorch for proper gradients
# Submit with --partition=gpu
```

### Subject-Specific Training

Train separate agents per subject:
```bash
# Loop over config.SUBJECTS
# Use subject's baseline EEG for validation
```

### Real-Time Deployment

Test trained policy on live neurofeedback:
```bash
# Load agent in MURFI callback
# Compute R(t) from real-time EEG
# Select action, modulate neurofeedback signal
```

---

## Citation

Hall, R., Jackson, M., Maleki, M., & Crogman, H. T. (2025). Modeling cognition through adaptive neural synchronization: a multimodal framework using EEG, fMRI, and reinforcement learning. *Frontiers in Computational Neuroscience*, 19:1616472.

---

## Contact

Clemens Bauer  
Email: bauer.cl@northeastern.edu  
GitHub: github.com/cccbauer/adaptive_sync_dmnelf

**Related Repos:**
- `microstate_pda`: EEG microstate → PDA decoding
- `neuro-bolt`: EEG fingerprinting foundation model
