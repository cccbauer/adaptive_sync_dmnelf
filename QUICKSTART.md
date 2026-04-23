# Quick Start Guide

## Overview

This repository implements the adaptive neural synchronization framework from:

> **Hall, R., Jackson, M., Maleki, M., & Crogman, H. T. (2025).** Modeling cognition through adaptive neural synchronization: a multimodal framework using EEG, fMRI, and reinforcement learning. *Frontiers in Computational Neuroscience*, 19:1616472.

Applied to **DMNELF simultaneous EEG-fMRI neurofeedback data**.

---

## What Does This Do?

The framework models **how neural synchronization emerges from random firing** and is **regulated by metabolic constraints** using reinforcement learning.

### Key Components

1. **Kuramoto Oscillators** (`src/models/kuramoto.py`)
   - Models neuronal phase synchronization
   - Transitions from random → synchronized states
   - Captures cognitive states (rest, focus, multitasking)

2. **Energy Function** (`src/models/energy.py`)
   - Metabolic cost = f(synchronization, EEG power, fMRI BOLD)
   - Integrates multimodal neurophysiology
   - Validates against real DMNELF data

3. **RL Agents** (`src/models/agents.py`)
   - Q-learning and DQN implementations
   - Learn to modulate external input
   - Optimize: maximize sync, minimize energy

4. **Data Loaders** (`src/data_loaders/`)
   - Interface with preprocessed DMNELF EEG/fMRI
   - Reuses `microstate_pda` preprocessing
   - Extracts band power, BOLD signals, PDA

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/cccbauer/adaptive_sync_dmnelf.git
cd adaptive_sync_dmnelf
```

### 2. Run Setup

```bash
bash setup.sh
```

This will:
- Initialize git repository
- Create symlinks to `microstate_pda` data
- Set up conda environment
- Verify data availability

### 3. Activate Environment

```bash
conda activate adaptive_sync
```

---

## Usage Examples

### Train RL Agent

Train a Q-learning agent to control Kuramoto synchronization:

```bash
python scripts/train_rl_agent.py
```

**Output:**
- `trained_qlearning_agent.pkl` - Saved Q-table
- `training_results.png` - Learning curves
- Terminal: Real-time training progress

**What it does:**
- Simulates 100 episodes of 10s each
- Agent learns to maintain R(t) ≈ 0.9 while minimizing energy
- Demonstrates adaptive control of neural dynamics

---

### Validate Against Real Data

Compare simulated dynamics with DMNELF EEG-fMRI:

```bash
python scripts/validate_empirical.py
```

**Tests:**
1. **Spectral Features**: Compare alpha/beta power
2. **BOLD Correlation**: Sync → BOLD vs. real PDA
3. **Energy Dynamics**: Resting vs. focused states

**Expected:**
- Moderate correlation (r ~ 0.3) for resting BOLD
- Higher energy in focused vs. resting
- Dominant alpha in simulated resting state

---

### Interactive Exploration

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

Includes:
- Load real EEG/fMRI from DMNELF
- Simulate Kuramoto for matched duration
- Compute PLV, circular statistics
- GLM analysis of simulated BOLD
- Visualization of sync-energy trade-offs

---

## Key Differences from Hall et al.

| Aspect | Hall et al. 2025 | This Implementation |
|--------|------------------|---------------------|
| **Data** | NatView dataset (3 subjects) | DMNELF (10 subjects, neurofeedback) |
| **Focus** | General cognitive states | CEN-DMN neurofeedback dynamics |
| **Integration** | Standalone framework | Extends `microstate_pda` pipeline |
| **Validation** | Task vs. rest | Baseline vs. neurofeedback runs |

---

## File Structure

```
adaptive_sync_dmnelf/
├── src/
│   ├── models/           # Core models
│   │   ├── kuramoto.py   # Phase synchronization
│   │   ├── energy.py     # Metabolic cost
│   │   └── agents.py     # Q-learning, DQN
│   ├── data_loaders/     # DMNELF interface
│   │   ├── eeg_loader.py
│   │   └── fmri_loader.py
│   └── analysis/         # PLV, GLM, circular stats
├── scripts/
│   ├── train_rl_agent.py      # Main training
│   └── validate_empirical.py  # Real data comparison
├── data/                 # Symlinks to microstate_pda
└── notebooks/            # Interactive analysis
```

---

## Expected Results

### After Training (100 episodes):

- **Final R(t)**: 0.85-0.95 (target: 0.9)
- **Convergence**: ~50-80 episodes
- **Energy**: Stable, lower than random baseline
- **ε**: Decayed to ~0.1 (reduced exploration)

### Validation Metrics:

- **PLV** (alpha): 0.90-0.99 across conditions
- **BOLD correlation**: r = 0.2-0.4 (resting)
- **Energy increase**: 30-50% (resting → focused)

---

## Troubleshooting

### "Data directory not found"
```bash
# Check symlinks
ls -la data/derivatives

# Re-run setup
bash setup.sh
```

### "Could not load EEG data"
```bash
# Verify microstate_pda preprocessing
ls ../microstate_pda/data/derivatives/preprocessed

# Check subjects
python -c "from src.data_loaders.eeg_loader import EEGLoader; print(EEGLoader().list_subjects())"
```

### "Module not found"
```bash
# Ensure environment active
conda activate adaptive_sync

# Reinstall
conda env remove -n adaptive_sync
conda env create -f environment.yml
```

---

## Next Steps

1. **Extend to all DMNELF subjects**: Modify validation script
2. **Compare neurofeedback runs**: Baseline vs. CEN-DMN runs
3. **DQN implementation**: Use PyTorch for proper gradients
4. **Real-time control**: Test on live neurofeedback sessions
5. **Microstate integration**: Link with microstate decoding

---

## Citation

If you use this framework, please cite:

```bibtex
@article{hall2025modeling,
  title={Modeling cognition through adaptive neural synchronization: a multimodal framework using EEG, fMRI, and reinforcement learning},
  author={Hall, Rashad and Jackson, Maury and Maleki, Maryam and Crogman, Horace T},
  journal={Frontiers in Computational Neuroscience},
  volume={19},
  pages={1616472},
  year={2025},
  publisher={Frontiers}
}
```

---

## Contact

Clemens Bauer  
Northeastern University EPIC Brain Lab / MIT McGovern Institute  
Email: bauer.cl@northeastern.edu

**Related Repositories:**
- `microstate_pda`: EEG microstate → fMRI PDA decoding
- `neuro-bolt`: EEG fingerprinting for DMNELF

---

**Status**: Initial implementation (April 2026)  
**License**: MIT (or specify)
