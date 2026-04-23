# Adaptive Neural Synchronization for DMNELF

**Modeling EEG-fMRI Neurofeedback Dynamics Using Kuramoto Oscillators and Reinforcement Learning**

## Overview

This repository implements the adaptive neural synchronization framework from Hall et al. (2025, *Frontiers in Computational Neuroscience*) applied to DMNELF simultaneous EEG-fMRI neurofeedback data.

### Key Components

1. **Kuramoto Oscillator Model**: Simulates neuronal phase synchronization
2. **Energy Function**: Metabolic cost model integrating EEG power and fMRI BOLD
3. **Reinforcement Learning**: Q-learning and DQN agents optimize synchronization
4. **Multimodal Validation**: Compare simulated vs. real EEG/fMRI dynamics

## Data Source

Leverages preprocessed DMNELF data from `microstate_pda` pipeline:
- **EEG**: 31-channel simultaneous recording (0.01-20 Hz, BCG-corrected)
- **fMRI**: fMRIPrep-processed BOLD signals (DiFuMo-64 parcellation)
- **Conditions**: Resting baseline, CEN-DMN neurofeedback runs

## Repository Structure

```
adaptive_sync_dmnelf/
├── data/                    # Symlinks to preprocessed DMNELF data
├── src/
│   ├── models/
│   │   ├── kuramoto.py      # Kuramoto oscillator implementation
│   │   ├── energy.py        # Energy function (EEG + fMRI)
│   │   └── agents.py        # Q-learning and DQN agents
│   ├── analysis/
│   │   ├── plv.py          # Phase-locking value analysis
│   │   ├── glm.py          # GLM for BOLD prediction
│   │   └── circular_stats.py  # Circular statistics
│   ├── data_loaders/
│   │   ├── eeg_loader.py    # Load preprocessed EEG
│   │   └── fmri_loader.py   # Load fMRIPrep BOLD
│   └── visualization/
│       └── plots.py         # Plotting utilities
├── scripts/
│   ├── 01_simulate_kuramoto.py
│   ├── 02_train_rl_agent.py
│   └── 03_validate_empirical.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── environment.yml          # Conda environment
└── README.md
```

## Relationship to microstate_pda

This work extends the microstate→PDA decoding pipeline by:
- Modeling **why** EEG patterns predict fMRI (via synchronization dynamics)
- Optimizing neurofeedback control using RL
- Simulating cognitive state transitions (rest → neurofeedback)

We reuse:
- Preprocessed EEG time series (BCG-corrected, filtered)
- fMRIPrep BOLD signals
- Subject/run metadata

## Installation

```bash
# Clone repository
git clone https://github.com/cccbauer/adaptive_sync_dmnelf.git
cd adaptive_sync_dmnelf

# Create conda environment
conda env create -f environment.yml
conda activate adaptive_sync

# Link to preprocessed DMNELF data
ln -s /path/to/microstate_pda/data ./data
```

## Quick Start

```python
from src.models.kuramoto import KuramotoModel
from src.models.energy import EnergyFunction
from src.data_loaders.eeg_loader import load_eeg_timeseries

# Load real EEG data
eeg = load_eeg_timeseries(subject='sub-001', run='baseline')

# Initialize Kuramoto model
model = KuramotoModel(n_oscillators=31, coupling_strength=5.0)

# Simulate synchronization
sync, phases = model.simulate(duration=300, dt=0.01)

# Compute energy cost
energy_fn = EnergyFunction(eeg_power=eeg, bold_signal=fmri)
cost = energy_fn.compute(sync)
```

## Citation

Hall, R., Jackson, M., Maleki, M., & Crogman, H. T. (2025). Modeling cognition through adaptive neural synchronization: a multimodal framework using EEG, fMRI, and reinforcement learning. *Frontiers in Computational Neuroscience*, 19:1616472.

## Contact

Clemens Bauer  
Northeastern University EPIC Brain Lab / MIT McGovern Institute

---

**Status**: Initial setup (April 2026)
