#!/bin/bash
#
# Setup Script for adaptive_sync_dmnelf Repository
#
# This script:
# 1. Initializes git repository
# 2. Creates symlinks to existing microstate_pda data
# 3. Sets up conda environment
# 4. Verifies data availability
#
# Author: Clemens Bauer
# Date: April 2026

set -e  # Exit on error

echo "=========================================="
echo "Adaptive Neural Synchronization Setup"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. Initialize git repository
echo "1. Initializing git repository..."
if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "Initial commit: Adaptive neural synchronization framework

- Kuramoto oscillator model for phase synchronization
- Energy function integrating EEG and fMRI
- Q-learning and DQN agents for control
- Data loaders for preprocessed DMNELF data
- Training scripts and analysis tools

Based on Hall et al. (2025) Frontiers in Computational Neuroscience."
    echo "  ✓ Git repository initialized"
else
    echo "  ✓ Git repository already exists"
fi

# 2. Create symlinks to microstate_pda data
echo ""
echo "2. Linking to microstate_pda data..."

# Prompt for microstate_pda path
read -p "Enter path to microstate_pda repository [../microstate_pda]: " MICROSTATE_PATH
MICROSTATE_PATH=${MICROSTATE_PATH:-"../microstate_pda"}

if [ -d "$MICROSTATE_PATH" ]; then
    # Create data directory structure
    mkdir -p data/derivatives
    
    # Link preprocessed EEG
    if [ -d "$MICROSTATE_PATH/data/derivatives/preprocessed" ]; then
        ln -sf "$(realpath $MICROSTATE_PATH/data/derivatives/preprocessed)" \
               data/derivatives/preprocessed
        echo "  ✓ Linked preprocessed EEG data"
    else
        echo "  ⚠ Warning: preprocessed EEG not found in microstate_pda"
    fi
    
    # Link fMRIPrep outputs
    if [ -d "$MICROSTATE_PATH/data/derivatives/fmriprep" ]; then
        ln -sf "$(realpath $MICROSTATE_PATH/data/derivatives/fmriprep)" \
               data/derivatives/fmriprep
        echo "  ✓ Linked fMRIPrep fMRI data"
    else
        echo "  ⚠ Warning: fMRIPrep outputs not found"
    fi
    
    # Link DiFuMo parcellations
    if [ -d "$MICROSTATE_PATH/data/derivatives/difumo" ]; then
        ln -sf "$(realpath $MICROSTATE_PATH/data/derivatives/difumo)" \
               data/derivatives/difumo
        echo "  ✓ Linked DiFuMo parcel time series"
    else
        echo "  ⚠ Warning: DiFuMo data not found"
    fi
else
    echo "  ⚠ Warning: microstate_pda path not found: $MICROSTATE_PATH"
    echo "    You can create symlinks manually later."
fi

# 3. Setup conda environment
echo ""
echo "3. Setting up conda environment..."
if command -v conda &> /dev/null; then
    read -p "Create conda environment 'adaptive_sync'? (y/n) [y]: " CREATE_ENV
    CREATE_ENV=${CREATE_ENV:-y}
    
    if [ "$CREATE_ENV" = "y" ]; then
        if conda env list | grep -q "adaptive_sync"; then
            echo "  ✓ Environment 'adaptive_sync' already exists"
        else
            conda env create -f environment.yml
            echo "  ✓ Environment created"
        fi
        
        echo ""
        echo "To activate the environment:"
        echo "  conda activate adaptive_sync"
    fi
else
    echo "  ⚠ conda not found - skipping environment creation"
fi

# 4. Verify data
echo ""
echo "4. Verifying data availability..."

if [ -d "data/derivatives/preprocessed" ]; then
    N_SUBJECTS=$(ls data/derivatives/preprocessed | grep -c "^sub-" || true)
    echo "  ✓ Found $N_SUBJECTS subjects with preprocessed EEG"
else
    echo "  ⚠ No preprocessed EEG data found"
fi

if [ -d "data/derivatives/fmriprep" ]; then
    N_FMRI=$(ls data/derivatives/fmriprep | grep -c "^sub-" || true)
    echo "  ✓ Found $N_FMRI subjects with fMRI data"
else
    echo "  ⚠ No fMRI data found"
fi

# 5. Create output directories
echo ""
echo "5. Creating output directories..."
mkdir -p results figures logs
echo "  ✓ Created: results/, figures/, logs/"

# Done
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate adaptive_sync"
echo "  2. Run test simulation: python scripts/train_rl_agent.py"
echo "  3. Check notebooks/ for analysis examples"
echo ""
echo "Repository structure:"
echo "  src/models/      - Kuramoto, energy, RL agents"
echo "  src/data_loaders/ - EEG and fMRI loaders"
echo "  scripts/         - Training and validation scripts"
echo "  data/            - Symlinks to microstate_pda"
echo ""
