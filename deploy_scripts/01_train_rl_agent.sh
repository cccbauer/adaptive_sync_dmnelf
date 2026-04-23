#!/bin/bash
#SBATCH --job-name=rl_shortrest
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/logs/rl_shortrest_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/logs/rl_shortrest_%j.err

echo "======================================================="
echo "Training RL Agent: shortrest"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================="

# Use base conda environment
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh

# Run training script
python /projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/scripts/01_train_rl_agent_cluster.py

echo "======================================================="
echo "Complete: shortrest"
echo "======================================================="