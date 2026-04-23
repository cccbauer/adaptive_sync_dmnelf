#!/bin/bash
#SBATCH --job-name=val_001
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/logs/val_sub-dmnelf001_rest_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/logs/val_sub-dmnelf001_rest_%j.err

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
python /projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/scripts/02_validate_rest_cluster.py