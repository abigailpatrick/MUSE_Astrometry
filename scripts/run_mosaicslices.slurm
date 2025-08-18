#!/bin/bash
#SBATCH --job-name=mosaic_slices
#SBATCH --output=logs/mosaic_%A_%a.out
#SBATCH --error=logs/mosaic_%A_%a.err
#SBATCH --array=1000-1020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# Load your Python environment
module load anaconda
source activate env39

# Paths
CUBE_DIR="/cephfs/apatrick/musecosmos/reduced_cubes/practices"
OFFSETS_FILE="/cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt"
OUTPUT_DIR="/cephfs/apatrick/musecosmos/dataproducts/mosaics"

# SLURM array variable is the slice index
SLICE_INDEX=$SLURM_ARRAY_TASK_ID

# Run the Python script for this slice
python mosaicslices.py "$CUBE_DIR" "$OFFSETS_FILE" "$SLICE_INDEX" --output_dir "$OUTPUT_DIR"
