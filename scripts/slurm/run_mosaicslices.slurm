#!/bin/bash
#SBATCH --job-name=mosaic_array
#SBATCH --array=0-49                # 50 array jobs (each one handles a different slice)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8           # One Python process per slice, with 8 CPU cores
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mosaic_%A_%a.out
#SBATCH --error=logs/mosaic_%A_%a.err

# Activate conda environment
. /usr/local/anaconda/3.9/etc/profile.d/conda.sh
conda activate env39

# Paths
CUBES_DIR="/cephfs/apatrick/musecosmos/reduced_cubes/norm"
OFFSETS_TXT="/cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt"
OUTPUT_DIR="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics"
TMP_BASE="/cephfs/apatrick/musecosmos/scripts/aligned/tmp_slices"

# Map array index to slice number
START_SLICE=$((3105 - 25))  
SLICE_NUM=$((START_SLICE + SLURM_ARRAY_TASK_ID))

echo "Job $SLURM_ARRAY_TASK_ID processing slice $SLICE_NUM"

# Create unique temp folder for this job
SLICE_TMP_DIR="${TMP_BASE}/slice_${SLICE_NUM}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SLICE_TMP_DIR"

# Run Python mosaic (single process per array task)
python /home/apatrick/P1/mosaic_chunk.py \
    --path "$CUBES_DIR" \
    --offsets_txt "$OFFSETS_TXT" \
    --slice $SLICE_NUM \
    --output_file "$OUTPUT_DIR/mosaic_slice_${SLICE_NUM}.fits" \
    --plotting \
    --tmp_dir "$SLICE_TMP_DIR"

# Clean up temp folder
rm -rf "$SLICE_TMP_DIR"
echo "Temporary directory $SLICE_TMP_DIR removed."

echo "Job $SLURM_ARRAY_TASK_ID completed slice $SLICE_NUM"
