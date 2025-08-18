#!/bin/bash
#SBATCH --job-name=astrom
#SBATCH --account=users
#SBATCH --time=02:00:00
#SBATCH --output=job_output_%A_%a.log
#SBATCH --error=job_error_%A_%a.log
#SBATCH --array=0-295 #
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Starting job on task ID $SLURM_ARRAY_TASK_ID"

# Move to your code directory
cd /cephfs/apatrick/musecosmos/scripts

# Activate your conda environment
. /usr/local/anaconda/3.9/etc/profile.d/conda.sh
conda activate env39

# Create an array of all FITS files found under reduced3 (recursively)
mapfile -t MUSE_FILES < <(find /cephfs/apatrick/musecosmos/reduced_cubes/white -name "*.fits" | sort)

# Validate task ID
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#MUSE_FILES[@]}" ]; then
  echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of available FITS files (${#MUSE_FILES[@]}), exiting."
  exit 1
fi

# Select the corresponding file for this task
MUSE_FILE=${MUSE_FILES[$SLURM_ARRAY_TASK_ID]}
NEW_FILE=$(basename "$MUSE_FILE" .fits)

# Run your Python script
echo "Running astrom.py on $MUSE_FILE"
python astrom.py "$MUSE_FILE" "$NEW_FILE"

echo "Done with $NEW_FILE"
