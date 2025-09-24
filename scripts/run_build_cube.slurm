#!/bin/bash
#SBATCH --job-name=build_cube
#SBATCH --output=build_cube_%j.log
#SBATCH --error=build_cube_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=480G
#SBATCH --cpus-per-task=1

# Load Python
. /usr/local/anaconda/3.9/etc/profile.d/conda.sh
conda activate env39

cd /cephfs/apatrick/musecosmos/scripts/

echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "---------------------------"

# Run your Python script with memory monitoring in background
/usr/bin/time -v python build_cube.py --start_id 0 --end_id 3681 2>&1 | tee build_cube_${SLURM_JOB_ID}_runtime.log

echo "---------------------------"
echo "Job finished at: $(date)"
