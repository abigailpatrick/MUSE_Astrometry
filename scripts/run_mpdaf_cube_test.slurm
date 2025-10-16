#!/bin/bash
#SBATCH --job-name=muse_wl
#SBATCH --output=logs/muse_wl_%j.out
#SBATCH --error=logs/muse_wl_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G

# Load Python
. /usr/local/anaconda/3.9/etc/profile.d/conda.sh
conda activate env39

cd /cephfs/apatrick/musecosmos/scripts/
srun python mpdaf_cube_test.py