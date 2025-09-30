#!/bin/bash
#SBATCH --job-name=muse_extract
#SBATCH --output=logs/muse_extract_%j.out
#SBATCH --error=logs/muse_extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G

# Load Python
. /usr/local/anaconda/3.9/etc/profile.d/conda.sh
conda activate env39

cd /cephfs/apatrick/musecosmos/scripts/
srun python /cephfs/apatrick/musecosmos/scripts/cube_extractions.py \
    --cube /cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits \
    --objects /home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic.csv \
    --spatial_radius 40.0 \
    --spectral_radius 50.0 \
    --rest_wavelength 1215.67 \
    --output /cephfs/apatrick/musecosmos/dataproducts/extractions/
