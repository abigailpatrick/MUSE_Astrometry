#!/bin/bash
#SBATCH --job-name=realign
#SBATCH --output=realign_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

python realignwave.py \
  --path /cephfs/apatrick/musecosmos/reduced_cubes/full \
  --offsets_txt /cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt \
  --slice 100 \
  --output_file mosaic.fits \
  --output_dir /cephfs/apatrick/musecosmos/reduced_cubes/norm