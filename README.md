
The scripts for MUSE COSMOS astrometry.

The slurm script runs astrom.py which imports functions from astromfunc.py

To Use:
* Ensure astrom.py, slurm, astromfunc.py and the exposures desired are on cuillin.
* Edit file paths in astrom.py to be to the correct to exposures
* Run slurm
* Run mosaicfunc.py to see automated mosaic
* Choose pointings to do manually. (object in header contains the pointing number also printed in the opening of file in astrom.py.)
* Open the cutout and exposure in ds9 follow manual instructions using manualds9adjusts.py to adjust the images until happy with alignment
* Scp the aligned_vis folder from this into aligned. It will overwrite the corrected files
* Re-run mosaicfunc.py to see updated final mosaic.
