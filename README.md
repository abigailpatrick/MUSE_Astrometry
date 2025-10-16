# Routine for MUSE COSMOS astrometry and 'MEGA-Cube' creation

_Note: This should be re-written when first re-ran to check for clarity._

## Using White light Images to find astrometric offsets
### Automated routine
* To run the automated routine on multiple exposures save the white light image for each to reduced_cubes/white
* Have astrom.py, run_astrom.slurm and astromfunc.py open and check the paths align. (The slurm script runs astrom.py which imports functions from astromfunc.py.)
* Have the hst file covering the area, the path to these written in top of astrom.py
* Run run_astrom.slurm
* Produces an offsets file showing the correction that needs applying to each from offset_txt file and aligned fits file for each whitelight image
* To stop clashing when running, it saves the offsets to seperate files so now in temrinal do ' cat offsets_*.txt | awk '!seen[$1]++' > offsets.txt ' to merge to one offsets.txt file.
* Then run mosaicfunc.py on the aligned white light aligned images. Visually inspect if any pointings (pointing number is under ‘OBJECT’ in header) have failed or need further corrections (generally those that need very large offsets fail.) Note these and go to the manual routine for any that need the correction
### Manual Routine 
* Open the original white light image and the hst cutout in ds9 tiled next to each other with wcs locked.
* Blink them to get a feel for what adjustment needs to be added in each direction 
  * (MUSE to HST : Right = +ve ra, Left = -ve ra, Up = -ve dec, DOwn = +ve dec)
* Open manualds9adjusts.py (on desktop) and input your first estimate of adjustment, save as aligned1.
* Open aligned1 and repeat above until happy with alignment. Then save as "aligned" - this is the white light image for a quick check of the alignment in mosaic.
* Note the adjustment made in manualoffsets.txt. when all complete copy to cuillin. Run join_manda.py to merge the manual and automatic offsets. Check updated_offsets looks correct th.
* Copy (or scp from aligned_vis) into the ‘aligned folder from before and it will overwrite the files correcting the white light images.
* Re-run mosaicfunc.py to test if the overall mosaic is improved. Once happy, move on.
## Align cubes in wavespace
* Some of the cubes have different wavelength axis lengths. But we are mosaicing them slice by slice so to maintain consistency between slices for now we will unify the wavelength axis.
* Some have an extra 50 A at the blue end and then there is a general variation +/- 1 at both ends. To fix this for now we have just trimmed to wavelength axis to the minimum length (which is standard in many MUSE papers it seems). This is 3681 slices with a 1.25 A step with a start at 4749.9 A and end at 9349.9 A.
* The script realignwave.py can do this and is run for all cubes through realign.slurm. This can be a slightly longer step run time wise.
* It will trim all the existing cubes ready to be aligned spatially, saved to ‘norm’.
* Apply offsets to each slice of each cube. 
* We now want to apply the offsets found from the white light images to each slice of the cube to create a mosaic for each slice. 
* Run mosaic_chunk.py by using run_mosaicslices.slurm to get aligned mosaic slices for every slice of the cube. You need to input a start wavelength slice and the array length, chunks of 1000 seem to take between 3 and 12 hours depending on cuillin busyness, each ‘job’ should take less than 20 minutes but can be as quick as 6 minutes. If they ran in sequence it would take over a week so quiet times are key!
* This script will check the slice_wavelength of each cube in the slice to check it is <1.0 A from the expected value ( the step is 1.25 A so it keeps it under that, they are in reality much closer than this I believe more like 0.1?). There will be an error if it does not pass this. The wavelengths of each cube in that slice are saved to a csv for use in cube construction. 
* This script then takes the offsets file and applies this to the allocated wavelength slice of each cube and then maps them all together into one slice mosaic which is saved for each slice.
* It saves mosaic_slice_{slice_num}.fits (the median stack) and mean_stack_{slice_num}.fits (the mean stack) for each slice.

## Stack into Mega Cube 
* This step takes all the slices and stacks them into one megacube.
* You run build_cube.py through run_build_cube.slurm, it needs a high mem node and takes a few hours but as long as you have the high mem node it's fine. 
* Currently the default is set to take the median slices but can run for the mean stacks.
* The wavelength associated with each slice in the MEGA Cube is from the median of the wavelengths for each cube that makes up that slice, the variations are small but still seems the best method. 
* The Primary header contains the normal header infor from the first slice and the additions I have made saying what cubes have gone into this build and the offsets applied etc should we want to go in to change it. It also retains the ZAP info etc.
* There is a smaller header associated with the actual cube hdu but as an mpdaf object there are strict header rules limiting any extras so that is why it's separate. The thinking is it will be universally useful for the cube to be compatible with standard MUSE MPDAF pipelines for future analysis. 
* It saves the cube to MEGA_CUBE.fits
------------
## Extractions from Megacube
