import sys
from astropy.io import fits
import numpy as np
from scipy.ndimage import shift

# Read slice range from command-line
start = int(sys.argv[1])
end   = int(sys.argv[2])

# Your 4 cubes
cube_path = "/cephfs/apatrick/musecosmos/reduced_cubes/"
cube_names = ["DATACUBE_FINAL_Autocal3687411a_1_ZAP.fits", " DATACUBE_FINAL_Autocal3687416a_1_ZAP.fits", " DATACUBE_FINAL_Autocal3687419a_1_ZAP.fits", " DATACUBE_FINAL_Autocal3688185a_1_ZAP.fits"]
cube_files = [cube_path + name for name in cube_names]
# Known x/y offsets (dx, dy for each cube) loaded from txt
offsets = np.loadtxt("offsets.txt")

# Memory-map cubes so you don’t load everything into RAM
cubes = [fits.open(f, memmap=True)[0].data for f in cube_files]

# Use header from the first cube as reference
header_ref = fits.open(cube_files[0])[0].header

# Function: apply shift to an image
def apply_offsets(image, dx, dy):
    return shift(image, shift=(dy, dx), order=1)

# Function: combine aligned images into one mosaic
def mosaic_images(list_of_images):
    return np.nanmean(list_of_images, axis=0)

# Collect mosaic slices for this chunk
mosaic_slices = []
for i in range(start, end+1):
    if i >= cubes[0].shape[0]: break  # don’t go beyond cube length
    
    # Take slice i from each cube
    slices = [cube[i, :, :] for cube in cubes]
    
    # Apply the known offsets
    shifted = [apply_offsets(s, offsets[j,0], offsets[j,1]) 
               for j,s in enumerate(slices)]
    
    # Mosaic them into one slice
    mosaic_slices.append(mosaic_images(shifted))

# Convert list → 3D array (nslice, ny, nx)
mosaic_slices = np.array(mosaic_slices)

# Save this chunk as a FITS file
fits.PrimaryHDU(mosaic_slices, header=header_ref).writeto(
    f"mosaic_{start}_{end}.fits", overwrite=True
)
