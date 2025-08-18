import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject import reproject_adaptive
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

# Parameters
cube = 'all'
method = 'new'

# Automatically list all matching FITS files in the aligned folder
file_path = '/home/apatrick/Code/aligned/'
cube_list = sorted([
    f for f in os.listdir(file_path)
    if f.endswith('_ZAP_img_aligned.fits') and f.startswith('DATACUBE_FINAL_Autocal')
])

print(f"Found {len(cube_list)} files to process.")
# Construct full file paths
file_list = [os.path.join(file_path, f) for f in cube_list]

# Step 1: Open all HDUs and get WCS info
hdus = [fits.open(f, memmap=True, ignore_missing_simple=True)[0] for f in file_list]
wcs_list = [WCS(h.header).celestial for h in hdus]

# Step 2: Find a common WCS that covers all images
wcs_out, shape_out = find_optimal_celestial_wcs(
    [(h.data, w) for h, w in zip(hdus, wcs_list)],
)

# Add 50 pixels of padding on all sides
pad_y, pad_x = 100, 100  # 50 each side
shape_out = (shape_out[0] + pad_y, shape_out[1] + pad_x)

# Shift CRPIX so that the original sky center remains centered
wcs_out.wcs.crpix[0] += pad_x // 2  # X
wcs_out.wcs.crpix[1] += pad_y // 2  # Y


# Step 3: Reproject all images to the common grid in parallel
def reproject_single(args, reproject_quick = False):
    data, wcs = args
    if reproject_quick == True:
        # Use reproject_interp for interpolation
        array, _ = reproject_interp((data, wcs), output_projection=wcs_out, shape_out=shape_out)
    else:
        # Use reproject_adaptive with flux conservation
        # This is slower but preserves flux
        array, _ = reproject_adaptive((data, wcs), output_projection=wcs_out, shape_out=shape_out, conserve_flux=True)
    
    return array

reproject_args = [(h.data, w) for h, w in zip(hdus, wcs_list)]
with ProcessPoolExecutor() as executor:
    reprojected_images = list(executor.map(reproject_single, reproject_args))

# Step 4: Stack via nanmedian
stack = np.array(reprojected_images)
mosaic = np.nanmedian(stack, axis=0)

# Step 5: Save the stacked mosaic as a FITS file
output_fits = f'/home/apatrick/Code/aligned/mosaic_whitelight_nanmedian_{cube}_{method}_full.fits'
hdu = fits.PrimaryHDU(mosaic, header=wcs_out.to_header())
hdu.writeto(output_fits, overwrite=True)
print(f"Saved full mosaic to {output_fits}")

# Step 6: Plot the result
norm = simple_norm(mosaic, 'sqrt', percent=99.5)

plt.figure(figsize=(10, 8))
plt.imshow(mosaic, origin='lower', cmap='viridis', norm=norm)
plt.title(f"Full Mosaic {cube} ({method}) - nanmedian stack")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.colorbar(label='Flux')
plt.tight_layout()

# Step 7: Save the plot
output_png = f'/home/apatrick/Code/aligned/mosaic_whitelight_nanmedian_{cube}_{method}_full.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved plot to {output_png}")



