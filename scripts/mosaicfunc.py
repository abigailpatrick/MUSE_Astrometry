import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_adaptive
from reproject.mosaicking import find_optimal_celestial_wcs
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, LinearStretch, AsinhStretch, SqrtStretch
from concurrent.futures import ProcessPoolExecutor

# --------------------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------------------
cube = 'all'
method = 'new'
mask_percent = 1.5  # percentage to mask (default 1%)

file_path = '/cephfs/apatrick/musecosmos/scripts/aligned'

# --------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------

def compute_mask(data, percent=1.0):
    """Return a boolean mask for pixels in the bottom `N`% of values."""
    valid_data = data[np.isfinite(data)]
    if valid_data.size == 0:
        return np.zeros_like(data, dtype=bool)
    threshold = np.nanpercentile(valid_data, percent)
    mask = data <= threshold
    return mask


def apply_mask(data, mask):
    """Apply a boolean mask to data (set masked pixels to NaN)."""
    masked_data = np.where(mask, np.nan, data)
    return masked_data


def save_mask(mask, filename, header=None):
    """Save a mask as a FITS file."""
    hdu = fits.PrimaryHDU(mask.astype(np.uint8), header=header)
    hdu.writeto(filename, overwrite=True)


# --------------------------------------------------------------------
# LOAD INPUT FILES
# --------------------------------------------------------------------
cube_list = sorted([
    f for f in os.listdir(file_path)
    if f.endswith('_ZAP_img_aligned.fits') and f.startswith('DATACUBE_FINAL_Autocal')
])

print(f"Found {len(cube_list)} files to process.")
file_list = [os.path.join(file_path, f) for f in cube_list]

hdus = [fits.open(f, memmap=True, ignore_missing_simple=True)[0] for f in file_list]
wcs_list = [WCS(h.header).celestial for h in hdus]

# --------------------------------------------------------------------
# CALCULATE GLOBAL PERCENTILE-BASED SCALING (before masking)
# --------------------------------------------------------------------
print("Computing global percentile-based scaling for consistent visualization...")

all_data = np.concatenate([
    h.data[np.isfinite(h.data)].ravel() for h in hdus if np.isfinite(h.data).any()
])

# Compute vmin/vmax from percentiles of unmasked data
vmin = np.nanpercentile(all_data, 0.5)   # 0.5th percentile
vmax = np.nanpercentile(all_data, 99.5)  # 99.5th percentile

print(f"Global percentiles (unmasked data):")
print(f"  0.5th percentile (vmin)  = {vmin:.3e}")
print(f" 99.5th percentile (vmax) = {vmax:.3e}")
print("These limits will be used for all runs to keep visualization consistent.")

# --------------------------------------------------------------------
# COMPUTE AND SAVE MASKS FOR EACH EXPOSURE
# --------------------------------------------------------------------
mask_dir = os.path.join(file_path, "masks")
os.makedirs(mask_dir, exist_ok=True)

for i, h in enumerate(hdus):
    mask = compute_mask(h.data, percent=mask_percent)
    mask_filename = os.path.join(mask_dir, cube_list[i].replace('.fits', f'_mask1_5p.fits'))
    save_mask(mask, mask_filename, header=h.header)
    print(f"Saved mask for {cube_list[i]} to {mask_filename}")
    # Apply mask to data
    h.data = apply_mask(h.data, mask)





# --------------------------------------------------------------------
# MOSAIC STACKING
# --------------------------------------------------------------------
wcs_out, shape_out = find_optimal_celestial_wcs(
    [(h.data, w) for h, w in zip(hdus, wcs_list)]
)

# Add padding
pad_y, pad_x = 100, 100
shape_out = (shape_out[0] + pad_y, shape_out[1] + pad_x)
wcs_out.wcs.crpix[0] += pad_x // 2
wcs_out.wcs.crpix[1] += pad_y // 2

def reproject_single(args, reproject_quick=False):
    data, wcs = args
    if reproject_quick:
        array, _ = reproject_interp((data, wcs), output_projection=wcs_out, shape_out=shape_out)
    else:
        array, _ = reproject_adaptive((data, wcs), output_projection=wcs_out, shape_out=shape_out, conserve_flux=True)
    return array

reproject_args = [(h.data, w) for h, w in zip(hdus, wcs_list)]
with ProcessPoolExecutor() as executor:
    reprojected_images = list(executor.map(reproject_single, reproject_args))

stack = np.array(reprojected_images)
mosaic = np.nanmedian(stack, axis=0)

# --------------------------------------------------------------------
# SAVE FINAL MOSAIC
# --------------------------------------------------------------------
output_fits = os.path.join(
    file_path, f'mosaic_whitelight_nanmedian_{cube}_{method}_masked1_5p.fits'
)
fits.PrimaryHDU(mosaic, header=wcs_out.to_header()).writeto(output_fits, overwrite=True)
print(f"Saved full mosaic to {output_fits}")

# --------------------------------------------------------------------
# PLOT WITH FIXED SCALE
# --------------------------------------------------------------------
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

plt.figure(figsize=(10, 8))
plt.imshow(mosaic, origin='lower', cmap='viridis', norm=norm)
plt.title(f"Full Mosaic {cube} ({method}) - nanmedian stack (masked {mask_percent}%)")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.colorbar(label='Flux')
plt.tight_layout()

output_png = output_fits.replace('.fits', '.png')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved plot to {output_png}")
