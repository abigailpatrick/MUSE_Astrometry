#!/usr/bin/env python3
"""
create_masks_and_save.py
------------------------
Computes bottom-percentile masks for MUSE exposures, applies them,
and saves both the mask and masked exposure as FITS files.
"""

import os
import numpy as np
from astropy.io import fits

# --------------------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------------------
mask_percent = 2.0  # bottom X% to mask
file_path = '/cephfs/apatrick/musecosmos/scripts/aligned'

# --------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------

def compute_mask(data, percent=1.0):
    """Return boolean mask for pixels in the bottom N% of values."""
    valid_data = data[np.isfinite(data)]
    if valid_data.size == 0:
        return np.zeros_like(data, dtype=bool)
    threshold = np.nanpercentile(valid_data, percent)
    return data <= threshold


def apply_mask(data, mask):
    """Apply boolean mask (set masked pixels to NaN)."""
    return np.where(mask, np.nan, data)


def save_fits(data, filename, header=None):
    """Save a FITS file."""
    fits.PrimaryHDU(data, header=header).writeto(filename, overwrite=True)


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Find all exposures
    cube_list = sorted([
        f for f in os.listdir(file_path)
        if f.endswith('_ZAP_img_aligned.fits') and f.startswith('DATACUBE_FINAL_Autocal')
    ])

    print(f"Found {len(cube_list)} exposures to process.")

    # Create directories
    mask_dir = os.path.join(file_path, "masks")
    masked_dir = os.path.join(file_path, "masked_exposures")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)

    # Process each exposure
    for cube in cube_list:
        in_path = os.path.join(file_path, cube)
        hdu = fits.open(in_path, memmap=True, ignore_missing_simple=True)[0]
        data = hdu.data
        header = hdu.header

        # Compute and save mask
        mask = compute_mask(data, percent=mask_percent)
        mask_name = cube.replace('.fits', f'_mask{mask_percent:.1f}p.fits')
        save_fits(mask.astype(np.uint8), os.path.join(mask_dir, mask_name), header)
        print(f"Saved mask: {mask_name}")

        # Apply mask and save masked FITS
        masked_data = apply_mask(data, mask)
        masked_name = cube.replace('.fits', f'_masked{mask_percent:.1f}p.fits')
        save_fits(masked_data, os.path.join(masked_dir, masked_name), header)
        print(f"Saved masked exposure: {masked_name}")

    print("\nAll masks and masked exposures saved successfully.")
