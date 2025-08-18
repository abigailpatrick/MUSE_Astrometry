import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

cube_dir = "/cephfs/apatrick/musecosmos/reduced_cubes"

# Find all cube files (excluding _img.fits)
cube_files = [
    os.path.join(cube_dir, f)
    for f in os.listdir(cube_dir)
    if f.endswith(".fits") and "_img" not in f
]

reference_step = None
reference_range = None

for f in cube_files:
    try:
        with fits.open(f) as hdul:
            hdr = hdul[1].header  # MUSE cube is usually in extension 1
            # Try direct CDELT3
            if "CDELT3" in hdr:
                cdelt3 = hdr["CDELT3"]
                n = hdr["NAXIS3"]
                crval3 = hdr["CRVAL3"]
                range_vals = (crval3, crval3 + cdelt3 * (n - 1))
            else:
                # Use WCS
                w = WCS(hdr)
                n = hdr["NAXIS3"]
                pix = np.arange(n)
                wavelengths = w.all_pix2world(np.zeros(n), np.zeros(n), pix, 0)[2]
                cdelt3 = np.mean(np.diff(wavelengths))
                range_vals = (wavelengths[0], wavelengths[-1])

            # Store reference if first file
            if reference_step is None:
                reference_step = cdelt3
                reference_range = range_vals
                print(f"Reference from {os.path.basename(f)}:")
                print(f"  Step: {reference_step:.6f}")
                print(f"  Range: {reference_range}")
            else:
                if not np.isclose(cdelt3, reference_step, rtol=1e-8):
                    print(f"⚠ Step mismatch in {os.path.basename(f)}: {cdelt3}")
                if not (np.isclose(range_vals[0], reference_range[0], rtol=1e-8) and
                        np.isclose(range_vals[1], reference_range[1], rtol=1e-8)):
                    print(f"⚠ Range mismatch in {os.path.basename(f)}: {range_vals}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
