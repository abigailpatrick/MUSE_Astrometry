#!/usr/bin/env python3
"""
plot_single_exposure.py
-----------------------
Plots one MUSE exposure (masked or unmasked) with fixed percentile scaling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, SqrtStretch

# --------------------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------------------
file_path = '/cephfs/apatrick/musecosmos/scripts/aligned/masked_exposures'  # path to masked FITS files
exposure_name = 'DATACUBE_FINAL_Autocal4010859a_2_ZAP_img_aligned_masked2.0p.fits'  # example
percentile_range = (0.5, 99.5)  # for display

# --------------------------------------------------------------------
# PLOT
# --------------------------------------------------------------------
if __name__ == "__main__":
    exposure_path = os.path.join(file_path, exposure_name)
    if not os.path.exists(exposure_path):
        raise FileNotFoundError(f"File not found: {exposure_path}")

    data = fits.getdata(exposure_path)

    # Compute global percentiles for scaling
    valid = data[np.isfinite(data)]
    vmin, vmax = np.nanpercentile(valid, percentile_range)
    print(f"Plotting {exposure_name} with vmin={vmin:.3e}, vmax={vmax:.3e}")

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

    plt.figure(figsize=(10, 8))
    plt.imshow(data, origin='lower', cmap='viridis', norm=norm)
    plt.title(exposure_name)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label='Flux')
    plt.tight_layout()
    plt.show()

    # save plot
    output_png = exposure_path.replace('.fits', '.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_png}")

# --------------------------------------------------------------------