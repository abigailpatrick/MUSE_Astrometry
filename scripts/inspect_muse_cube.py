#!/usr/bin/env python
import sys
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

def inspect_muse_cube(fits_file):
    print(f"\nInspecting: {fits_file}\n{'='*60}")

    # Open the FITS file
    with fits.open(fits_file) as hdul:
        print("HDU List:")
        hdul.info()

        # Loop over HDUs and print headers
        for i, hdu in enumerate(hdul):
            print(f"\n--- HDU {i} ---")
            print(f"Name: {hdu.name}")
            print(f"Type: {type(hdu)}")
            print(f"NAXIS: {hdu.header.get('NAXIS', 'None')}")
            print(f"Shape of data: {None if hdu.data is None else hdu.data.shape}")
            print("Important spectral keys if they exist:")
            for key in ["CRVAL3", "CDELT3", "CRPIX3", "CTYPE3", "NAXIS3"]:
                print(f"  {key}: {hdu.header.get(key, 'Not found')}")
            # Show first 5 keywords of header for quick inspection
            print("First 5 header keywords:")
            for k, v in list(hdu.header.items())[:5]:
                print(f"  {k} = {v}")

        # Try reading WCS from cube HDU (usually [1])
        if len(hdul) > 1 and hdul[1].data is not None:
            print("\nWCS inspection from HDU 1:")
            wcs = WCS(hdul[1].header)
            naxis3 = hdul[1].header.get("NAXIS3", 0)
            print(f"Number of spectral slices (NAXIS3): {naxis3}")
            if naxis3 > 0:
                zpix = np.arange(naxis3)
                xpix = np.zeros(naxis3)
                ypix = np.zeros(naxis3)
                world = wcs.all_pix2world(xpix, ypix, zpix, 0)
                print(f"First 5 wavelengths (world coords): {world[2][:5]}")
                print(f"Last 5 wavelengths (world coords): {world[2][-5:]}")
                cdelt3 = np.mean(np.diff(world[2]))
                print(f"Approximate wavelength step: {cdelt3}")
        else:
            print("No data in HDU 1 to inspect WCS.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_muse_cube.py <fits_file>")
        sys.exit(1)
    fits_file = sys.argv[1]
    inspect_muse_cube(fits_file)


#DATACUBE_FINAL_Autocal3688204a_2_ZAP.fits 
#DATACUBE_FINAL_Autocal3692904b_1_ZAP.fits 
#DATACUBE_FINAL_Autocal3692936a_2_ZAP.fits