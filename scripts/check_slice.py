from astropy.io import fits
import numpy as np
# cp /cephfs/apatrick/musecosmos/scripts/check_slice.py /home/apatrick/P1
"""
# Path to your file
cube_file = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full/mosaic_slice_860.fits"

with fits.open(cube_file) as hdul:
    print("=== FITS file structure ===")
    hdul.info()

    # Access the only HDU (PrimaryHDU)
    hdu = hdul[0]
    data = hdu.data
    header = hdu.header

    # Print the data shape
    print("\n=== Data shape ===")
    print(data.shape)
    # Check data values 
    print("\n==== Number of Nans ====")
    print(np.isnan(data).sum())
    print("\n===== Number of Data Values ====")
    print(np.count_nonzero(~np.isnan(data)))
    print(f" Data Values: {data[~np.isnan(data)]}")



    # Print the full header
    print("\n=== Full header ===")
    print(repr(header))
"""

# this section is checking full 
# Path to your 3D cube
cube_file = "/cephfs/apatrick/musecosmos/reduced_cubes/norm/DATACUBE_FINAL_Autocal3693040b_1_ZAP_norm.fits"

with fits.open(cube_file) as hdul:
    print("=== FITS file structure ===")
    hdul.info()

    
    hdu = hdul[1]


    data = hdu.data
    header = hdu.header

    # Print the data shape
    #print("\n=== Data shape ===")
    #print(data.shape)  # (n_lambda, ny, nx) or (nz, ny, nx)

    """ 
    # Check for NaNs
    slicenum = 1536
    print(f"\n=== Number of NANS in wavelength slice {slicenum} ===")
    print(np.isnan(data[slicenum, :, :]).sum())
    print("\n===== Number of Data Values ====")
    print(np.count_nonzero(~np.isnan(data[slicenum, :, :])))
    print(f" Number of pixels in slice: {data[slicenum, :, :].size}")
    print(f" Fraction of valid data in slice: {np.count_nonzero(~np.isnan(data[slicenum, :, :])) / data[slicenum, :, :].size:.4f}")
    print(f" Data Values: {data[slicenum, :, :][~np.isnan(data[slicenum, :, :])]}")
    print(f"\n==== Number of slices with all NANs ====")
    print(np.count_nonzero(np.all(np.isnan(data), axis=(1, 2))))
    print(f" Slices with all NANs: {np.where(np.all(np.isnan(data), axis=(1, 2)))[0]}")
    print (f"\n===== Slice wavelength ===== ")
    """

    # Print basic info from the header
    print("\n=== Header info ===")
    print(f"NAXIS: {header.get('NAXIS')}")
    print(f"NAXIS1: {header.get('NAXIS1')}, NAXIS2: {header.get('NAXIS2')}, NAXIS3: {header.get('NAXIS3')}")
    print(f"CTYPE1: {header.get('CTYPE1')}, CTYPE2: {header.get('CTYPE2')}, CTYPE3: {header.get('CTYPE3')}")
    print(f"CRVAL1/2/3: {header.get('CRVAL1')}, {header.get('CRVAL2')}, {header.get('CRVAL3')}")
    print(f"CRPIX1/2/3: {header.get('CRPIX1')}, {header.get('CRPIX2')}, {header.get('CRPIX3')}")
    print(f"CDELT1/2/3: {header.get('CDELT1')}, {header.get('CDELT2')}, {header.get('CDELT3')}")

    # Optionally print the full header
    print("\n=== Full header ===")
    print(repr(header))

#"""