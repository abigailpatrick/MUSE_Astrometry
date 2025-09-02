from astropy.io import fits
#"""
# Path to your file
cube_file = "/cephfs/apatrick/musecosmos/scripts/mosaic.fits"

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

    # Print the full header
    print("\n=== Full header ===")
    print(repr(header))
"""

from astropy.io import fits

# Path to your 3D cube
cube_file = "/cephfs/apatrick/musecosmos/reduced_cubes/norm/DATACUBE_FINAL_Autocal3821806b_1_ZAP_norm.fits"

with fits.open(cube_file) as hdul:
    print("=== FITS file structure ===")
    hdul.info()

    # Usually 3D cubes are in the first extension (HDU 1), but check hdul.info()
    if len(hdul) > 1 and hdul[1].data is not None:
        hdu = hdul[1]
    else:
        hdu = hdul[0]  # fallback to primary HDU

    data = hdu.data
    header = hdu.header

    # Print the data shape
    print("\n=== Data shape ===")
    print(data.shape)  # (n_lambda, ny, nx) or (nz, ny, nx)

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

"""