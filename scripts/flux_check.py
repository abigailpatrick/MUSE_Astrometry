from html import parser
from astropy.table import Table
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import argparse
import matplotlib.pyplot as plt
from photutils import CircularAperture
from photutils.aperture import aperture_photometry

"""
Open hst catalog and print columns
Find sources in mosaic and restrict to this 
Restrict to stars (CLASS_STAR > 0.9)
Print/save column of the F606W fluxes for these sources
Load a 100 A cube from the full cube, either side of 6060 A
Create an image averaging this cube the Bunit is 1e-20 erg/s/cm2/A
Measure FLambda flux in 0.1, 0.3, 0.5 and 1 arcsec apertures for the stars
Convert these to fNu in microJy
Plot curve of growth for brightest 3 stars
Plot comparison of F606W fluxes from HST and MUSE
Print out first 10 rows of catalog with both fluxes




"""


# Inputs
hst_path = "/cephfs/apatrick/musecosmos/dataproducts/hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits"
mosaic_wl_slice = '/cephfs/apatrick/musecosmos/scripts/aligned/mosaic_whitelight_nanmedian_all_new_full.fits' # for shape
cube_file    = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits"
output_csv   = "muse_hst_flux_comparison.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Flux conservation check between HST and MUSE")

    parser.add_argument('--hst_catalog', type=str, default=hst_path,
                        help='Path to the HST catalog file')
    parser.add_argument('--mosaic_wl_slice', type=str, default=mosaic_wl_slice,
                        help='Path to the mosaic white-light slice FITS file')
    parser.add_argument('--muse_cube', type=str, default=cube_file,
                        help='Path to the MUSE cube file')
    parser.add_argument('--output_csv', type=str, default=output_csv,
                        help='Path to the output CSV file')
    args = parser.parse_args()

    return args

def select_stars(hst_path, n_brightest=300):
    """Select the brightest star candidates from HST catalog."""
    hst = Table.read(hst_path)
    hst = hst.to_pandas()

    print(f"\nTotal number of HST sources: {len(hst)}")

    # Select likely stars
    star_candidates = hst[hst['CLASS_STAR'] > 0.9].copy()
    print(f"Number of star candidates (CLASS_STAR > 0.9): {len(star_candidates)}")

    # Keep only those with positive flux
    star_candidates = star_candidates[star_candidates['ACS_F606W_FLUX'] > 0]

    # Select the N brightest by F606W flux
    brightest_stars = star_candidates.nlargest(n_brightest, 'ACS_F606W_FLUX').copy()
    print(f"Selected the {len(brightest_stars)} brightest stars by ACS_F606W_FLUX.")

    return brightest_stars



def sources_in_mosaic(source_catalog, wcs, mosaic_data, output_csv=None):
    """
    Identify sources from a catalog that fall within a mosaic and on valid pixels.

    Parameters
    ----------
    source_catalog : pandas.DataFrame
        Catalog with 'RA' and 'DEC' columns (in degrees).
    wcs : astropy.wcs.WCS
        WCS object of the mosaic.
    mosaic_data : 2D np.ndarray
        Image data of the mosaic.
    output_csv : str or None
        If provided, saves the valid sources to this CSV path.

    Returns
    -------
    valid_sources : pandas.DataFrame
        Subset of source_catalog within mosaic bounds and on valid pixels,
        with 'x_pix' and 'y_pix' columns added.
    """
    # Convert RA/Dec to SkyCoord
    sky_coords = SkyCoord(ra=source_catalog['RA'].values * u.deg,
                          dec=source_catalog['DEC'].values * u.deg,
                          frame='icrs')
    
    # Convert to pixel coordinates
    x_pix, y_pix = wcs.world_to_pixel(sky_coords)
    ny, nx = mosaic_data.shape
    
    # Mask for in-bounds coordinates
    in_bounds_mask = (x_pix >= 0) & (x_pix < nx) & (y_pix >= 0) & (y_pix < ny)
    x_pix_in = x_pix[in_bounds_mask].astype(int)
    y_pix_in = y_pix[in_bounds_mask].astype(int)
    
    # Mask out NaN pixels
    not_nan_mask = ~np.isnan(mosaic_data[y_pix_in, x_pix_in])
    
    # Combine masks
    valid_mask = np.zeros(len(source_catalog), dtype=bool)
    valid_mask[in_bounds_mask] = not_nan_mask
    
    # Filter valid sources
    valid_sources = source_catalog[valid_mask].copy()
    valid_sources['x_pix'] = x_pix[valid_mask]
    valid_sources['y_pix'] = y_pix[valid_mask]
    
    print(f"{len(valid_sources)} sources found in mosaic")
    
    # Save to CSV if requested
    if output_csv is not None:
        valid_sources.to_csv(output_csv, index=False)
        print(f"Saved valid sources to {output_csv}")
    
    return valid_sources

def extract_subcube(cube_file, central_wl=6060, delta_slices=100, output_file=None):
    """
    Extract a subcube from a MUSE cube centered around central_wl ± delta_slices.

    Parameters
    ----------
    cube_file : str
        Path to the full MUSE cube FITS file.
    central_wl : float
        Central wavelength in Angstroms.
    delta_slices : int
        Number of slices on each side of central wavelength to include.
    output_file : str or None
        If provided, saves the subcube to this FITS path.

    Returns
    -------
    subcube_data : np.ndarray
        Extracted subcube (shape: [nz, ny, nx]).
    subcube_header : astropy.io.fits.Header
        Header of the subcube with updated CRPIX3.
    """
    with fits.open(cube_file) as hdul:
        cube_data = hdul[0].data  # shape (NAXIS3, NAXIS2, NAXIS1) = (wave, y, x)
        header = hdul[0].header
        header_full = hdul[1].header  
        # Print header information
        print("Cube header info:")
        print(repr(header))
        print("Full header info:")
        print(repr(header_full))

        # Wavelength axis info
        CRVAL3 = header['CRVAL3']
        CRPIX3 = header['CRPIX3']
        CD3_3 = header['CD3_3']

        nz = cube_data.shape[0]
        wl_array = CRVAL3 + (np.arange(nz) + 1 - CRPIX3) * CD3_3

        # Find central slice index
        central_idx = np.argmin(np.abs(wl_array - central_wl))

        start_idx = max(0, central_idx - delta_slices)
        end_idx = min(nz, central_idx + delta_slices + 1)

        subcube_data = cube_data[start_idx:end_idx, :, :]

        # Update header
        subcube_header = header.copy()
        subcube_header['CRPIX3'] = CRPIX3 - start_idx

        if output_file is not None:
            fits.writeto(output_file, subcube_data, subcube_header, overwrite=True)
            print(f"Saved subcube to {output_file} with shape {subcube_data.shape}")

    return subcube_data, subcube_header

def add_muse_pixel_coords(valid_sources, wcs):
    """
    Add MUSE pixel coordinates to a catalog using RA/DEC and WCS.
    """
    sky_coords = SkyCoord(ra=valid_sources['RA'].values*u.deg,
                          dec=valid_sources['DEC'].values*u.deg)
    x_pix, y_pix = wcs.world_to_pixel(sky_coords)
    valid_sources = valid_sources.copy()
    valid_sources['x_pix'] = x_pix
    valid_sources['y_pix'] = y_pix
    return valid_sources

def cube_to_image(subcube_data, subcube_header, output_file=None):
    """
    Create a 2D image by averaging a MUSE subcube along the wavelength axis.

    Parameters
    ----------
    subcube_data : np.ndarray
        Subcube of shape (nwavelength, ny, nx).
    subcube_header : astropy.io.fits.Header
        Header from the subcube, for WCS and saving.
    output_file : str or None
        If provided, save the 2D image as a FITS file.

    Returns
    -------
    image_2d : np.ndarray
        2D image averaged over wavelength.
    image_header : astropy.io.fits.Header
        Header for the 2D image (wavelength axis removed).
    """

    # Average along wavelength axis
    image_2d = np.nanmean(subcube_data, axis=0)  # shape (ny, nx)

    # Create a new header without the wavelength axis
    image_header = subcube_header.copy()
    for key in ['CRVAL3', 'CRPIX3', 'CDELT3', 'CD3_3', 'CTYPE3', 'CUNIT3']:
        if key in image_header:
            del image_header[key]

    if output_file is not None:
        fits.writeto(output_file, image_2d, image_header, overwrite=True)
        print(f"Saved 2D image to {output_file} with shape {image_2d.shape}")

    return image_2d, image_header



def measure_fluxes_fixed_scale(image_2d, valid_sources, apertures_arcsec, pixel_scale=0.2):
    """
    Measure FLambda fluxes for sources in a 2D image using circular apertures
    with a fixed pixel scale.

    Parameters
    ----------
    image_2d : np.ndarray
        2D image in FLambda (BUNIT = 1e-20 erg/s/cm²/Å).
    valid_sources : pandas.DataFrame
        Sources with 'x_pix' and 'y_pix' columns.
    apertures_arcsec : list of float
        Aperture radii in arcseconds.
    pixel_scale : float
        Pixel scale in arcsec/pixel.

    Returns
    -------
    flux_table : pandas.DataFrame
        Input sources DataFrame with additional columns:
        'FLambda_0.1', 'FLambda_0.3', 'FLambda_0.5', 'FLambda_1.0'
    """
    flux_table = valid_sources.copy()

    for ap in apertures_arcsec:
        radius_pix = ap / pixel_scale
        colname = f"FLambda_{ap:.1f}"
        fluxes = []
        for x, y in zip(valid_sources['x_pix'], valid_sources['y_pix']):
            aperture = CircularAperture((x, y), r=radius_pix)
            phot_table = aperture_photometry(image_2d, aperture)
            fluxes.append(phot_table['aperture_sum'][0])
        flux_table[colname] = fluxes

    return flux_table

def flambda_to_fnu_microjy(flambda, wavelength_angstrom):
    """
    Convert FLambda [erg/s/cm²/Å] to Fnu [microJy] at a given wavelength.

    Parameters
    ----------
    flambda : float or np.ndarray
        FLambda in erg/s/cm²/Å
    wavelength_angstrom : float
        Wavelength in Angstroms

    Returns
    -------
    fnu_microjy : float or np.ndarray
        Fnu in microJy
    """
    c = 2.99792458e18  # speed of light in Å/s
    fnu = flambda * (wavelength_angstrom**2) / c  # erg/s/cm²/Hz
    fnu_microjy = fnu * 1e6 / 1e-23  # convert to μJy
    # Simplify: 1 erg/s/cm²/Hz = 1e23 Jy, 1 Jy = 1e6 μJy
    return fnu_microjy

def plot_curve_of_growth(flux_table, source_idx, apertures_arcsec, 
                         flux_column_prefix='Fnu', flux_unit='μJy'):
    """
    Plot the curve of growth for a single source from a flux table.

    Parameters
    ----------
    flux_table : pandas.DataFrame
        Table containing flux measurements for different apertures.
        Must have columns like 'Fnu_0.1_uJy', 'Fnu_0.3_uJy', etc.
    source_idx : int
        Index of the source in the table to plot.
    apertures_arcsec : list of float
        Aperture radii in arcseconds.
    flux_column_prefix : str
        Prefix of the flux columns in the table ('FLambda' or 'Fnu').
    flux_unit : str
        Units for the flux for labeling the plot.
    """
    source_row = flux_table.iloc[source_idx]
    
    fluxes = []
    for ap in apertures_arcsec:
        colname = f"{flux_column_prefix}_{ap:.1f}_uJy" if flux_column_prefix=='Fnu' else f"{flux_column_prefix}_{ap:.1f}"
        fluxes.append(source_row[colname])
    
    plt.figure(figsize=(6,5))
    plt.plot(apertures_arcsec, fluxes, marker='o', linestyle='-', color='tab:blue')
    plt.xlabel("Aperture radius [arcsec]")
    plt.ylabel(f"Flux [{flux_unit}]")
    plt.xlim(0, max(apertures_arcsec)*1.1)
    plt.title(f"Curve of growth for source index {source_idx}")
    plt.grid(True)
    plt.show()
    plt.savefig(f"curve_of_growth_source_{source_idx}.png", dpi=300)
    print(f"Saved curve of growth plot to curve_of_growth_source_{source_idx}.png")

import matplotlib.pyplot as plt
import numpy as np

def compare_hst_muse_fluxes(flux_table, aperture_radius=1.0):
    """
    Compare HST (F606W) and MUSE fluxes (in μJy) for all sources.
    
    Parameters
    ----------
    flux_table : pandas.DataFrame
        Table containing ACS_F606W_FLUX (μJy) and MUSE Fν columns (Fnu_<radius>_uJy).
    aperture_radius : float
        Aperture radius (arcsec) used for MUSE flux comparison.
    """

    muse_col = f"Fnu_{aperture_radius:.1f}_uJy"
    if muse_col not in flux_table.columns:
        raise ValueError(f"{muse_col} not found in flux_table columns.")

    fnu_hst = flux_table["ACS_F606W_FLUX"]
    fnu_muse = flux_table[muse_col]

    # Remove invalid values
    mask = np.isfinite(fnu_hst) & np.isfinite(fnu_muse)
    fnu_hst, fnu_muse = fnu_hst[mask], fnu_muse[mask]

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(fnu_hst, fnu_muse, color='dodgerblue', alpha=0.7, edgecolor='k', s=50)

    # 1:1 line
    lims = [
        min(fnu_hst.min(), fnu_muse.min()),
        max(fnu_hst.max(), fnu_muse.max())
    ]
    plt.plot(lims, lims, 'k--', lw=1.2, label="1:1 line")

    plt.xlabel("HST F606W Flux (μJy)", fontsize=12)
    plt.ylabel(f"MUSE Flux (μJy, {aperture_radius:.1f}″ aperture)", fontsize=12)
    plt.title("HST vs MUSE Flux Comparison", fontsize=13)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"hst_muse_flux_comparison_{aperture_radius:.1f}arcsec.png", dpi=300)
    print(f"Saved flux comparison plot to hst_muse_flux_comparison_{aperture_radius:.1f}arcsec.png")

    # Summary stats
    ratio = fnu_muse / fnu_hst
    print(f"\nFlux ratio (MUSE / HST) for {aperture_radius:.1f}″ aperture:")
    print(f"  Mean   = {np.nanmean(ratio):.3f}")
    print(f"  Median = {np.nanmedian(ratio):.3f}")
    print(f"  Std    = {np.nanstd(ratio):.3f}")
    print(f"  N      = {len(ratio)} sources")

    return ratio





def main():

    args = parse_args()

    # Select star candidates from HST catalog
    star_candidates = select_stars(args.hst_catalog)

    with fits.open(args.mosaic_wl_slice) as hdul:
        mosaic_data = hdul[0].data
        wcs = WCS(hdul[0].header)

    valid_sources = sources_in_mosaic(star_candidates, wcs, mosaic_data)
    
    # Restrict to flux column only 
    valid_sources = valid_sources[['RA', 'DEC', 'ACS_F606W_FLUX', 'ACS_F606W_FLUXERR']].copy()

    # Compute pixel positions for MUSE image
    valid_sources = add_muse_pixel_coords(valid_sources, wcs)

    print(f"Measuring fluxes for {len(valid_sources)} star candidates in mosaic slice.")

    # Load a 100 A cube around 6060 A
    subcube, subcube_hdr = extract_subcube(args.muse_cube, central_wl=6060, delta_slices=100,
                                       output_file="subcube_6060A.fits")
    print(subcube.shape)  # should be roughly (201, 2027, 2540)

    # Create a 2D image by averaging this cube
    image_2d, image_hdr = cube_to_image(subcube, subcube_hdr, output_file="subcube_avg_6060A.fits")
    print(image_2d.shape)  # should be (2027, 2540)

    apertures = np.arange(0.2, 2.0, 0.1)  # arcsec
    # Measure fluxes in fixed apertures (assuming 0.2 arcsec/pixel scale)
    pixel_scale = 0.2  # arcsec/pixel
    flux_table = measure_fluxes_fixed_scale(image_2d, valid_sources,
                                        apertures_arcsec=apertures,
                                        pixel_scale=pixel_scale)
    #print(flux_table.head())

    # Convert FLambda to Fnu in microJy at 6060 A
    central_wl = 6060  # Angstroms, wavelength of the MUSE slice

    for ap in apertures:
        colname = f"FLambda_{ap:.1f}"
        newcol = f"Fnu_{ap:.1f}_uJy"
        flux_table[newcol] = flambda_to_fnu_microjy(flux_table[colname] * 1e-20, central_wl)

    print(flux_table.head())

    # Plot growth curve 
    plot_curve_of_growth(flux_table, source_idx=8, 
                     apertures_arcsec= apertures, 
                     flux_column_prefix='Fnu', flux_unit='μJy')
    
    # Compare HST and MUSE fluxes
    compare_hst_muse_fluxes(flux_table, aperture_radius=1.0)

    
if __name__ == "__main__":
    main()



""".   
with fits.open(mosaic_wl_slice) as hdul:
    mosaic_data = hdul[0].data
    wcs = WCS(hdul[0].header)

valid_sources = sources_in_mosaic(star_candidates, wcs, mosaic_data)

# F606W fluxes for the valid sources
f606_fluxes = valid_sources['ACS_F606W_FLUX'].values
f606_flux_errors = valid_sources['ACS_F606W_FLUXERR'].values

# print first 10 fluxes
print("F606W fluxes of first 10 sources in mosaic:")
print(f606_fluxes[:10])

slice_fits = '/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full/mosaic_slice_1048.fits' # 6060A
aperture_radius_arcsec = 0.5  # 0.5 arcsec radius

with fits.open(slice_fits) as hdul:
    slice_data = hdul[0].data
    slice_wcs = WCS(hdul[0].header)

# Compute pixel scale from CD matrix
cd = slice_wcs.wcs.cd
pix_scale = np.sqrt(np.abs(cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0])) * 3600.0  # arcsec/pixel
ap_radius_pix = aperture_radius_arcsec / pix_scale
print(f"Using aperture radius of {ap_radius_pix:.2f} pixels (~{aperture_radius_arcsec} arcsec)")

slice_wcs_2d = slice_wcs.celestial

# Convert catalog positions to pixels
sky_coords = SkyCoord(ra=valid_sources['RA'].values*u.deg,
                      dec=valid_sources['DEC'].values*u.deg)
x_pix, y_pix = slice_wcs_2d.world_to_pixel(sky_coords)


positions = np.transpose([x_pix, y_pix])
apertures = CircularAperture(positions, r=ap_radius_pix)
phot_table = aperture_photometry(slice_data, apertures)

# Add fluxes to catalog
valid_sources['flux_6060'] = phot_table['aperture_sum']

print(f"Measured flux for {len(valid_sources)} star candidates in mosaic slice")

valid_sources.to_csv('star_fluxes_6060A.csv', index=False)
print("Saved fluxes to star_fluxes_6060A.csv")

# Print both flux measures for 606
print("\nFirst 10 sources with F606W and 6060A fluxes:")
print(valid_sources[['ACS_F606W_FLUX', 'flux_6060']].head(10))

# Constants
c_ang_per_s = 2.998e18  # speed of light in Å/s
lambda_f606 = 6060.0     # central wavelength of F606W in Å

# Convert ACS flux (µJy) to erg/s/cm²/Å
# 1 µJy = 1e-32 erg/s/cm²/Hz
acs_flux_microjy = valid_sources['ACS'].values
f_nu_erg = acs_flux_microjy * 1e-32
f_lambda = f_nu_erg * c_ang_per_s / (lambda_f606**2)

# Add new column to DataFrame
valid_sources['FLUX_APER_8_F606W_cgs'] = f_lambda

# Convert MUSE slice fluxes to erg/s/cm²/Å
valid_sources['flux_6060_cgs'] = valid_sources['flux_6060'] * 1e-20

# Print comparison
print(valid_sources[['FLUX_APER_8_F606W_cgs', 'flux_6060_cgs']].head(10))

"""