from mpdaf.obj import Cube


from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from mpdaf.MUSE import FSFModel, MoffatModel2
from mpdaf.obj import Cube
from mpdaf.MUSE.fsf import get_images
import numpy as np
import logging


# ======================
# CONFIGURATION
# ======================

hst_path = "/cephfs/apatrick/musecosmos/dataproducts/hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits"
mosaic_wl_slice = '/cephfs/apatrick/musecosmos/scripts/aligned/mosaic_whitelight_nanmedian_all_new_full.fits'
cube_file = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits"
output_csv = "muse_hst_fsf_results.csv"

# ======================
# ARGUMENTS
# ======================

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FSF models for multiple stars in a MUSE cube")
    parser.add_argument('--hst_catalog', type=str, default=hst_path)
    parser.add_argument('--mosaic_wl_slice', type=str, default=mosaic_wl_slice)
    parser.add_argument('--muse_cube', type=str, default=cube_file)
    parser.add_argument('--output_csv', type=str, default=output_csv)
    return parser.parse_args()

# ======================
# STAR SELECTION
# ======================

def select_stars(hst_path, n_brightest=100):
    """Select the brightest star candidates from HST catalog."""
    hst = Table.read(hst_path).to_pandas()
    star_candidates = hst[hst['CLASS_STAR'] > 0.9].copy()
    star_candidates = star_candidates[star_candidates['ACS_F606W_FLUX'] > 0]
    brightest_stars = star_candidates.nlargest(n_brightest, 'ACS_F606W_FLUX')
    return brightest_stars

# ======================
# SOURCE VALIDATION
# ======================

def sources_in_mosaic(source_catalog, wcs, mosaic_data):
    sky_coords = SkyCoord(ra=source_catalog['RA'].values * u.deg,
                          dec=source_catalog['DEC'].values * u.deg)
    x_pix, y_pix = wcs.world_to_pixel(sky_coords)
    ny, nx = mosaic_data.shape
    in_bounds = (x_pix >= 0) & (x_pix < nx) & (y_pix >= 0) & (y_pix < ny)
    x_pix_in = x_pix[in_bounds].astype(int)
    y_pix_in = y_pix[in_bounds].astype(int)
    not_nan = ~np.isnan(mosaic_data[y_pix_in, x_pix_in])
    valid_mask = np.zeros(len(source_catalog), dtype=bool)
    valid_mask[in_bounds] = not_nan
    valid_sources = source_catalog[valid_mask].copy()
    valid_sources['x_pix'] = x_pix[valid_mask]
    valid_sources['y_pix'] = y_pix[valid_mask]
    print(f"{len(valid_sources)} valid sources found in the mosaic")
    return valid_sources

# ======================
# FSF FITTING
# ======================

def fit_fsf_for_source(cube, ra, dec, fwhmdeg=3, betadeg=2, plot=True):
    """Fit FSF model (Moffat) for one source at given RA, Dec."""
    try:
        fsf = MoffatModel2.from_starfit(cube, (dec, ra), fwhmdeg=fwhmdeg, betadeg=betadeg)
        params = {
            "RA": ra,
            "DEC": dec,
            "FWHM_poly": fsf.fit['fwhmpol'],
            "BETA_poly": fsf.fit['betapol'],
            "FWHM_median": np.median(fsf.fit['fwhmfit']),
            "BETA_median": np.median(fsf.fit['betafit']),
        }
        if plot:
            plot_fsf_fit(fsf, ra, dec)
        return params
    except Exception as e:
        print(f"[WARNING] FSF fit failed for RA={ra}, DEC={dec}: {e}")
        return None

# ======================
# PLOTTING
# ======================

def plot_fsf_fit(fsf, ra, dec):
    """Plot FSF fit quality (FWHM and BETA vs wavelength)."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fsf.fit['wave'], fsf.fit['fwhmfit'], 'ok', label='data')
    ax[0].plot(fsf.fit['wave'], fsf.fit['fwhmpol'], '-r', label='poly fit')
    ax[0].set_xlabel("Wavelength (Å)")
    ax[0].set_ylabel("FWHM")
    ax[0].legend()
    
    ax[1].plot(fsf.fit['wave'], fsf.fit['betafit'], 'ok', label='data')
    ax[1].plot(fsf.fit['wave'], fsf.fit['betapol'], '-r', label='poly fit')
    ax[1].set_xlabel("Wavelength (Å)")
    ax[1].set_ylabel("BETA")
    #ax[1].set_ylim(2, 3)
    ax[1].legend()
    
    fig.suptitle(f"FSF Fit at RA={ra:.5f}, DEC={dec:.5f}")
    plt.tight_layout()
    plt.savefig(f"fsf_fit_RA{ra:.5f}_DEC{dec:.5f}.png")
    plt.close()
    print(f"Saved FSF fit plot to /cephfs/apatrick/musecosmos/scripts/fsf_fit_RA{ra:.5f}_DEC{dec:.5f}.png")

# ======================
# FSF MODEL CREATION
# ======================

def get_cube_wave_range(cube_file):
    """Get min and max wavelength from a MUSE cube header."""
    with fits.open(cube_file) as hdul:
        # Check which HDU has the cube; usually 1 if multi-extension
        header = hdul[1].header if len(hdul) > 1 else hdul[0].header
        crval = header['CRVAL3']    # reference wavelength
        cdelt = header['CDELT3']    # step per pixel
        crpix = header['CRPIX3']    # reference pixel
        naxis3 = header['NAXIS3']   # number of spectral pixels
        wave_min = crval + (1 - crpix) * cdelt
        wave_max = crval + (naxis3 - crpix) * cdelt
    return wave_min, wave_max


def create_fsfmodel_from_df(df, cube_file):
    """
    Create a FSFModel from a DataFrame of FSF fits.
    Uses median FWHM and BETA across sources.
    """
    # Median polynomials across stars
    fwhm_polys = np.array(df['FWHM_poly'].tolist())  # shape (nstars, ncoeff)
    beta_polys = np.array(df['BETA_poly'].tolist())
    fwhm_median_poly = np.median(fwhm_polys, axis=0)
    beta_median_poly = np.median(beta_polys, axis=0)

    # Get wavelength range
    wave_min, wave_max = get_cube_wave_range(cube_file)
    print(f"Cube wavelength range: {wave_min:.2f} Å – {wave_max:.2f} Å")

    # Create Moffat model
    moffat = MoffatModel2(fwhm=fwhm_median_poly, beta=beta_median_poly)

    # Create FSFModel for the full cube wavelength range
    fsfmodel = FSFModel(model=moffat, wave=[wave_min, wave_max])

    return fsfmodel

# ======================
# MAIN PIPELINE
# ======================

def main():
    args = parse_args()

    # Step 1: Load cube
    print("Loading MUSE cube...")
    cube = Cube(args.muse_cube, mmap=True)

    # Step 2: Select bright stars
    print("Selecting stars from HST catalog...")
    stars = select_stars(args.hst_catalog)

    # Step 3: Validate which fall within mosaic
    print("Filtering sources inside mosaic...")
    with fits.open(args.mosaic_wl_slice) as hdul:
        mosaic_data = hdul[0].data
        wcs = WCS(hdul[0].header)
    valid_sources = sources_in_mosaic(stars, wcs, mosaic_data)

    # Step 4: Fit FSFs
    results = []
    print(f"\nFitting FSFs for {len(valid_sources)} sources...\n")

    for _, src in valid_sources.iterrows():
        res = fit_fsf_for_source(cube, src['RA'], src['DEC'], plot=True)
        if res is not None:
            results.append(res)

    # Step 5: Store results in a DataFrame
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved FSF results to {args.output_csv}")

    # Step 6: Print summary stats
    print("\n=== FSF Summary ===")
    print("Number of successful fits:", len(df))
    pd.set_option('display.max_rows', None)   # show all rows
    pd.set_option('display.max_columns', None)  # show all columns
    #print(df)  
    for col in ['FWHM_median', 'BETA_median']:
        print(f"{col}: mean={df[col].mean():.3f}, min={df[col].min():.3f}, max={df[col].max():.3f}")

    # Step 7: Create FSF model for ORIGIN
    print("\nCreating FSF model for the cube...")
    fsfmodel = create_fsfmodel_from_df(df, args.muse_cube)
    fsfmodel_file = "muse_fsfmodel.fits"
    fsfmodel.write(fsfmodel_file)
    print(f"Saved FSF model to {fsfmodel_file}")


# ======================
# ENTRY POINT
# ======================

if __name__ == "__main__":
    main()
