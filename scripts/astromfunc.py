# imports
import os
import numpy as np
import matplotlib.pyplot as plt

from numpy import ma, loadtxt
from scipy import integrate, ndimage
from scipy.ndimage import binary_erosion
from scipy.interpolate import interp1d

from astropy import units as u
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn, hstack, vstack, pprint
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
from astropy.convolution import convolve, Gaussian2DKernel, Moffat2DKernel
from astropy.stats import sigma_clip
from astropy.visualization import (
    SqrtStretch,
    PercentileInterval,
    ImageNormalize,
    simple_norm
)

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

from photutils import aperture_photometry
from photutils.aperture import CircularAperture, ApertureStats
from photutils.segmentation import (
    detect_sources,
    deblend_sources,
    make_2dgaussian_kernel,
    SourceCatalog,
    SourceFinder
)
from photutils.background import Background2D, MedianBackground

from mpdaf.obj import Cube, Image


# Plotting functions removed for speed and space but can be added back in for diagnostics

def get_wcs_info(im_muse, wcs_muse):
    """
    Extract WCS and image information from a MUSE image.

    Parameters
    ----------
    im_muse : np.ndarray
        The data array of the MUSE image (e.g., after applying a bandpass filter).

    wcs_muse : astropy.wcs.WCS
        The WCS information extracted from the MUSE image.

    Returns
    -------
    shape_muse : tuple
        The shape of the MUSE image (ny, nx).

    wcs_muse : astropy.wcs.WCS
        The WCS object associated with the MUSE image.

    central_pixel : tuple
        The (y, x) coordinates of the central pixel in the image.

    ra_muse : float
        The right ascension (RA) of the central pixel, in degrees.

    dec_muse : float
        The declination (Dec) of the central pixel, in degrees.

    width_deg : float
        The image width in degrees (RA axis).

    height_deg : float
        The image height in degrees (Dec axis).

    fwhm_nyquist : astropy.units.quantity.Quantity
        The FWHM in arcseconds needed for Nyquist sampling 
        (2 *pixel scale in arcseconds).

    pixscale : astropy.units.quantity.Quantity
        The pixel scale of the image in arcseconds.

    """
    # Image shape and central pixel
    shape_muse = im_muse.shape
    central_pixel = (shape_muse[0] // 2, shape_muse[1] // 2)

    # Central world coordinates (RA, Dec)
    ra_muse, dec_muse = wcs_muse.all_pix2world(central_pixel[1], central_pixel[0], 0)

    # Width and height in degrees (from opposite corners)
    ra1, dec1 = wcs_muse.all_pix2world(0, 0, 0)
    ra2, dec2 = wcs_muse.all_pix2world(shape_muse[1], shape_muse[0], 0)
    width_deg = abs(ra2 - ra1)
    height_deg = abs(dec2 - dec1)

    # Pixel scale from CD matrix (assumes square pixels)
    cd1_1 = abs(wcs_muse.wcs.cd[0, 0])
    pixscale = (cd1_1 * u.deg).to(u.arcsec)

    # Nyquist sampling FWHM
    fwhm_nyquist = 2 * pixscale
    print (" Ran get_wcs_info ")

    return (
        shape_muse,
        wcs_muse,
        central_pixel,
        ra_muse,
        dec_muse,
        width_deg,
        height_deg,
        fwhm_nyquist,
        pixscale
    )


def source_catalog(data, wcs_muse, photplam, pixel_scale, compact_only=True, min_sep_arcsec=0.0, 
                   npixels=10, radii=[3.0, 4.0, 5.0], 
                   fout='outputs/source_catalog_MUSE.fits'):
    """
    Detects and catalogs sources in a 2D MUSE  image.

    Parameters
    ----------
    data : np.ndarray
        The 2D bandpass-convolved image array (e.g. from mpdaf).
    wcs_muse : astropy.wcs.WCS
        WCS information for coordinate conversion.
    photplam : float
        Central wavelength of the bandpass (for Jy conversion).
    pixel_scale : float
        Image pixel scale in arcsec/pixel.
    compact_only : bool, optional
        If True, returns only compact sources. Default is True.
    min_sep_arcsec : float, optional
        Minimum separation between sources in arcseconds. Default is 0.0.
    npixels : int, optional
        Minimum number of connected pixels to define a source.
    radii : list of float, optional
        List of aperture radii (pixels) for photometry.
    fout : str, optional
        Output FITS filename for the catalog.

    Returns
    -------
    data : np.ndarray
        Background-subtracted image.
    tbl : astropy.table.Table
        Table of source properties and photometry.
    segment_map : np.ndarray
        Initial segmentation map.
    segm_deblend : np.ndarray
        Deblended segmentation map.
    cat : photutils.segmentation.SourceCatalog
        Full source catalog object.
    aperture_phot_tbl : astropy.table.Table
        Raw aperture photometry results.
    extent : list
        Image extent in RA/Dec coordinates for plotting.
    """

    # --- Clean and pre-process data ---
    data = np.where(data == 0.0, np.nan, data)

    # Erode edges to remove noise near boundaries
    erosion_mask = ndimage.binary_erosion(data, iterations=20)
    data = np.where(erosion_mask, data, np.nan)
    print(f"Valid pixels after erosion: {np.sum(~np.isnan(data))}")

    # Mask non-finite or extreme values
    bad_vals = (~np.isfinite(data)) | (data < -100) | (data > 100)
    data_clean = np.copy(data)
    data_clean[bad_vals] = np.nan

    # --- Background estimation and subtraction ---
    bkg = Background2D(
        data_clean, (25, 25), mask=bad_vals, filter_size=(3, 3),
        bkg_estimator=MedianBackground()
    )
    data[bad_vals] = np.nan
    data -= bkg.background

    print("Background median:", np.nanmedian(bkg.background))
    print("Background RMS:", np.nanmedian(bkg.background_rms))
    print(f"Post-background data: min={np.nanmin(data)}, max={np.nanmax(data)}")

    # --- Source detection ---
    threshold = 2 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    segment_map = detect_sources(convolved_data, threshold, npixels)
    segm_deblend = deblend_sources(
        convolved_data, segment_map, npixels,
        nlevels=32, contrast=0.001, progress_bar=False
    )

    finder = SourceFinder(npixels, progress_bar=True)
    segment_map = finder(convolved_data, threshold)

    # --- Build source catalog ---
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data, wcs=wcs_muse)

    # Filter out sources with invalid centroids
    valid_mask = ~(np.isnan(cat.xcentroid) | np.isnan(cat.ycentroid))
    cat = cat[valid_mask]

    # --- Tabulate source properties ---
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'

    tbl['kron_radius'] = cat.kron_radius
    tbl['fwhm'] = cat.fwhm
    tbl['semi_major_axis'] = cat.semimajor_sigma
    tbl['semi_minor_axis'] = cat.semiminor_sigma

    # Aperture area (approximate elliptical Kron aperture)
    tbl['aperture_area'] = (
        np.pi * tbl['semi_major_axis'] * tbl['semi_minor_axis'] * tbl['kron_radius']**2
    )

    # Convert Kron radius to arcsec
    tbl['kron_radius_arcsec'] = tbl['kron_radius'] * pixel_scale

    # --- Aperture photometry ---
    positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
    apertures = [CircularAperture(positions, r=r) for r in radii]
    aperture_phot_tbl = aperture_photometry(data, apertures)

    for i, r in enumerate(radii):
        tbl[f'aperture_sum_{r}'] = aperture_phot_tbl[f'aperture_sum_{i}']


    # --- Flux unit conversion (to uJy) ---
    flux_erg = tbl['kron_flux'] * 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
    flux_jy = flux_erg.to(u.Jy, equivalencies=u.spectral_density(photplam * u.AA))
    tbl['kron_flux_uJy'] = flux_jy.to(u.microjansky)

    # --- Compactness filtering ---
    fwhm_arcsec = tbl['fwhm']
    tbl['is_compact'] = fwhm_arcsec.value < 8.0
    

    if compact_only:
        pre_filter_len = len(tbl)
        tbl = tbl[tbl['is_compact']]
        cat = cat[fwhm_arcsec.value < 8.0]
        print(f"Filtered compact sources: {pre_filter_len} -> {len(tbl)}")

    # --- World coordinate conversion ---
    ra, dec = wcs_muse.all_pix2world(tbl['xcentroid'], tbl['ycentroid'], 0)
    sky_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    tbl['RA'] = sky_coords.ra
    tbl['Dec'] = sky_coords.dec

    # --- Minimum separation filtering ---
    if min_sep_arcsec > 0:
        coords = SkyCoord(ra=tbl['RA'], dec=tbl['Dec'], unit='deg')

        idx1, idx2, sep2d, _ = search_around_sky(coords, coords, min_sep_arcsec * u.arcsec)

        # Remove self-matches
        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]

        # Mark indices to remove (all sources involved in close pairs)
        bad_indices = np.unique(np.concatenate([idx1, idx2]))

        pre_filter_len = len(tbl)
        keep_mask = np.ones(len(tbl), dtype=bool)
        keep_mask[bad_indices] = False

        tbl = tbl[keep_mask]
        cat = cat[keep_mask]

        print(f"Filtered close sources (<{min_sep_arcsec}\" separation): {pre_filter_len} -> {len(tbl)}")

    # --- Filter sources within radius of image center (increased strictness from boundary erosion) ---
    center_x = data.shape[1] / 2
    center_y = data.shape[0] / 2

    dx = tbl['xcentroid'] - center_x
    dy = tbl['ycentroid'] - center_y
    distance_pixels = np.sqrt(dx**2 + dy**2)
    
    edge_filtering_radii = 140
    within_radius_mask = distance_pixels.value <= edge_filtering_radii
    pre_filter_len = len(tbl)
    tbl = tbl[within_radius_mask]
    cat = cat[within_radius_mask]

    print(f"Filtered by radius (<= {edge_filtering_radii} pixels from center): {pre_filter_len} -> {len(tbl)}")


    # --- Compute image extent in RA/Dec ---
    ny, nx = data.shape
    pixel_corners = np.array([[0, 0], [nx, 0], [nx, ny], [0, ny]])
    world_corners = wcs_muse.all_pix2world(pixel_corners, 0)

    ra_vals = world_corners[:, 0]
    dec_vals = world_corners[:, 1]

    ra_min, ra_max = ra_vals.min(), ra_vals.max()
    dec_min, dec_max = dec_vals.min(), dec_vals.max()

    extent = [ra_max, ra_min, dec_min, dec_max] # For correct RA order in imshow

    # --- Save catalog ---
    
    tbl.write(fout, format='fits', overwrite=True)

    print(f"Ran source_catalog. Source catalog saved to {fout}")

    return data, tbl, segment_map, segm_deblend, cat, aperture_phot_tbl, extent



def create_cutout(image_data, wcs, width_deg, height_deg, ra_center, dec_center, fout=None):
    """
    Create a WCS-preserving cutout from an image centered at a given RA/Dec.

    Parameters
    ----------
    image_data : 2D ndarray
        The full image data (e.g., HST image).
    wcs : WCS
        WCS of the image.
    width_deg : float
        Width of the cutout in degrees.
    height_deg : float
        Height of the cutout in degrees.
    ra_center : float
        Right Ascension of the cutout center (degrees).
    dec_center : float
        Declination of the cutout center (degrees).
    fout : str or None
        If set, the cutout will be written to this FITS file.

    Returns
    -------
    cutout : Cutout2D object
    cutout.wcs : WCS object
    """

    position = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)
    size = (height_deg * u.deg, width_deg * u.deg)

    cutout = Cutout2D(image_data, position, size, wcs=wcs, mode='trim')

    if fout:
        hdu = fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header())
        hdu.writeto(fout, overwrite=True)

    print ("Ran create_cutout")

    return cutout.data, cutout.wcs


def convolve_image(image_data_hst, fwhm, gamma):
    """
    Convolve the HST image with a Moffat kernel.

    The Moffat kernel is used to simulate the point spread function (PSF) of an imaging system. Convolution with this kernel helps blur the image, 
    which is useful for simulating the effects of atmospheric seeing or telescope optics.

    Parameters
    ----------
    image_data_hst : np.ndarray
        The data array of the HST image.

    fwhm : float
        The full width at half maximum (FWHM) of the Moffat kernel. This defines the scale of the kernel's spatial extent.

    gamma : float
        The gamma parameter of the Moffat kernel. It controls the "sharpness" of the kernel's profile. Higher values of gamma result in a narrower kernel.

    Returns
    -------
    convolved_image : np.ndarray
        The convolved image.


    """
    # Calculate the alpha parameter of the Moffat kernel based on FWHM and gamma
    alpha = fwhm / (2 * np.sqrt(2**(1/gamma) - 1))  

    # Create the Moffat kernel using the calculated alpha and provided gamma
    kernel_M = Moffat2DKernel(gamma=gamma, alpha=alpha)

    # Convolve the HST image with the Moffat kernel
    convolved_image = convolve(image_data_hst, kernel_M)

    print("Ran convolve_image")

    return convolved_image

def source_catalog_HST(data, wcs_hst, photflam, photplam, radii, npixels=10, fout='outputs/source_catalog_HST.fits'):
    """  
    Generates a source catalog from an HST bandpass image.

    Parameters
    ----------
    data : np.ndarray
        The image data (2D array).
    
    wcs_hst : astropy.wcs.WCS
        WCS information from the HST image.
    
    photflam : float
        The flux calibration factor.
    
    photplam : float
        The central wavelength of the filter used.
    
    radii : list of float
        Aperture radii for aperture photometry.
    
    npixels : int, optional
        Minimum source size (default is 10).
    
    fout : str, optional
        Output FITS file path for the catalog (default is 'outputs/source_catalog_HST.fits').

    Returns
    -------
    data : np.ndarray
        The background-subtracted image data.
    
    tbl : astropy.table.Table
        The source catalog table.
    
    segment_map : np.ndarray
        The segmentation map of the image.
    
    segm_deblend : np.ndarray
        The deblended segmentation map.
    
    cat : photutils.segmentation.SourceCatalog
        The source catalog object.
    
    aperture_phot_tbl : astropy.table.Table
        Table containing aperture photometry results for each radius.
    """
    
    # Background subtraction using median background estimator
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (30, 30), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data -= bkg.background  # Subtract background

    # Threshold for source detection (3-sigma above background)
    threshold = 3 * bkg.background_rms

    # Convolve data with a 2D Gaussian kernel
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    data = convolve(data, kernel)

    # Detect sources in the convolved data
    segment_map = detect_sources(data, threshold, npixels)

    # Deblend sources
    segm_deblend = deblend_sources(data, segment_map, npixels, nlevels=32, contrast=0.001, progress_bar=False)

    # Create a source catalog
    cat = SourceCatalog(data, segm_deblend, convolved_data=data, wcs=wcs_hst)

    # Convert the catalog to an Astropy Table
    tbl = cat.to_table()

    # Set formatting for certain columns
    tbl['xcentroid'].info.format = '.2f'
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'

    # Add FWHM to the table
    tbl['fwhm'] = cat.fwhm

    # Perform aperture photometry for each radius
    positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
    apertures = [CircularAperture(positions, r=r) for r in radii]
    aperture_phot_tbl = aperture_photometry(data, apertures)

    # Add aperture photometry results to the table
    for i, r in enumerate(radii):
        colname = f'aperture_sum_{r}'
        tbl[colname] = aperture_phot_tbl[f'aperture_sum_{i}']

    # Convert pixel coordinates to RA and Dec
    ra, dec = wcs_hst.all_pix2world(tbl['xcentroid'], tbl['ycentroid'], 0)
    sky_coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    # Add RA and Dec to the table
    tbl['RA'] = sky_coords.ra
    tbl['Dec'] = sky_coords.dec

    # Flux conversion to microjansky for each aperture
    for i, r in enumerate(radii):
        aperture_col = f'aperture_sum_{r}'
        arr_es = tbl[aperture_col] * (u.electron / u.s)
        arr_pflam = arr_es * (photflam * u.erg / u.cm**2 / u.AA / u.electron)
        arr_jy = arr_pflam.to(u.Jy, equivalencies=u.spectral_density(photplam * u.AA))
        arr_ujy = arr_jy.to(u.microjansky)
        tbl[f'{aperture_col}_uJy'] = arr_ujy

    # Convert Kron flux to microjansky
    aperture_col = 'kron_flux'
    arr_es = tbl[aperture_col] * (u.electron / u.s)
    arr_pflam = arr_es * (photflam * u.erg / u.cm**2 / u.AA / u.electron)
    arr_jy = arr_pflam.to(u.Jy, equivalencies=u.spectral_density(photplam * u.AA))
    arr_ujy = arr_jy.to(u.microjansky)
    tbl[f'{aperture_col}_uJy'] = arr_ujy

    # --- magnitude limit filtering ---
  
    # Convert Kron flux to AB magnitude
    kron_fluxes_ujy = tbl['kron_flux_uJy']
    with np.errstate(divide='ignore', invalid='ignore'):
        kron_mags_ab = -2.5 * np.log10(kron_fluxes_ujy.value) + 23.9

    # Add AB mag column to the table for reference
    tbl['kron_mag_AB'] = kron_mags_ab
    
    # Filter based on AB magnitude threshold (e.g., mag < 27)
    mag_limit = 27.0
    keep_mask = kron_mags_ab < mag_limit

    print(f"Sources before filtering: {len(tbl)}")
    tbl = tbl[keep_mask]
    print(f"Sources after filtering (mag < {mag_limit}): {len(tbl)}")
    print(f"Brightest source: {np.nanmin(kron_mags_ab):.2f} mag")
    print(f"Faintest retained source: {np.nanmax(tbl['kron_mag_AB']):.2f} mag")

    # Save the catalog as a FITS file
    tbl.write(fout, format='fits', overwrite=True)

    # Compute extent in RA/Dec for WCS plotting
    ny, nx = data.shape
    pixel_corners = np.array([[0, 0], [nx, 0], [nx, ny], [0, ny]])
    world_corners = wcs_hst.all_pix2world(pixel_corners, 0)

    ra_vals = world_corners[:, 0]
    dec_vals = world_corners[:, 1]

    ra_min, ra_max = ra_vals.min(), ra_vals.max()
    dec_min, dec_max = dec_vals.min(), dec_vals.max()

    extent_hst = [ra_max, ra_min, dec_min, dec_max]  # For imshow in RA/Dec

    print(f"Ran source_catalog_HST. Source catalog saved to {fout}")
    return data, tbl, segment_map, segm_deblend, cat, aperture_phot_tbl, extent_hst


def crossmatch_catalogs(catalog1, catalog2, tolerance_arcsec=2.0):
    """
    Crossmatches two catalogs of sources based on their RA and Dec coordinates.

    Parameters
    ----------
    catalog1 : astropy.table.Table  
        The first catalog of sources.
    
    catalog2 : astropy.table.Table
        The second catalog of sources.

    tolerance_arcsec : float
        The matching tolerance in arcseconds. Default is 2.0.

    Returns
    -------
    matched_catalog : astropy.table.Table
        A new catalog containing only the matched sources.
    """
    
    # Extract RA and Dec columns (case-insensitive)
    ra1 = catalog1[[col for col in catalog1.colnames if col.lower() == 'ra'][0]]
    dec1 = catalog1[[col for col in catalog1.colnames if col.lower() == 'dec'][0]]
    
    ra2 = catalog2[[col for col in catalog2.colnames if col.lower() == 'ra'][0]]
    dec2 = catalog2[[col for col in catalog2.colnames if col.lower() == 'dec'][0]]
    
    # Create SkyCoord objects
    coords1 = SkyCoord(ra1, dec1, unit='deg')
    coords2 = SkyCoord(ra2, dec2, unit='deg')
    
    # Perform the crossmatch
    idx, d2d, _ = match_coordinates_sky(coords1, coords2)
    
    # Convert tolerance to degrees
    tolerance = tolerance_arcsec * u.arcsec
    
    # Find matches within the specified tolerance
    matched_mask = d2d <= tolerance
    
    # Get indices of matched sources in both catalogs
    idx1 = np.arange(len(catalog1))[matched_mask]
    idx2 = idx[matched_mask]
    
    # Extract the matched rows from both catalogs
    matched_catalog1 = catalog1[matched_mask]
    matched_catalog2 = catalog2[idx2]
    
    # Combine the catalogs with prefixed field names for uniqueness
    combined_dtype = [(f'cat1_{name}', dtype) for name, dtype in matched_catalog1.dtype.descr]
    combined_dtype += [(f'cat2_{name}', dtype) for name, dtype in matched_catalog2.dtype.descr if f'cat2_{name}' not in combined_dtype]
    
    # Create a structured array for the combined catalog
    combined_catalog = np.empty(len(matched_catalog1), dtype=combined_dtype)

    
    # Populate the combined catalog with data from both matched catalogs
    for name in matched_catalog1.dtype.names:
        combined_catalog[f'cat1_{name}'] = matched_catalog1[name]
    for name in matched_catalog2.dtype.names:
        combined_catalog[f'cat2_{name}'] = matched_catalog2[name]

    
    # Convert the combined structured array back to an Astropy Table

    print ("ran crossmatch_catalogs")
    return Table(combined_catalog)


def add_offsets(muse_file,matched_catalog, log_file,pixel_scale=0.2):
    """
    Adds columns to the matched catalog with the RA and Dec offsets between the two catalogs.
    It also calculates and saves the median RA and Dec offsets.

    Parameters
    ----------
    muse_file : str
        The path to the MUSE FITS file (used for logging).

    matched_catalog : astropy.table.Table
        The matched catalog of sources.

    log_file : str
        The path to the log file where median offsets will be saved.

    pixel_scale : float, optional
        The pixel scale in arcseconds per pixel. Default is 0.2 arcsec/pixel.

    Returns
    -------
    matched_catalog : astropy.table.Table
        The matched catalog with additional columns for RA and Dec offsets.
    """
    # Calculate the RA and Dec offsets
    ra_offset = matched_catalog['cat1_RA'] - matched_catalog['cat2_RA']
    dec_offset = matched_catalog['cat1_Dec'] - matched_catalog['cat2_Dec']

    # Add the offsets to the matched catalog
    matched_catalog['RA_offset'] = ra_offset
    matched_catalog['Dec_offset'] = dec_offset

    # Convert the offsets from degrees to arcseconds
    ra_offset_arcsec = ra_offset * 3600
    dec_offset_arcsec = dec_offset * 3600

    # Convert arcsec to pixel offsets
    ra_offset_pix = ra_offset_arcsec / pixel_scale
    dec_offset_pix = dec_offset_arcsec / pixel_scale

    # Add the offsets in arcseconds to the matched catalog
    matched_catalog['RA_offset_arcsec'] = ra_offset_arcsec
    matched_catalog['Dec_offset_arcsec'] = dec_offset_arcsec
    matched_catalog['RA_offset_pix'] = ra_offset_pix
    matched_catalog['Dec_offset_pix'] = dec_offset_pix

    # Calculate the median offsets in arcseconds
    median_ra_offset = np.median(ra_offset_arcsec)
    median_dec_offset = np.median(dec_offset_arcsec)
    median_ra_pix = np.median(ra_offset_pix)
    median_dec_pix = np.median(dec_offset_pix)
    print("Median RA offset (arcsec):", median_ra_offset)
    print("Median Dec offset (arcsec):", median_dec_offset)
    print("Median RA offset (pixels):", median_ra_pix)
    print("Median Dec offset (pixels):", median_dec_pix)

    
    #--------- apply sigma clipping
    clipped_ra = sigma_clip(ra_offset_pix, sigma=3, maxiters=5)  # 3-sigma clip, up to 5 iterations
    clipped_dec = sigma_clip(dec_offset_pix, sigma=3, maxiters=5)

    # calculate the sigma-clipped means
    sigma_clipped_mean_ra = np.mean(clipped_ra.data[~clipped_ra.mask])
    sigma_clipped_mean_dec = np.mean(clipped_dec.data[~clipped_dec.mask])
    # calculate the sigma-clipped medians
    sigma_clipped_median_ra = np.median(clipped_ra.data[~clipped_ra.mask])
    sigma_clipped_median_dec = np.median(clipped_dec.data[~clipped_dec.mask])

    print("Sigma-clipped mean RA offset (pix):", sigma_clipped_mean_ra)
    print("Sigma-clipped mean Dec offset (pix):", sigma_clipped_mean_dec)
    print("Sigma-clipped median RA offset (pix):", sigma_clipped_median_ra)
    print("Sigma-clipped median Dec offset (pix):", sigma_clipped_median_dec)


    # number of points after sigma clipping
    n_ra_points = np.sum(~clipped_ra.mask)
    n_dec_points = np.sum(~clipped_dec.mask)

    #print("Number of points in RA before sigma clipping:", len(ra_offset_pix))
    #print("Number of points in Dec before sigma clipping:", len(dec_offset_pix))
    print(f"Number of points in RA after sigma clipping: {n_ra_points}")
    print(f"Number of points in Dec after sigma clipping: {n_dec_points}")

    # Create mask where both RA and Dec are not clipped
    combined_mask = (~clipped_ra.mask) & (~clipped_dec.mask)

    # Subset the matched catalog
    matched_catalog = matched_catalog[combined_mask]

    # Prepare summary log line
    log_line = (
        f"{os.path.basename(muse_file)}\n"
        f"  Median RA offset (arcsec): {median_ra_offset:.6f}\n"
        f"  Median Dec offset (arcsec): {median_dec_offset:.6f}\n"
        f"  Median RA offset (pixels): {median_ra_pix:.3f}\n"
        f"  Median Dec offset (pixels): {median_dec_pix:.3f}\n\n"
    )

    # Write to log file
    with open(log_file, 'w') as f:
        f.write(log_line)

    # Print to screen
    print(log_line.strip())

    log_file=f'outputs/pix_offset_log.txt'

    log_line = (
        f"{os.path.basename(muse_file)} "
        f"{median_ra_pix:.3f} "
        f"{median_dec_pix:.3f}\n"
    )

    # Write to log file
    with open(log_file, 'a') as f:
        f.write(log_line)

    print ("ran add_offsets")

    return matched_catalog, n_ra_points, n_dec_points


def align_wcs_using_pixel_shift(musefile, matched_catalog, log_file='offsets.log', output_dir='aligned'):
    """
    Apply WCS alignment by shifting CRPIX based on pixel offsets from a matched catalog.
    
    Parameters:
    - musefile: str, path to input MUSE FITS file
    - matched_catalog: Table with 'RA_offset_pix', 'Dec_offset_pix' columns
    - log_file: str, file to log applied shifts
    - output_dir: str, directory to save aligned FITS

    Returns
    -------
    output_path : str
        Path to the aligned FITS file.
    data_a_sky : np.ndarray
        Sky coordinate grid (RA, Dec in degrees) for each pixel in the aligned image.
    """

    # Compute median or mean pixel shift
    dx = np.median(matched_catalog['RA_offset_pix'])
    dy = np.median(matched_catalog['Dec_offset_pix'])
    dxa = np.median(matched_catalog['RA_offset_arcsec'])
    dya = np.median(matched_catalog['Dec_offset_arcsec'])

    # Open and modify WCS
    with fits.open(musefile, ignore_missing_simple=True) as hdul:
        hdr = hdul[0].header
        wcs = WCS(hdr, naxis=2)

        # Apply shift to reference pixel
        wcs.wcs.crpix[0] -= dx  # RA axis (usually X)
        wcs.wcs.crpix[1] += dy  # Dec axis (usually Y)

        # Update header
        new_hdr = wcs.to_header()
        for key in new_hdr:
            hdr[key] = new_hdr[key]

        # Output
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(musefile).replace('.fits', '_aligned.fits'))
        hdul.writeto(output_path, overwrite=True)

        # Logging
        # Extract OBJECT name
        object_name = hdr.get('OBJECT', 'UNKNOWN')

        # Compute pixel offset ranges
        ra_pix_range = matched_catalog['RA_offset_pix'].max() - matched_catalog['RA_offset_pix'].min()
        dec_pix_range = matched_catalog['Dec_offset_pix'].max() - matched_catalog['Dec_offset_pix'].min()

        # Write to log
        with open(log_file, 'a') as f:
            f.write(
                f"{os.path.basename(musefile)}\t"
                f"OBJECT={object_name}\t"
                f"dx={dx:.3f}\t dy={dy:.3f}\t"
                f"RA_range={ra_pix_range:.3f}\t"
                f"Dec_range={dec_pix_range:.3f}\n"
            )


        print(f"Shift applied (dx={dx:.3f}, dy={dy:.3f}) → saved: {output_path}")

        # --- Apply pixel shifts to matched catalog RA/Dec, ignore just approximate for plotting ---
        # Convert pixel shifts from arcsec to degrees
        dxa_deg = (dxa * u.arcsec).to(u.deg).value
        dya_deg = (dya * u.arcsec).to(u.deg).value

        # Apply the shifts
        matched_catalog['cat1_RA_a'] = matched_catalog['cat1_RA'] - dxa_deg
        matched_catalog['cat1_Dec_a'] = matched_catalog['cat1_Dec'] - dya_deg

        # --- Generate RA/Dec sky coordinate grid from shifted WCS ---
        data_shape = (hdr['NAXIS2'], hdr['NAXIS1'])  # (ny, nx)
        ny, nx = data_shape

        pixel_corners = np.array([[0, 0], [nx, 0], [nx, ny], [0, ny]])
        world_corners = wcs.all_pix2world(pixel_corners, 0)

        ra_vals = world_corners[:, 0]
        dec_vals = world_corners[:, 1]

        ra_min, ra_max = ra_vals.min(), ra_vals.max()
        dec_min, dec_max = dec_vals.min(), dec_vals.max()


        extent_a = [ra_max, ra_min, dec_min, dec_max]

        print ("ran align_wcs_using_pixel_shift")

        return output_path, extent_a


def offset_txt(matched_catalog, musefile, output_dir='aligned'):
    """
    Take a median pixel shift a musefile name and add as a line to a text file.
    
    Parameters:
    - matched_catalog: Table with 'RA_offset_pix', 'Dec_offset_pix' columns
    - musefile: str, path to input MUSE FITS file
    - output_dir: str, directory to save the output file
    """

    # Compute median pixel shift
    dx = np.median(matched_catalog['RA_offset_pix'])
    dy = np.median(matched_catalog['Dec_offset_pix'])

    # Create txt file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'offsets.txt')
    with open(output_file, 'a') as f:
        f.write(f"{musefile} {dx:.5f} {dy:.5f} {'a'}\n")          
    print(f"Added offsets for {musefile} to {output_file}")

    print ("ran offset_txt")

    return output_file

