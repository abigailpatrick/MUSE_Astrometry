
# imports

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from numpy import ma
import numpy as np
from astropy.table import Table, Column, MaskedColumn, pprint
from astropy.io import fits
from astropy.wcs import WCS
from scipy import integrate
from astropy.stats import sigma_clip

import os

# Imports necessary for Source Catalog
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog
from scipy import ndimage
from scipy.ndimage import binary_erosion
from numpy import loadtxt

# Imports necessary for visulisation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import simple_norm


from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.convolution import convolve, Moffat2DKernel
from mpdaf.obj import Cube, Image


from astropy.io import fits
from astropy.table import Table, hstack
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from photutils.aperture import CircularAperture, ApertureStats
import astropy.units as u


from photutils import detect_sources, deblend_sources, SourceCatalog, aperture_photometry, CircularAperture
from astropy.convolution import convolve
from scipy import ndimage
from astropy.coordinates import SkyCoord, search_around_sky


from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord, match_coordinates_sky
from matplotlib.colors import Normalize
from astropy.visualization import PercentileInterval, SqrtStretch, ImageNormalize
from astropy.table import Table, vstack

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

    return (
        shape_muse,
        wcs_muse,
        central_pixel,
        ra_muse,
        dec_muse,
        width_deg,
        height_deg,
        fwhm_nyquist,
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


    # --- Flux unit conversion (to µJy) ---
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

    # --- Filter sources within radius of image center ---
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

    return data, tbl, segment_map, segm_deblend, cat, aperture_phot_tbl, extent


def visulisation(segment_map, data, segm_deblend, cat, fout=None):
    """
    Create a 3-panel plot showing the original image, segmentation map,
    and deblended sources, with source centroids overlaid.

    Parameters
    ----------
    segment_map : photutils.segmentation.SegmentationImage
        Segmentation map from detect_sources.
    data : 2D ndarray
        Original image data.
    segm_deblend : photutils.segmentation.SegmentationImage
        Deblended segmentation map from deblend_sources.
    cat : photutils SourceCatalog or astropy Table
        Source catalog with centroid positions.
    fout : str or None
        Filename to save the figure, or None to show interactively.
    """
    # If `cat` is a SourceCatalog, convert to table for easy indexing
    if not hasattr(cat, 'to_table'):
        # Assume it's already a table or compatible
        tbl = cat
    else:
        tbl = cat.to_table()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Normalize original image with sqrt stretch
    norm = simple_norm(data, 'sqrt', percent=99)

    # Panel 1: Original image
    im0 = axes[0].imshow(data, origin='lower', cmap='gray', norm=norm)
    axes[0].set_title('Original Image')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Flux')

    # Panel 2: Segmentation map with categorical colormap
    im1 = axes[1].imshow(segment_map.data, origin='lower', cmap=segment_map.cmap)
    axes[1].set_title('Segmentation Map')

    # Panel 3: Deblended segmentation with source centroids overlay
    im2 = axes[2].imshow(segm_deblend.data, origin='lower', cmap=segm_deblend.cmap)
    axes[2].scatter(tbl['xcentroid'], tbl['ycentroid'], s=30, edgecolor='white', facecolor='none', linewidth=1.2)
    axes[2].set_title('Deblended Sources + Centroids')

    # Remove ticks on all panels for clarity
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Source Detection and Segmentation', fontsize=16, y=1.02)
    plt.tight_layout()

    if fout:
        plt.savefig(fout, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



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

    return cutout.data, cutout.wcs



def plot_cutout(cutout_data, cutout_wcs, fout=None):
    """
    Plot the HST cutout image with WCS axes.

    Parameters
    ----------
    cutout_data : 2D ndarray
        Image cutout data.
    cutout_wcs : WCS
        WCS object corresponding to the cutout.
    fout : str or None
        If set, saves the figure to this path.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(projection=cutout_wcs)

    norm = simple_norm(cutout_data, 'sqrt', percent=99)
    im = ax.imshow(cutout_data, origin='lower', cmap='gray', norm=norm)

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title('HST Cutout')

    plt.colorbar(im, ax=ax, label='Flux')

    if fout:
        plt.savefig(fout)
        plt.close()
    else:
        plt.show()


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


    return convolved_image


def plot_cutout_convolved(image, fout=None):
    """
    Plot the convolved HST cutout image.

    Parameters
    ----------
    image : 2D ndarray
        The convolved image data.
    fout : str or None
        If set, saves the figure to this path.
    """
    plt.figure(figsize=(6, 6))
    norm = simple_norm(image, 'sqrt', percent=99)
    plt.imshow(image, origin='lower', cmap='gray', norm=norm)
    plt.title('Convolved HST Cutout')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.colorbar(label='Flux')

    if fout:
        plt.savefig(fout)
        plt.close()
    else:
        plt.show()

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
    """
    kron_fluxes = tbl['kron_flux_uJy']
    valid_fluxes = kron_fluxes[np.isfinite(kron_fluxes)]
    flux_cut = np.nanpercentile(valid_fluxes, 10)

    print(f"10th percentile flux: {flux_cut:.2f} µJy")
    print(f"Sources before filtering: {len(tbl)}")

    keep_mask = kron_fluxes > flux_cut
    tbl = tbl[keep_mask]
    print(f"Sources after filtering: {len(tbl)}")
    print(f"Minimum flux post-filtering: {np.nanmin(tbl['kron_flux_uJy']):.2f} µJy")
    """
    # Convert Kron flux to AB magnitude
    kron_fluxes_ujy = tbl['kron_flux_uJy']
    with np.errstate(divide='ignore', invalid='ignore'):
        kron_mags_ab = -2.5 * np.log10(kron_fluxes_ujy.value) + 23.9

    # Add AB mag column to the table for reference
    tbl['kron_mag_AB'] = kron_mags_ab
    
    # plot the distribution of AB magnitudes
    plt.figure(figsize=(8, 6))
    plt.hist(kron_mags_ab, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Kron Magnitude (AB)')
    plt.ylabel('Number of Sources')
    plt.title('Distribution of Kron Magnitudes (AB)')
    plt.legend()
    plt.tight_layout()

    # Save the histogram plot
    plt.savefig('outputs/kron_magnitude_distribution_HST.png')

    # Filter based on AB magnitude threshold (e.g., mag < 27)
    mag_limit = 27.0
    keep_mask = kron_mags_ab < mag_limit

    print(f"Sources before filtering: {len(tbl)}")
    tbl = tbl[keep_mask]
    print(f"Sources after filtering (mag < {mag_limit}): {len(tbl)}")
    print(f"Brightest source: {np.nanmin(kron_mags_ab):.2f} mag")
    print(f"Faintest retained source: {np.nanmax(tbl['kron_mag_AB']):.2f} mag")
    

    # Print column names and table for inspection
    print(tbl.colnames)
    print(tbl)

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


    return data, tbl, segment_map, segm_deblend, cat, aperture_phot_tbl, extent_hst

def visualisation_HST(segment_map, data, segm_deblend, cat, fout='outputs/visualisation_HST.pdf'):
    """
    Generates plots of the data, segmentation map, and deblended segmentation map
    of the sources in the image. It also overlays the sources selected in the source catalog.  
    
    This provides a visual representation of the sources selected so the selection criteria
    can be adjusted by eye.

    Parameters
    ----------
    segment_map : np.ndarray
        The segmentation map of the image.

    data : np.ndarray   
        The data array of the image.
    
    segm_deblend : np.ndarray       
        The deblended segmentation map of the image.

    cat : photutils.segmentation.SourceCatalog
        The source catalog of the image.
    
    fout : str, optional
        The output file path for saving the visualization plots. Default is 'outputs/visualisation_HST.pdf'.

    Returns
    -------
    None
        Saves the plots to the specified PDF file.
    """
    
    # Compute the percentiles for contrast adjustment
    vmin, vmax = np.percentile(data, 5), np.percentile(data, 95)

    # Create a PDF to save the plots
    pdf_pages = PdfPages(fout)

    # Background-subtracted data and segmentation map
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=vmin, vmax=vmax)
    ax1.set_title('Background-subtracted Data')
    ax2.imshow(segment_map, origin='lower', cmap='Spectral', interpolation='nearest')
    ax2.set_title('Segmentation Image')

    # Deblended segmentation map
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    ax.imshow(segm_deblend, origin='lower', cmap='Spectral', interpolation='nearest')
    ax.set_title('Deblended Segmentation Image')
    plt.tight_layout()

    # Data with overlaid apertures from the source catalog
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=vmin, vmax=vmax)
    ax1.set_title('Data')
    ax2.imshow(segm_deblend, origin='lower', cmap='Spectral', interpolation='nearest')
    ax2.set_title('Segmentation Image')

    # Plot Kron apertures on the data and segmentation images
    cat.plot_kron_apertures(ax=ax1, color='white', lw=1.5)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)

    # Save to PDF
    pdf_pages.savefig()
    pdf_pages.close()



def crossmatch_catalogs(catalog1, catalog2, tolerance_arcsec=1.0):
    """
    Crossmatches two catalogs of sources based on their RA and Dec coordinates.

    Parameters
    ----------
    catalog1 : astropy.table.Table  
        The first catalog of sources.
    
    catalog2 : astropy.table.Table
        The second catalog of sources.

    tolerance_arcsec : float
        The matching tolerance in arcseconds. Default is 1.0.

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

    # print column names for debugging


    # Convert the combined structured array back to an Astropy Table
    return Table(combined_catalog)



def add_offsets(muse_file,matched_catalog, log_file,pixel_scale=0.2):
    """
    Adds columns to the matched catalog with the RA and Dec offsets between the two catalogs.
    It also calculates and saves the median RA and Dec offsets.

    Parameters
    ----------
    matched_catalog : astropy.table.Table
        The matched catalog of sources.

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

    print("Number of points in RA before sigma clipping:", len(ra_offset_pix))
    print("Number of points in Dec before sigma clipping:", len(dec_offset_pix))
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


    return matched_catalog, n_ra_points, n_dec_points



def plot_offset(matched_catalog, fout='outputs/offset.pdf'):
    """
    Plot the RA and Dec offsets between two matched catalogs, including histograms and error bars.

    Parameters
    ----------
    matched_catalog : astropy.table.Table
        The matched catalog of sources with RA and Dec offsets.

    fout : str
        Path to the output PDF file where the plot will be saved.
    """
    ra_offsets = matched_catalog['RA_offset_arcsec']
    dec_offsets = matched_catalog['Dec_offset_arcsec']


    # Compute medians and percentiles
    ra_median = np.median(ra_offsets)
    dec_median = np.median(dec_offsets)
    ra_err = np.percentile(ra_offsets, [16, 84])
    dec_err = np.percentile(dec_offsets, [16, 84])

    ra_error = [[ra_median - ra_err[0]], [ra_err[1] - ra_median]]
    dec_error = [[dec_median - dec_err[0]], [dec_err[1] - dec_median]]

    with PdfPages(fout) as pdf:
        fig = plt.figure(figsize=(8, 8))
        grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
        fig.suptitle('RA and Dec Offsets between Catalogs', fontsize=16)

        # Main scatter plot
        main_ax = fig.add_subplot(grid[1:, :-1])
        main_ax.plot(ra_offsets, dec_offsets, 'ob', markersize=3)
        main_ax.errorbar(ra_median, dec_median, xerr=ra_error, yerr=dec_error,
                         fmt='or', ecolor='r', elinewidth=2, capsize=4)
        main_ax.set_xlim(-0.85, 0.85)
        main_ax.set_ylim(-0.85, 0.85)
        main_ax.set_xlabel('RA Offset (arcsec)')
        main_ax.set_ylabel('Dec Offset (arcsec)')
        main_ax.grid(True)
        main_ax.tick_params(labelsize=10)

        # RA histogram (top)
        x_hist = fig.add_subplot(grid[0, :-1], sharex=main_ax)
        x_hist.hist(ra_offsets, bins=50, color='blue', alpha=0.7, density=True)
        x_hist.set_xlim(main_ax.get_xlim())
        x_hist.axis('off')  # Hide all ticks and labels

        # Dec histogram (right)
        y_hist = fig.add_subplot(grid[1:, -1], sharey=main_ax)
        y_hist.hist(dec_offsets, bins=50, orientation='horizontal', color='blue', alpha=0.7, density=True)
        y_hist.set_ylim(main_ax.get_ylim())
        y_hist.axis('off')  # Hide all ticks and labels

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Offset plot saved to {fout}")



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

        return output_path, extent_a


def source_comp_plot(catalog, fout, data,aligned_data ,extent, extent_a):
    """
    Plot MUSE and HST source positions before and after alignment to visually assess WCS correction.

    Parameters
    ----------
    catalog : astropy.table.Table
        The source catalog containing RA and Dec coordinates of sources.

    fout : str  
        Path to the output PDF file for saving the comparison plot.

    data : np.ndarray
        The 2D image array.

    extent : list of float
        [RA_min, RA_max, Dec_min, Dec_max] extent of the image in degrees.

    Returns
    -------
    None
    """
  

    pdf_pages = PdfPages(fout)

    # Extract coordinates
    x = catalog['cat1_RA']
    y = catalog['cat1_Dec']
    xa = catalog['cat1_RA_a']
    ya = catalog['cat1_Dec_a']
    xh = catalog['cat2_RA']
    yh = catalog['cat2_Dec']


    interval = PercentileInterval(99.5)
    vmin, vmax = interval.get_limits(data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    
    # Create plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Original MUSE
    axs[0].imshow(data, origin='lower', cmap='Greys_r', extent=extent, norm=norm)
    axs[0].plot(x, y, 'r+', markersize=3, label='Original Sources')
    axs[0].plot(xh, yh, 'b+', markersize=3, label='HST Sources')
    axs[0].set_title("Original MUSE")
    axs[0].legend()

    # Aligned MUSE
    axs[1].imshow(aligned_data, origin='lower', cmap='Greys_r', extent=extent_a, norm=norm)
    axs[1].plot(xa, ya, 'r+', markersize=3, label='Aligned Sources')
    axs[1].plot(xh, yh, 'b+', markersize=3, label='HST Sources')
    axs[1].set_title("Aligned MUSE")
    axs[1].legend()

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")

    plt.tight_layout()
    pdf_pages.savefig()
    pdf_pages.close()

    print(f"Source comparison plot saved to {fout}")


def compute_and_record_offset(catalog, musefile, results_file='astrometry_offsets.csv'):
    """
    Computes MUSE-HST positional offsets and appends results to a persistent table.

    Parameters
    ----------
    catalog : Table or DataFrame
        Catalog with MUSE and HST RA/Dec coordinates.
    musefile : str
        Path to the MUSE FITS file for this catalog.
    results_file : str
        CSV file to store cumulative offset results.
    """
    # Extract RA/Dec columns
    xa = catalog['cat1_RA_a']
    ya = catalog['cat1_Dec_a']
    xh = catalog['cat2_RA']
    yh = catalog['cat2_Dec']
    
    coords_muse = SkyCoord(ra=xa*u.deg, dec=ya*u.deg)
    coords_hst = SkyCoord(ra=xh*u.deg, dec=yh*u.deg)
    separations = coords_muse.separation(coords_hst)
    
    avg_offset = np.mean(separations.arcsec)
    median_offset = np.median(separations.arcsec)
    max_offset = np.max(separations.arcsec)
    
    fname = os.path.basename(musefile)
    new_row = Table([[fname], [avg_offset], [median_offset], [max_offset]],
                    names=('filename', 'avg_offset_arcsec', 'median_offset_arcsec', 'max_offset_arcsec'))

    # Check if the results file exists
    if os.path.exists(results_file):
        existing_table = Table.read(results_file, format='csv')
        updated_table = vstack([existing_table, new_row])
    else:
        updated_table = new_row

    # Save back to disk
    updated_table.write(results_file, format='csv', overwrite=True)
    print(f"Appended results to {results_file}")


def apply_wcs_alignment_and_save_fits(original_fits_path, tform, output_dir="alignment_results"):
    """
    Applies a Euclidean shift and rotation to the WCS of a FITS file and saves the result.

    Parameters
    ----------
    original_fits_path : str
        Path to the original FITS file.
    tform : astroalign.Transform
        Transform object containing the rotation and shift to apply.
    output_dir : str
        Directory to save the updated FITS file and log.

    Returns
    -------
    output_fits_path : str
        Path to the saved aligned FITS file.
    """
    import os
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    os.makedirs(output_dir, exist_ok=True)

    # Extract base filename and define output path
    basename = os.path.splitext(os.path.basename(original_fits_path))[0]
    output_fits_name = f"{basename}_align.fits"
    output_fits_path = os.path.join(output_dir, output_fits_name)

    # Open original FITS file
    with fits.open(original_fits_path, ignore_missing_simple=True) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data.copy()

        # Full WCS, including all axes (e.g. spectral if present)
        wcs_full = WCS(header)
        wcs2d = wcs_full.celestial

        # Decompose transformation
        matrix = tform.params
        dx = matrix[0, 2]
        dy = matrix[1, 2]
        rotation_matrix = matrix[:2, :2]

        # Get original CD matrix
        if wcs2d.wcs.has_cd():
            cd = wcs2d.wcs.cd
        else:
            cd = np.dot(np.diag(wcs2d.wcs.cdelt), wcs2d.wcs.pc)

        print(f"Original CD matrix:\n{cd}")
        print(f"Applied rotation matrix:\n{rotation_matrix}")

        # Apply rotation
        new_cd = rotation_matrix.T @ cd
        print(f"New CD matrix:\n{new_cd}")
        wcs2d.wcs.cd = new_cd

        # Apply shift (assuming no sign flip; adjust if convention differs)
        print(f"Original CRPIX: {wcs2d.wcs.crpix}")
        wcs2d.wcs.crpix -= [dx, dy]
        print(f"Shifted CRPIX by dx={dx}, dy={dy} → {wcs2d.wcs.crpix}")

        # Update full WCS header (celestial only)
        new_header = wcs2d.to_header()
        for key, val in new_header.items():
            header[key] = val

        # Save updated FITS file
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_fits_path, overwrite=True)
        print(f"Aligned FITS saved to: {output_fits_path}")

        # Log path and transform details
        log_path = os.path.join(output_dir, "aligned_fits_log.txt")
        with open(log_path, "a") as log_file:
            log_file.write(f"{output_fits_name}\tdx={dx:.3f}\tdy={dy:.3f}\n")

    return output_fits_path




def plot_alignment_diagnostics(muse_pixels, hst_in_muse_pixels, transformed_muse, tform, basename, out_dir="alignment_results"):
    """
    Generate and save a diagnostic plot for the alignment of MUSE to HST using astroalign,
    and print the shift and rotation values.

    Parameters
    ----------
    muse_pixels : ndarray
        Nx2 array of original MUSE centroid pixel positions.
    hst_in_muse_pixels : ndarray
        Nx2 array of HST RA/Dec positions transformed into MUSE pixel coordinates.
    transformed_muse : ndarray
        Nx2 array of MUSE pixels after transformation.
    tform : astroalign.Transform
        The estimated transform object returned by astroalign.
    basename : str
        Base name used for labeling and saving the plot.
    out_dir : str
        Directory to save the output plot and logs.
    """

    os.makedirs(out_dir, exist_ok=True)

    matrix = tform.params
    shift_x = matrix[0, 2]
    shift_y = matrix[1, 2]
    theta_rad = np.arctan2(matrix[1, 0], matrix[0, 0])
    theta_deg = np.degrees(theta_rad)

    # Print transform summary
    print("\n=== Alignment Diagnostics ===")
    print("Euclidean transformation matrix (3x3):")
    print(matrix)
    print("\nEstimated shifts and rotation (in MUSE pixel space):")
    print(f"Shift in x: {shift_x:.3f} pixels")
    print(f"Shift in y: {shift_y:.3f} pixels")
    print(f"Rotation: {theta_deg:.3f} degrees")

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(hst_in_muse_pixels[:, 0], hst_in_muse_pixels[:, 1], label='Reference (HST)', color='black', marker='o')
    plt.scatter(muse_pixels[:, 0], muse_pixels[:, 1], label='Original (MUSE)', color='blue', marker='x')
    plt.scatter(transformed_muse[:, 0], transformed_muse[:, 1], label='Transformed (MUSE → HST)', color='red', marker='+')
    plt.legend()
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.title(f'Alignment: {basename}')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(out_dir, f"{basename}_alignment_diagnostic.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Diagnostic plot saved to: {plot_path}")

    # Also log values to a .txt file
    log_line = f"{basename}\t{shift_x:.6f}\t{shift_y:.6f}\t{theta_deg:.6f}\n"
    log_path = os.path.join(out_dir, "source_aa_log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(log_line)
    print(f"Logged alignment info to: {log_path}")



def get_offsets_1(muse_ra, muse_dec, hst_ra, hst_dec, pixel_scale=0.2):
    """
    Calculate the offsets in RA and Dec between MUSE and HST coordinates.
    
    Parameters:
    muse_ra (float): RA of the MUSE source in degrees.
    muse_dec (float): Dec of the MUSE source in degrees.
    hst_ra (float): RA of the HST source in degrees.
    hst_dec (float): Dec of the HST source in degrees.
    
    Returns:
    tuple: Offsets in RA and Dec in arcseconds.
    """
    ra_offset_arcsec = (hst_ra - muse_ra) * 3600  # Convert to arcseconds
    dec_offset_arcsec = (hst_dec - muse_dec) * 3600  # Convert to arcseconds

    # convert to pixel offsets
    ra_offset_pix = ra_offset_arcsec / pixel_scale
    dec_offset_pix = dec_offset_arcsec / pixel_scale

    print(f"RA offset (pixels): {ra_offset_pix}, type: {type(ra_offset_pix)}")
    print(f"Dec offset (pixels): {dec_offset_pix}, type: {type(dec_offset_pix)}")


    return ra_offset_pix, dec_offset_pix

def align_1(musefile, ra_offset_pix, dec_offset_pix, output_dir='aligned'):
    """
    Align the MUSE FITS file by applying pixel offsets.

    """

    
    ra_offset_pix = float(np.asarray(ra_offset_pix).item())
    dec_offset_pix = float(np.asarray(dec_offset_pix).item()
                           )
    # Load the MUSE data
    with fits.open(musefile, ignore_missing_simple=True) as hdul:
        hdr = hdul[0].header
        wcs = WCS(hdr, naxis=2)

        # Apply offsets to the WCS
        wcs.wcs.crpix[0] -= ra_offset_pix
        wcs.wcs.crpix[1] += dec_offset_pix

        # Update the header with the new WCS
        new_hdr = wcs.to_header()
        for key in new_hdr:
            hdr[key] = new_hdr[key]

        # Output 
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(musefile).replace('.fits', '_aligned_1.fits'))
        
        # Write while still inside the context manager
        hdul.writeto(output_path, overwrite=True)
        print(f'Saving aligned FITS file to: {output_path}')

    return output_path


import os

def log_crossmatch_stats(musefile, catalog1, catalog2, matched_catalog, tolerance_arcsec, ra_sig,dec_sig):
    """
    Logs crossmatching statistics to a text file.

    Parameters
    ----------
    musefile : str
        The path to the MUSE catalog file, used to name the log file.
    
    catalog1 : astropy.table.Table
        The original MUSE catalog.
    
    catalog2 : astropy.table.Table
        The original HST catalog.
    
    matched_catalog : astropy.table.Table
        The catalog of matched sources.

    tolerance_arcsec : float
        The matching tolerance used (in arcseconds).

    ra_sig : float
        The number of sources after sigma clipping.
    
    dec_sig : float
        The number of sources after sigma clipping.
    """
    muse_filename_base = os.path.splitext(os.path.basename(musefile))[0]
    log_filename = f"crossmatch_log_{muse_filename_base}.txt"

    with open(log_filename, "a") as log_file:
        log_file.write(f"MUSE catalog: {muse_filename_base}\n")
        log_file.write(f"Tolerance: {tolerance_arcsec} arcsec\n")
        log_file.write(f"Number of sources in MUSE catalog: {len(catalog1)}\n")
        log_file.write(f"Number of sources in HST catalog: {len(catalog2)}\n")
        log_file.write(f"Number of matched sources: {len(matched_catalog)}\n")
        log_file.write(f"Number of sources after RA sigma clipping: {ra_sig}\n")
        log_file.write(f"Number of sources after Dec sigma clipping: {dec_sig}\n")

    print(f"Crossmatch statistics logged to {log_filename}")
