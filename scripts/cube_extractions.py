
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from mpdaf.obj import Cube
import argparse
import os

from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Circle
from astropy.visualization import quantity_support
import matplotlib.image as mpimg
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm

import warnings
from astropy.wcs import FITSFixedWarning

# Suppress specific warnings
warnings.simplefilter('ignore', FITSFixedWarning)
warnings.filterwarnings('ignore', message=".*'partition' will ignore the 'mask'.*")


"""
Inputs 
cube_path (maybe just laod a subsube for time)
A list of ra and dec and z for each object to extract
A spatial radii to extract around each object (in arcsec)
A spectral radii to extract around each object (in Angstroms)

Outputs
For each object:
1. A pseudo-narrowband image (sum over a small wavelength range around the redshifted line)
2. A 1D spectrum over the spatial region (sum over a small spatial region around the object)
These images and spectra should be side by side in a single figure for each object

Steps 
1. load cube 
2. for each object:
    a. convert ra,dec to x,y using wcs
    b. convert z to wavelength using rest-frame wavelength of line
    c. extract subcube around x,y, wavelength with given spatial and spectral radii
    d. extract pseudo-narrowband image
    e. extract 1D spectrum
    f. plot side by side and save figure - left as don't look good.



"""

def parse_args():
    parser = argparse.ArgumentParser(description="Extract pseudo-narrowband images and 1D spectra from a MUSE cube for a list of objects.")

    parser.add_argument("--cube", type=str, 
                        default="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits", 
                        help="Path to the MUSE cube FITS file.")
    
    parser.add_argument("--objects", type=str,
                        default="/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all.csv",
                        help="Path to a CSV file containing RA, Dec, and redshift (z).")
    
    parser.add_argument("--spatial_width", type=float, 
                        default=30.0, 
                        help="Spatial width for extraction in arcsec.")

    parser.add_argument("--spectral_width", type=float, 
                        default=2000,
                        help="Spectral width for subcube extraction in Angstroms (default is 600 Å).")

    parser.add_argument("--rest_wavelength", type=float,
                        default=1215.7, 
                        help="Rest-frame wavelength of the emission line in Angstroms (default is Lyman-alpha).")
    
    parser.add_argument("--wavelengths", type=float, nargs='+', default=[1215.7, 1240.0, 1260.0, 1303, 1336, 1395, 1548.2],
                        help="List of rest-frame wavelengths of interest in Angstroms (default includes Lyα, NV, CIV).")

    parser.add_argument("--spectrum_radius", type=float,
                        default=0.5,
                        help="Radius for 1D spectrum aperture in arcsec (default is 1 arcsec).")

    parser.add_argument("--spectrum_width", type=float,
                        default=2000,
                        help="Width around central wavelength for 1D spectrum in Angstroms (default is 600 Å).")

    parser.add_argument("--pixel_scale", type=float,
                        default=0.2, 
                        help="Pixel scale of the cube in arcsec/pixel (default is 0.2 for MUSE).")
    
    parser.add_argument("--nb_image_width", type=float,
                        default=60.0,
                        help="Width of the pseudo-narrowband image in Angstroms (default is 60 Å).")

    parser.add_argument("--b_region", type=float,
                        default=30.0,
                        help="Width of the background region in Angstroms (default is 30 Å).")

    parser.add_argument("--output", type=str,
                        default="/cephfs/apatrick/musecosmos/dataproducts/extractions/", 
                        help="Path to the output directory.")
    
    parser.add_argument("--use_saved_subcubes", action="store_true",
                    help="If set, use already saved subcubes instead of extracting them again.")

    
    args = parser.parse_args()
    return args

def ra_dec_to_xy(ra, dec, cube):
    # sky2pix expects an (n,2) array with (dec, ra) order
    coords = np.array([[dec, ra]])
    pix = cube.wcs.sky2pix(coords, nearest=True, unit='deg')  # returns (y, x)
    y, x = pix[0]
    return int(x), int(y)  # convert to (x, y) order for indexing

def z_to_wavelength(z, rest_wavelength):
    obs_wavelength = rest_wavelength * (1 + z) * u.AA
    return obs_wavelength

def extract_subcube(cube, x, y, obs_wavelength, spatial_width, spectral_width, pixel_scale):
    """
    Extract a subcube for a single source if within bounds.
    Returns: subcube or None if outside bounds.
    """

    # Convert spatial width from arcsec to pixels
    spatial_width = int(spatial_width / pixel_scale)
    half_size = spatial_width // 2
    x_min, x_max = int(max(x-half_size, 0)), int(min(x+half_size, cube.shape[2]))
    y_min, y_max = int(max(y-half_size, 0)), int(min(y+half_size, cube.shape[1]))

    wave_min = obs_wavelength.value - spectral_width / 4
    wave_max = obs_wavelength.value + (spectral_width / 4) * 3

    # Check wavelength coverage
    if wave_max < cube.wave.coord()[0] or wave_min > cube.wave.coord()[-1]:
        print(f"Source at {obs_wavelength:.1f} Å is outside wavelength coverage.")
        return None

    # Extract subcube
    subcube = cube.select_lambda(wave_min, wave_max)[:, y_min:y_max, x_min:x_max]

    # Check spatial region bounds
    if subcube.shape[1] == 0 or subcube.shape[2] == 0:
        print(f"Source outside spatial bounds: x={x}, y={y}")
        return None
    
    return subcube

def create_nb_image(subcube, central_wavelength, width, pixel_scale, spectrum_radius,
                    smooth_sigma=1, vmin=None, vmax=None, output_path="nb_image.png"):
    """
    Create pseudo-narrowband image around central_wavelength with given width.
    Sums over wavelength range to get 2D image, smooths it with a Gaussian kernel,
    overlays a red circle at the image center, and adds a small legend.
    """

    # --- Set black background plot style ---
    #plt.rcParams['figure.facecolor'] = 'black'
    #plt.rcParams['axes.facecolor'] = 'black'
    #plt.rcParams['axes.edgecolor'] = 'white'
    #plt.rcParams['axes.labelcolor'] = 'white'
    #plt.rcParams['axes.titlecolor'] = 'white'
    #plt.rcParams['xtick.color'] = 'white'
    #plt.rcParams['ytick.color'] = 'white'
    #plt.rcParams['xtick.labelcolor'] = 'white'
    #plt.rcParams['ytick.labelcolor'] = 'white'
    #plt.rcParams['legend.facecolor'] = 'black'
    #plt.rcParams['legend.edgecolor'] = 'white'
    #plt.rcParams['text.color'] = 'white'


    # Convert to Ångström units
    central_wavelength = ensure_angstrom(central_wavelength)
    width = ensure_angstrom(width)

    # Define wavelength range
    wave_min = central_wavelength - width / 2
    wave_max = central_wavelength + width / 2

    # Sum over wavelength range → 2D NB image
    nb_image = subcube.select_lambda(wave_min.value, wave_max.value).sum(axis=0)

    # Smooth with a Gaussian kernel
    smoothed_data = gaussian_filter(nb_image.data, sigma=smooth_sigma)

    # Compute auto scale if not provided
    if vmin is None or vmax is None:
        med = np.nanmedian(smoothed_data)
        std = np.nanstd(smoothed_data)
        vmin = med - 1.5 * std if vmin is None else vmin
        vmax = med + 5 * std if vmax is None else vmax

    # Create coordinate grid in arcseconds
    ny, nx = nb_image.shape
    x = (np.arange(nx) - nx / 2) * pixel_scale
    y = (np.arange(ny) - ny / 2) * pixel_scale

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # Use a purple colormap — 'plasma' or 'magma' 
    im = ax.imshow(
        smoothed_data,
        origin='lower',
        cmap='magma',  # purple-yellow colormap
        extent=[x.max(), x.min(), y.min(), y.max()],  # flipped RA axis
        vmin=vmin,
        vmax=vmax
    )

    # Colorbar with white text
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Axis labels and title
    ax.set_xlabel('ΔRA [arcsec]')
    ax.set_ylabel('ΔDec [arcsec]')
    ax.set_title(f'Pseudo-NB Image: {central_wavelength.value:.1f} ± {width.value/2:.1f} Å')

    # Add circle at the center for the aperture
    spectrum_diameter = 2 * spectrum_radius
    circle_color = 'white'
    circle = Circle(
        (0, 0),
        radius=spectrum_radius,
        edgecolor=circle_color,
        facecolor='none',
        lw=2,
        label=f'Spectrum Aperture = {spectrum_diameter:.1f}"'
    )
    ax.add_patch(circle)

    # Add legend with matching color
    legend = ax.legend(loc='lower left', frameon=False, fontsize=10, handlelength=0)
    for text in legend.get_texts():
        text.set_color(circle_color)
    for handle in legend.legend_handles:
        handle.set_visible(False)

    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight') # , facecolor='black' for dark plot
    plt.close(fig)
    print(f"Saved pseudo-NB image to {output_path}")



def create_continuum_image(subcube, central_wavelength, width, pixel_scale,
                           spectrum_radius, smooth_sigma=1, vmin=None, vmax=None, output_path="continuum_image.png"):
    """
    Create a continuum image redward of Lyα using the median flux along the spectral axis.
    Smoothed with a Gaussian kernel, overlays a red circle, and adds a legend.

    Parameters
    ----------
    subcube : mpdaf.obj.Cube
        MUSE subcube.
    central_wavelength : Quantity
        Central wavelength for the continuum region (in Å or convertible to Å).
    width : Quantity
        Width of the continuum region (in Å or convertible to Å).
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    spectrum_radius : float
        Radius of the aperture circle in arcsec.
    smooth_sigma : float
        Gaussian smoothing sigma in pixels.
    output_path : str
        Path to save the resulting image.
    """

    # Ensure units
    central_wavelength = ensure_angstrom(central_wavelength)
    width = ensure_angstrom(width)

    # Define wavelength range for continuum (redward of central_wavelength)
    wave_min = central_wavelength
    wave_max = central_wavelength + width

    # Extract subcube and compute median along wavelength axis
    continuum_subcube = subcube.select_lambda(wave_min.value, wave_max.value)
    continuum_image = np.median(continuum_subcube.data, axis=0)

    # Smooth with Gaussian
    smoothed_data = gaussian_filter(continuum_image, sigma=smooth_sigma)

    # Compute auto scale only if not provided
    if vmin is None or vmax is None:
        med = np.nanmedian(smoothed_data)
        std = np.nanstd(smoothed_data)
        vmin = med - 1.5 * std if vmin is None else vmin
        vmax = med + 5 * std if vmax is None else vmax

    # Coordinate grid in arcseconds
    ny, nx = smoothed_data.shape
    x = (np.arange(nx) - nx / 2) * pixel_scale
    y = (np.arange(ny) - ny / 2) * pixel_scale

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(smoothed_data, origin='lower', cmap='magma',
                   extent=[x.max(), x.min(), y.min(), y.max()], vmin=vmin, vmax=vmax) # flipped ra

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]')

    # Labels and title
    ax.set_xlabel('ΔRA [arcsec]')
    ax.set_ylabel('ΔDec [arcsec]')
    ax.set_title(f'Continuum Image: {wave_min.value:.1f} – {wave_max.value:.1f} Å')

    # Red circle at center
    spectrum_diameter = 2 * spectrum_radius
    circle_color = 'white'
    circle = Circle((0, 0), radius=spectrum_radius, edgecolor=circle_color,
                    facecolor='none', lw=2,
                    label=f'Spectrum Aperture = {spectrum_diameter:.1f}"')
    ax.add_patch(circle)

   # Add legend inside plot
    legend = ax.legend(loc='lower left', frameon=False, fontsize=10, handlelength=0)
    for text in legend.get_texts():
        text.set_color(circle_color)
    for handle in legend.legend_handles:
        handle.set_visible(False)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved continuum image to {output_path}")

def ensure_angstrom(value):
    """Ensure value is an astropy Quantity in Å."""
    return value if isinstance(value, u.Quantity) else value * u.AA


def create_spectrum(subcube, spectrum_radius, central_wavelength, spectrum_width,
                    pixel_scale, b_region, fwhm, wavelengths, output_path="spectrum.png"):
    """
    Create 1D spectrum from circular aperture in subcube.
    Also overlays a Gaussian-smoothed version and marks key spectral features.
    """

    # --- Set black background plot style ---
    #plt.rcParams['figure.facecolor'] = 'black'
    #plt.rcParams['axes.facecolor'] = 'black'
    #plt.rcParams['axes.edgecolor'] = 'white'
    #plt.rcParams['axes.labelcolor'] = 'white'
    #plt.rcParams['axes.titlecolor'] = 'white'
    #plt.rcParams['xtick.color'] = 'white'
    #plt.rcParams['ytick.color'] = 'white'
    #plt.rcParams['xtick.labelcolor'] = 'white'
    #plt.rcParams['ytick.labelcolor'] = 'white'
    #plt.rcParams['legend.facecolor'] = 'black'
    #plt.rcParams['legend.edgecolor'] = 'white'
    #plt.rcParams['text.color'] = 'white'

    # Convert radius from arcsec to pixels
    spectrum_radius_pix = int(spectrum_radius / pixel_scale)

    # Create circular aperture mask
    y, x = np.indices(subcube.data.shape[1:])
    center_x, center_y = subcube.data.shape[2] // 2, subcube.data.shape[1] // 2
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    aperture_mask = r <= spectrum_radius_pix

    # Define wavelength range
    central_wavelength = ensure_angstrom(central_wavelength)
    spectrum_width = ensure_angstrom(spectrum_width)
    wave_min = central_wavelength - spectrum_width / 4
    wave_max = central_wavelength + (spectrum_width / 4) * 3

    # Extract spectral region
    spectrum = subcube.select_lambda(wave_min.value, wave_max.value)
    data = spectrum.data  # shape: (Nλ, Ny, Nx)
    wave = spectrum.wave.coord()  # wavelength array in Å

    # Apply aperture mask and sum spatially
    masked_data = data * aperture_mask
    spectrum_1d = masked_data.sum(axis=(1, 2))

    # --- Smooth the spectrum with Gaussian kernel ---
    sigma_A = fwhm / 2.355          # convert FWHM to sigma in Å
    dw = np.median(np.diff(wave))   # wavelength step in Å/pixel
    sigma_pix = sigma_A / dw        # convert sigma to pixel units
    spectrum_smooth = gaussian_filter1d(spectrum_1d, sigma_pix)
    smooth_color = "#B63DE2"  # colour for the smoothed spectrum

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.step(wave, spectrum_1d, color='grey', where='mid', lw=1, alpha=0.4, label='Original') #make white for dark plot
    plt.plot(wave, spectrum_smooth, color=smooth_color, lw=1.8, alpha=0.8,
             label=f'Smoothed (FWHM = {fwhm} Å)')

    # Highlight expected Lyα region
    lymin = central_wavelength.value - b_region
    lymax = central_wavelength.value + b_region
    plt.axvspan(lymin, lymax, color='lightblue', alpha=0.2,
                label=f'Expected Lyα region (±{b_region} Å)') # deepskyblue for dark plot

    # Add horizontal dashed line at y = 0
    plt.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.6) # white for dark plot

    # Mark rest-frame lines (Lyα, NV, CIV, etc.)
    line_labels = ["Lyα", "N V","Si II", "O I","C II","Si IV","C IV"]
    
    for i, wl in enumerate(wavelengths):
        wl = wl.value if hasattr(wl, 'value') else wl
        if wl < wave_min.value or wl > wave_max.value:
            continue
        label = line_labels[i] if i < len(line_labels) else f"{wl:.1f} Å"
        plt.axvline(wl, color='orange', linestyle='--', lw=1.0, alpha=0.8)
        plt.text(wl + 3, plt.ylim()[1]*0.85, label, color='orange',
                 rotation=90, va='top', ha='left', fontsize=9)

    plt.xlabel("Wavelength [Å]")
    plt.ylabel(r"Flux Density [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    spectrum_diameter = 2 * spectrum_radius
    plt.title(f"1D Spectrum: Aperture = {spectrum_diameter:.1f} arcsec")
    plt.xlim(wave_min.value, wave_max.value)
    plt.legend(loc='lower left', frameon=False)
    plt.savefig(output_path, dpi=300, pad_inches=0.2) # add , facecolor='black' for dark plot
    plt.close()
    print(f"Saved spectrum to {output_path}")


def create_spectrum_with_inset(subcube, spectrum_radius, central_wavelength, spectrum_width,
                               pixel_scale, b_region, fwhm, wavelengths,
                               nb_width, smooth_sigma=1, vmin=None, vmax=None,
                               output_path="spectrum_with_inset.png"):
    """
    Create 1D spectrum with a small NB image inset in the top-right corner.

    Parameters
    ----------
    subcube : mpdaf.obj.Cube
        MUSE subcube.
    spectrum_radius : float
        Aperture radius in arcseconds.
    central_wavelength : Quantity
        Central wavelength (in Å or convertible to Å).
    spectrum_width : Quantity
        Wavelength span for the extracted spectrum (in Å).
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    b_region : float
        Half-width (in Å) of expected Lyα region.
    fwhm : float
        FWHM (in Å) for Gaussian smoothing kernel.
    wavelengths : list of float
        Rest-frame wavelengths of lines to mark.
    nb_width : Quantity
        Width of narrowband image around central_wavelength (in Å).
    smooth_sigma : float
        Gaussian smoothing for the NB image (pixels).
    vmin, vmax : float
        Intensity limits for NB image display.
    output_path : str
        File path for output figure.
    """

    # --- Shared black & white style ---
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['axes.titlecolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['legend.facecolor'] = 'black'
    plt.rcParams['legend.edgecolor'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams.update({'font.size': 16}) # big for poster

    # --- Convert to Ångström ---
    central_wavelength = ensure_angstrom(central_wavelength)
    spectrum_width = ensure_angstrom(spectrum_width)
    nb_width = ensure_angstrom(nb_width)

    # --- Extract spectral region ---
    wave_min = central_wavelength - spectrum_width / 4
    wave_max = central_wavelength + (spectrum_width / 4) * 3
    spectrum = subcube.select_lambda(wave_min.value, wave_max.value)
    data = spectrum.data
    wave = spectrum.wave.coord()

    # --- Aperture mask ---
    spectrum_radius_pix = int(spectrum_radius / pixel_scale)
    y, x = np.indices(subcube.data.shape[1:])
    cx, cy = subcube.data.shape[2] // 2, subcube.data.shape[1] // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    aperture_mask = r <= spectrum_radius_pix

    # --- Extract 1D spectrum ---
    masked_data = data * aperture_mask
    spec_1d = masked_data.sum(axis=(1, 2))

    # --- Smooth spectrum ---
    sigma_A = fwhm / 2.355
    dw = np.median(np.diff(wave))
    sigma_pix = sigma_A / dw
    spec_smooth = gaussian_filter1d(spec_1d, sigma_pix)
    smooth_color = "#B63DE2"

    # --- Create main spectrum plot ---
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.subplots_adjust(top=0.95, right=0.88)  # top: up = lower number, right left = lower number

    ax.step(wave, spec_1d, color='white', where='mid', lw=1, alpha=0.4, label='Original')
    ax.plot(wave, spec_smooth, color=smooth_color, lw=1.8, alpha=0.8,
            label=f'Smoothed (FWHM = {fwhm} Å)')

    # Highlight expected Lyα region
    lymin = central_wavelength.value - b_region
    lymax = central_wavelength.value + b_region
    ax.axvspan(lymin, lymax, color='deepskyblue', alpha=0.25,
               label=f'Expected Lyα region (±{b_region} Å)')

    ax.axhline(0, color='white', linestyle='--', lw=0.8, alpha=0.6)

    # Mark rest-frame lines
    line_labels = ["Lyα", "N V", "Si II", "O I", "C II", "Si IV", "C IV"]
    for i, wl in enumerate(wavelengths):
        wl = wl.value if hasattr(wl, 'value') else wl
        if wave_min.value <= wl <= wave_max.value:
            label = line_labels[i] if i < len(line_labels) else f"{wl:.1f} Å"
            ax.axvline(wl, color='orange', linestyle='--', lw=1.0, alpha=0.8)
            ax.text(wl + 3, ax.get_ylim()[1]*0.85, label, color='orange',
                    rotation=90, va='top', ha='left', fontsize=14)

    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel(r"Flux Density [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax.set_xlim(wave_min.value, wave_max.value)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.8)
    ax.legend(loc='upper left', frameon=False)
    ax.set_title(f"1D Spectrum with NB Image (Aperture = {2*spectrum_radius:.1f}\" )")

    # --- Create inset NB image ---
    inset_ax = fig.add_axes([0.68, 0.58, 0.28, 0.33])  # [left, bottom, width, height]
    wave_min_nb = central_wavelength - nb_width / 2
    wave_max_nb = central_wavelength + nb_width / 2
    nb_image = subcube.select_lambda(wave_min_nb.value, wave_max_nb.value).sum(axis=0)
    smoothed_data = gaussian_filter(nb_image.data, sigma=smooth_sigma)

    # Auto scaling if needed
    if vmin is None or vmax is None:
        med = np.nanmedian(smoothed_data)
        std = np.nanstd(smoothed_data)
        vmin = med - 1.5 * std
        vmax = med + 5 * std

    ny, nx = nb_image.shape
    x = (np.arange(nx) - nx / 2) * pixel_scale
    y = (np.arange(ny) - ny / 2) * pixel_scale

    im = inset_ax.imshow(smoothed_data, origin='lower', cmap='magma',
                         extent=[x.max(), x.min(), y.min(), y.max()],
                         vmin=vmin, vmax=vmax)
    inset_ax.set_title(f"{central_wavelength.value:.0f} Å", color='white', fontsize=10)
    inset_ax.set_xlabel("ΔRA [\"]", fontsize=14)
    inset_ax.set_ylabel("ΔDec [\"]", fontsize=14)
    inset_ax.tick_params(axis='both', colors='white', labelsize=14)

    # Add white circle for aperture
    circle = Circle((0, 0), radius=spectrum_radius, edgecolor='white',
                    facecolor='none', lw=1.5)
    inset_ax.add_patch(circle)

    # --- Save figure ---
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    # Reset Matplotlib style for future plots
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Saved combined spectrum+NB image to {output_path}")

def plot_primer_rgb_cutout(primer_path, ra, dec, cutout_size, pixel_scale=0.03, output_path="primer_cutout.png"):
    """
    Create RGB PRIMER cutout centered on RA,Dec, axes in ΔRA/ΔDec (arcsec).
    cutout_size : tuple
        (nx, ny) in pixels, e.g., use JELS cutout size
    pixel_scale : float
        Arcsec per pixel (default 0.03")
    """
    # Load FITS cube and WCS
    with fits.open(primer_path) as hdul:
        rgb_data = hdul[0].data  # shape (3, ny, nx)
        wcs = WCS(hdul[0].header).celestial  # only RA/Dec part

    position = SkyCoord(ra, dec, unit="deg")

    # Perform cutout per channel
    cutouts = []
    for channel in rgb_data:
        cut = Cutout2D(channel, position=position, size=cutout_size, wcs=wcs)
        cutouts.append(cut.data)

    # Stack channels into RGB array
    rgb_cutout = np.stack(cutouts, axis=0)

    # Normalize & stretch
    rgb_norm = np.zeros_like(rgb_cutout)
    for i in range(3):
        norm = simple_norm(rgb_cutout[i], 'asinh', percent=99.5)
        rgb_norm[i] = (rgb_cutout[i] - norm.vmin) / (norm.vmax - norm.vmin)
    rgb_norm = np.clip(rgb_norm, 0, 1)
    rgb_image = np.moveaxis(rgb_norm, 0, -1)
    rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert pixel grid to ΔRA/ΔDec in arcsec
    ny, nx = cutout_size
    x = (np.arange(nx) - nx / 2) * pixel_scale
    y = (np.arange(ny) - ny / 2) * pixel_scale
    extent = [x.max(), x.min(), y.min(), y.max()]  # flip x-axis for RA

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb_image, origin='lower', extent=extent)
    ax.set_xlabel('ΔRA [arcsec]')
    ax.set_ylabel('ΔDec [arcsec]')
    ax.set_title('PRIMER RGB Cutout')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PRIMER RGB cutout to {output_path}")

    return rgb_cutout



def unwrap(wl, cube_data, outfile):
    """
    Unwrap a 3D MUSE subcube (nz, ny, nx) into a 2D FITS file (ny*nx, nz)
    for DS9 or 2D spectral visualization.

    Parameters
    ----------
    wl : array-like
        Wavelength array in Å.
    cube_data : numpy.ndarray
        3D cube data with shape (nz, ny, nx).
    outfile : str
        Output FITS file path.
    """
    nz, ny, nx = cube_data.shape
    twod = np.zeros((ny * nx, nz))

    c = 0
    for i in range(nx):
        for j in range(ny):
            spectrum = cube_data[:, j, i]
            if np.nansum(spectrum) != 0:
                twod[c, :] = np.nan_to_num(spectrum, nan=0.0)
                c += 1

    twod = twod[:c, :]

    hdu = fits.PrimaryHDU(twod)
    hdu.header['CRVAL1'] = wl[0]
    hdu.header['CRPIX1'] = 1
    hdu.header['CDELT1'] = wl[1] - wl[0]
    hdu.header['CTYPE1'] = 'LINEAR'
    hdu.header['BUNIT'] = 'Flux'
    hdu.header['COMMENT'] = 'Unwrapped 2D spectrum from subcube'
    hdu.writeto(outfile, overwrite=True)
    print(f"Saved unwrapped 2D spectrum to {outfile}")

    return twod, hdu.header

def plot_unwrapped_spectrum(twod, hdr, vmin=None, vmax=None, cmap='inferno', save_path=None):
    """
    Plot a 2D spectrum from unwrapped data (returned by unwrap()).

    Parameters
    ----------
    twod : np.ndarray
        Unwrapped 2D data (spatial pixels × wavelength).
    hdr : fits.Header
        FITS header containing wavelength calibration.
    vmin, vmax : float, optional
        Display range (auto-scaled if None).
    cmap : str, optional
        Matplotlib colormap (default: 'inferno').
    save_path : str, optional
        Path to save figure (if None, shows interactively).
    """
    if twod.size == 0:
        print("Empty unwrapped data.")
        return

    # Build wavelength axis
    wl0 = hdr.get('CRVAL1', 0)
    dwl = hdr.get('CDELT1', 1)
    nlam = twod.shape[1]
    wavelength = wl0 + np.arange(nlam) * dwl

    # Auto scale
    if vmin is None or vmax is None:
        med, std = np.nanmedian(twod), np.nanstd(twod)
        vmin = med - 1.5 * std if vmin is None else vmin
        vmax = med + 6 * std if vmax is None else vmax

    # Plot
    plt.figure(figsize=(12, 4))
    plt.imshow(twod, aspect='auto', origin='lower',
               extent=[wavelength[0], wavelength[-1], 0, twod.shape[0]],
               vmin=vmin, vmax=vmax, cmap=cmap)

    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Spatial pixel index")
    plt.title("Unwrapped 2D Spectrum")
    cbar = plt.colorbar()
    cbar.set_label("Flux")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, pad_inches=0.2)
        plt.close()
        print(f"Saved plot to {save_path}")
    else:
        plt.show()




def save_combined_pdf(df, output_dir, ID, pdf_name="combined.pdf", sources=None, suffix=""):
    """
    Combines PNG outputs for each source into a single PDF with consistent layout.
    Each page shows:
      - Top: unwrap (2D spectrum)
      - Middle: 1D spectrum
      - 3rd row: NB image + continuum
      - Bottom row: JELS cutout + PRIMER continuum cutout (same size)
    Automatically supports '_manual' suffix for manual adjustments.
    """
    pdf_path = os.path.join(output_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        if sources is None:
            sources = range(1, len(df) + 1)

        for i in sources:
            base = f"{output_dir}source_{i}"
            suffix_part = f"{suffix}" if suffix else ""

            unwrap_path = f"{base}_unwrap{suffix_part}.png"
            spec_path   = f"{base}_spec{suffix_part}.png"
            nb_path     = f"{base}_nb_image_from_z{suffix_part}.png"
            cont_path   = f"{base}_continuum{suffix_part}.png"
            jels_path   = f"{base}_JELS{suffix_part}.png"
            primer_path = f"{base}_PRIMER{suffix_part}.png"
            # Skip if essential file missing
            if not os.path.exists(spec_path):
                print(f"Skipping source {i}: missing spectrum figure")
                continue

            # --- Figure layout: 4 rows total ---
            fig = plt.figure(figsize=(12, 12))
            fig.suptitle(f"Source {i} (JELS ID: {ID})", fontsize=14, fontweight='bold')

            # Create a 4x2 grid (for side-by-side bottom rows)
            gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])

            # --- Row 1: Unwrapped 2D spectrum (span both columns) ---
            ax1 = fig.add_subplot(gs[0, :])
            if os.path.exists(unwrap_path):
                img1 = mpimg.imread(unwrap_path)
                ax1.imshow(img1)
            else:
                ax1.text(0.5, 0.5, "Unwrap image missing", ha='center', va='center', fontsize=9)
            ax1.axis('off')

            # --- Row 2: 1D Spectrum (span both columns) ---
            ax2 = fig.add_subplot(gs[1, :])
            img2 = mpimg.imread(spec_path)
            ax2.imshow(img2)
            ax2.axis('off')

            # --- Row 3: NB + Continuum ---
            ax3 = fig.add_subplot(gs[2, 0])
            if os.path.exists(nb_path):
                img3 = mpimg.imread(nb_path)
                ax3.imshow(img3)
            else:
                ax3.text(0.5, 0.5, "NB image missing", ha='center', va='center', fontsize=9)
            ax3.axis('off')

            ax4 = fig.add_subplot(gs[2, 1])
            if os.path.exists(cont_path):
                img4 = mpimg.imread(cont_path)
                ax4.imshow(img4)
            else:
                ax4.text(0.5, 0.5, "Continuum image missing", ha='center', va='center', fontsize=9)
            ax4.axis('off')

            # --- Row 4: JELS cutout + PRIMER continuum cutout ---
            ax5 = fig.add_subplot(gs[3, 0])
            if os.path.exists(jels_path):
                img5 = mpimg.imread(jels_path)
                ax5.imshow(img5)
            else:
                ax5.text(0.5, 0.5, "JELS cutout missing", ha='center', va='center', fontsize=9)
            ax5.axis('off')

            ax6 = fig.add_subplot(gs[3, 1])
            if os.path.exists(primer_path):
                img6 = mpimg.imread(primer_path)
                ax6.imshow(img6)
            else:
                ax6.text(0.5, 0.5, "PRIMER continuum cutout missing", ha='center', va='center', fontsize=9)
            ax6.axis('off')

            # --- Layout & save ---
            plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

    print(f" Combined PDF saved to: {pdf_path}")


def adjust_lines(central_wavelength, wavelengths):
    diff = wavelengths[0].value - central_wavelength.value
    return np.array(wavelengths) - diff

def JELS__cutout(JELS_image_file, ra, dec, cutout_size, band, pixel_scale=0.03, output_path="jels_cutout.png"):
    """
    Create a JWST JELS cutout centered on given RA/Dec with axes in ΔRA and ΔDec (arcsec).
    
    Parameters
    ----------
    JELS_image_file : str
        Path to JWST JELS i2d FITS image.
    ra, dec : float
        Source coordinates in degrees.
    cutout_size : tuple
        (nx, ny) size of the cutout in pixels.
    band : str
        JWST band name for title.
    pixel_scale : float
        Pixel scale in arcsec/pixel (default = 0.03″).
    output_path : str
        Path to save PNG output.
    """
    # --- Set black background plot style ---
    #plt.rcParams['figure.facecolor'] = 'black'
    #plt.rcParams['axes.facecolor'] = 'black'
    #plt.rcParams['axes.edgecolor'] = 'white'
    #plt.rcParams['axes.labelcolor'] = 'white'
    #plt.rcParams['axes.titlecolor'] = 'white'
    #plt.rcParams['xtick.color'] = 'white'
    #plt.rcParams['ytick.color'] = 'white'
    #plt.rcParams['xtick.labelcolor'] = 'white'
    #plt.rcParams['ytick.labelcolor'] = 'white'
    #plt.rcParams['legend.facecolor'] = 'black'
    #plt.rcParams['legend.edgecolor'] = 'white'
    #plt.rcParams['text.color'] = 'white'
    #plt.rcParams.update({'font.size': 16}) # big for poster

    # --- Load FITS and WCS ---
    jels_hdul = fits.open(JELS_image_file)
    jels_data = jels_hdul[1].data
    jels_wcs = WCS(jels_hdul[1].header)
    position = SkyCoord(ra, dec, unit='deg')

    # --- Extract cutout ---
    cutout = Cutout2D(jels_data, position, cutout_size, wcs=jels_wcs)
    data = cutout.data

    # --- Create ΔRA, ΔDec coordinate grid (in arcsec) ---
    ny, nx = data.shape
    x = (np.arange(nx) - nx / 2) * pixel_scale  # arcsec
    y = (np.arange(ny) - ny / 2) * pixel_scale  # arcsec

    # Note: RA increases to the left → flip x-axis
    extent = [x.max(), x.min(), y.min(), y.max()]

    # --- Plot ---
    norm = simple_norm(data, 'asinh', percent=99.5)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, origin='lower', cmap='magma', extent=extent, norm=norm)

    # Axes labels in ΔRA and ΔDec
    ax.set_xlabel('ΔRA [arcsec]')
    ax.set_ylabel('ΔDec [arcsec]')
    ax.set_title(f'JWST {band} Cutout')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux')

    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved JWST cutout to {output_path}")

    return cutout





def main():
    args = parse_args()

    # --- Load MUSE cube ---
    cube = Cube(args.cube, ext=0, memmap=True)
    print("Full Cube Object Loaded")

    # --- Load object catalog ---
    df = pd.read_csv(args.objects)
    print(f"Columns: {df.columns}")
    print(f"Loaded {len(df)} objects from {args.objects}")

    # --- Process each object ---
    for i, row in df.iterrows():
        ra, dec, z , id, band = row['ra'], row['dec'], row['z1_median'], row['ID'], row['band']
        subcube_path = f"{args.output}source_{i+1}_subcube_{args.spatial_width}_{args.spectral_width}.fits"

        # Use existing subcube if available
        if args.use_saved_subcubes and os.path.exists(subcube_path):
            print(f"Using existing subcube: {subcube_path}")
            subcube = Cube(subcube_path, ext=1)
        else:
            # Convert coordinates and extract region
            x, y = ra_dec_to_xy(ra, dec, cube)
            obs_wavelength = z_to_wavelength(z, args.rest_wavelength)
            wavelengths = z_to_wavelength(z, np.array(args.wavelengths))

            print(f"Object {i+1}: RA={ra:.6f}, Dec={dec:.6f}, z={z:.3f} → x={x:.1f}, y={y:.1f}, λ={obs_wavelength:.2f} Å")

            subcube = extract_subcube(
                cube, x, y, obs_wavelength,
                spatial_width=args.spatial_width,
                spectral_width=args.spectral_width,
                pixel_scale=args.pixel_scale
            )

            # Save extracted subcube
            if subcube is not None:
                subcube.write(subcube_path)
                print(f"Saved subcube to {subcube_path}")
        """ 
        if subcube is None:
            print(f"Skipping object {i+1}: out of cube bounds.")
            continue
 
        # Observed-frame line wavelengths
        obs_wavelength = z_to_wavelength(z, args.rest_wavelength)
        wavelengths = z_to_wavelength(z, np.array(args.wavelengths))

        # --- Generate outputs ---
        wl = subcube.wave.coord()
        twod, hdr = unwrap(wl, subcube.data, outfile=f"{args.output}source_{i+1}_unwrap.fits")
        plot_unwrapped_spectrum(twod, hdr, save_path=f"{args.output}source_{i+1}_unwrap.png")

        # --- Compute shared scaling between NB and continuum ---
        nb_data = subcube.select_lambda(
            (obs_wavelength.value - args.nb_image_width/2),
            (obs_wavelength.value + args.nb_image_width/2)
        ).sum(axis=0).data

        cont_central_wl = z_to_wavelength(z, 1260.0)
        cont_data = np.median(
            subcube.select_lambda(
                cont_central_wl.value, 
                cont_central_wl.value + args.nb_image_width
            ).data,
            axis=0
        )

        # Smooth and get shared vmin/vmax
        sm_nb = gaussian_filter(nb_data, sigma=1)
        sm_cont = gaussian_filter(cont_data, sigma=1)
        combined = np.concatenate([sm_nb.flatten(), sm_cont.flatten()])
        med = np.nanmedian(combined)
        std = np.nanstd(combined)
        vmin = med - 1.5 * std
        vmax = med + 5 * std

        create_spectrum(
            subcube, spectrum_radius=args.spectrum_radius,
            central_wavelength=obs_wavelength,
            spectrum_width=args.spectrum_width,
            pixel_scale=args.pixel_scale, b_region=args.b_region,
            fwhm=5.0, wavelengths=wavelengths, 
            output_path=f"{args.output}source_{i+1}_spec.png"
        )

        create_nb_image(
            subcube, central_wavelength=obs_wavelength,
            width=args.nb_image_width, pixel_scale=args.pixel_scale,
            spectrum_radius=args.spectrum_radius, smooth_sigma=1, vmin=vmin, vmax=vmax,
            output_path=f"{args.output}source_{i+1}_nb_image_from_z.png"
        )

        create_spectrum_with_inset(
            subcube,
            spectrum_radius=args.spectrum_radius,
            central_wavelength=obs_wavelength,
            spectrum_width=args.spectrum_width,
            pixel_scale=args.pixel_scale,
            b_region=args.b_region,
            fwhm=5.0,
            wavelengths=wavelengths,
            nb_width=args.nb_image_width,
            smooth_sigma=1,
            vmin=vmin,
            vmax=vmax,
            output_path=f"{args.output}source_{i+1}_spec_with_nb.png"
        )

        cont_central_wl = z_to_wavelength(z, 1260.0)
        create_continuum_image(
            subcube, central_wavelength=cont_central_wl,
            width=args.nb_image_width, pixel_scale=args.pixel_scale,
            spectrum_radius=args.spectrum_radius, smooth_sigma=1, vmin=vmin, vmax=vmax,
            output_path=f"{args.output}source_{i+1}_continuum.png"
        )

    
        # JELS cutout
        jels_pixel_scale = 0.03  # arcsec/pixel (30mas)
        cutoutsize_jels = (int(args.spatial_width / jels_pixel_scale),
                        int(args.spatial_width / jels_pixel_scale))

        jels_cutout = JELS__cutout(
            f'/home/apatrick/P1/JELSDP/JELS_v1_{band}_30mas_i2d.fits',
            ra, dec, cutoutsize_jels, band,
            pixel_scale=jels_pixel_scale,
            output_path=f"{args.output}source_{i+1}_JELS.png"
        )

        # --- PRIMER RGB cutout ---
        primer_pixel_scale = 0.03  # arcsec/pixel (30mas)
        cutoutsize_primer = (int(args.spatial_width / primer_pixel_scale),
                        int(args.spatial_width / primer_pixel_scale))
        primer_cutout = plot_primer_rgb_cutout(
            primer_path="/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_rgb.fits",  # update to your PRIMER file
            ra=ra,
            dec=dec,
            cutout_size=cutoutsize_primer,
            output_path=f"{args.output}source_{i+1}_PRIMER.png"
        )

    # --- Combine all default outputs into a single PDF ---
    save_combined_pdf(df, args.output, id, pdf_name=f"all_sources_combined_{args.spectral_width}.pdf")

    # --- Manual wavelength adjustments ---
    sources = [1, 3, 7, 8, 10, 11, 12, 13, 15, 16, 23]
    new_central_wavelengths = {
        1: 8599, 3: 8609, 7: 8612, 8: 8635, 10: 8595,
        11: 8620, 12: 8633, 13: 8598, 15: 8623, 16: 8581, 23: 8637
    }

    for i in sources:
        row = df.iloc[i - 1]
        ra, dec, z , id, band = row['ra'], row['dec'], row['z1_median'], row['ID'], row['band']
        subcube_path = f"{args.output}source_{i}_subcube.fits"

        if not os.path.exists(subcube_path):
            print(f"Skipping source {i}: subcube not found")
            continue

        subcube = Cube(subcube_path, ext=1)
        central_wavelength = new_central_wavelengths[i] * u.AA

        # Adjust all emission line markers to align with the new Lyα
        wavelengths = z_to_wavelength(z, np.array(args.wavelengths))
        wavelengths = adjust_lines(central_wavelength, wavelengths)
        cont_central_wl = z_to_wavelength(z, 1260.0)

        # --- Compute shared scaling between NB and continuum ---
        nb_data = subcube.select_lambda(
            (obs_wavelength.value - args.nb_image_width/2),
            (obs_wavelength.value + args.nb_image_width/2)
        ).sum(axis=0).data

        cont_central_wl = z_to_wavelength(z, 1260.0)
        cont_data = np.median(
            subcube.select_lambda(
                cont_central_wl.value, 
                cont_central_wl.value + args.nb_image_width
            ).data,
            axis=0
        )

        # Smooth and get shared vmin/vmax
        sm_nb = gaussian_filter(nb_data, sigma=1)
        sm_cont = gaussian_filter(cont_data, sigma=1)
        combined = np.concatenate([sm_nb.flatten(), sm_cont.flatten()])
        med = np.nanmedian(combined)
        std = np.nanstd(combined)
        vmin = med - 1.5 * std
        vmax = med + 5 * std

        create_nb_image(
            subcube, central_wavelength=central_wavelength,
            width=20, pixel_scale=args.pixel_scale,
            spectrum_radius=args.spectrum_radius, smooth_sigma=1, vmin=vmin, vmax=vmax,
            output_path=f"{args.output}source_{i}_nb_image_from_z_manual.png"
        )

        create_continuum_image(
            subcube, central_wavelength=cont_central_wl,
            width=20, pixel_scale=args.pixel_scale,
            spectrum_radius=args.spectrum_radius, smooth_sigma=1, vmin=vmin, vmax=vmax,
            output_path=f"{args.output}source_{i}_continuum_manual.png"
        )

        create_spectrum(
            subcube, spectrum_radius=args.spectrum_radius,
            central_wavelength=central_wavelength,
            spectrum_width=args.spectrum_width,
            pixel_scale=args.pixel_scale, b_region=10,
            fwhm=5.0, wavelengths=wavelengths,
            output_path=f"{args.output}source_{i}_spec_manual.png"
        )

        create_spectrum_with_inset(
            subcube,
            spectrum_radius=args.spectrum_radius,
            central_wavelength=central_wavelength,
            spectrum_width=args.spectrum_width,
            pixel_scale=args.pixel_scale,
            b_region=10,
            fwhm=5.0,
            wavelengths=wavelengths,
            nb_width=args.nb_image_width,
            smooth_sigma=1,
            vmin=vmin,
            vmax=vmax,
            output_path=f"{args.output}source_{i}_spec_with_nb_manual.png"
        )

        wl = subcube.wave.coord()
        twod, hdr = unwrap(wl, subcube.data, outfile=f"{args.output}source_{i}_unwrap_manual.fits")
        plot_unwrapped_spectrum(twod, hdr, save_path=f"{args.output}source_{i}_unwrap_manual.png")

        # JELS cutout
        jels_pixel_scale = 0.03  # arcsec/pixel (30mas)
        cutoutsize_jels = (int(args.spatial_width / jels_pixel_scale),
                        int(args.spatial_width / jels_pixel_scale))
        
        print(f"Creating JWST cutout for source {i}, RA={ra}, Dec={dec}")

        jels_cutout = JELS__cutout(
            f'/home/apatrick/P1/JELSDP/JELS_v1_{band}_30mas_i2d.fits',
            ra, dec, cutoutsize_jels, band,
            pixel_scale=jels_pixel_scale,
            output_path=f"{args.output}source_{i}_JELS_manual.png"
        )
        
        # --- PRIMER RGB cutout (manual) ---
        primer_pixel_scale = 0.03  # arcsec/pixel (30mas)
        cutoutsize_primer = (int(args.spatial_width / primer_pixel_scale),
                        int(args.spatial_width / primer_pixel_scale))
        primer_cutout = plot_primer_rgb_cutout(
            primer_path="/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_rgb.fits",  # update to your PRIMER file
            ra=ra,
            dec=dec,
            cutout_size=cutoutsize_primer,
            output_path=f"{args.output}source_{i}_PRIMER_manual.png"
        )


    # --- Combine manual adjusted outputs into a single PDF ---
    save_combined_pdf(df, args.output, id ,pdf_name=f"manual_adjustments_{args.spectral_width}.pdf", sources=sources, suffix="_manual")
     """

if __name__ == "__main__":
    main()


        


