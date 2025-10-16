
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
                        default="/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic.csv",
                        help="Path to a CSV file containing RA, Dec, and redshift (z).")
    
    parser.add_argument("--spatial_width", type=float, 
                        default=3.0, 
                        help="Spatial width for extraction in arcsec.")

    parser.add_argument("--spectral_width", type=float, 
                        default=800.0,
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
                        default=800.0,
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

    wave_min = obs_wavelength.value - spectral_width
    wave_max = obs_wavelength.value + spectral_width

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


def create_nb_image(subcube, central_wavelength, width, pixel_scale, spectrum_radius, smooth_sigma=1, output_path="nb_image.png"):
    """
    Create pseudo-narrowband image around central_wavelength with given width.
    Sums over wavelength range to get 2D image, smooths it with a Gaussian kernel,
    overlays a red circle at the image center, and adds a small legend.

    Parameters
    ----------
    subcube : mpdaf.obj.Cube
        MUSE subcube.
    central_wavelength : Quantity
        Central wavelength for NB image (in Å or convertible to Å).
    width : Quantity
        Width of the NB filter (in Å or convertible to Å).
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    output_path : str
        Path to save the resulting image.
    """

    # Convert to Ångström units
    central_wavelength = ensure_angstrom(central_wavelength)
    width = ensure_angstrom(width)

    # Define wavelength range
    wave_min = central_wavelength - width / 2
    wave_max = central_wavelength + width / 2

    # Sum over wavelength range → 2D NB image
    nb_image = subcube.select_lambda(wave_min.value, wave_max.value).sum(axis=0)

    # Smooth with a Gaussian kernel (sigma = 1 pixel)
    smoothed_data = gaussian_filter(nb_image.data, sigma=smooth_sigma)

    # Create coordinate grid in arcseconds
    ny, nx = nb_image.shape
    x = (np.arange(nx) - nx / 2) * pixel_scale
    y = (np.arange(ny) - ny / 2) * pixel_scale

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(smoothed_data, origin='lower', cmap='viridis',
                   extent=[x.min(), x.max(), y.min(), y.max()])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]')
    ax.set_xlabel('ΔRA [arcsec]')
    ax.set_ylabel('ΔDec [arcsec]')
    ax.set_title(f'Pseudo-NB Image: {central_wavelength.value:.1f} ± {width.value/2:.1f} Å')

    # Add red circle at center with radius = spectrum_radius
    spectrum_diameter = 2 * spectrum_radius
    circle_color = 'red'
    circle = Circle((0, 0), radius=spectrum_radius, edgecolor=circle_color, facecolor='none', lw=2, label='Spectrum Aperture = {:.1f}"'.format(spectrum_diameter))
    ax.add_patch(circle)

    # Add legend inside plot
    legend = ax.legend(loc='lower left', frameon=False, fontsize=10, handlelength=1.5)
    for text in legend.get_texts():
        text.set_color(circle_color)

    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved pseudo-NB image to {output_path}")

def create_continuum_image(subcube, central_wavelength, width, pixel_scale,
                           spectrum_radius, smooth_sigma=1, output_path="continuum_image.png"):
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

    # Coordinate grid in arcseconds
    ny, nx = smoothed_data.shape
    x = (np.arange(nx) - nx / 2) * pixel_scale
    y = (np.arange(ny) - ny / 2) * pixel_scale

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(smoothed_data, origin='lower', cmap='viridis',
                   extent=[x.min(), x.max(), y.min(), y.max()])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]')

    # Labels and title
    ax.set_xlabel('ΔRA [arcsec]')
    ax.set_ylabel('ΔDec [arcsec]')
    ax.set_title(f'Continuum Image: {wave_min.value:.1f} – {wave_max.value:.1f} Å')

    # Red circle at center
    spectrum_diameter = 2 * spectrum_radius
    circle_color = 'red'
    circle = Circle((0, 0), radius=spectrum_radius, edgecolor=circle_color,
                    facecolor='none', lw=2,
                    label=f'Spectrum Aperture = {spectrum_diameter:.1f}"')
    ax.add_patch(circle)

    # Legend inside plot
    legend = ax.legend(loc='lower left', frameon=False, fontsize=10, handlelength=1.5)
    for text in legend.get_texts():
        text.set_color(circle_color)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved continuum image to {output_path}")

def ensure_angstrom(value):
    """Ensure value is an astropy Quantity in Å."""
    return value if isinstance(value, u.Quantity) else value * u.AA



def create_spectrum(subcube, spectrum_radius, central_wavelength, spectrum_width, pixel_scale, b_region, fwhm, wavelengths, output_path="spectrum.png"):
    """
    Create 1D spectrum from circular aperture in subcube.
    Also overlays a Gaussian-smoothed version and marks key spectral features.
    
    Parameters
    ----------
    subcube : mpdaf.obj.Cube
        MUSE subcube.
    spectrum_radius : float
        Aperture radius in arcseconds.
    central_wavelength : Quantity
        Central wavelength for spectrum (in Å or convertible to Å).
    spectrum_width : Quantity
        Spectral width to extract (in Å or convertible to Å).
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    b_region : float
        Half-width (in Å) of the expected Lyα region to highlight.
    fwhm : float
        FWHM (in Å) for Gaussian smoothing kernel.
    wavelengths : list of float, optional
        Rest-frame wavelengths of lines to mark (e.g. [1215.7, 1240.0, 1548.2]).
    output_path : str
        Path to save the spectrum figure.
    """

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
    wave_min = central_wavelength - spectrum_width / 2
    wave_max = central_wavelength + spectrum_width / 2

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

    # Plotting
    smooth_color = "#217544"  # soft green

    plt.figure(figsize=(12, 4))
    plt.step(wave, spectrum_1d, color='grey', where='mid', lw=1, label='Original')
    plt.plot(wave, spectrum_smooth, color=smooth_color, lw=1.8, alpha=0.7,
             label=f'Smoothed (FWHM = {fwhm} Å)')

    # Highlight expected Lyα region
    lymin = central_wavelength.value - b_region
    lymax = central_wavelength.value + b_region
    plt.axvspan(lymin, lymax, color='lightblue', alpha=0.3,
                label=f'Expected Lyα region (±{b_region} Å)')

    # Add horizontal dashed line at y = 0
    plt.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.6)

    # Mark rest-frame lines (Lyα, NV, CIV, etc.)
    line_labels = ["Lyα", "N V","Si II", "O I","C II","Si IV","C IV"]
    
    for i, wl in enumerate(wavelengths):
        wl = wl.value if hasattr(wl, 'value') else wl
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
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, pad_inches=0.2)
    plt.close()
    print(f"Saved spectrum to {output_path}")



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
    plt.figure(figsize=(12, 3))
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


def save_combined_pdf(df, output_dir, pdf_name="all_sources.pdf", sources=None, suffix=""):
    """
    Combine NB, 1D spectrum, and 2D unwrapped spectrum images into a single PDF.

    Parameters
    ----------
    df : pandas.DataFrame
        Source catalog with at least as many rows as sources.
    output_dir : str
        Directory containing the image files.
    pdf_name : str, optional
        Name of the output PDF file.
    sources : list of int, optional
        Specific source indices (1-indexed). If None, use all rows in df.
    suffix : str, optional
        Optional suffix for distinguishing manual versions (e.g. '_manual').
    """
    pdf_path = os.path.join(output_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        # Decide which indices to use
        indices = sources if sources is not None else range(1, len(df) + 1)

        for i in indices:
            nb_path = os.path.join(output_dir, f"source_{i}_nb_image_from_z{suffix}.png")
            spec_path = os.path.join(output_dir, f"source_{i}_spec{suffix}.png")
            unwrap_path = os.path.join(output_dir, f"source_{i}_unwrap{suffix}.png")

            if not os.path.exists(nb_path) or not os.path.exists(spec_path) or not os.path.exists(unwrap_path):
                print(f"Skipping source {i}: one or more image files missing.")
                continue

            # Load images
            nb_img = Image.open(nb_path)
            spec_img = Image.open(spec_path)
            unwrap_img = Image.open(unwrap_path)

            # Layout: NB (left), 1D (top-right), 2D (bottom-right)
            fig = plt.figure(figsize=(18, 7))
            gs = gridspec.GridSpec(2, 2, width_ratios=[7, 12], height_ratios=[3, 3])

            # NB image
            ax0 = fig.add_subplot(gs[:, 0])
            ax0.imshow(nb_img)
            ax0.axis('off')
            ax0.set_title(f"Source {i}", fontsize=16)

            # 1D Spectrum
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(spec_img)
            ax1.axis('off')


            # 2D Spectrum
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.imshow(unwrap_img)
            ax2.axis('off')
        

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Combined PDF saved to {pdf_path}")



def main():
    args = parse_args()

    # Load cube
    cube = Cube(args.cube, ext=0, memmap=True)
    print("Full Cube Object Loaded")
    

    # Load objects from CSV
    df = pd.read_csv(args.objects)
    print(f"columns: {df.columns}")
    print(f"Loaded {len(df)} objects from {args.objects}")



    for i, row in df.iterrows():
        ra, dec, z = row['ra'], row['dec'], row['z1_median']
        subcube_path = f"{args.output}source_{i+1}_subcube.fits"

        # Check if we should use a saved subcube
        if args.use_saved_subcubes and os.path.exists(subcube_path):
            print(f"Using existing subcube: {subcube_path}")
            subcube = Cube(subcube_path, ext=1)
        else:
            x, y = ra_dec_to_xy(ra, dec, cube)
            obs_wavelength = z_to_wavelength(z, args.rest_wavelength)
            wavelengths = z_to_wavelength(z, np.array(args.wavelengths))


            print(f"Object {i+1}: RA={ra:.6f}, Dec={dec:.6f}, z={z:.3f} to  x={x}, y={y}, λ={obs_wavelength:.2f} Å")

            subcube = extract_subcube(cube, x, y, obs_wavelength,
                            spatial_width=args.spatial_width, spectral_width=args.spectral_width, pixel_scale=args.pixel_scale)

             # save subcube as fits if extracted
            if subcube is not None:
                subcube_path = f"{args.output}source_{i+1}_subcube.fits"
                subcube.write(subcube_path)
                print(f"Saved subcube to {subcube_path}")

        if subcube is None:
            continue  # skip sources outside bounds
        
        obs_wavelength = z_to_wavelength(z, args.rest_wavelength)
        wavelengths = z_to_wavelength(z, np.array(args.wavelengths))
        print(wavelengths)

        # Create unwrapped 2D spectrum
        wl = subcube.wave.coord()
        twod, hdr = unwrap(wl, subcube.data, outfile=f"{args.output}source_{i+1}_unwrap.fits")
        plot_unwrapped_spectrum(twod, hdr, save_path=f"{args.output}source_{i+1}_unwrap.png")

        create_spectrum(subcube, spectrum_radius=args.spectrum_radius, central_wavelength=obs_wavelength, spectrum_width=args.spectrum_width, pixel_scale=args.pixel_scale, b_region=args.b_region, fwhm=5.0, wavelengths=wavelengths, output_path=f"{args.output}source_{i+1}_spec.png")
        
        # Default central wavelength is from redshift but manually alter from observed spectrum if needed
        create_nb_image(subcube, central_wavelength=obs_wavelength, width=args.nb_image_width, pixel_scale=args.pixel_scale, spectrum_radius=args.spectrum_radius, smooth_sigma=1, output_path=f"{args.output}source_{i+1}_nb_image_from_z.png")

        # Create continuum image redward of Lyα
        cont_central_wl = z_to_wavelength(z, 1260.0)
        create_continuum_image(subcube, central_wavelength=cont_central_wl,
                               width=100 * u.AA, pixel_scale=args.pixel_scale,
                               spectrum_radius=args.spectrum_radius, smooth_sigma=1,
                               output_path=f"{args.output}source_{i+1}_continuum.png")

    # Combine all into a single PDF
    save_combined_pdf(df, args.output, pdf_name="all_sources_combined.pdf")

    # Manual adjustments
    sources = [1, 3, 7, 8, 10, 11, 12, 13, 15, 16, 23]
    new_central_wavelengths = {1: 8599, 3: 8609, 7: 8612, 8: 8635, 10: 8595, 11: 8620, 12: 8633, 13: 8598, 15: 8623, 16: 8581, 23: 8637}  # in Å

    for i in sources:
        row = df.iloc[i-1]
        z = row['z1_median'] 
        subcube_path = f"{args.output}source_{i}_subcube.fits"
        if not os.path.exists(subcube_path):
            print(f"Skipping source {i}: subcube not found")
            continue

        subcube = Cube(subcube_path, ext=1)
        central_wavelength = new_central_wavelengths[i] * u.AA


        # Convert rest-frame wavelengths to observed-frame for this source
        wavelengths = z_to_wavelength(z, np.array(args.wavelengths))
        cont_central_wl = z_to_wavelength(z, 1260.0)

        create_nb_image(subcube, central_wavelength, width=20, pixel_scale=args.pixel_scale,
                        spectrum_radius=args.spectrum_radius, smooth_sigma=1, output_path=f"{args.output}source_{i}_nb_image_from_z_manual.png")

        create_continuum_image(subcube, central_wavelength=cont_central_wl,
                               width=100 * u.AA, pixel_scale=args.pixel_scale,
                               spectrum_radius=args.spectrum_radius, smooth_sigma=1,
                               output_path=f"{args.output}source_{i}_continuum_manual.png")

        create_spectrum(subcube, spectrum_radius=args.spectrum_radius,
                        central_wavelength=central_wavelength,
                        spectrum_width=args.spectrum_width,
                        pixel_scale=args.pixel_scale, b_region=10,
                        fwhm=5.0, wavelengths=wavelengths, output_path=f"{args.output}source_{i}_spec_manual.png")

        wl = subcube.wave.coord()
        twod, hdr = unwrap(wl, subcube.data, outfile=f"{args.output}source_{i}_unwrap_manual.fits")
        plot_unwrapped_spectrum(twod, hdr, save_path=f"{args.output}source_{i}_unwrap_manual.png")

    # Combine manual ones into a separate PDF
    save_combined_pdf(df, args.output, pdf_name="manual_adjustments.pdf", sources=sources, suffix="_manual")

        

if __name__ == "__main__":
    main()
        



