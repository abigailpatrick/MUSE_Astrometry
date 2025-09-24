from astropy.table import Table
import numpy as np

def extract_images_spectra(cube, catalog_csv, width=30):
    """
    Extract a 30 A image and spectrum around Lyα for each source in catalog.

    Parameters:
    - cube : MPDAF Cube object
    - catalog_csv : path to CSV with 'ra', 'dec', 'z1_median'
    - width : float, width in Angstrom for both image and spectrum

    Returns:
    - results : list of dicts with 'ra', 'dec', 'z', 'lambda_lya', 'image', 'spectrum'
    """
    catalog = Table.read(catalog_csv)
    results = []

    for src in catalog:
        ra, dec, z = src['ra'], src['dec'], src['z1_median']

        # Compute observed Lyα
        lambda_lya = 1215.67 * (1 + z)

        # Define wavelength range
        wave_min = lambda_lya - width/2
        wave_max = lambda_lya + width/2

        # Extract narrow-band image
        img = cube.get_image(center=(ra, dec), wave=(wave_min, wave_max))

        # Extract spectrum
        # The get_spectrum method allows aperture=1 to sum over one spaxel, or larger if needed
        spec = cube.get_spectrum(center=(ra, dec), wave=(wave_min, wave_max))

        results.append({
            'ra': ra,
            'dec': dec,
            'z': z,
            'lambda_lya': lambda_lya,
            'image': img,
            'spectrum': spec
        })

    return results
