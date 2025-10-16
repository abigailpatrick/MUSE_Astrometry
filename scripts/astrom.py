# imports 

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from numpy import ma
from astropy.table import Table, Column, MaskedColumn, pprint
from astropy.io import fits
from astropy.wcs import WCS
from scipy import integrate


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
from matplotlib.backends.backend_pdf import PdfPages
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import simple_norm


from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.convolution import convolve, Moffat2DKernel
from mpdaf.obj import Cube, Image


from astropy.visualization import simple_norm
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle
import re
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
import sys
import os


from astromfunc import * # Custom functions for this script


# Plotting functions removed for speed and space but can be added back in for diagnostics

# Define variables 
gamma = 2.5 
fwhm_arcs = 0.3 # FWHM in arcseconds 
radii = [1.0] # Radii in arcseconds for the circular apertures (onky need more to plot light curves)
tolerence_arcsec = 2.0  # Tolerance for cross-matching in arcseconds

# Files from slurm script
m_file = sys.argv[1] # Full MUSE fits file
m_filename = sys.argv[2] # MUSE file name without the path and extension (for output file-naming)

# Define paths 
MUSE_fits_path = os.path.join('/cephfs/apatrick/musecosmos/reduced_cubes/white', m_file) # edit 
hst_file = f'primer_cosmos_acswfc_f814w_30mas_sci.fits.gz'
HST_fits_path = os.path.join('/home/apatrick/HST', hst_file) # edit 
os.chdir('/cephfs/apatrick/musecosmos/scripts') # Where the main script works from


# Open the MUSE file
with fits.open(MUSE_fits_path, ignore_missing_simple=True) as hdul:
    # Extract the data and header
    m_data = hdul[0].data
    m_hdr = hdul[0].header
    
    # Extract the WCS information
    m_wcs = WCS(m_hdr, naxis=2)
    print('pixel shape:',m_wcs.pixel_shape)
    print('naxis (should be 2:)',m_wcs.wcs.naxis)
    print('crpix:',m_wcs.wcs.crpix) 
    object_str = m_hdr['OBJECT']  
    match = re.search(r'Pointing\s+(\d+)', object_str)
    pointing = int(match.group(1)) 
    print('Pointing:', pointing)

# Open the HST file
with fits.open(HST_fits_path) as hdul:
    # Get the primary header and data
    h_data = hdul[0].data
    h_hdr = hdul[0].header
  

    # Extract PHOTFLAM and PHOTPLAM from the header
    photflam = h_hdr['PHOTFLAM']
    photplam = h_hdr['PHOTPLAM']

    # Print the values to confirm
    print(f'PHOTFLAM: {photflam}')
    print(f'PHOTPLAM: {photplam}')

    # Get the WCS information
    h_wcs = WCS(h_hdr)

# ========= MUSE ==========
print (' Creating MUSE source catalog...')

# Extract more WCS information from the MUSE data
shape_muse, m_wcs, central_pixel, ra_muse, dec_muse, width_deg, height_deg, fwhm_nyquist, pixscale = get_wcs_info(m_data, m_wcs) 
print (f' Pixel scale: {pixscale}, FWHM Nyquist: {fwhm_nyquist}')

# Create a MUSE Soure Catalog
m_data, MUSEtab, segment_map, segm_deblend, m_cat, aperture_phot_tbl, extent = source_catalog(m_data,m_wcs,photplam,fwhm_nyquist,compact_only=True,min_sep_arcsec=2.5,npixels=10, radii=radii,fout=f'outputs/source_catalog_MUSE_{m_filename}.fits')

# =========================

# ========= HST ==========
print (' Creating HST cutout and source catalog...')

# Create a cutout of the HST image
cutout, cutout_wcs = create_cutout(h_data, h_wcs, width_deg, height_deg, ra_muse, dec_muse, fout=f'outputs/hst_cutout_{m_filename}.fits')

# Convolve the HST image with a Moffat kernel
fwhm_pix = fwhm_arcs/fwhm_nyquist.value 
convolved_image = convolve_image(cutout, fwhm_pix, gamma)

# Create a HST Source Catalog
datah, HSTcat, segment_map, segm_deblend, cat, aperture_phot_tbl, extent_hst = source_catalog_HST(convolved_image, cutout_wcs, photflam, photplam ,npixels=10, radii= radii,fout=f'outputs/source_catalog_HST_{m_filename}.fits')

# =========================

# ========= Crossmatch ==========
print ('Crossmatching MUSE and HST catalogs...')

# Define the paths to the FITS files
catalogm_path = f'outputs/source_catalog_MUSE_{m_filename}.fits'
catalogh_path = f'outputs/source_catalog_HST_{m_filename}.fits'

# Load the catalogs from FITS files
catalogm = fits.open(catalogm_path)[1].data
catalogh = fits.open(catalogh_path)[1].data

# Crossmatch the catalogs
matched_catalog = crossmatch_catalogs(Table(catalogm), Table(catalogh), tolerance_arcsec=tolerence_arcsec)

# =========================

# ========= Find and apply offsets ==========
print ('Finding offsets between MUSE and HST catalogs...')

# Add offsets to the matched catalog
matched_catalog_a, n_ra_points, n_dec_points = add_offsets(MUSE_fits_path,matched_catalog, log_file=f'outputs/median_offset_log.txt') 

# Functions to create addiotnal logs removed as well

# Save the matched catalog to a FITS file
#matched_catalog.write(f'outputs/matched_catalog_{m_filename}.fits', overwrite=True)

output_path, extent_a = align_wcs_using_pixel_shift(MUSE_fits_path, matched_catalog_a, log_file=f'outputs/offset_log.txt')


task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
out = offset_txt(matched_catalog_a, MUSE_fits_path, task_id=task_id)


print('Astrometry checks complete')




