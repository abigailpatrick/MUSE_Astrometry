
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
import matplotlib.pyplot as plt
import numpy as np
from mpdaf.obj import Cube, Image

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Circle
import re
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
import numpy as np


import sys
import os


from museprimer_astrometry import *





# INPUTS
muse_no = 27
new_file = 'cube_27_whitelight_exp3'
#int(sys.argv[1])+27
band = 814

# For HST Convolution
gamma = 2.5 
fwhm_arcs = 0.3   # 0.7 from https://astro.dur.ac.uk/~ams/MUSEcubes/MUSE_OII_short.pdf but ken says 0.3      - seeing fwhm?

radii=[1.0, 2.0, 3.0] # radii for aperture photometry, doesn't matter unless using for light curves

#"""

# Change directory to the location of the FITS file
os.chdir('/home/apatrick/MUSE_NEW')  # Adjust this path if necessary


muse_file = sys.argv[1]
new_file = sys.argv[2]

MUSE_fits_path = os.path.join('/home/apatrick/MUSE_NEW', muse_file)

#"""
#c
#DATACUBE_FINAL_Autocal3688204a_2_ZAP_img.fits
#DATACUBE_FINAL_Autocal3688204b_1_ZAP_img.fits

#MUSE_fits_path = f'/home/apatrick/MUSE_NEW/DATACUBE_FINAL_Autocal3688204b_1_ZAP_img.fits' # add 814 for the 814 band 

# Load the updated FITS file
with fits.open(MUSE_fits_path, ignore_missing_simple=True) as hdul:
    # Extract the data and header
    im_data = hdul[0].data
    header = hdul[0].header
    
    # Extract the WCS information
    im_wcs = WCS(header, naxis=2)
    print('pixel shape:',im_wcs.pixel_shape)
    print('naxis (should be 2:)',im_wcs.wcs.naxis)
    print('crpix:',im_wcs.wcs.crpix) 
    object_str = header['OBJECT']  
    match = re.search(r'Pointing\s+(\d+)', object_str)
    pointing = int(match.group(1)) 
    print('Pointing:', pointing)



# Change directory to the location of the FITS file
os.chdir('/home/apatrick/HST')  # Adjust this path if necessary


# Define the path to the FITS file

fits_file = f'primer_cosmos_acswfc_f{band}w_30mas_sci.fits.gz'

# Open the FITS file
with fits.open(fits_file) as hdul:
    # Print the information about the FITS file
    hdul.info()

    # Get the primary header and data
    primary_header = hdul[0].header
    image_data = hdul[0].data

    # Extract PHOTFLAM and PHOTPLAM from the header
    photflam = primary_header['PHOTFLAM']
    photplam = primary_header['PHOTPLAM']

    # Print the values to confirm
    print(f'PHOTFLAM: {photflam}')
    print(f'PHOTPLAM: {photplam}')

    # Get the WCS information
    wcs_hst = WCS(primary_header)



# Change directory to the location of the FITS file
os.chdir('/home/apatrick/Code')  

# Get the WCS information from the MUSE image
shape_muse, wcs_muse, central_pixel, ra_muse, dec_muse, width_deg, height_deg, pixscale_nyquist = get_wcs_info(im_data, im_wcs) 

# Create a MUSE Soure Catalog
data, MUSEcat, segment_map, segm_deblend, cat, aperture_phot_tbl, extent = source_catalog(im_data,wcs_muse,photplam,pixscale_nyquist,compact_only=True,min_sep_arcsec=2.5,npixels=10, radii=radii,fout=f'outputs/source_catalog_MUSE_{new_file}.fits')

visulisation(segment_map, data, segm_deblend, cat, fout=f'outputs/visualisation_{new_file}.pdf')



# Create a cutout of the HST image
cutout, cutout_wcs = create_cutout(image_data, wcs_hst, width_deg, height_deg, ra_muse, dec_muse, fout=f'outputs/hst_cutout_{new_file}.fits')


# Plot the cutout of the HST image
plot_cutout(cutout, cutout_wcs,fout=f'outputs/hst_cutout_{new_file}.pdf') 


# Convolve the HST image with a Moffat kernel
fwhm_pix = fwhm_arcs/pixscale_nyquist.value
convolved_image = convolve_image(cutout, fwhm_pix, gamma)

# Plot the convolved cutout of the HST image
plot_cutout_convolved(convolved_image, fout=f'outputs/hst_convolved_cutout_{new_file}.pdf')

#data1, tbl1, segment_map1, segm_deblend1, cat1, aperture_phot_tbl1 = source_catalog_HST(cutout, cutout_wcs, photflam, photplam ,npixels=10, radii=[1.0,2.0,3.0,4.0,5.0,6.0,7.0],fout=f'outputs/source_catalog_HST_{muse_no}_{band}_raw.fits')

#print (f'Number of sources in MUSE image: {len(tbl)}')
#print (f'Number of sources in HST raw image: {len(tbl1)}')

data1, HSTcat, segment_map, segm_deblend, cat, aperture_phot_tbl, extent_hst = source_catalog_HST(convolved_image, cutout_wcs, photflam, photplam ,npixels=10, radii= radii,fout=f'outputs/source_catalog_HST_{new_file}.fits')

#print (f'Number of sources in HST convolved image: {len(tbl)}')
print (f'Muse = {muse_no}, HST = {band}')

visualisation_HST(segment_map, data, segm_deblend, cat,fout=f'outputs/visualisation_HST_{new_file}.pdf')

#plot_flux_vs_aperture(tbl,tbl1, radii=[1.0,2.0,3.0,4.0,5.0,6.0,7.0],source_index=40, fout=f'outputs/flux_vs_aperture_{muse_no}_2025.pdf')

# Define the paths to the FITS files
catalog1_path = f'outputs/source_catalog_MUSE_{new_file}.fits'
catalog2_path = f'outputs/source_catalog_HST_{new_file}.fits'

 # Load the catalogs from FITS files
catalog1 = fits.open(catalog1_path)[1].data
print (len(catalog1))
catalog2 = fits.open(catalog2_path)[1].data


tolerence_arcsec = 2.0  # Tolerance for cross-matching in arcseconds
# Crossmatch the catalogs
matched_catalog = crossmatch_catalogs(Table(catalog1), Table(catalog2), tolerance_arcsec=tolerence_arcsec)


# Add offsets to the matched catalog
matched_catalog_a, n_ra_points, n_dec_points = add_offsets(MUSE_fits_path,matched_catalog, log_file=f'outputs/median_offset_log.txt')

# Log the result
log_crossmatch_stats(MUSE_fits_path,catalog1,catalog2, matched_catalog, tolerence_arcsec,n_ra_points,n_dec_points)

#align_wcs_using_catalog(MUSE_fits_path, matched_catalog, log_file=f'outputs/offset_log.txt')

plot_offset(matched_catalog_a, fout=f'outputs/offsets_{new_file}.pdf')

# Save the matched catalog to a FITS file
matched_catalog.write(f'outputs/matched_catalog_{new_file}.fits', overwrite=True)


output_path, extent_a = align_wcs_using_pixel_shift(MUSE_fits_path, matched_catalog_a, log_file=f'outputs/offset_log.txt')


with fits.open(output_path, ignore_missing_simple=True) as hdul:
    aligned_data = hdul[0].data

source_comp_plot(matched_catalog_a, f'outputs/source_comp_{new_file}.pdf', data, aligned_data,  extent, extent_a)

compute_and_record_offset(matched_catalog_a, MUSE_fits_path)

print('astrometry checks complete')






