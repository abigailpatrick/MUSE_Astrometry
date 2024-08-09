import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from functions import crossmatch_catalogs

# Change directory to the location of the FITS file
os.chdir('/home/apatrick/cats')  # Adjust this path if necessary

# Path to the FITS file
fits_file = 'jels_F470N_detected_source_catalogue_v0.3.fits'

# Open the FITS file and load the data into an Astropy Table
with fits.open(fits_file) as hdul:
    # Assuming the data is in the first extension (index 1)
    data = hdul[1].data
    cat = Table(data)

# Display the table
#print(cat)

# Display columns in the table
#print(cat.colnames)

# Filter the table to select H alpha sources 

halpha_cat = cat[(cat['z1_median'] >= 5.5) &
                              (cat['z1_median'] <= 6.5) &
                              (cat['Emission_line_goodz'] == True) ]
                              
                             # & (cat['Visual_Score'] == 1)]

# Display the filtered table
#print(halpha_cat)
print(len(halpha_cat))


# Change directory to the location of the FITS file
os.chdir('/home/apatrick/Code')  # Adjust this path if necessary

# Path to the FITS file
fits_file = 'outputs/source_catalog_MUSE_26_814.fits'

# Open the FITS file and load the data into an Astropy Table
with fits.open(fits_file) as hdul:
    # Assuming the data is in the first extension (index 1)
    data = hdul[1].data
    muse_cat = Table(data)


# Assuming catalog1 and catalog2 are Astropy Tables
halpha_cat = np.array(halpha_cat)
muse_cat = np.array(muse_cat)

tolerance_arcsec = 4.0  # tolerance for crossmatching in arcseconds

# Crossmatch the catalogs
matched_catalog = crossmatch_catalogs(halpha_cat, muse_cat, tolerance_arcsec)

# Display the matched catalog
print(matched_catalog)
print(len(matched_catalog))