import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Image, Cube
from mpdaf.sdetect import Catalog

import muse_origin
from muse_origin import ORIGIN
import photutils
from mpdaf.MUSE import FSFModel
from pathlib import Path
from photutils.segmentation import SegmentationImage

import os
from astropy.io import fits


path = "/cephfs/apatrick/musecosmos/dataproducts/extractions/source_7_subcube_30.0_2000.fits"
with fits.open(path) as hdul:
    hdul.info()
    print("Data shape:", [hdu.data.shape if hdu.data is not None else None for hdu in hdul])



def make_origin_compatible(input_fits, output_fits=None):
    """
    Convert a FITS cube with data in extension 1 into an ORIGIN-compatible cube
    with data in PrimaryHDU (HDU0).
    
    Parameters
    ----------
    input_fits : str
        Path to the input FITS cube (data may be in HDU1).
    output_fits : str, optional
        Path to save the ORIGIN-compatible cube. If None, will overwrite input_fits.
    
    Returns
    -------
    output_fits : str
        Path to the ORIGIN-compatible FITS cube.
    """
    if output_fits is None:
        output_fits = input_fits

    # Open the original FITS and extract data + header from HDU1
    with fits.open(input_fits) as hdul:
        data = hdul[1].data.copy()    # make a full copy
        header = hdul[1].header.copy()

    # Create a new PrimaryHDU with the cube data
    primary_hdu = fits.PrimaryHDU(data=data, header=header)

    # Create a new HDUList containing only HDU0
    hdul_new = fits.HDUList([primary_hdu])

    # Remove existing file if present
    if os.path.exists(output_fits):
        os.remove(output_fits)

    # Write to disk
    hdul_new.writeto(output_fits)

    # Verify
    cube = Cube(output_fits)
    cube.info()

    return output_fits

def add_fsf_keywords(
    cube_path,
    FSFMODE=2,
    FSFLB1=5000,
    FSFLB2=9000,
    FSF00FNC=2,
    FSF00F00=-0.120267451303792,
    FSF00F01=0.6209395273959001,
    FSF00BNC=1,
    FSF00B00=2.8,
    verbose=True
):
    """
    Add or update FSF (PSF) keywords in a MUSE datacube FITS header.

    Parameters
    ----------
    cube_path : str or Path
        Path to the FITS cube to update.
    FSFMODE, FSFLB1, FSFLB2, FSF00FNC, FSF00F00, FSF00F01, FSF00BNC, FSF00B00 : float or int
        FSF model parameters (defaults correspond to UDF-10).
    verbose : bool
        If True, print confirmation of added keywords.
    """
    fsf_values = {
        "FSFMODE": FSFMODE,
        "FSFLB1": FSFLB1,
        "FSFLB2": FSFLB2,
        "FSF00FNC": FSF00FNC,
        "FSF00F00": FSF00F00,
        "FSF00F01": FSF00F01,
        "FSF00BNC": FSF00BNC,
        "FSF00B00": FSF00B00,
    }

    with fits.open(cube_path, mode="update") as hdul:
        hdr = hdul[0].header
        for key, val in fsf_values.items():
            hdr[key] = val
            if verbose:
                print(f"Added/Updated {key} = {val}")
        hdul.flush()

    if verbose:
        print(f"\nFSF keywords successfully written to: {cube_path}")

input_path = "/cephfs/apatrick/musecosmos/dataproducts/extractions/source_7_subcube_30.0_2000.fits"
output_path = "/cephfs/apatrick/musecosmos/dataproducts/extractions/source_7_subcube_origin.fits"

make_origin_compatible(input_path, output_path)

add_fsf_keywords(output_path)

DATACUBE = Path("/cephfs/apatrick/musecosmos/dataproducts/extractions/source_7_subcube_origin.fits")

h = fits.getheader("/cephfs/apatrick/musecosmos/dataproducts/extractions/source_7_subcube_30.0_2000.fits")
print(h)


NAME = DATACUBE.stem

print(NAME)

Cube(output_path).info()
fsfmodel = FSFModel.read(output_path) # stuck because no fsf info in subcube
fsfmodel.to_header()

orig = ORIGIN.init(output_path, name=NAME, loglevel='DEBUG', logcolor=True)

# step 1
orig.set_loglevel('INFO')
orig.step01_preprocessing()

orig.ima_white, orig.ima_dct, orig.ima_std

# step 2
orig.step02_areas(minsize=80, maxsize=120)  

# step 3
orig.step03_compute_PCA_threshold(pfa_test=0.0001)

# step 4
orig.step04_compute_greedy_PCA()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
orig.plot_mapPCA(ax=ax1)
ima_faint = orig.cube_faint.max(axis=0)
ima_faint.plot(ax=ax2, colorbar='v',zscale=True, title='White image for cube_faint')
fig.savefig(f'{output_path}PCA_{NAME}.png', dpi=300, bbox_inches="tight")
print (f'Figure saved to {output_path}PCA_{NAME}.png')

pcut = 1e-08

# step 5
orig.step05_compute_TGLR(ncpu=1, pcut=pcut)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14, 4))
orig.maxmap.plot(ax=ax1, title='maxmap', cmap='Spectral_r')
(-1*orig.minmap).plot(ax=ax2, title='minmap', cmap='Spectral_r')

fig.savefig(f'{output_path}TGLR_{NAME}.png', dpi=300, bbox_inches="tight")
print (f'Figure saved to {output_path}TGLR_{NAME}.png')

# step 6
orig.step06_compute_purity_threshold(purity=0.98, purity_std=0.99)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14, 4))
orig.plot_purity(ax=ax1)
orig.plot_purity(ax=ax2, comp=True)
fig.savefig(f'{output_path}purity_{NAME}.png', dpi=300, bbox_inches="tight")
print (f'Figure saved to {output_path}purity_{NAME}.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True, sharex=True)
orig.plot_min_max_hist(ax=ax1)
orig.plot_min_max_hist(ax=ax2, comp=True)
fig.savefig(f'{output_path}hist_{NAME}.png', dpi=300, bbox_inches="tight")
print (f'Figure saved to {output_path}hist_{NAME}.png')

orig.threshold_correl, orig.threshold_std

# Convert MPDAF Image to a NumPy array
segmap_array = orig.segmap_merged.data  # Extract data from MPDAF Image

# Convert to SegmentationImage
segmap = SegmentationImage(segmap_array)

# Assign the converted segmap back to orig
orig.segmap_merged = segmap  # Explicitly assign it

# Check type again
print("Segmap type before deblending:", type(orig.segmap_merged))


# step 7


orig.step07_detection()

cat0 = Catalog(orig.Cat0)
cat1 = Catalog(orig.Cat1)
print(type(cat0))
print(type(cat1))

orig.Cat0.write(f'{output_path}Cat0.fits', overwrite=True)
orig.Cat1.write(f'{output_path}Cat1.fits', overwrite=True)
print (f'Catalogs saved to {output_path}Cat0.fits and {output_path}Cat1.fits')

fig,(ax1, ax2) = plt.subplots(1,2,figsize=(20,10))

orig.maxmap.plot(ax=ax1, zscale=True, colorbar='v', cmap='gray', scale='asinh', title='maxmap')
orig.ima_white.plot(ax=ax2, zscale=True, colorbar='v', cmap='gray', scale='asinh', title='white image')

for ax in (ax1, ax2):
    #cat0.plot_symb(ax, orig.maxmap.wcs, ecol='g', esize=1.0, ra='ra', dec='dec')
    if len(cat1) > 0:
        cat1.plot_symb(ax, orig.maxmap.wcs, ecol='r', esize=1.0, ra='ra', dec='dec')

fig.savefig(f'{output_path}segmap7.png', dpi=300, bbox_inches="tight")
print (f'Figure saved to {output_path}segmap7.png')


# step 8
orig.step08_compute_spectra()

# step 9
orig.step09_clean_results()

# step 10
orig.step10_create_masks()


orig.Cat3_sources
orig.Cat3_lines
print (orig.Cat3_sources)
print (orig.Cat3_lines)
orig.Cat3_lines.write(f'{output_path}Cat3_lines.fits', overwrite=True)
orig.Cat3_sources.write(f'{output_path}Cat3_sources.fits', overwrite=True)
print (f'Catalogs saved to {output_path}Cat3_lines.fits and {output_path}Cat3_sources.fits')
orig.write()

# step 11

orig.step11_save_sources('0.1', n_jobs=1)

orig.write()
orig.stat()
orig.timestat()