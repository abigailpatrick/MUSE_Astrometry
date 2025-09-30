
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord

from mpdaf.obj import Cube
from mpdaf.obj import Image
import argparse

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
                        default="/cephfs/apatrick/musecosmos/dataproducts/extractions/subcube_500_1500_500_1500.fits", 
                        help="Path to the MUSE cube FITS file.")
    
    parser.add_argument("--objects", type=str,
                        default="/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic.csv",
                        help="Path to a CSV file containing RA, Dec, and redshift (z).")
    
    parser.add_argument("--spatial_radius", type=float, 
                        default=40.0, 
                        help="Spatial radius for extraction in Pixels.")

    parser.add_argument("--spectral_radius", type=float, 
                        default=100.0, 
                        help="Spectral radius for extraction in Angstroms.")

    parser.add_argument("--rest_wavelength", type=float,
                        default=1215.7, 
                        help="Rest-frame wavelength of the emission line in Angstroms (default is Lyman-alpha).")

    parser.add_argument("--output", type=str,
                        default="/cephfs/apatrick/musecosmos/dataproducts/extractions/", 
                        help="Path to the output directory.")

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

def extract_subcube(cube, x, y, obs_wavelength, spatial_size=40, spectral_width=30):
    """
    Extract a subcube for a single source if within bounds.
    Returns: subcube or None if outside bounds.
    """
    half_size = spatial_size // 2
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

def create_nb_image(subcube, output_path="nb_image.png"):
    """
    Create pseudo-narrowband image from subcube.
    Sums over wavelength range to get 2D image.
    """
    nb_image = subcube.sum(axis=0)  # collapse wavelength for spatial image

    plt.figure(figsize=(6, 5))
    plt.imshow(nb_image.data, origin='lower', cmap='viridis')
    plt.colorbar(label='Flux')
    plt.title("Pseudo-NB Image")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved pseudo-NB image to {output_path}")

def create_spectrum(subcube, output_path="spectrum.png"):
    """
    Create 1D spectrum from subcube.
    Sums over spatial axes to get 1D spectrum.
    """
    spectrum = subcube.sum(axis=(1, 2))  # collapse spatial axes to get 1D

    plt.figure(figsize=(8, 4))
    plt.plot(spectrum.wave.coord(), spectrum.data, color='k', lw=1)
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux")
    plt.title("1D Spectrum")
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved spectrum to {output_path}")


def main():
    args = parse_args()

    # Load cube
    cube = Cube(args.cube, ext=0, memmap=True)
    print("Full Cube Object Loaded")
    cube.info()

    # Load objects from CSV
    df = pd.read_csv(args.objects)
    print(f"Loaded {len(df)} objects from {args.objects}")



    for i, row in df.iterrows():
        ra, dec, z = row['ra'], row['dec'], row['z1_median']
        x, y = ra_dec_to_xy(ra, dec, cube)
        obs_wavelength = z_to_wavelength(z, args.rest_wavelength)

        print(f"Object {i+1}: RA={ra:.6f}, Dec={dec:.6f}, z={z:.3f} to  x={x}, y={y}, λ={obs_wavelength:.2f} Å")

        subcube = extract_subcube(cube, x, y, obs_wavelength,
                            spatial_size=args.spatial_radius, spectral_width=args.spectral_radius)

        # save subcube as fits if extracted
        if subcube is not None:
            subcube_path = f"{args.output}source_{i+1}_subcube.fits"
            subcube.write(subcube_path, overwrite=True)
            print(f"Saved subcube to {subcube_path}")

        if subcube is None:
            continue  # skip sources outside bounds

        create_nb_image(subcube, output_path=f"{args.output}source_{i+1}_nb.png")
        create_spectrum(subcube, output_path=f"{args.output}source_{i+1}_spec.png")

if __name__ == "__main__":
    main()
        



