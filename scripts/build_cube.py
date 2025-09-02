

"""
cp /cephfs/apatrick/musecosmos/scripts/build_cube.py /home/apatrick/P1

python build_cube.py /cephfs/apatrick/musecosmos/scripts/aligned/mosaics


need to take args :
- path to a directory of 2d slice fits files 
- path to csv with wavelengths/slice numbers 
- output path to save new cube

i want to take indivual 2d slices and create one big muse cube object
i want to use mpdaf so it is a cube object
each 2d slice fits file contains spatial 2d array of that slice and 2d wcs for that slice
the csv contains many rows showing what smaller slices made the 2d slice, the slice number and many options for slice wavelength
The slice wavelength for each slice to be included in the cube wcs should be a median of the slice_wavelength column of the the csv for that slice.
The slice column of the csv can be used to order the slices, the whole column is the same number so just take the first.

the fits files have names mosaic_slice_3035.fits and the csv's 3035_wave.csv the number in the names links them.

the default path to fits files is /cephfs/apatrick/musecosmos/scripts/aligned/mosaics
the default path to csv files is /home/apatrick/P1/slurm
the default output path is /cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube

so i need to have a parse args function
a function to find the slice number from the csv and the median wavelength for that slice
a function to open the fits file and get the data and 2d wcs information
a function to create the data stack  in the correct order and if needed get the wcs/ wavelengths into the correct format/order
a function to put these into the mpdaf cube creation
a main to run all this


"""
import argparse
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from mpdaf.obj import Cube, WCS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a MUSE cube from 2D slice FITS files and CSV wavelength tables.")
    parser.add_argument(
        "--fits_dir",
        type=str,
        default="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics",
        help="Path to directory with 2D slice FITS files.",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="/home/apatrick/P1/slurm",
        help="Path to directory with CSV files containing slice/wavelength info.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/big_cube.fits",
        help="Output path for the combined cube.",
    )
    return parser.parse_args()


def get_slice_info(csv_path):
    """Find slice number and median wavelength for a slice from CSV."""
    df = pd.read_csv(csv_path)
    slice_num = int(df["slice"].iloc[0])
    median_wave = np.median(df["slice_wavelength"].values)
    return slice_num, median_wave


def load_slice_fits(fits_path):
    """Open FITS file and return data (2D array) and MPDAF WCS object."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # Use MPDAF WCS directly from header (2D spatial)
    wcs = WCS(header)

    return data, wcs


def create_data_stack(fits_dir, csv_dir):
    """Create data stack, wavelength list, and WCS from slices."""
    data_stack = []
    wave_list = []
    wcs_list = []

    fits_files = [f for f in os.listdir(fits_dir) if f.startswith("mosaic_slice_") and f.endswith(".fits")]
    fits_files.sort()

    for fits_file in fits_files:
        slice_id = fits_file.split("_")[-1].replace(".fits", "")
        csv_file = f"{slice_id}_wave.csv"
        csv_path = os.path.join(csv_dir, csv_file)
        fits_path = os.path.join(fits_dir, fits_file)

        if not os.path.exists(csv_path):
            print(f"Missing CSV for {fits_file}, skipping.")
            continue

        slice_num, median_wave = get_slice_info(csv_path)
        data, wcs = load_slice_fits(fits_path)

        data_stack.append(data)
        wave_list.append(median_wave)
        wcs_list.append(wcs)

    cube_data = np.array(data_stack)
    wave_array = np.array(wave_list)

    # Sort by wavelength
    order = np.argsort(wave_array)
    cube_data = cube_data[order, :, :]
    wave_array = wave_array[order]
    wcs_ref = wcs_list[order[0]]  # first slice WCS as reference

    return cube_data, wave_array, wcs_ref


def make_muse_cube(cube_data, wave_array, wcs):
    """Put stacked data, WCS, and wavelengths into an MPDAF Cube."""
    cube = Cube(data=cube_data, wcs=wcs, wave=wave_array, copy=False)
    return cube


def main():
    args = parse_args()
    cube_data, wave_array, wcs = create_data_stack(args.fits_dir, args.csv_dir)
    cube = make_muse_cube(cube_data, wave_array, wcs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cube.write(args.output, savemask="none", overwrite=True)
    print(f"Saved cube to {args.output}")


if __name__ == "__main__":
    main()
