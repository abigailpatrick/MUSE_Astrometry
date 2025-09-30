


"""
cp /cephfs/apatrick/musecosmos/scripts/build_cube.py /home/apatrick/P1

python build_cube.py 


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
from mpdaf.obj import Cube, WCS, WaveCoord


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a MUSE cube from 2D slice FITS files and CSV wavelength tables.")
    parser.add_argument(
        "--fits_dir",
        type=str,
        default="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full",
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
        default="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_mean_stack.fits",
        help="Output path for the combined cube.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=None,
        help="Optional starting slice ID (inclusive).",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="Optional ending slice ID (inclusive).",
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

    wcs = WCS(header)
    
    return data, wcs


def list_fits_files(fits_dir, start_id=None, end_id=None):
    """List and filter FITS files by optional slice ID range."""
    fits_files = [f for f in os.listdir(fits_dir) if f.startswith("mean_stack_") and f.endswith(".fits")]
    filtered = []

    for f in fits_files:
        slice_id_str = f.split("_")[-1].replace(".fits", "")
        try:
            slice_id = int(slice_id_str)
        except ValueError:
            print(f"Cannot parse slice ID from {f}, skipping.")
            continue

        if (start_id is not None and slice_id < start_id) or (end_id is not None and slice_id > end_id):
            continue

        filtered.append((slice_id, f))

    filtered.sort(key=lambda x: x[0])
    return [f[1] for f in filtered]


def load_slice(fits_file, fits_dir, csv_dir):
    """Load a single FITS slice and median wavelength."""
    fits_path = os.path.join(fits_dir, fits_file)
    slice_id = os.path.basename(fits_file).split("_")[-1].replace(".fits", "")
    csv_file = f"{slice_id}_wave.csv"
    csv_path = os.path.join(csv_dir, csv_file)

    if not os.path.exists(csv_path):
        print(f"Missing CSV for {fits_file}, skipping.")
        return None, None, None

    slice_num, median_wave = get_slice_info(csv_path)
    data, wcs = load_slice_fits(fits_path)
    return data, wcs, median_wave


def stack_slices(fits_files, fits_dir, csv_dir):
    """Stack multiple slices into cube_data, wave_array, and reference WCS."""
    data_stack, wave_list, slice_list, wcs_list = [], [], [], []

    for f in fits_files:
        data, wcs, median_wave = load_slice(f, fits_dir, csv_dir)
        if data is None:
            continue
        
        # Extract slice number from filename
        slice_num = int(os.path.basename(f).split("_")[-1].replace(".fits", ""))
        slice_list.append(slice_num)

        data_stack.append(data)
        wave_list.append(median_wave)
        wcs_list.append(wcs)

    if len(data_stack) == 0:
        raise RuntimeError("No valid slices found.")

    cube_data = np.array(data_stack)
    wave_array = np.array(wave_list)
    slice_array = np.array(slice_list)
    wcs_ref = wcs_list[0]

    # Sort by wavelength
    order = np.argsort(wave_array)
    cube_data = cube_data[order, :, :]
    wave_array = wave_array[order]
    slice_array = slice_array[order]
    print(f"Number of stacked slices: {len(wave_array)}")

    print(f"Stacked cube shape: {cube_data.shape}")


    return cube_data, wave_array, slice_array, wcs_ref


def create_data_stack(fits_dir, csv_dir, start_id=None, end_id=None):
    """Main wrapper to list, load, and stack slices."""
    fits_files = list_fits_files(fits_dir, start_id, end_id)
    return stack_slices(fits_files, fits_dir, csv_dir)


def make_muse_cube(cube_data, wave_array, slice_array, wcs):
    """Create MPDAF Cube with linear wavelength axis, enforcing 1.25 Å spacing."""
    cdelt_array = np.diff(wave_array)
    cdelt_median = np.median(cdelt_array)

    bad_idx = np.where(~np.isclose(cdelt_array, 1.25, rtol=1e-3))[0]
    if bad_idx.size > 0:
        # Report slice numbers corresponding to the spacing problem
        bad_slices = slice_array[bad_idx]
        raise ValueError(
            f"Wavelength spacing is not 1.25 Å: median={cdelt_median}\n"
            f"Problematic slices: {bad_slices}\n"
            f"Problematic spacings: {cdelt_array[bad_idx]}"
        )
    
    wave = WaveCoord(
        crval=wave_array[0],
        cdelt=cdelt_median,
        cunit='Angstrom',
        crpix=1.,
        shape=(len(wave_array),)
    )

    cube = Cube(data=cube_data, wcs=wcs, wave=wave, copy=False)
    return cube


def main():
    args = parse_args()

    # --- Build the cube ---
    cube_data, wave_array, slice_array, wcs = create_data_stack(
        args.fits_dir,
        args.csv_dir,
        start_id=args.start_id,
        end_id=args.end_id
    )
    cube = make_muse_cube(cube_data, wave_array, slice_array, wcs)

    # --- Load first slice header (for extension 1) ---
    fits_files = list_fits_files(args.fits_dir, args.start_id, args.end_id)
    first_slice_path = os.path.join(args.fits_dir, fits_files[0])
    with fits.open(first_slice_path) as hdul:
        long_header = hdul[0].header.copy()

    # --- Write temporary MPDAF cube ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tmpfile = args.output + ".tmp"
    cube.write(tmpfile, savemask="none")

    # --- Reopen temp cube and reorganize HDUs ---
    with fits.open(tmpfile) as hdul:
        # hdul[1] contains cube data + MPDAF header
        mpdaf_hdu = hdul[1]

        # --- Ensure BUNIT is set explicitly in HDU 0 ---
        mpdaf_hdu.header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom'

        # Promote to PrimaryHDU
        primary_hdu = fits.PrimaryHDU(data=mpdaf_hdu.data, header=mpdaf_hdu.header)

        # --- Header-only extension with long header + comments ---
        long_header['OBJECT'] = "COSMOS_MEGA_CUBE_mean_stack"
        long_header.add_comment("Spatial WCS from first slice")
        long_header.add_comment("Wavelength axis constructed from median slice wavelengths")
        header_hdu = fits.ImageHDU(header=long_header, name="FULLHEADER")

        # --- Build final HDU list ---
        new_hdul = fits.HDUList([primary_hdu, header_hdu])
        new_hdul.writeto(args.output, overwrite=True)

    # Clean up temporary file
    os.remove(tmpfile)

    print(f"Saved cube to {args.output}")







if __name__ == "__main__":
    main()