"""
This is to realign the wavelength axis of MUSE cubes.
only needs running on each cube once and then can run the mosaic slice script on 
these new cubes
run like this: python realignwave.py --path /cephfs/apatrick/musecosmos/reduced_cubes/full --offsets_txt /cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt --slice 100 --output_file mosaic.fits --plotting
"""
from concurrent.futures import ProcessPoolExecutor
import os
import argparse
from astropy.io import fits
from astropy.wcs import WCS
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_interp
from reproject import reproject_adaptive
import matplotlib.pyplot as plt
import numpy as np      
from astropy.visualization import simple_norm            
from dataclasses import dataclass
import csv
import pandas
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="MUSE slice mosaicking pipeline")

    parser.add_argument('--path', type=str, required=True,
                        help='Path to directory containing the cubes')
    parser.add_argument('--offsets_txt', type=str, required=True,
                        help='Text file containing cube offsets')
    parser.add_argument('--slice', type=int, required=True,
                        help='Wavelength slice to extract from each cube')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output FITS file path for the mosaic')
    parser.add_argument('--plotting', action='store_true',
                        help='Enable plotting of the mosaic')

    args = parser.parse_args()
    return args

@dataclass
class CubeEntry:
    file_id: str       # e.g. "Autocal_3687411a_1"
    x_offset: float
    y_offset: float
    flag: str          # 'a' or 'm'


def paths_ids_offsets(offsets_txt):
    """
    Parses the offsets text file and extracts cube paths, IDs, and offsets.

    Parameters
    -----------
    offsets_txt : str
        Path to the text file containing cube offsets.

    Returns
    -------
    cubes : list of CubeEntry
        List of CubeEntry objects containing the extracted information.
    """
    cubes = []
    with open(offsets_txt, "r") as f:
        for line in f:
            if line.strip():
                parts = line.split()
                image_path = parts[0]
                x_offset = float(parts[1])
                y_offset = float(parts[2])
                flag = parts[3]

                # Extract file_id from the image filename
                filename = os.path.basename(image_path)
                file_id = filename.replace("DATACUBE_FINAL_", "").replace("_ZAP_img.fits", "")

                cubes.append(CubeEntry(file_id, x_offset, y_offset, flag))
    return cubes

def check_cube_wavelength_axis(cube_path, slice_number, file_id, log_file="cube_wavelength_log.csv"):
    """
    Checks the wavelength axis of a cube and logs the information in a CSV file.
    
    Parameters
    ----------
    cube_path : str
        Path to the cube FITS file.
    slice_number : int
        Index of the slice to check wavelength.
    file_id : str
        Identifier for this cube (e.g., filename or short ID).
    log_file : str
        CSV file path to save or append the info.
        
    Returns
    -------
    info : dict
        Dictionary containing cube wavelength information.
    """

    # Expected values AFTER trimming
    expected_n_slices = 3681
    expected_step = 1.25        # Å
    expected_start = 4749.9     # Å (starts 50 Å redder than 4700.4 after trimming)
    expected_end   = 9349.9     # Å

    # Open the cube
    with fits.open(cube_path) as hdul:
        header = hdul[1].header
        naxis3 = header['NAXIS3']

        # Read wavelength info from header if available
        if "CDELT3" in header and "CRVAL3" in header:
            crval3 = header['CRVAL3']
            cdelt3 = header['CDELT3']
            start_wavelength = crval3
            end_wavelength = crval3 + cdelt3 * (naxis3 - 1)
            slice_wavelength = crval3 + cdelt3 * slice_number
        else:
            # Use WCS if no CDELT3/CRVAL3
            w = WCS(header)
            pix = np.arange(naxis3)
            wavelengths = w.all_pix2world(np.zeros(naxis3), np.zeros(naxis3), pix, 0)[2]
            cdelt3 = np.mean(np.diff(wavelengths)) * 1e10
            start_wavelength = wavelengths[0] * 1e10
            end_wavelength = wavelengths[-1] * 1e10
            slice_wavelength = wavelengths[slice_number] * 1e10

    # Expected slice value
    expected_slice_wavelength = expected_start + expected_step * slice_number

    # --- Consistency checks ---
    if naxis3 != expected_n_slices:
        raise ValueError(f"[{file_id}] Wrong number of slices: got {naxis3}, expected {expected_n_slices}")
    if abs(cdelt3 - expected_step) > 0.1:
        raise ValueError(f"[{file_id}] Wavelength step mismatch: got {cdelt3}, expected ~{expected_step}")
    if abs(start_wavelength - expected_start) > 1.0:
        raise ValueError(f"[{file_id}] Start wavelength mismatch: got {start_wavelength}, expected ~{expected_start}")
    if abs(end_wavelength - expected_end) > 1.0:
        raise ValueError(f"[{file_id}] End wavelength mismatch: got {end_wavelength}, expected ~{expected_end}")
    if abs(slice_wavelength - expected_slice_wavelength) > 1.0:
        raise ValueError(f"[{file_id}] Slice {slice_number} mismatch: got {slice_wavelength}, expected ~{expected_slice_wavelength}")

    print(f"[{file_id}] Cube wavelength axis check PASSED.")

    # Collect info
    info = {
        "file_id": file_id,
        "n_slices": naxis3,
        "step": cdelt3,
        "start": start_wavelength,
        "end": end_wavelength,
        "slice_number": slice_number,
        "slice_wavelength": slice_wavelength,
    }

    # --- Write to CSV log ---
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=info.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(info)

    return info

def normalise_cube_slices(cube_path, output_path="/cephfs/apatrick/musecosmos/reduced_cubes/norm"):
    """
    Normalise the number of spectral slices in a MUSE cube.

    Rules:
    - If NAXIS3 == 3681 → leave as is
    - If NAXIS3 == 3721 → remove the first 40 slices (→ 3681)
    - If NAXIS3 == 3682 → remove the first 40 slices and the last 1 slice (→ 3681)
    - Otherwise → exit with error

    Parameters
    ----------
    cube_path : str
        Path to input cube FITS file.
    output_path : str, optional
        Path to save corrected cube. If None, saves to "<cube_path>_norm.fits".

    Returns
    -------
    new_path : str
        Path to the corrected cube.
    """
    with fits.open(cube_path, mode="readonly") as hdul:
        # Assume DATA is in ext 1
        data = hdul["DATA"].data
        header = hdul["DATA"].header
        naxis3 = header["NAXIS3"]

        if naxis3 == 3681:
            print(f"{cube_path}: already 3681 slices, no change.")
            return cube_path

        elif naxis3 == 3721:
            print(f"{cube_path}: trimming first 40 slices (3721 → 3681).")
            sl = slice(40, None)

        elif naxis3 == 3722:
            print(f"{cube_path}: trimming first 40 and last slice (3722 → 3681).")
            sl = slice(40, -1)

        else:
            sys.exit(f"ERROR: {cube_path} has {naxis3} slices (unsupported).")

        # Apply trimming consistently to all 3D extensions
        for extname in ["DATA", "STAT", "EXPTIME"]:
            if extname in hdul:
                cube_data = hdul[extname].data
                hdul[extname].data = cube_data[sl, :, :]

                # Update header
                hdr = hdul[extname].header
                hdr["NAXIS3"] = hdul[extname].data.shape[0]

                                # Figure out wavelength step (delta λ)
                if "CDELT3" in hdr:
                    delta = hdr["CDELT3"]
                elif "CD3_3" in hdr:
                    delta = hdr["CD3_3"]
                else:
                    raise KeyError("Neither CDELT3 nor CD3_3 found in FITS header")

                # Adjust CRVAL3 after trimming
                hdr["CRVAL3"] += delta * (sl.start if sl.start else 0)
                

        # Decide output path
        if output_path is None:
            output_path = cube_path.replace(".fits", "_norm.fits")

        hdul.writeto(output_path, overwrite=True)

    return output_path


def main():
    # Parse command-line arguments
    args = parse_args()

    # Extract cube entries from offsets file
    cubes = paths_ids_offsets(args.offsets_txt)

    # Loop over cubes: standardize and check before extracting slices
    for cube in cubes:
        cube_path = os.path.join(args.path, f"DATACUBE_FINAL_{cube.file_id}_ZAP.fits")

        # --- Step 1: Standardize number of slices ---
        standardized_path = normalise_cube_slices(cube_path)

        # Save standardized path back into the cube object
        cube.cube_path = standardized_path

        # --- Step 2: Check wavelength axis consistency ---
        check_cube_wavelength_axis(standardized_path, args.slice, cube.file_id)


if __name__ == "__main__":
    main()
