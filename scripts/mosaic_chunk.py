"""
run like this: python mosaic_chunk.py --path /cephfs/apatrick/musecosmos/reduced_cubes/norm --offsets_txt /cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt --slice 100 --output_file mosaic.fits --plotting

do this before running from P1 : cp /cephfs/apatrick/musecosmos/scripts/mosaic_chunk.py /home/apatrick/P1
- move all norm.fits to new file and then update input path to take that
- then run like this : python mosaic_chunk.py --path /cephfs/apatrick/musecosmos/reduced_cubes/norm --offsets_txt /cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt --slice 100 --output_file mosaic.fits --plotting


- copy sbatch cp /cephfs/apatrick/musecosmos/scripts/slurm/run_mosaicslices.slurm /home/apatrick/P1/slurm
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
from functools import partial
import sys
from dataclasses import dataclass
import csv
from astropy.visualization import simple_norm
import pandas
import shutil
from mpdaf.obj import Cube
import time

start_time = time.time()  # record start


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
    parser.add_argument('--tmp_dir', type=str, default='/cephfs/apatrick/musecosmos/scripts/aligned/tmp_slice', help='Optional temp dir for slices')

    args = parser.parse_args()
    return args




@dataclass
class CubeEntry:
    file_id: str       # e.g. "Autocal_3687411a_1"
    x_offset: float
    y_offset: float
    flag: str          # 'a' or 'm'


def paths_ids_offsets(offsets_txt, cubes_dir):
    cubes = {}
    with open(offsets_txt, "r") as f:
        for line in f:
            if line.strip():
                parts = line.split()
                image_path = parts[0]
                x_offset = float(parts[1])
                y_offset = float(parts[2])
                flag = parts[3]

                filename = os.path.basename(image_path)
                file_id = filename.replace("DATACUBE_FINAL_", "").replace("_ZAP_img.fits", "")
                norm_filename = f"DATACUBE_FINAL_{file_id}_ZAP_norm.fits"
                cube_path = os.path.join(cubes_dir, norm_filename)

                if os.path.exists(cube_path) and file_id not in cubes:
                    cube_entry = CubeEntry(file_id, x_offset, y_offset, flag)
                    cube_entry.cube_path = cube_path
                    cubes[file_id] = cube_entry
                elif not os.path.exists(cube_path):
                    print(f"Skipping {norm_filename}, not found in {cubes_dir}")

    cubes = list(cubes.values())
    return cubes




def i_slice(cubes, slice_number):

    """
    Extract a 2D slice from each cube in the list.

    Parameters
    ----------
    cubes : list of CubeEntry
        Metadata entries (file_id, offsets, flag).
    slice_number : int
        Index of the spectral slice to extract.

    Returns
    -------
    i_slice : dict
        Dictionary {file_id: {"data": 2D array, "wcs": WCS object}}.
    """
    i_slice = {}

    for cube in cubes:
        with fits.open(cube.cube_path) as hdul:
            data = hdul[1].data[slice_number]
            header = hdul[1].header.copy()
        
            # 2D WCS for mosaicking / alignment (celestial only)
            wcs = WCS(header).celestial


            i_slice[cube.file_id] = {
                'data': data,
                'wcs': wcs,
                'wcs_e': header  # full FITS header stored as wcs_e
            }

    return i_slice






def slice_wavelength_check(cube_path, slice_number, expected_start=4749.9, expected_step=1.25):
    """
    Checks if the wavelength of a given slice is within 1 Å of the expected value.

    Parameters
    ----------
    cube_path : str
        Path to the cube FITS file.
    slice_number : int
        Index of the slice to check.
    expected_start : float
        Expected starting wavelength of slice 0 (Å). Default is 4749.9.
    expected_step : float
        Expected wavelength step per slice (Å). Default is 1.25.

    Returns
    -------
    slice_wavelength : float
        Calculated wavelength of the given slice.
    within_tolerance : bool
        True if slice wavelength is within 1 Å of expected value.
    """
    
    with fits.open(cube_path) as hdul:
        header = hdul[1].header
        naxis3 = header['NAXIS3']
        
        if "CDELT3" in header and "CRVAL3" in header:
            crval3 = header['CRVAL3']
            cdelt3 = header['CDELT3']
            slice_wavelength = crval3 + cdelt3 * slice_number
        else:
            w = WCS(header)
            pix = np.arange(naxis3)
            wavelengths = w.all_pix2world(np.zeros(naxis3), np.zeros(naxis3), pix, 0)[2]
            slice_wavelength = wavelengths[slice_number] * 1e10  # convert to Å

    expected_slice_wavelength = expected_start + expected_step * slice_number
    within_tolerance = abs(slice_wavelength - expected_slice_wavelength) <= 1.0
    if not within_tolerance:
        raise ValueError(f"WARNING: {cube_path} slice {slice_number} wavelength {slice_wavelength} Å "
              f"not within tolerance of expected {expected_slice_wavelength} Å")
    
    return slice_wavelength


def align_i_slices(i_slice_data, i_slice_wcs, offsets):
    """
    Applies pixel offsets to a list of i_slice data and WCS objects.

    Parameters
    ----------
    i_slice_data : list of 2D ndarrays
        The extracted slices from each cube.
    
    i_slice_wcs : list of WCS
        WCS objects corresponding to each slice.
    
    offsets : list of tuples
        List of (dx, dy) pixel shifts for each slice.

    Returns
    -------
    aligned_i_slice : list of dicts
        Each entry contains:
            'data' : 2D ndarray (image data)
            'wcs' : aligned WCS object
            'applied_offset' : (dx, dy) tuple
    """
    aligned_i_slice = []

    for data, wcs, (dx, dy) in zip(i_slice_data, i_slice_wcs, offsets):
        # Make a copy of the WCS to avoid modifying the original
        new_wcs = wcs.deepcopy()
        
        # Apply pixel shifts
        new_wcs.wcs.crpix[0] -= dx  # RA axis
        new_wcs.wcs.crpix[1] += dy  # Dec axis
        
        aligned_i_slice.append({
            'data': data,
            'wcs': new_wcs,
            'applied_offset': (dx, dy)
        })

    return aligned_i_slice



def common_wcs_area(aligned_slices):
    """
    Finds the common WCS area for all aligned slices.

    Inputs
    -------
    aligned_slices : list of dict
        Each dict should have keys:
            'data' : 2D ndarray of the slice
            'wcs'  : WCS object for the slice (already aligned)

    Outputs
    -------
    wcs_out : WCS
        The WCS object for the common area.
    shape_out : tuple
        The shape of the common area (ny, nx).
    """

    
    # Extract (data, WCS) tuples from aligned slices
    slice_list = [(s['data'], s['wcs'].celestial) for s in aligned_slices]
    

    # Compute optimal common celestial WCS and shape
    wcs_out, shape_out = find_optimal_celestial_wcs(slice_list)
    

    # Optional: add padding (50 pixels each side → 100 total)
    pad_y, pad_x = 100, 100
    shape_out = (shape_out[0] + pad_y, shape_out[1] + pad_x)

    # Shift CRPIX to keep original sky center roughly centered
    wcs_out.wcs.crpix[0] += pad_x // 2  # X
    wcs_out.wcs.crpix[1] += pad_y // 2  # Y

    return wcs_out, shape_out



def reproject_and_save_single(i, slice_dict, wcs_out, shape_out, output_dir, reproject_quick=False):
    """
    Reprojects a single aligned slice onto the common WCS grid and saves to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    data, wcs = slice_dict['data'], slice_dict['wcs'].celestial
    

    if reproject_quick:
        array, _ = reproject_interp((data, wcs), output_projection=wcs_out, shape_out=shape_out)
    else:
        array, _ = reproject_adaptive((data, wcs), output_projection=wcs_out, shape_out=shape_out, conserve_flux=True)

    header = wcs_out.to_header()

    fname = os.path.join(output_dir, f"reproj_slice_{i:03d}_pid{os.getpid()}.fits")
    fits.writeto(fname, array, header=header, overwrite=True)

    return fname

# Top-level wrapper for executor.map
def _map_reproject_and_save(args):
    return reproject_and_save_single(*args)

# Main reproject + save function
def reproject_and_save_slices(aligned_slices, wcs_out, shape_out, output_dir, reproject_quick=False):
    """
    Reprojects all aligned slices onto a common WCS grid, writes each to disk,
    and returns a list of file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build argument tuples for each slice
    args_list = [
        (i, slice_dict, wcs_out, shape_out, output_dir, reproject_quick)
        for i, slice_dict in enumerate(aligned_slices)
    ]

    # Reproject in parallel
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        file_list = list(executor.map(_map_reproject_and_save, args_list))

    return file_list



def mosaic_from_files(file_list):
    """
    Median-stack reprojected slices stored on disk using memmap.

    Inputs
    ------
    file_list : list of str
        List of file paths to the reprojected slice FITS files.

    Outputs
    -------
    mosaic_data : ndarray
        The final mosaic data array (median stacked).
    mosaic_wcs : WCS
        The WCS object for the final mosaic (same as slices).
    """
    arrays = [fits.open(f, memmap=True)[0].data for f in file_list]

    # Stack into a single 3D array (memmap references)
    stack = np.stack(arrays, axis=0)

    # Compute nanmedian along slices axis
    mosaic_data = np.nanmedian(stack, axis=0)


    # Use WCS from first file
    wcs = WCS(fits.getheader(file_list[0]))
    
    return mosaic_data, wcs


def save_mosaic(mosaic, mosaic_wcs, output_file, file_ids, offsets, a_m, wcs_e):
    """
    Save the mosaic to a FITS file and add slice info to the header.

    Parameters
    ----------
    mosaic : ndarray
        The 2D mosaic data array.
    mosaic_wcs : WCS
        WCS object of the mosaic.
    output_file : str
        Path to save the FITS file.
    file_ids : list of str
        List of file IDs used in the mosaic.
    offsets : list of tuple
        List of (x, y) offsets applied to each slice.
    a_m : list of str
        List indicating 'a' or 'm' type for each slice.
    wcs_e : WCS
        Extra WCS object containing additional header keywords.

    Returns
    -------
    None 
    """

    # Convert headers
    mosaic_header = mosaic_wcs.to_header()

    for card in wcs_e.cards:
        key = card.keyword
        value = card.value

        # Skip spectral axis keys
        if key.endswith('3'):
            continue

        # Handle COMMENT and HISTORY specially
        if key == 'COMMENT' and value is not None:
            mosaic_header.add_comment(str(value))
        elif key == 'HISTORY' and value is not None:
            mosaic_header.add_history(str(value))
        elif key not in mosaic_header:
            mosaic_header[key] = value


    # Create primary HDU with data and merged header
    hdu = fits.PrimaryHDU(data=mosaic, header=mosaic_header)

    # Add slice info to header
    for i, (fid, off, typ) in enumerate(zip(file_ids, offsets, a_m), 1):
        hdu.header[f'FILE{i}'] = fid
        hdu.header[f'OFF{i}'] = str(off)  # store as string (x, y)
        hdu.header[f'TYPE{i}'] = typ

    # Save to FITS
    hdu.writeto(output_file, overwrite=True)
    print(f"Saved mosaic to {output_file}")




def plot_mosaic(mosaic, slice_wavelength, output_path=None):
    """
    Plots the mosaic image for a specific slice and saves it.

    Parameters
    ----------
    mosaic : ndarray
        The mosaic image data.
    slice_wavelength : float or int
        The central wavelength of the slice.
    output_path : str, optional
        The output directory path where the plot will be saved.
        Defaults to '/cephfs/apatrick/musecosmos/reduced_cubes/practice'.

    Returns
    -------
    output_png : str
        Path to the saved PNG file.
    """
    if output_path is None:
        output_path = "/cephfs/apatrick/musecosmos/reduced_cubes/slices"

    
    norm = simple_norm(mosaic, 'sqrt', percent=99.5)

    plt.figure(figsize=(10, 8))
    plt.imshow(mosaic, origin='lower', cmap='viridis', norm=norm)
    plt.title(f"Full Mosaic of slice {round(slice_wavelength, 1)} Å - nanmedian stack")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label='Flux')
    plt.tight_layout()

    # Make sure directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save plot
    output_png = os.path.join(
        output_path, f"mosaic_nanmedian_slice_{round(slice_wavelength, 1)}_full.png"
    )
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_png}")

    return output_png


def main():
    # Parse command-line arguments
    args = parse_args()

    # cubes_dir is the folder with _norm files
    cubes_dir = args.path  

    # Extract cube entries from offsets file, only keeping ones that exist in cubes_dir
    cubes = paths_ids_offsets(args.offsets_txt, cubes_dir)
    unique_cubes = {}
    for c in cubes:
        if c.file_id not in unique_cubes:
            unique_cubes[c.file_id] = c
    cubes = list(unique_cubes.values())
    print(f"Found {len(cubes)} unique cubes with offsets.")

    if len(cubes) == 0:
        raise RuntimeError(f"No matching _norm cubes found in {cubes_dir}. Check your paths.")

    print(f"Checking wavelength alignment of slices")
    # Loop over cubes: set path to _norm cubes and check slice wavelength
    for cube in cubes:
        cube.cube_path = os.path.join(cubes_dir, f"DATACUBE_FINAL_{cube.file_id}_ZAP_norm.fits")
        slice_wavelength = slice_wavelength_check(cube.cube_path, args.slice)
        csv_filename = f"{int(args.slice)}_wave.csv"
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            if not file_exists:
                csvfile.write("file_id,slice,slice_wavelength\n")
            csvfile.write(f"{cube.file_id},{args.slice},{slice_wavelength:.4f}\n")
    print(f"Completed writing {csv_filename}")

    # Extract 2D slices from _norm cubes
    slices = i_slice(cubes, args.slice)  # assumes each cube has .cube_path
    print(f"Extracted {len(slices)} slices.")

    wcs_e = slices[cubes[0].file_id]['wcs_e']  # Full WCS from first cube
    print (wcs_e)

    # Align slices using pixel offsets
    i_slice_data = [slices[cube.file_id]['data'] for cube in cubes]
    i_slice_wcs = [slices[cube.file_id]['wcs'] for cube in cubes]
    offsets = [(cube.x_offset, cube.y_offset) for cube in cubes]
    aligned_slices = align_i_slices(i_slice_data, i_slice_wcs, offsets)
    print(f"Applied pixel offsets to slices.")

    # Find common WCS area
    wcs_out, shape_out = common_wcs_area(aligned_slices)

    # Prepare tmp directory
    tmp_dir = args.tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)  # ensure it exists

    print(f"Reprojecting and saving aligned slices")
    # Reproject and save each aligned slice to disk (parallelized)
    file_list = reproject_and_save_slices(aligned_slices, wcs_out, shape_out, tmp_dir)

    # Median stack from disk using memmap
    mosaic_data, mosaic_wcs = mosaic_from_files(file_list)
    print(f"Created mosaic from {len(file_list)} reprojected slices.")

    # Prepare lists for FITS header
    file_ids = [cube.file_id for cube in cubes]
    a_m = [cube.flag for cube in cubes]  # assuming cubes have 'flag' attribute

    # Save the mosaic to FITS
    save_mosaic(mosaic_data, mosaic_wcs, args.output_file, file_ids, offsets, a_m, wcs_e)
    print(f"Saved mosaic to {args.output_file}")

    # Plot the mosaic if --plotting is True
    if getattr(args, 'plotting', False):
        print(f"Plotting mosaic for slice {round(slice_wavelength, 1)} Å")
        plot_mosaic(mosaic_data, slice_wavelength)

    # Clean up temporary directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Temporary directory {tmp_dir} removed.")

    print(f"Finished slice {args.slice} processing.")

    end_time = time.time()
    elapsed = end_time - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"\nMosaic completed in {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
