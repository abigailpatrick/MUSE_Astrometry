"""
run like this: python mosaic_chunk.py --path /cephfs/apatrick/musecosmos/reduced_cubes/full --offsets_txt /cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt --slice 100 --output_file mosaic.fits --plotting

- move all norm.fits to new file and then update input path to take that
- then run like this : python mosaic_chunk.py --path /cephfs/apatrick/musecosmos/reduced_cubes/norm --offsets_txt /cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt --slice 100 --output_file mosaic.fits --plotting

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
            wcs = WCS(hdul[1].header).celestial
        i_slice[cube.file_id] = {'data': data, 'wcs': wcs}
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



# Top-level function (not nested)
def _reproject_single(slice_dict, wcs_out, shape_out, reproject_quick):
    data, wcs = slice_dict['data'], slice_dict['wcs'].celestial
    if reproject_quick:
        array, _ = reproject_interp((data, wcs), output_projection=wcs_out, shape_out=shape_out)
    else:
        array, _ = reproject_adaptive((data, wcs), output_projection=wcs_out, shape_out=shape_out, conserve_flux=True)
    return {'data': array, 'wcs': wcs_out}

def reproject_aligned_slices(aligned_slices, wcs_out, shape_out, reproject_quick=False):
    """
    Reprojects all aligned slices onto a common WCS grid.

    Inputs
    ------
    aligned_slices : list of dict
        Each dict should have keys:
            'data' : 2D ndarray of the slice
            'wcs'  : WCS object of the aligned slice
    wcs_out : WCS
        WCS object for the common area (output grid)
    shape_out : tuple
        Shape of the common area (ny, nx)
    reproject_quick : bool, optional
        If True, use fast interpolation (reproject_interp).
        If False, use flux-conserving reproject_adaptive (slower).

    Outputs
    -------
    reprojected_slices : list of dict
        Each dict has:
            'data' : 2D ndarray reprojected onto common WCS
            'wcs'  : common WCS (wcs_out)
    """
    # Use partial to pass extra arguments to top-level function
    reproject_func = partial(_reproject_single, wcs_out=wcs_out, shape_out=shape_out, reproject_quick=reproject_quick)

    # Reproject in parallel
    with ProcessPoolExecutor() as executor:
        reprojected_slices = list(executor.map(reproject_func, aligned_slices))

    return reprojected_slices



def mosaic_reprojected_slices(reprojected_slices):
    """
    Creates a mosaic from reprojected slices.

    Inputs
    ------
    reprojected_slices : list of dict
        Each dict should have keys:
            'data' : 2D ndarray of the reprojected slice
            'wcs'  : WCS object of the reprojected slice (all the same)

    Outputs
    -------
    mosaic_data : ndarray
        The final mosaic data array (median stacked).
    mosaic_wcs : WCS
        The WCS object for the final mosaic (same as slices).
    """

    # Stack all data arrays
    stack = np.array([s['data'] for s in reprojected_slices])

    # Combine using nanmedian
    mosaic_data = np.nanmedian(stack, axis=0)

    # WCS is the same for all reprojected slices
    mosaic_wcs = reprojected_slices[0]['wcs']

    return mosaic_data, mosaic_wcs




def save_mosaic(mosaic, mosaic_wcs, output_file, file_ids, offsets, a_m):
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

    Returns
    -------
    None
    """

    # Create primary HDU with data and WCS header
    hdu = fits.PrimaryHDU(data=mosaic, header=mosaic_wcs.to_header())

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
        output_path = "/cephfs/apatrick/musecosmos/reduced_cubes/practice"

    method = 'combined'
    norm = simple_norm(mosaic, 'sqrt', percent=99.5)

    plt.figure(figsize=(10, 8))
    plt.imshow(mosaic, origin='lower', cmap='viridis', norm=norm)
    plt.title(f"Full Mosaic {slice_wavelength} ({method}) - nanmedian stack")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label='Flux')
    plt.tight_layout()

    # Make sure directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save plot
    output_png = os.path.join(
        output_path, f"mosaic_nanmedian_slice_{slice_wavelength}_{method}_full.png"
    )
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_png}")

    return output_png


def main():
    # Parse command-line arguments
    args = parse_args()

    # Extract cube entries from offsets file
    cubes = paths_ids_offsets(args.offsets_txt)

    # Loop over cubes: set path to _norm cubes and check slice wavelength
    for cube in cubes:
        cube.cube_path = os.path.join(args.path, f"DATACUBE_FINAL_{cube.file_id}_ZAP_norm.fits") # Need to used the cubes with the realignwaves.py applied

        # Check wavelength axis for the selected slice
        slice_wavelength = slice_wavelength_check(cube.cube_path, args.slice)
        print(f"[{cube.file_id}] Slice {args.slice} wavelength = {slice_wavelength:.2f} Å")

    # Extract 2D slices from _norm cubes
    slices = i_slice(cubes, args.slice)   # assumes each cube has .cube_path

    # Align slices using pixel offsets
    i_slice_data = [slices[cube.file_id]['data'] for cube in cubes]
    i_slice_wcs = [slices[cube.file_id]['wcs'] for cube in cubes]
    offsets = [(cube.x_offset, cube.y_offset) for cube in cubes]
    aligned_slices = align_i_slices(i_slice_data, i_slice_wcs, offsets)

    # Find common WCS area
    wcs_out, shape_out = common_wcs_area(aligned_slices)

    # Reproject all aligned slices onto the common WCS grid
    reprojected_slices = reproject_aligned_slices(aligned_slices, wcs_out, shape_out)

    # Create a mosaic from reprojected slices
    mosaic_data, mosaic_wcs = mosaic_reprojected_slices(reprojected_slices)

    # Prepare lists for FITS header
    file_ids = [cube.file_id for cube in cubes]
    a_m = [cube.flag for cube in cubes]  # assuming cubes have 'flag' attribute

    # Save the mosaic to FITS
    save_mosaic(mosaic_data, mosaic_wcs, args.output_file, file_ids, offsets, a_m)

    # Plot the mosaic if --plotting is True
    if getattr(args, 'plotting', False):
        plot_mosaic(mosaic_data, args.slice)


if __name__ == "__main__":
    main()
