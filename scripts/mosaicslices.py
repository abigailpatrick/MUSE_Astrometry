"""
inputs
 - a path to a directory full of fits files (cubes)
 - a txt files full of offsetsto be applied to each cube 

i want to find the first wavelength slice (2d image) from every cube input 
then assign these slices to one job 
in this job i want to :
- use my apply_offsets function to apply the txt files offsets corresponding to each cube to the right slice to get an 'aligned slice'.
- then I want to use  my reproject and mosaic function to create one 2d slice mosaic of all these aligned images combined
- I then want to plot the mosaic slice 

later i want to run a new job in a new script that takes all these 2d slices and stacks them into a megacube
output
- a full mosaic 2d slice for each wavelength in the cubes 
- a full mega cube of all the 2d slice mosaics stacked back into a cube


"""
 # script 1

import os
import glob
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_adaptive
from reproject.mosaicking import find_optimal_celestial_wcs
from concurrent.futures import ProcessPoolExecutor


def apply_offsets(image_2d, hdr, offset):
    dx, dy = offset
    wcs = WCS(hdr, naxis=2)
    wcs.wcs.crpix[0] -= dx  # RA axis
    wcs.wcs.crpix[1] -= dy  # Dec axis
    hdr.update(wcs.to_header())
    hdr['ASTROM_OFF'] = f'dx={dx:.5f}, dy={dy:.5f}'
    return image_2d, hdr

def reproject_single(args, wcs_out, shape_out, use_interp=False):
    data, wcs = args
    if use_interp:
        array, _ = reproject_interp((data, wcs), output_projection=wcs_out, shape_out=shape_out)
    else:
        array, _ = reproject_adaptive((data, wcs), output_projection=wcs_out,
                                      shape_out=shape_out, conserve_flux=True)
    return array

def reproject_and_mosaic(images, headers, use_interp=False, padding=100):
    wcs_list = [WCS(h).celestial for h in headers]
    wcs_out, shape_out = find_optimal_celestial_wcs(zip(images, wcs_list))

    # Add padding
    shape_out = (shape_out[0] + padding, shape_out[1] + padding)
    wcs_out.wcs.crpix[0] += padding // 2
    wcs_out.wcs.crpix[1] += padding // 2

    # Reproject all images in parallel
    reproject_args = list(zip(images, wcs_list))
    with ProcessPoolExecutor() as executor:
        reprojected_images = list(executor.map(lambda a: reproject_single(a, wcs_out, shape_out, use_interp),
                                               reproject_args))

    # Stack with nanmedian
    stack = np.array(reprojected_images)
    mosaic = np.nanmedian(stack, axis=0)

    # Convert WCS to header
    mosaic_hdr = wcs_out.to_header()
    return mosaic, mosaic_hdr

def main(cube_dir, offsets_file, slice_index, output_dir):
    # Load offsets
    offsets = {}
    with open(offsets_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                fname, dx, dy = parts
                offsets[fname] = (float(dx), float(dy))

    cube_files = sorted(glob.glob(os.path.join(cube_dir, "*.fits")))
    aligned_slices = []
    headers = []

    for cube_file in cube_files:
        with fits.open(cube_file) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header

            if slice_index >= data.shape[0]:
                print(f"WARNING: slice {slice_index} out of range for {cube_file}")
                continue

            slice_2d = data[slice_index, :, :]
            fname = os.path.basename(cube_file)
            if fname in offsets:
                slice_2d, hdr = apply_offsets(slice_2d, hdr, offsets[fname])

            aligned_slices.append(slice_2d)
            headers.append(hdr)

    if not aligned_slices:
        raise ValueError(f"No valid slices found for index {slice_index}")

    mosaic, mosaic_hdr = reproject_and_mosaic(aligned_slices, headers)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"mosaic_slice_{slice_index:04d}.fits")
    fits.writeto(output_file, mosaic, header=mosaic_hdr, overwrite=True)
    print(f"Saved mosaic slice {slice_index} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mosaic one wavelength slice from all cubes using WCS offsets")
    parser.add_argument("cube_dir", help="Path to directory with input FITS cubes")
    parser.add_argument("offsets_file", help="Path to text file with offsets")
    parser.add_argument("slice_index", type=int, help="Index of wavelength slice to process")
    parser.add_argument("--output_dir", default="mosaics", help="Directory to save mosaic slices")

    args = parser.parse_args()
    main(args.cube_dir, args.offsets_file, args.slice_index, args.output_dir)
