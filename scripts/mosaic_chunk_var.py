import os
import argparse
import csv
import time
import shutil
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_adaptive


start_time = time.time()  # record start


def parse_args():
    parser = argparse.ArgumentParser(description="MUSE variance slice mosaicking pipeline (variance-only)")

    parser.add_argument('--path', type=str, required=True,
                        help='Path to directory containing the cubes')
    parser.add_argument('--offsets_txt', type=str, required=True,
                        help='Text file containing cube offsets')
    parser.add_argument('--slice', type=int, required=True,
                        help='Wavelength slice to extract from each cube (0-indexed)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output FITS file path for the summed-variance mosaic')
    parser.add_argument('--mask_percent', type=float, default=1.0,
                        help='Percentage of lowest pixels to mask (default 1.0); must match white-light mask creation step')
    parser.add_argument('--plotting', action='store_true',
                        help='Enable plotting of the variance mosaic')
    parser.add_argument('--tmp_dir', type=str,
                        default='/cephfs/apatrick/musecosmos/scripts/aligned/tmp_slice_var',
                        help='Temporary directory for reprojected slices')

    return parser.parse_args()


@dataclass
class CubeEntry:
    file_id: str       # e.g. "Autocal_3687411a_1"
    x_offset: float
    y_offset: float
    flag: str          # 'a' or 'm'
    cube_path: str = ""


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
                    cubes[file_id] = CubeEntry(file_id=file_id, x_offset=x_offset, y_offset=y_offset,
                                               flag=flag, cube_path=cube_path)
                elif not os.path.exists(cube_path):
                    print(f"Skipping {norm_filename}, not found in {cubes_dir}")

    return list(cubes.values())


def var_slice(cubes, slice_number):
    """
    Extract a 2D variance slice from each cube (HDU 2: STAT).
    Returns dict: {file_id: {'data': 2D array, 'wcs': celestial WCS, 'wcs_e': full header}}
    """
    out = {}
    for cube in cubes:
        with fits.open(cube.cube_path) as hdul:
            data = hdul[2].data[slice_number]
            header = hdul[2].header.copy()
            wcs2d = WCS(header).celestial
            out[cube.file_id] = {'data': data, 'wcs': wcs2d, 'wcs_e': header}
    return out


def apply_saved_masks(aligned_slices, cubes, mask_dir, mask_percent=1.0):
    """
    Apply precomputed white-light masks to each aligned slice.
    True mask pixels are set to NaN, excluded from sum.
    """
    masked_slices = []
    for i, cube in enumerate(cubes):
        mask_filename = f"DATACUBE_FINAL_{cube.file_id}_ZAP_img_aligned_mask{int(mask_percent)}p.fits"
        mask_path = os.path.join(mask_dir, mask_filename)

        if os.path.exists(mask_path):
            mask_data = fits.getdata(mask_path).astype(bool)
            masked_data = np.where(mask_data, np.nan, aligned_slices[i]['data'])
            masked_slices.append({
                'data': masked_data,
                'wcs': aligned_slices[i]['wcs'],
                'applied_offset': aligned_slices[i]['applied_offset']
            })
            print(f"Applied saved mask to {cube.file_id}")
        else:
            masked_slices.append(aligned_slices[i])
            print(f"WARNING: No mask found for {cube.file_id}, skipping mask")
    return masked_slices


def slice_wavelength_check(cube_path, slice_number, expected_start=4749.9, expected_step=1.25):
    """
    Validate wavelength using HDU 1 (DATA) spectral WCS (independent of variance).
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


def align_slices(slice_data, slice_wcs, offsets):
    """
    Apply pixel offsets to WCS only (data remain unchanged).
    """
    aligned = []
    for data, wcs, (dx, dy) in zip(slice_data, slice_wcs, offsets):
        new_wcs = wcs.deepcopy()
        new_wcs.wcs.crpix[0] -= dx  # RA
        new_wcs.wcs.crpix[1] += dy  # Dec
        aligned.append({'data': data, 'wcs': new_wcs, 'applied_offset': (dx, dy)})
    return aligned


def common_wcs_area(aligned_slices):
    """
    Find optimal common WCS area with optional padding.
    """
    slice_list = [(s['data'], s['wcs'].celestial) for s in aligned_slices]
    wcs_out, shape_out = find_optimal_celestial_wcs(slice_list)

    # Optional padding
    pad_y, pad_x = 100, 100
    shape_out = (shape_out[0] + pad_y, shape_out[1] + pad_x)
    wcs_out.wcs.crpix[0] += pad_x // 2
    wcs_out.wcs.crpix[1] += pad_y // 2
    return wcs_out, shape_out


def reproject_and_save_single(i, slice_dict, wcs_out, shape_out, output_dir):
    """
    Reproject a single aligned slice (variance) onto common WCS and save.
    """
    os.makedirs(output_dir, exist_ok=True)
    data, wcs = slice_dict['data'], slice_dict['wcs'].celestial

    # Using same reprojection method as data path - I should maybe try false for conserve_flux?
    array, _ = reproject_adaptive((data, wcs), output_projection=wcs_out,
                                  shape_out=shape_out, conserve_flux=False)

    header = wcs_out.to_header()
    fname = os.path.join(output_dir, f"reproj_slice_{i:03d}_pid{os.getpid()}.fits")
    fits.writeto(fname, array, header=header, overwrite=True)
    return fname


def _map_reproject_and_save(args):
    return reproject_and_save_single(*args)


def reproject_and_save_slices(aligned_slices, wcs_out, shape_out, output_dir):
    """
    Parallel reprojection of all slices. Returns list of file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    args_list = [(i, s, wcs_out, shape_out, output_dir) for i, s in enumerate(aligned_slices)]
    with ProcessPoolExecutor(max_workers=8) as executor:
        file_list = list(executor.map(_map_reproject_and_save, args_list))
    return file_list


def variance_sum_from_files(file_list):
    """
    Combine reprojected variance slices using inverse-variance summation.

    For each pixel position across all input variance maps:
      - Compute the sum of inverse variances:  inv_sum = Σ (1 / var_i)
        where var_i are valid (finite and > 0) variance values from each exposure.
      - The combined variance is then given by:  combined_var = 1 / inv_sum
        if inv_sum > 0, otherwise it is set to NaN.

    This calculation returns the propagated variance that corresponds to an
    inverse-variance–weighted mean of the underlying data values.
    It naturally decreases in regions of overlap where multiple exposures
    contribute, reflecting higher confidence (lower uncertainty) there.

    """
    sum_inv = None
    first_header = None

    for i, f in enumerate(file_list):
        data = fits.getdata(f, memmap=True)  # reprojected variance slice

        # Mask out bad values: NaN, non-finite, or non-positive variances don't contribute
        good = np.isfinite(data) & (data > 0.0)

        # compute inverse variance safely: set inv where good, else zero
        inv = np.zeros_like(data, dtype=float)
        if np.any(good):
            inv[good] = 1.0 / data[good]

        if sum_inv is None:
            sum_inv = np.zeros_like(inv, dtype=float)
            first_header = fits.getheader(f)

        # accumulate inverse-variance
        sum_inv += inv

    # convert sum_inv -> combined variance
    combined_var = np.full_like(sum_inv, np.nan, dtype=float)
    positive = sum_inv > 0.0
    combined_var[positive] = 1.0 / sum_inv[positive]

    wcs = WCS(first_header) if first_header is not None else None
    return combined_var, wcs



def save_variance_mosaic(mosaic, mosaic_wcs, output_file, file_ids, offsets, a_m, wcs_e):
    """
    Save the summed-variance mosaic to FITS with merged headers and provenance.
    """
    mosaic_header = mosaic_wcs.to_header()

    # Merge non-spectral header keys from the variance HDU
    for card in wcs_e.cards:
        key = card.keyword
        value = card.value

        if key.endswith('3'):
            continue
        if key == 'COMMENT' and value is not None:
            mosaic_header.add_comment(str(value))
        elif key == 'HISTORY' and value is not None:
            mosaic_header.add_history(str(value))
        elif key not in mosaic_header:
            mosaic_header[key] = value

    # Annotate that this is a variance sum
    mosaic_header.add_history("Variance mosaic created by summing STAT (HDU 2) slices after reprojection.")
    mosaic_header['VARCOMB'] = ('INVVAR', 'Variance combined via inverse-variance weighting (1/sum(1/var))')


    hdu = fits.PrimaryHDU(data=mosaic, header=mosaic_header)
    for i, (fid, off, typ) in enumerate(zip(file_ids, offsets, a_m), 1):
        hdu.header[f'FILE{i}'] = fid
        hdu.header[f'OFF{i}'] = str(off)
        hdu.header[f'TYPE{i}'] = typ

    hdu.writeto(output_file, overwrite=True)
    print(f"Saved variance mosaic to {output_file}")


def plot_mosaic(mosaic, slice_wavelength, output_path=None):
    """
    Plot the variance mosaic image for a specific slice and save it.
    """
    if output_path is None:
        output_path = "/cephfs/apatrick/musecosmos/reduced_cubes/slices"

    norm = simple_norm(mosaic, 'sqrt', percent=99.5)

    plt.figure(figsize=(10, 8))
    plt.imshow(mosaic, origin='lower', cmap='viridis', norm=norm)
    plt.title(f"Full Variance Mosaic of slice {round(slice_wavelength, 1)} Å - sum of variances")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label='Variance')
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    output_png = os.path.join(output_path, f"mosaic_varsum_slice_{round(slice_wavelength, 1)}_full.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_png}")
    return output_png


def main():
    args = parse_args()

    cubes_dir = args.path
    cubes = paths_ids_offsets(args.offsets_txt, cubes_dir)

    # Ensure unique by file_id
    unique = {}
    for c in cubes:
        unique.setdefault(c.file_id, c)
    cubes = list(unique.values())

    print(f"Found {len(cubes)} unique cubes with offsets.")
    if len(cubes) == 0:
        raise RuntimeError(f"No matching _norm cubes found in {cubes_dir}. Check your paths.")

    print("Checking wavelength alignment of slices")
    for cube in cubes:
        cube.cube_path = os.path.join(cubes_dir, f"DATACUBE_FINAL_{cube.file_id}_ZAP_norm.fits")
        slice_wavelength = slice_wavelength_check(cube.cube_path, args.slice)
        csv_filename = f"{int(args.slice)}_wave.csv"
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            if not file_exists:
                csvfile.write("file_id,slice,slice_wavelength\n")
            csvfile.write(f"{cube.file_id},{args.slice},{slice_wavelength:.4f}\n")
    print(f"Completed writing {int(args.slice)}_wave.csv")

    # Extract variance slices from HDU 2
    slices = var_slice(cubes, args.slice)
    print(f"Extracted {len(slices)} variance slices (HDU 2).")

    wcs_e = slices[cubes[0].file_id]['wcs_e']  # full header from STAT HDU

    # Align via offsets
    slice_data = [slices[c.file_id]['data'] for c in cubes]
    slice_wcs = [slices[c.file_id]['wcs'] for c in cubes]
    offsets = [(c.x_offset, c.y_offset) for c in cubes]
    aligned_slices = align_slices(slice_data, slice_wcs, offsets)
    print("Applied pixel offsets to variance slices.")

    # Apply saved masks
    mask_dir = '/cephfs/apatrick/musecosmos/scripts/aligned/masks'
    aligned_slices = apply_saved_masks(aligned_slices, cubes, mask_dir, args.mask_percent)
    print(f"Applied {args.mask_percent}% masks to variance slices.")

    # Find common WCS
    wcs_out, shape_out = common_wcs_area(aligned_slices)

    # Reproject all slices to common grid
    tmp_dir = args.tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    print("Reprojecting and saving aligned variance slices")
    file_list = reproject_and_save_slices(aligned_slices, wcs_out, shape_out, tmp_dir)

    # Sum variance across reprojected slices
    mosaic_data, mosaic_wcs = variance_sum_from_files(file_list)
    print(f"Created variance-sum mosaic from {len(file_list)} reprojected slices.")

    # Save mosaic
    file_ids = [c.file_id for c in cubes]
    a_m = [c.flag for c in cubes]
    save_variance_mosaic(mosaic_data, mosaic_wcs, args.output_file, file_ids, offsets, a_m, wcs_e)

    # Plot if requested
    if getattr(args, 'plotting', False):
        print(f"Plotting variance mosaic for slice {round(slice_wavelength, 1)} Å")
        plot_mosaic(mosaic_data, slice_wavelength)

    # Cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Temporary directory {tmp_dir} removed.")

    print(f"Finished variance slice {args.slice} processing.")

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"\nVariance mosaic completed in {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()


"""
to run:
python mosaic_chunk_var.py --path /cephfs/apatrick/musecosmos/reduced_cubes/norm/ --offsets_txt /cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt --slice 123 --output_file /cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full/var_mosaic_slice123.fits --mask_percent 1.0 --plotting

"""