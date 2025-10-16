#!/usr/bin/env python
import os
from astropy.io import fits

# Folder where your slice outputs are saved
SLICE_DIR = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full"

# Slice range to check
START_SLICE = 1532
END_SLICE = 1732

# List all FITS files in that directory
slice_files = sorted([f for f in os.listdir(SLICE_DIR) if f.endswith(".fits")])

if not slice_files:
    print(f"No FITS files found in {SLICE_DIR}")
    exit(1)

print(f"Found {len(slice_files)} total files. Filtering slices {START_SLICE}-{END_SLICE}...\n")

# Filter files by slice number in filename
filtered_files = []
for f in slice_files:
    try:
        # Extract slice number from filename
        num_part = f.split("_")[2]  # assumes "mosaic_slice_1532.fits"
        slice_num = int(num_part.split(".")[0])
        if START_SLICE <= slice_num <= END_SLICE:
            filtered_files.append((slice_num, f))
    except:
        continue

print(f"Checking {len(filtered_files)} slices...\n")

# Loop over files and print the shape of each slice
for slice_num, f in filtered_files:
    path = os.path.join(SLICE_DIR, f)
    try:
        with fits.open(path, memmap=True) as hdulist:
            data = hdulist[0].data
            if data is None:
                print(f"{f}: data is None")
            else:
                print(f"Slice {slice_num}: shape = {data.shape}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
print("\nDone.")