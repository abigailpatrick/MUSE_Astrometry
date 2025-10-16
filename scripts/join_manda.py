import re

# input files
offsets_file = "/cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt"
manual_file = "/cephfs/apatrick/musecosmos/scripts/aligned/manualoffsets.txt"
output_file = "/cephfs/apatrick/musecosmos/scripts/aligned/updated_offsets.txt"

def deg_to_pix(ra_deg, dec_deg, pixel_scale=0.2):
    """
    Convert RA/Dec offsets in degrees to pixel offsets.
    """
    dx_pix = (ra_deg * 3600) / pixel_scale
    dy_pix = (dec_deg * 3600) / pixel_scale
    return dx_pix, dy_pix

# parse manualoffsets into a dict
manual_dict = {}
pattern = re.compile(
    r"offset_wcs_2d\('.*?(DATACUBE_FINAL_[^']+)',\s*ra_offset_deg=([-\d.eE]+),\s*dec_offset_deg=([-\d.eE]+)\)"
)

with open(manual_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            fname, ra, dec = match.groups()
            ra, dec = float(ra), float(dec)
            dx_pix, dy_pix = deg_to_pix(ra, dec)
            manual_dict[fname] = (dx_pix, dy_pix)

# process offsets.txt and replace where matches found
updated, unchanged = 0, 0

with open(offsets_file, "r") as f, open(output_file, "w") as out:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            path, xoff, yoff, letter = parts
            fname = path.split("/")[-1]  # extract FITS filename

            if fname in manual_dict:
                dx_pix, dy_pix = manual_dict[fname]
                new_line = f"{path} {dx_pix:.5f} {dy_pix:.5f} m\n"
                out.write(new_line)
                print(f"UPDATED: {fname} → {dx_pix:.5f}, {dy_pix:.5f} pixels")
                updated += 1
            else:
                out.write(line)
                print(f"UNCHANGED (no match): {fname}")
                unchanged += 1
        else:
            out.write(line)  # malformed → just copy
            print(f"SKIPPED malformed line: {line.strip()}")

print("\nSummary:")
print(f"  Updated:   {updated}")
print(f"  Unchanged: {unchanged}")
print(f"Output written to {output_file}")