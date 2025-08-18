from astropy.table import Table, vstack
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def get_halpha_candidates(band, mosaic_fits, wcs, mosaic_data):
    # Read catalog
    fits_path = f"/home/apatrick/P1/JELSDP/jels_{band}_detected_source_catalogue_v1.0.fits"
    catalog = Table.read(fits_path)

    # Filter by redshift and quality
    halpha_candidates = catalog[
        (catalog['z1_median'] >= 5.5) &
        (catalog['z1_median'] <= 6.5) &
        (catalog['Emission_line_goodz'] == True)
    ]

    print (f"Found {len(halpha_candidates)} H-alpha candidates for band {band}")

    # Exclude manually flagged IDs
    if band == "F466N":
        exclude_ids = [2394, 3962, 7164, 8303, 9336]
    elif band == "F470N":
        exclude_ids = [6228]
    else:
        exclude_ids = []

    halpha_candidates = halpha_candidates[~np.isin(halpha_candidates['ID'], exclude_ids)]
    print(f"After excluding flagged IDs, {len(halpha_candidates)} candidates remain for band {band}")

    # Convert to SkyCoord
    ra_values = halpha_candidates['ra'].data
    dec_values = halpha_candidates['dec'].data
    sky_coords = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg, frame='icrs')

    # Convert RA/Dec to pixel coords
    x_pix, y_pix = wcs.world_to_pixel(sky_coords)

    ny, nx = mosaic_data.shape

    # Mask for in-bounds coords
    in_bounds_mask = (
        (x_pix >= 0) & (x_pix < nx) &
        (y_pix >= 0) & (y_pix < ny)
    )

    x_pix_in = x_pix[in_bounds_mask]
    y_pix_in = y_pix[in_bounds_mask]
    x_int = x_pix_in.astype(int)
    y_int = y_pix_in.astype(int)

    # Mask out NaN pixels
    not_nan_mask = ~np.isnan(mosaic_data[y_int, x_int])

    # Combine masks to get final valid mask
    valid_mask = np.zeros(len(halpha_candidates), dtype=bool)
    valid_mask[in_bounds_mask] = not_nan_mask

    # Filter candidates on valid pixels
    halpha_candidates['in_mosaic'] = valid_mask
    valid_candidates = halpha_candidates[valid_mask]

    # Add pixel coords to table
    valid_candidates['x_pix'] = x_pix[valid_mask]
    valid_candidates['y_pix'] = y_pix[valid_mask]

    valid_candidates['band'] = band

    print(f"{len(valid_candidates)} valid H-alpha candidates found for band {band}")

    # Save valid candidates as csv
    csv_path = f"/home/apatrick/P1/outputfiles/jels_{band}_halpha_candidates.csv"
    valid_candidates.write(csv_path, format='csv', overwrite=True)
    print(f"Saved valid candidates to {csv_path}")

    return valid_candidates

# Load MUSE mosaic once
mosaic_fits = '/home/apatrick/Code/aligned/mosaic_whitelight_nanmedian_all_new_full.fits'
with fits.open(mosaic_fits) as hdul:
    mosaic_data = hdul[0].data
    wcs = WCS(hdul[0].header)

# Get candidates for both bands
candidates_466 = get_halpha_candidates("F466N", mosaic_fits, wcs, mosaic_data)
candidates_470 = get_halpha_candidates("F470N", mosaic_fits, wcs, mosaic_data)

# Combine tables
all_candidates = vstack([candidates_466, candidates_470])

print(f"Total combined candidates: {len(all_candidates)}")

# Print list of candidate IDs
print("Combined candidate IDs:", np.array(all_candidates['ID']))

# ---------- PLOTTING ----------
fig, ax = plt.subplots(figsize=(10, 10))
vmin, vmax = np.nanpercentile(mosaic_data, [5, 99])
ax.imshow(mosaic_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

# Plot all candidates with different colors by band
colors = {'F466N': 'red', 'F470N': 'blue'}
for i, row in enumerate(all_candidates):
    x = row['x_pix']
    y = row['y_pix']
    band = row['band']
    color = colors.get(band, 'black')

    circ = Circle((x, y), radius=5, edgecolor=color, facecolor='none', lw=1.0)
    ax.add_patch(circ)
    ax.text(x + 7, y + 7, str(i + 1), color=color, fontsize=7)


ax.set_title('Combined H-alpha candidates on MUSE mosaic (Red=F466N, Blue=F470N)')
ax.set_xlabel('X pixel')
ax.set_ylabel('Y pixel')
plt.grid(False)
plt.tight_layout()

save_path = "/home/apatrick/P1/plots/halpha_candidates_combined.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved combined plot to {save_path}")
plt.show()
