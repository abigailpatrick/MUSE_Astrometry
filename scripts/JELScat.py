from astropy.table import Table, vstack
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



def get_halpha_candidates(band, z_low=5.5, z_high=6.5, strict=True):
    # Read catalog
    fits_path = f"/home/apatrick/P1/JELSDP/jels_{band}_detected_source_catalogue_v1.0.fits"
    catalog = Table.read(fits_path)
    

    # Define emission line condition
    if strict:
        em_cond = catalog['Emission_line_goodz'] == True
    else:
        em_cond = catalog['Emission_line'] == True

    # Filter by redshift and emission line
    halpha_candidates = catalog[
        (catalog['z1_median'] >= z_low) &
        (catalog['z1_median'] <= z_high) &
        em_cond
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

    return halpha_candidates

# Get candidates for both bands
candidates_466 = get_halpha_candidates("F466N", z_low=5.5, z_high=6.5, strict=True)
candidates_470 = get_halpha_candidates("F470N", z_low=5.5, z_high=6.5, strict=True)

# Combine tables
all_candidates = vstack([candidates_466, candidates_470])

print(f"Total combined JELS H-alpha candidates 5.5<z<6.5: {len(all_candidates)}")

# Get candidates for both bands
candidates_466_w = get_halpha_candidates("F466N", z_low=4.92, z_high=6.69, strict=False)
candidates_470_w = get_halpha_candidates("F470N", z_low=4.92, z_high=6.69, strict=False)

# Combine tables
all_candidates_w = vstack([candidates_466_w, candidates_470_w]) 

print(f"Total combined JELS H-alpha candidates 4.92<z<6.69: {len(all_candidates_w)}")
#""" 
def get_candidates_in_mosaic(band,halpha_candidates, wcs, mosaic_data):

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
mosaic_fits = '/cephfs/apatrick/musecosmos/scripts/aligned/mosaic_whitelight_nanmedian_all_new_full.fits'
with fits.open(mosaic_fits) as hdul:
    mosaic_data = hdul[0].data
    wcs = WCS(hdul[0].header)

# Get candidates for both bands
candidates_466_mosaic = get_candidates_in_mosaic("F466N", candidates_466, wcs, mosaic_data)
candidates_470_mosaic = get_candidates_in_mosaic("F470N", candidates_470, wcs, mosaic_data)

# Combine tables
all_candidates_mosaic = vstack([candidates_466_mosaic, candidates_470_mosaic])
# limit to just ra dec and z columns 
ha_candidates_mosaic = all_candidates_mosaic['ra', 'dec', 'z1_median']

# Save combined valid candidates as csv
csv_path = "/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic.csv"
ha_candidates_mosaic.write(csv_path, format='csv', overwrite=True)
print(f"Saved combined valid candidates to {csv_path}")
"""

print(f"Total combined candidates in current MUSE mosaic: {len(all_candidates_mosaic)}")

# Print list of candidate IDs
print("Combined candidates in mosaic IDs:", np.array(all_candidates_mosaic['ID']))

# ---------- PLOTTING ----------
fig, ax = plt.subplots(figsize=(10, 10))
vmin, vmax = np.nanpercentile(mosaic_data, [5, 99])
ax.imshow(mosaic_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

# Plot all candidates with different colors by band
colors = {'F466N': 'red', 'F470N': 'blue'}
for i, row in enumerate(all_candidates_mosaic):
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

save_path = "/cephfs/apatrick/musecosmos/scripts/aligned/halpha_candidates_combined.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved combined plot to {save_path}")
plt.show()

"""
