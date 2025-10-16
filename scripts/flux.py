import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, aperture_photometry
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

# -----------------------------------------------------------
# User inputs
# -----------------------------------------------------------
catalog_file = "/cephfs/apatrick/musecosmos/dataproducts/hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits"
mosaic_fits  = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaic_whitelight_nanmedian_all_new_full.fits"
cube_file    = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits"
output_csv   = "muse_hst_flux_comparison.csv"

wl_center = 6060.0     # Angstrom, center of 100 Å range
wl_width  = 100.0      # Å width
initial_ap_radii = np.arange(0.1, 1.2, 0.1)  # arcsec radii to test for curve of growth
BUNIT_factor = 1e-20   # from MUSE header
c_cgs = 2.998e10       # cm/s
lambda_A = wl_center   # Å

# -----------------------------------------------------------
# Find sources inside mosaic
# -----------------------------------------------------------
def sources_in_mosaic(source_catalog, wcs, mosaic_data):
    sky_coords = SkyCoord(ra=source_catalog["RA"].values*u.deg,
                          dec=source_catalog["DEC"].values*u.deg)
    x_pix, y_pix = wcs.world_to_pixel(sky_coords)
    ny, nx = mosaic_data.shape
    in_bounds = (x_pix >= 0) & (x_pix < nx) & (y_pix >= 0) & (y_pix < ny)
    valid = source_catalog[in_bounds].copy()
    valid["x_pix"] = x_pix[in_bounds]
    valid["y_pix"] = y_pix[in_bounds]
    print(f"{len(valid)} sources inside mosaic footprint.")
    return valid

# -----------------------------------------------------------
# Step 1: Read HST catalog and filter for point sources
# -----------------------------------------------------------
print("Reading HST catalog...")
hst_table = Table.read(catalog_file)
df = hst_table.to_pandas()
stars = df[df["CLASS_STAR"] > 0.9]
print(f"{len(stars)} sources with CLASS_STAR > 0.9")

# -----------------------------------------------------------
# Step 2: Restrict to sources inside mosaic
# -----------------------------------------------------------
with fits.open(mosaic_fits) as hdul:
    mosaic_data = hdul[0].data
    mosaic_wcs = WCS(hdul[0].header)

valid_sources = sources_in_mosaic(stars, mosaic_wcs, mosaic_data)

# -----------------------------------------------------------
# Step 3: Load only 100 Å of cube (memory safe)
# -----------------------------------------------------------
with fits.open(cube_file, memmap=True) as hdul:
    hdr = hdul[0].header      
    cube = hdul[0].data       

    # MUSE uses CD3_3 instead of CDELT3
    crval3 = hdr["CRVAL3"]
    cdelt3 = hdr["CD3_3"]
    crpix3 = hdr["CRPIX3"]

    # Build wavelength axis
    n_wave = cube.shape[0]
    wave = crval3 + (np.arange(n_wave) + 1 - crpix3) * cdelt3
    print(f"Wavelength range in cube: {wave.min():.1f}–{wave.max():.1f} Å")

    # Pick your wavelength window (e.g. around 6060 Å)
    sel = (wave > 6060 - 100/2) & (wave < 6060 + 100/2)
    
    print(f" slice index of selected center: {np.where(sel)[0][len(sel)//2]}")
    print(f"Number of selected wavelength planes: {np.sum(sel)}")
    print(f"Selected wavelength range: {wave[sel].min():.1f}–{wave[sel].max():.1f} Å")


    if not np.any(sel):
        raise ValueError(f"No wavelength planes found near {wl_center} Å!")

    # Average the selected wavelength slice
    avg_image = np.nanmedian(cube[sel, :, :], axis=0)

    # Set up WCS and pixel scale
    muse_wcs = WCS(hdr).celestial
    pixscale = np.abs(hdr["CD1_1"]) * 3600.0  # arcsec/pix
    print(f"Pixel scale: {pixscale:.3f} arcsec/pix")

# -----------------------------------------------------------
# Step 4: Curve of growth test on bright source(s)
# -----------------------------------------------------------
# Pick a bright isolated star (or median flux star)
test_star = valid_sources.iloc[len(valid_sources)//2]  # pick one in middle of flux range
sky = SkyCoord(test_star["RA"]*u.deg, test_star["DEC"]*u.deg)
x0, y0 = muse_wcs.world_to_pixel(sky)
print(f"Testing curve of growth at pixel ({x0:.1f}, {y0:.1f})")

r_pix = initial_ap_radii / pixscale
total_fluxes = []

for r in r_pix:
    ap = CircularAperture((x0, y0), r=r)
    phot = aperture_photometry(avg_image, ap)
    total_fluxes.append(phot["aperture_sum"][0])

total_fluxes = np.array(total_fluxes)
norm_flux = total_fluxes / total_fluxes[-1]

plt.figure()
plt.plot(initial_ap_radii, norm_flux, "o-", color="royalblue")
plt.axhline(0.9, color="r", ls="--", label="90% flux")
plt.xlabel("Aperture radius (arcsec)")
plt.ylabel("Normalized enclosed flux")
plt.title("MUSE PSF curve of growth")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Find radius enclosing ~90% flux
r90_arcsec = np.interp(0.9, norm_flux, initial_ap_radii)
print(f"Approximate 90% flux radius: {r90_arcsec:.2f} arcsec")

# -----------------------------------------------------------
# Step 5: Measure flux for all stars with this radius
# -----------------------------------------------------------
sky_coords = SkyCoord(valid_sources["RA"]*u.deg, valid_sources["DEC"]*u.deg)
x_pix, y_pix = muse_wcs.world_to_pixel(sky_coords)
apertures = CircularAperture(np.transpose([x_pix, y_pix]), r=r90_arcsec/pixscale)
phot_table = aperture_photometry(avg_image, apertures)
f_lambda = phot_table["aperture_sum"] * BUNIT_factor  # erg/s/cm²/Å
f_nu = f_lambda * (lambda_A*1e-8)**2 / c_cgs
f_nu_uJy = f_nu / 1e-29
valid_sources["MUSE_FNU_UJY"] = f_nu_uJy

# -----------------------------------------------------------
# Step 6: Compare to HST F606W flux
# -----------------------------------------------------------
hst_flux = valid_sources["ACS_F606W_FLUX"]
plt.figure(figsize=(6,6))
plt.scatter(hst_flux, f_nu_uJy, s=12, alpha=0.7)
lims = [1e-3, 1e3]
plt.plot(lims, lims, "k--")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("HST F606W fν (µJy)")
plt.ylabel("MUSE fν (µJy)")
plt.title("Flux conservation check: HST vs MUSE (100 Å @ 6060 Å)")
plt.grid(alpha=0.3, which="both")
plt.show()

# Save table
valid_sources[["RA","DEC","ACS_F606W_FLUX","MUSE_FNU_UJY"]].to_csv(output_csv, index=False)
print(f"Saved results to {output_csv}")

def main():

    args = parse_args()

    # Select star candidates from HST catalog
    star_candidates = select_stars(args.hst_catalog)

    with fits.open(mosaic_wl_slice) as hdul:
        mosaic_data = hdul[0].data
        wcs = WCS(hdul[0].header)

    valid_sources = sources_in_mosaic(star_candidates, wcs, mosaic_data)
    
    # Restrict to flux column only 
    valid_sources = valid_sources[['RA', 'DEC', 'ACS_F606W_FLUX', 'ACS_F606W_FLUXERR']].copy()
    print(f"Measuring fluxes for {len(valid_sources)} star candidates in mosaic slice.")

    
"""


cube_file = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaic_cube.fits"

# Pick a slice in the range you want to check
slice_index = 1000  # e.g., somewhere in the middle of the cube

with fits.open(cube_file, memmap=True) as hdul:
    hdr = hdul[0].header
    cube = hdul[0].data
    slice_data = cube[slice_index, :, :]  # 2D image of that wavelength
    wcs = WCS(hdr).celestial

# Optional: get wavelength of this slice
crval3 = hdr["CRVAL3"]
cdelt3 = hdr["CD3_3"]
crpix3 = hdr["CRPIX3"]
wave = crval3 + (slice_index + 1 - crpix3) * cdelt3
print(f"Slice {slice_index} wavelength: {wave:.1f} Å")

# Plot the slice
plt.figure(figsize=(8,6))
plt.imshow(slice_data, origin='lower', cmap='inferno', interpolation='none')
plt.colorbar(label='10^-20 erg/s/cm^2/Å')
plt.title(f"MUSE cube slice {slice_index}, λ = {wave:.1f} Å")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.show()

# save plt
output_png = f"muse_cube_slice_{slice_index}.png"
plt.savefig(output_png, dpi=300)
print(f"Saved plot to {output_png}")




"""