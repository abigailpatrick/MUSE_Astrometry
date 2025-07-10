
from astropy.io import fits
from astropy.wcs import WCS
import os


"""
Take input dec offset and ra offset in deg as well as muse image with wcs.
output a new fits file with the WCS updated by the offsets.

need to convert offest degree to arcsec. divde by plate scale 

get crval1 and crval2 from header
apply offset to crval1 and crval2


"""


def offset_wcs_2d(musefile, ra_offset_deg, dec_offset_deg):
    # Convert offsets to arcsec and then to pixels
    ra_offset_arcsec = ra_offset_deg * 3600
    dec_offset_arcsec = dec_offset_deg * 3600
    pixel_scale = 0.2  # arcsec/pixel

    dx = ra_offset_arcsec / pixel_scale  # RA → X axis
    dy = dec_offset_arcsec / pixel_scale  # Dec → Y axis

    with fits.open(musefile, ignore_missing_simple=True) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data

        # Use only celestial WCS (RA/Dec)
        wcs = WCS(hdr).celestial
        wcs.wcs.crpix[0] -= dx
        wcs.wcs.crpix[1] += dy  # Subtract for consistency in pixel space

        # Build clean 2D WCS header
        new_wcs_header = wcs.to_header()

        # Remove old WCS keywords to avoid 3D confusion
        for key in list(hdr.keys()):
            if key.startswith(('CTYPE3', 'CRVAL3', 'CRPIX3', 'CDELT3', 'CD3_', 'PC3_', 'NAXIS3')):
                del hdr[key]

        # Replace WCS keywords with new 2D ones
        for key in new_wcs_header:
            hdr[key] = new_wcs_header[key]

        # Save new file
        output_dir = 'aligned_vis'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            os.path.basename(musefile).replace('.fits', '_aligned.fits')
        )
        fits.writeto(output_path, data, hdr, overwrite=True)
        print(f"Aligned file saved to: {output_path}")


"""
Muse to hst
Right = +ve ra 
Left = -ve ra 

Up = -ve dec
Down = +ve dec
"""


offset_wcs_2d('MUSE_NEW/images/DATACUBE_FINAL_Autocal3821786b_1_ZAP_img.fits', ra_offset_deg=0.00004, dec_offset_deg=0.00018) # Change the file and shifts




