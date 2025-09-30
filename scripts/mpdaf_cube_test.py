
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord

from mpdaf.obj import Cube
from mpdaf.obj import Image

 
#mcube_path = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits"
#mcube = Cube(mcube_path, ext=1, memmap=True)
#print ("Full Cube Object Loaded")

#subcube = mcube[:, 500:1500, 500:1500]
# (wavelength, y, x)
#print ("Subcube 500:1500,500:1500 extracted")
#subcube.info()
#subcube.write("/cephfs/apatrick/musecosmos/dataproducts/extractions/subcube_500_1500_500_1500.fits")



# Load subcube from file to verify
#subcube = Cube("/cephfs/apatrick/musecosmos/dataproducts/extractions/subcube_500_1500_500_1500.fits", ext=1)
##print ("Subcube re-loaded from file")
#subcube.info()

# save primary and extension headers to text file
with open("/cephfs/apatrick/musecosmos/dataproducts/extractions/new_headers.txt", "w") as f:
    with fits.open("/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits") as hdul:
        f.write("Primary Header:\n")
        f.write(repr(hdul[0].header))
        f.write("\n\nExtension Header:\n")
        f.write(repr(hdul[1].header))   
print (f"Subcube headers saved to /cephfs/apatrick/musecosmos/dataproducts/extractions/new_headers.txt")


#wcs = subcube.wcs


""" 
# White-light image (sum over all wavelengths to get 2D image)
white_img = subcube.sum(axis=0)  # axis=0 = wavelength axis
plt.figure(figsize=(8, 6))
plt.imshow(white_img.data, origin='lower', cmap='viridis')
plt.colorbar(label='Flux')
plt.title("White-light Subcube Image")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.savefig("/cephfs/apatrick/musecosmos/dataproducts/extractions/wl_subcube_image.png", dpi=300)
print (f"White-light subcube image saved to /cephfs/apatrick/musecosmos/dataproducts/extractions/wl_subcube_image.png")
plt.close()

# 2. Full-cube 1D spectrum (sum over all spatial pixels to get 1D spectrum)
full_spectrum = subcube.sum(axis=(1, 2))  # collapse x and y
plt.figure(figsize=(8, 4))
plt.plot(full_spectrum.wave.coord(), full_spectrum.data, color='k', lw=1)
plt.xlabel("Wavelength [Å]")
plt.ylabel("Flux")
plt.title("Full-cube Spectrum")
plt.grid(alpha=0.3)
plt.savefig("/cephfs/apatrick/musecosmos/dataproducts/extractions/full_subcube_spectrum.png", dpi=300)
print (f"Full-cube spectrum saved to /cephfs/apatrick/musecosmos/dataproducts/extractions/full_subcube_spectrum.png")
plt.close()

"""