import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from mpdaf.obj import Cube, Image
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground


def mkcol(b, v, r, ff, gamma, xlo, xhi, ylo, yhi):
    # (keep function exactly as your supervisor sent)
    tmpb = b[ylo:yhi, xlo:xhi]
    tmpv = v[ylo:yhi, xlo:xhi]
    tmpr = r[ylo:yhi, xlo:xhi]

    bmean,bmedian,bstd,bmax = getstats(tmpb, ff)
    vmean,vmedian,vstd,vmax = getstats(tmpv, ff)
    rmean,rmedian,rstd,rmax = getstats(tmpr, ff)

    print('rescaling...')
    bmin, vmin, rmin = bmean, vmean, rmean

    gdb = np.where((b != 0) & (b != np.nan))
    gdv = np.where((v != 0) & (v != np.nan))
    gdr = np.where((r != 0) & (r != np.nan))

    b[gdb] = (b[gdb]-bmin)/(bmax-bmin)
    v[gdv] = (v[gdv]-vmin)/(vmax-vmin)
    r[gdr] = (r[gdr]-rmin)/(rmax-rmin)

    lo, hi = 0., 1.
    b[b <= lo] = 0; b[b >= hi] = 1
    v[v <= lo] = 0; v[v >= hi] = 1
    r[r <= lo] = 0; r[r >= hi] = 1

    b = b**gamma; v = v**gamma; r = r**gamma

    print('writing to array')
    sz = b.shape
    col = np.zeros((sz[0], sz[1], 3))
    col[:,:,0] = r
    col[:,:,1] = v
    col[:,:,2] = b
    print('mkcol complete.')
    return col

def getstats(img,ff):
   #from photutils import make_source_mask
   gd = np.where((img != 0) & np.isfinite(img))
   print('there are ',len(gd[0]),' elements')
   arr = img[gd]
   arr = sorted(arr)
   n = len(arr)
   print('array is ',n,' elements')
   i = round(ff*n)
   vmax = arr[i]
   print(ff,' signal range value is ',vmax)

   print('making mask')
   #mask = make_source_mask(img, nsigma=2, npixels=5, dilate_size=11)
   mask = make_source_mask(img)
   print('calculating stats')
   vmean, vmedian, vstd = sigma_clipped_stats(img, sigma=3.0, mask=mask, mask_value=0)
   print('mean: ',vmean)
   print('median: ',vmedian)
   print('sigma: ',vstd)
   return vmean,vmedian,vstd,vmax

def make_source_mask(img):

   bkg_estimator = MedianBackground()
   
   bkg = Background2D(img, (50, 50), filter_size=(3, 3),bkg_estimator=bkg_estimator)
   imgb = img.copy()
   imgb = img - bkg.background  # subtract the background
   #amstools.writefits_nohdr('imgb.fits',imgb)
   threshold = 2 * bkg.background_rms
   segment_map = detect_sources(imgb, threshold, npixels=10)
   segment_map = segment_map.data
   #segment_map = segment_map.astype(float)
   bad = np.where(segment_map >= 1)
   gd = np.where(segment_map == 0)
   segment_map[bad] = True
   segment_map[gd] = False
   #amstools.writefits_nohdr('seg.fits',segment_map)
   return(segment_map)

def wl(fits_path, out_fits, out_png, out_png_rgb):
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[0].data  
        header = hdul[0].header
        data = data[np.r_[0:800, 1000:3680], :, :]
        header_f = hdul[1].header
        print("Full header info:")  
        print(repr(header_f))

        
        white_data = np.sum(data, axis=0, dtype=np.float32)
        blue = np.sum(data[0:843, :, :], axis=0)
        green = np.sum(data[843:843+1175, :, :], axis=0)
        red = np.sum(data[843+1175:, :, :], axis=0)

    hdu = fits.PrimaryHDU(data=white_data, header=header)
    hdu.writeto(out_fits, overwrite=True)
    print(f"White-light image saved to {out_fits}")

    # Make white-light plot
    plt.figure(figsize=(8, 6))
    norm = simple_norm(white_data, 'sqrt', percent=99)
    plt.imshow(white_data, origin='lower', cmap='viridis', norm=norm)
    plt.colorbar(label='Flux')
    plt.title("White-light Image from FITS Cube")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"White-light image plot saved to {out_png}")

    # Create RGB composite
    print("Creating color image with mkcol...")
    ff = 0.99
    gamma = 0.5
    xlo, xhi, ylo, yhi = 0, white_data.shape[1], 0, white_data.shape[0]
    col = mkcol(blue.copy(), green.copy(), red.copy(), ff, gamma, xlo, xhi, ylo, yhi)

    plt.figure(figsize=(8, 6))
    plt.imshow(col, origin='lower')
    plt.title("MUSE RGB Composite")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.savefig(out_png_rgb, dpi=300)
    plt.close()
    print(f"RGB composite image saved to {out_png_rgb}")



if __name__ == "__main__":
    cube_path = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits"
    out_fits = "/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_wl.fits"
    out_png  = "/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_wl_image.png"
    out_png_rgb = "/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_rgb.png"

    wl(cube_path, out_fits, out_png, out_png_rgb)



#subcube = mcube[:, 500:1500, 500:1500]
# (wavelength, y, x)
#print ("Subcube 500:1500,500:1500 extracted")
#subcube.info()
#subcube.write("/cephfs/apatrick/musecosmos/dataproducts/extractions/subcube_500_1500_500_1500.fits")

"""

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